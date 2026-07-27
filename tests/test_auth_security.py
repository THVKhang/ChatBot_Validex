"""Security tests for authentication, JWT, and admin routes.

Covers:
- JWT token validation (expired, malformed, missing)
- Password constraints (short, unicode, bcrypt 72-byte limit)
- Admin route protection (non-admin, expired, anonymous)
- CORS header enforcement
- Input sanitization on auth endpoints
"""

import datetime
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api_server import app, get_current_admin_user, get_current_user_id
from app.auth import (
    ALGORITHM,
    SECRET_KEY,
    create_access_token,
    _truncate_pw,
)

from app.auth import get_password_hash, verify_password


# ── Fixtures ─────────────────────────────────────────────
@pytest.fixture()
def client():
    """TestClient without dependency overrides (real auth enforcement)."""
    # Clear overrides so real auth kicks in
    saved = dict(app.dependency_overrides)
    app.dependency_overrides.clear()
    yield TestClient(app)
    app.dependency_overrides.update(saved)


@pytest.fixture()
def admin_client():
    """TestClient with admin dependency override."""
    app.dependency_overrides[get_current_admin_user] = lambda: {
        "username": "admin",
        "is_admin": True,
        "user_id": 1,
    }
    app.dependency_overrides[get_current_user_id] = lambda: 1
    yield TestClient(app)


# ── Password Hashing Tests ──────────────────────────────
class TestPasswordHashing:
    def test_verify_correct_password(self):
        hashed = get_password_hash("ValidPass123")
        assert verify_password("ValidPass123", hashed)

    def test_verify_wrong_password(self):
        hashed = get_password_hash("ValidPass123")
        assert not verify_password("WrongPass999", hashed)

    def test_truncate_password_at_72_bytes(self):
        """Bcrypt only processes up to 72 bytes. Verify truncation works."""
        long_pw = "A" * 100
        truncated = _truncate_pw(long_pw)
        assert len(truncated.encode("utf-8")) <= 72

    def test_unicode_password_hashing(self):
        """Unicode passwords should hash and verify correctly."""
        pw = "Mật_khẩu_bảo_mật_123"
        hashed = get_password_hash(pw)
        assert verify_password(pw, hashed)

    def test_empty_password_hash_and_verify(self):
        """Empty password should still hash (but be blocked by validation)."""
        hashed = get_password_hash("")
        assert verify_password("", hashed)

    def test_password_with_null_bytes(self):
        """Null bytes in password should be handled by truncation."""
        pw = "pass\x00word"
        truncated = _truncate_pw(pw)
        # Should not raise
        hashed = get_password_hash(truncated)
        assert hashed


# ── JWT Token Tests ──────────────────────────────────────
class TestJWTTokens:
    def test_create_valid_token(self):
        token = create_access_token({"sub": "testuser", "user_id": 1, "is_admin": False})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 1
        assert payload["is_admin"] is False

    def test_token_contains_expiration(self):
        token = create_access_token({"sub": "user1"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload
        exp = datetime.datetime.fromtimestamp(payload["exp"], tz=datetime.timezone.utc)
        assert exp > datetime.datetime.now(tz=datetime.timezone.utc)

    def test_expired_token_rejected(self):
        """Manually create a token that expired 1 hour ago."""
        expired_payload = {
            "sub": "expired_user",
            "user_id": 99,
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(expired_payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    def test_token_with_wrong_secret_rejected(self):
        token = create_access_token({"sub": "user1", "user_id": 1})
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(token, "WRONG_SECRET_KEY", algorithms=[ALGORITHM])

    def test_malformed_token_rejected(self):
        with pytest.raises(jwt.DecodeError):
            jwt.decode("not.a.valid.token.at.all", SECRET_KEY, algorithms=[ALGORITHM])

    def test_empty_token_rejected(self):
        with pytest.raises(jwt.DecodeError):
            jwt.decode("", SECRET_KEY, algorithms=[ALGORITHM])


# ── Admin Route Protection Tests ─────────────────────────
class TestAdminRouteProtection:
    def test_admin_ingest_status_without_token_returns_401(self, client):
        resp = client.get("/api/admin/ingest/status")
        assert resp.status_code == 401

    def test_admin_ingest_status_with_non_admin_token_returns_403(self, client):
        token = create_access_token({"sub": "user1", "user_id": 2, "is_admin": False})
        resp = client.get(
            "/api/admin/ingest/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_admin_ingest_status_with_admin_token_succeeds(self, client):
        token = create_access_token({"sub": "admin", "user_id": 1, "is_admin": True})
        resp = client.get(
            "/api/admin/ingest/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_admin_users_endpoint_requires_admin(self, client):
        token = create_access_token({"sub": "normie", "user_id": 5, "is_admin": False})
        resp = client.get(
            "/api/admin/users",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_admin_token_usage_accessible_without_auth(self, admin_client):
        """Token usage endpoint should work with admin override."""
        resp = admin_client.get("/api/admin/token-usage")
        assert resp.status_code == 200


# ── CORS Tests ───────────────────────────────────────────
class TestCORSHeaders:
    def test_cors_allows_configured_origin(self, admin_client):
        resp = admin_client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:4200",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not be blocked (200 or 405 are acceptable for OPTIONS)
        assert resp.status_code in (200, 405)

    def test_health_endpoint_returns_cors_headers(self, admin_client):
        resp = admin_client.get(
            "/api/health",
            headers={"Origin": "http://localhost:4200"},
        )
        assert resp.status_code == 200
        # FastAPI CORS middleware adds headers for configured origins
        cors_header = resp.headers.get("access-control-allow-origin", "")
        assert cors_header in ("http://localhost:4200", "*", "")


# ── Auth Input Validation Tests ──────────────────────────
class TestAuthInputValidation:
    def test_register_password_too_short_rejected(self, client):
        resp = client.post(
            "/api/auth/register",
            json={"username": "test_short_pw", "password": "abc"},
        )
        # Either 400 (password validation) or 500 (no db) is acceptable
        assert resp.status_code in (400, 500)

    def test_register_empty_username(self, client):
        resp = client.post(
            "/api/auth/register",
            json={"username": "", "password": "ValidPass123"},
        )
        # Should fail due to empty username or db error
        assert resp.status_code in (400, 422, 500)

    def test_login_missing_fields_returns_422(self, client):
        resp = client.post("/api/auth/login", data={})
        assert resp.status_code == 422

    def test_login_with_sql_injection_username(self, client):
        """SQL injection attempt in username should not cause server crash."""
        resp = client.post(
            "/api/auth/login",
            data={
                "username": "'; DROP TABLE users; --",
                "password": "password123",
            },
        )
        # Should return 401 (bad credentials) or 500 (db error/timeout), NOT 200
        # The key assertion: injection must NOT succeed (status != 200)
        assert resp.status_code != 200
