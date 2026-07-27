"""Redis-backed session store with in-memory fallback.

When Redis is available (``USE_REDIS_SESSIONS=1`` and ``REDIS_URL`` set),
sessions are stored in Redis with automatic TTL expiration.
If Redis is unavailable, falls back to the legacy in-memory dict.

Session data is JSON-serialized for portability across FastAPI instances.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from app.config import settings
from app.session_manager import SessionManager
from app.session_store import _turns_to_json, _json_to_turns

logger = logging.getLogger(__name__)

_USE_REDIS_SESSIONS = os.getenv("USE_REDIS_SESSIONS", "0") == "1"

# ── Redis client singleton ───────────────────────────────
_redis = None
_redis_lock = threading.Lock()
_redis_init_attempted = False


def _init_redis():
    """Lazily create the Redis client."""
    global _redis, _redis_init_attempted
    if _redis_init_attempted:
        return _redis

    if not _USE_REDIS_SESSIONS:
        logger.info("redis_session: Redis sessions disabled (USE_REDIS_SESSIONS=0).")
        _redis_init_attempted = True
        return None

    redis_url = settings.redis_url
    if not redis_url:
        logger.warning("redis_session: REDIS_URL not set — falling back to in-memory sessions.")
        _redis_init_attempted = True
        return None

    try:
        import redis as redis_mod
        client = redis_mod.Redis.from_url(redis_url, decode_responses=True, socket_timeout=3)
        client.ping()
        logger.info("redis_session: Connected to Redis at %s", redis_url.split("@")[-1] if "@" in redis_url else redis_url)
        _redis = client
    except ImportError:
        logger.warning("redis_session: redis package not installed — using in-memory sessions.")
    except Exception as exc:
        logger.warning("redis_session: Cannot connect to Redis (%s) — using in-memory sessions.", exc)

    _redis_init_attempted = True
    return _redis


def _get_redis():
    """Return Redis client or None."""
    global _redis
    if _redis is not None:
        return _redis
    with _redis_lock:
        return _init_redis()


def check_redis_connectivity() -> dict[str, Any]:
    """Ping Redis for health check."""
    r = _get_redis()
    if r is None:
        return {"status": "disabled" if not _USE_REDIS_SESSIONS else "unavailable"}
    try:
        r.ping()
        info = r.info("memory")
        return {
            "status": "ok",
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "connected_clients": r.info("clients").get("connected_clients", 0),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ── Session Serialization ────────────────────────────────
_SESSION_PREFIX = "session:"


def _serialize_session(session: SessionManager) -> str:
    """Serialize a SessionManager to JSON string."""
    return _turns_to_json(session)


def _deserialize_session(data: str) -> SessionManager:
    """Deserialize JSON string back to SessionManager."""
    turns = _json_to_turns(data)
    session = SessionManager()
    session.turns = turns
    return session


# ── RedisSessionStore ────────────────────────────────────
class RedisSessionStore:
    """Session store with Redis primary and in-memory fallback.

    When Redis is available, sessions expire automatically via Redis TTL.
    When Redis is unavailable, sessions are stored in a local dict with
    manual TTL eviction.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = max(60, ttl_seconds)
        # Fallback in-memory store
        self._memory_store: dict[str, tuple[SessionManager, float]] = {}
        self._memory_lock = threading.Lock()

    @property
    def _redis(self):
        return _get_redis()

    @property
    def is_redis_active(self) -> bool:
        return self._redis is not None

    def get(self, session_id: str) -> SessionManager | None:
        """Retrieve a session by ID."""
        r = self._redis
        if r is not None:
            try:
                data = r.get(f"{_SESSION_PREFIX}{session_id}")
                if data:
                    # Refresh TTL on access
                    r.expire(f"{_SESSION_PREFIX}{session_id}", self.ttl)
                    return _deserialize_session(data)
                return None
            except Exception as exc:
                logger.warning("redis_session.get error: %s", exc)

        # Fallback: in-memory
        with self._memory_lock:
            entry = self._memory_store.get(session_id)
            if entry:
                session, ts = entry
                if time.time() - ts > self.ttl:
                    del self._memory_store[session_id]
                    return None
                self._memory_store[session_id] = (session, time.time())
                return session
        return None

    def set(self, session_id: str, session: SessionManager) -> None:
        """Store or update a session."""
        r = self._redis
        if r is not None:
            try:
                data = _serialize_session(session)
                r.setex(f"{_SESSION_PREFIX}{session_id}", self.ttl, data)
                return
            except Exception as exc:
                logger.warning("redis_session.set error: %s", exc)

        # Fallback: in-memory
        with self._memory_lock:
            self._memory_store[session_id] = (session, time.time())

    def delete(self, session_id: str) -> None:
        """Remove a session."""
        r = self._redis
        if r is not None:
            try:
                r.delete(f"{_SESSION_PREFIX}{session_id}")
            except Exception:
                pass

        with self._memory_lock:
            self._memory_store.pop(session_id, None)

    def evict_stale(self) -> int:
        """Remove expired sessions from in-memory store. Returns count evicted.

        Redis sessions expire automatically via TTL — this only handles the fallback.
        """
        if self._redis is not None:
            return 0  # Redis handles TTL natively

        now = time.time()
        evicted = 0
        with self._memory_lock:
            stale = [sid for sid, (_, ts) in self._memory_store.items() if now - ts > self.ttl]
            for sid in stale:
                del self._memory_store[sid]
                evicted += 1

        if evicted:
            logger.info("redis_session: Evicted %d stale in-memory sessions.", evicted)
        return evicted

    def count(self) -> int:
        """Return approximate session count."""
        r = self._redis
        if r is not None:
            try:
                keys = r.keys(f"{_SESSION_PREFIX}*")
                return len(keys)
            except Exception:
                pass

        with self._memory_lock:
            return len(self._memory_store)

    def stats(self) -> dict[str, Any]:
        """Return store statistics for health endpoint."""
        return {
            "backend": "redis" if self.is_redis_active else "memory",
            "session_count": self.count(),
            "ttl_seconds": self.ttl,
        }
