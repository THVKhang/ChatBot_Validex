"""Token Usage Tracker — tracks LLM token consumption per model per day.

Persists usage data to PostgreSQL for dashboard visualization and
budget-aware routing decisions.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelUsage:
    """Tracks token usage for a single model within a time window."""
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    error_count: int = 0
    last_request_at: float = 0.0


class TokenTracker:
    """Thread-safe token usage tracker with DB persistence."""

    def __init__(self, daily_limits: dict[str, int] | None = None) -> None:
        self._lock = threading.Lock()
        # { "model_name": ModelUsage }
        self._usage: dict[str, ModelUsage] = defaultdict(ModelUsage)
        self._current_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Configurable daily limits per model
        self._daily_limits: dict[str, int] = daily_limits or {
            "llama-3.3-70b-versatile": 100_000,
            "models/gemini-2.5-flash": 1_000_000,  # Gemini is much more generous
        }

    def _reset_if_new_day(self) -> None:
        """Reset counters if a new UTC day has started."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            logger.info("Token tracker: new day detected, resetting counters")
            self._flush_to_db()  # Save yesterday's data first
            self._usage.clear()
            self._current_date = today

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        is_error: bool = False,
    ) -> None:
        """Record token usage for a model."""
        with self._lock:
            self._reset_if_new_day()
            usage = self._usage[model]
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.request_count += 1
            usage.last_request_at = time.time()
            if is_error:
                usage.error_count += 1

    def get_remaining_tokens(self, model: str) -> int:
        """Get estimated remaining tokens for a model today."""
        with self._lock:
            self._reset_if_new_day()
            usage = self._usage.get(model)
            if not usage:
                return self._daily_limits.get(model, 100_000)
            total_used = usage.input_tokens + usage.output_tokens
            limit = self._daily_limits.get(model, 100_000)
            return max(0, limit - total_used)

    def get_usage_percent(self, model: str) -> float:
        """Get percentage of daily budget consumed (0.0 - 1.0)."""
        with self._lock:
            self._reset_if_new_day()
            usage = self._usage.get(model)
            if not usage:
                return 0.0
            total_used = usage.input_tokens + usage.output_tokens
            limit = self._daily_limits.get(model, 100_000)
            if limit <= 0:
                return 1.0
            return min(1.0, total_used / limit)

    def should_use_fallback(self, model: str, threshold: float = 0.85) -> bool:
        """Check if the model has exceeded the budget threshold."""
        return self.get_usage_percent(model) >= threshold

    def get_dashboard_data(self) -> dict[str, Any]:
        """Return data for the admin dashboard."""
        with self._lock:
            self._reset_if_new_day()
            models = []
            for model_name, usage in self._usage.items():
                limit = self._daily_limits.get(model_name, 100_000)
                total_used = usage.input_tokens + usage.output_tokens
                models.append({
                    "model": model_name,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": total_used,
                    "daily_limit": limit,
                    "usage_percent": round(min(1.0, total_used / limit) * 100, 1) if limit > 0 else 100.0,
                    "remaining_tokens": max(0, limit - total_used),
                    "request_count": usage.request_count,
                    "error_count": usage.error_count,
                    "last_request_at": datetime.fromtimestamp(usage.last_request_at, tz=timezone.utc).isoformat() if usage.last_request_at else None,
                })

            # Include models that haven't been used yet
            for model_name, limit in self._daily_limits.items():
                if model_name not in self._usage:
                    models.append({
                        "model": model_name,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "daily_limit": limit,
                        "usage_percent": 0.0,
                        "remaining_tokens": limit,
                        "request_count": 0,
                        "error_count": 0,
                        "last_request_at": None,
                    })

            return {
                "date": self._current_date,
                "models": models,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    def _flush_to_db(self) -> None:
        """Persist current usage data to PostgreSQL."""
        dsn = os.getenv("DATABASE_URL", "").strip()
        if not dsn:
            return

        try:
            import psycopg
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS token_usage_log (
                            id SERIAL PRIMARY KEY,
                            date TEXT NOT NULL,
                            model TEXT NOT NULL,
                            input_tokens INT NOT NULL DEFAULT 0,
                            output_tokens INT NOT NULL DEFAULT 0,
                            request_count INT NOT NULL DEFAULT 0,
                            error_count INT NOT NULL DEFAULT 0,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                    """)
                    for model_name, usage in self._usage.items():
                        cur.execute(
                            """INSERT INTO token_usage_log (date, model, input_tokens, output_tokens, request_count, error_count)
                               VALUES (%s, %s, %s, %s, %s, %s)""",
                            (self._current_date, model_name, usage.input_tokens,
                             usage.output_tokens, usage.request_count, usage.error_count),
                        )
                conn.commit()
            logger.info("Token usage flushed to database for %s", self._current_date)
        except Exception as exc:
            logger.warning("Failed to flush token usage to DB: %s", exc)

    def get_historical_data(self, days: int = 7) -> list[dict[str, Any]]:
        """Fetch historical token usage from the database."""
        dsn = os.getenv("DATABASE_URL", "").strip()
        if not dsn:
            return []

        try:
            import psycopg
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT date, model, SUM(input_tokens)::int, SUM(output_tokens)::int,
                               SUM(request_count)::int, SUM(error_count)::int
                        FROM token_usage_log
                        WHERE created_at > NOW() - INTERVAL '1 day' * %s
                        GROUP BY date, model
                        ORDER BY date DESC, model
                    """, (days,))
                    rows = cur.fetchall()
                    return [
                        {
                            "date": row[0],
                            "model": row[1],
                            "input_tokens": row[2],
                            "output_tokens": row[3],
                            "request_count": row[4],
                            "error_count": row[5],
                        }
                        for row in rows
                    ]
        except Exception as exc:
            logger.warning("Failed to fetch historical token data: %s", exc)
            return []

    # ── Per-User Token Quota ──────────────────────────────────────
    # Tracks token usage per user (session_id) per day for quota enforcement.

    def _ensure_user_table(self, conn: Any) -> None:
        """Create user_token_usage table if it doesn't exist."""
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_token_usage (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    tokens_used INT NOT NULL DEFAULT 0,
                    request_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(user_id, date)
                )
            """)
            conn.commit()

    def record_user_usage(self, user_id: str, tokens: int) -> None:
        """Record token usage for a specific user (upsert daily)."""
        if not user_id:
            return
        dsn = os.getenv("DATABASE_URL", "").strip()
        if not dsn:
            # In-memory fallback
            with self._lock:
                key = f"user:{user_id}"
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if not hasattr(self, "_user_usage"):
                    self._user_usage: dict[str, dict] = {}
                record = self._user_usage.get(key)
                if not record or record.get("date") != today:
                    self._user_usage[key] = {"date": today, "tokens": 0, "requests": 0}
                self._user_usage[key]["tokens"] += tokens
                self._user_usage[key]["requests"] += 1
            return

        try:
            import psycopg
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with psycopg.connect(dsn) as conn:
                self._ensure_user_table(conn)
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO user_token_usage (user_id, date, tokens_used, request_count)
                        VALUES (%s, %s, %s, 1)
                        ON CONFLICT (user_id, date) DO UPDATE SET
                            tokens_used = user_token_usage.tokens_used + EXCLUDED.tokens_used,
                            request_count = user_token_usage.request_count + 1
                    """, (user_id, today, tokens))
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to record user token usage: %s", exc)

    def get_user_budget(self, user_id: str, tier: str = "free") -> dict[str, Any]:
        """Get remaining token budget for a user today.

        Returns: {remaining, total, used, percent, tier, requests}
        """
        from app.config import settings
        quota = settings.user_tier_quotas.get(tier, settings.user_tier_quotas["free"])
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        used = 0
        requests = 0

        dsn = os.getenv("DATABASE_URL", "").strip()
        if dsn:
            try:
                import psycopg
                with psycopg.connect(dsn) as conn:
                    self._ensure_user_table(conn)
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT tokens_used, request_count FROM user_token_usage WHERE user_id = %s AND date = %s",
                            (user_id, today),
                        )
                        row = cur.fetchone()
                        if row:
                            used, requests = row[0], row[1]
            except Exception as exc:
                logger.warning("Failed to get user budget from DB: %s", exc)
        else:
            # In-memory fallback
            with self._lock:
                key = f"user:{user_id}"
                if hasattr(self, "_user_usage"):
                    record = self._user_usage.get(key, {})
                    if record.get("date") == today:
                        used = record.get("tokens", 0)
                        requests = record.get("requests", 0)

        remaining = max(0, quota - used)
        percent = round(used / quota * 100, 1) if quota > 0 else 100.0

        return {
            "remaining": remaining,
            "total": quota,
            "used": used,
            "percent": percent,
            "tier": tier,
            "requests": requests,
            "date": today,
        }

    def check_user_quota(self, user_id: str, tier: str = "free") -> bool:
        """Check if user has remaining quota. Returns True if OK, False if exceeded."""
        budget = self.get_user_budget(user_id, tier)
        return budget["remaining"] > 0


# Module-level singleton
token_tracker = TokenTracker()

