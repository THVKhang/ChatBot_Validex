"""Analytics module — aggregates system metrics from PostgreSQL."""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def _get_dsn() -> str | None:
    return os.environ.get("DATABASE_URL", "").strip() or None


def fetch_token_analytics(days: int = 30) -> dict[str, Any]:
    """Fetch token usage grouped by model and day."""
    dsn = _get_dsn()
    if not dsn:
        return {"status": "no_db", "data": []}

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS token_usage (
                        id SERIAL PRIMARY KEY,
                        model TEXT NOT NULL,
                        input_tokens INT DEFAULT 0,
                        output_tokens INT DEFAULT 0,
                        request_count INT DEFAULT 0,
                        usage_date DATE DEFAULT CURRENT_DATE,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                conn.commit()

                since = (datetime.utcnow() - timedelta(days=days)).date()
                cur.execute("""
                    SELECT usage_date, model, SUM(input_tokens), SUM(output_tokens), SUM(request_count)
                    FROM token_usage
                    WHERE usage_date >= %s
                    GROUP BY usage_date, model
                    ORDER BY usage_date DESC
                """, (since,))
                rows = cur.fetchall()
                return {
                    "status": "ok",
                    "data": [
                        {
                            "date": str(r[0]),
                            "model": r[1],
                            "input_tokens": r[2],
                            "output_tokens": r[3],
                            "request_count": r[4],
                        }
                        for r in rows
                    ],
                }
    except Exception as exc:
        logger.error("fetch_token_analytics failed: %s", exc)
        return {"status": "error", "data": [], "error": str(exc)}


def fetch_quality_stats(days: int = 30) -> dict[str, Any]:
    """Fetch editor verdict distribution from A/B test logs."""
    dsn = _get_dsn()
    if not dsn:
        return {"status": "no_db", "data": {}}

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                since = (datetime.utcnow() - timedelta(days=days)).isoformat()
                cur.execute("""
                    SELECT editor_verdict, COUNT(*) 
                    FROM prompt_ab_tests
                    WHERE created_at >= %s
                    GROUP BY editor_verdict
                    ORDER BY COUNT(*) DESC
                """, (since,))
                rows = cur.fetchall()
                return {
                    "status": "ok",
                    "data": {r[0]: r[1] for r in rows},
                }
    except Exception as exc:
        logger.error("fetch_quality_stats failed: %s", exc)
        return {"status": "error", "data": {}, "error": str(exc)}


def fetch_cache_stats() -> dict[str, Any]:
    """Fetch semantic cache statistics."""
    dsn = _get_dsn()
    if not dsn:
        return {"status": "no_db"}

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM validex_semantic_cache")
                total = cur.fetchone()[0]
                return {"status": "ok", "total_cached": total}
    except Exception as exc:
        logger.error("fetch_cache_stats failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def fetch_feedback_stats() -> dict[str, Any]:
    """Fetch user feedback statistics (thumbs up/down)."""
    dsn = _get_dsn()
    if not dsn:
        return {"status": "no_db"}

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS blog_feedback (
                        id SERIAL PRIMARY KEY,
                        report_id TEXT NOT NULL,
                        user_id INT,
                        rating INT CHECK (rating IN (1, -1)),
                        comment TEXT DEFAULT '',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                conn.commit()

                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE rating = 1) AS thumbs_up,
                        COUNT(*) FILTER (WHERE rating = -1) AS thumbs_down,
                        COUNT(*) AS total
                    FROM blog_feedback
                """)
                row = cur.fetchone()
                return {
                    "status": "ok",
                    "thumbs_up": row[0],
                    "thumbs_down": row[1],
                    "total": row[2],
                }
    except Exception as exc:
        logger.error("fetch_feedback_stats failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def save_feedback(report_id: str, rating: int, comment: str = "", user_id: int | None = None) -> bool:
    """Save user feedback for a report."""
    dsn = _get_dsn()
    if not dsn:
        return False

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS blog_feedback (
                        id SERIAL PRIMARY KEY,
                        report_id TEXT NOT NULL,
                        user_id INT,
                        rating INT CHECK (rating IN (1, -1)),
                        comment TEXT DEFAULT '',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute(
                    "INSERT INTO blog_feedback (report_id, user_id, rating, comment) VALUES (%s, %s, %s, %s)",
                    (report_id, user_id, rating, comment),
                )
            conn.commit()
        return True
    except Exception as exc:
        logger.error("save_feedback failed: %s", exc)
        return False
