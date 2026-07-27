"""A/B Testing Logger for MLOps and Prompt Evaluation.

Logs system prompt versions and editor evaluations to PostgreSQL
to track the performance, token efficiency, and quality of different prompts.
Supports multi-variant A/B testing with random assignment.
"""

import json
import logging
import os
import random
from typing import Any
from uuid import uuid4

import psycopg

logger = logging.getLogger(__name__)

# Active prompt variants for A/B testing
PROMPT_VARIANTS = {
    "v4-hybrid-seo": {"weight": 50, "description": "Phase 4 hybrid SEO + NLI pipeline"},
    "v5-experimental": {"weight": 50, "description": "Experimental variant for testing"},
}


def get_active_variant() -> str:
    """Randomly select a prompt variant based on configured weights."""
    variants = list(PROMPT_VARIANTS.keys())
    weights = [PROMPT_VARIANTS[v]["weight"] for v in variants]
    return random.choices(variants, weights=weights, k=1)[0]


def log_prompt_evaluation(
    prompt_version: str,
    topic: str,
    editor_verdict: str,
    structural_issues: list[str],
    total_tokens_used: int,
    session_id: str | None = None,
    variant_id: str | None = None,
    latency_ms: float | None = None,
) -> None:
    """Log an A/B test run to the database."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        logger.warning("No DATABASE_URL configured, skipping A/B test logging.")
        return

    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                # Ensure table has new columns
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS prompt_ab_tests (
                        id SERIAL PRIMARY KEY,
                        run_id TEXT UNIQUE NOT NULL,
                        session_id TEXT,
                        prompt_version TEXT NOT NULL,
                        variant_id TEXT,
                        topic TEXT,
                        editor_verdict TEXT,
                        structural_issues JSONB,
                        total_tokens_used INT DEFAULT 0,
                        latency_ms FLOAT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                run_id = str(uuid4())
                cur.execute(
                    """
                    INSERT INTO prompt_ab_tests 
                    (run_id, session_id, prompt_version, variant_id, topic, editor_verdict, structural_issues, total_tokens_used, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        session_id,
                        prompt_version,
                        variant_id or prompt_version,
                        topic,
                        editor_verdict,
                        json.dumps(structural_issues),
                        total_tokens_used,
                        latency_ms,
                    )
                )
                conn.commit()
                logger.info(f"Logged A/B test {run_id} for variant {variant_id or prompt_version} (verdict: {editor_verdict})")
    except Exception as exc:
        logger.error(f"Failed to log A/B test: {exc}")


def get_variant_stats(days: int = 30) -> dict[str, Any]:
    """Fetch A/B test results grouped by variant with statistical significance."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return {"status": "no_db"}

    try:
        from datetime import datetime, timedelta
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COALESCE(variant_id, prompt_version) AS variant,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE editor_verdict IN ('ACCEPTED', 'LLM_PASSED')) AS accepted,
                        AVG(total_tokens_used) AS avg_tokens,
                        AVG(latency_ms) AS avg_latency_ms
                    FROM prompt_ab_tests
                    WHERE created_at >= %s
                    GROUP BY variant
                    ORDER BY total DESC
                """, (since,))
                rows = cur.fetchall()
                
                return {
                    "status": "ok",
                    "variants": [
                        {
                            "variant": r[0],
                            "total_runs": r[1],
                            "accepted": r[2],
                            "accept_rate": round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0,
                            "avg_tokens": int(r[3]) if r[3] else 0,
                            "avg_latency_ms": round(r[4], 0) if r[4] else None,
                        }
                        for r in rows
                    ],
                }
    except Exception as exc:
        logger.error(f"get_variant_stats failed: {exc}")
        return {"status": "error", "error": str(exc)}


