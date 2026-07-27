from __future__ import annotations

from datetime import date
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any

from app.collect_au_sources import collect_sources
from app.config import settings
from app.ingest_pgvector import ingest_jsonl_to_pgvector


LOG_FILE = Path("logs/ingestion.log")


def _build_logger(log_file: Path = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger("ingestion_worker")
    if logger.handlers:
        return logger

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def is_scheduled_window(now: datetime, last_run_date: date | None) -> bool:
    if now.weekday() != 6:  # Sunday
        return False
    if now.hour != 2:
        return False
    if last_run_date == now.date():
        return False
    return True


def _contains_blocking_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return "403" in lowered or "forbidden" in lowered or "timeout" in lowered


def run_ingestion_job(logger: logging.Logger | None = None) -> dict[str, Any]:
    active_logger = logger or _build_logger()
    active_logger.info("ingestion job started")

    # Phase 1: AI Discovery Agent — find new URLs
    discovery_summary: dict[str, Any] = {}
    try:
        from app.agents.discovery_agent import discover_new_sources
        if settings.google_search_api_key and settings.google_search_cx:
            active_logger.info("Running AI Discovery Agent...")
            discovery_summary = discover_new_sources()
            approved = discovery_summary.get("approved_urls", [])
            active_logger.info("Discovery completed: %d new sources approved", len(approved))

            # Inject approved URLs into collect_sources targets
            if approved:
                new_urls = [item["url"] for item in approved if item.get("url")]
                from app.collect_au_sources import DEFAULT_TARGETS
                combined_targets = list(DEFAULT_TARGETS) + new_urls
            else:
                combined_targets = None
        else:
            active_logger.info("Google Search API not configured, skipping discovery.")
            combined_targets = None
    except Exception as exc:
        active_logger.warning("Discovery Agent failed (non-fatal): %s", exc)
        combined_targets = None

    # Phase 2: Collect sources (with expanded targets if discovery found new ones)
    collect_summary = collect_sources(
        target_urls=combined_targets,
        incremental=True,
    )
    active_logger.info(
        "collect_sources completed: chunks_total=%s changed_urls=%s unchanged_urls=%s errors_total=%s",
        collect_summary.get("chunks_total", 0),
        collect_summary.get("changed_urls", 0),
        collect_summary.get("unchanged_urls", 0),
        collect_summary.get("errors_total", 0),
    )

    for item in collect_summary.get("errors", []):
        if not isinstance(item, dict):
            continue
        message = str(item.get("error", ""))
        source = str(item.get("url", "unknown"))
        if _contains_blocking_error(message):
            active_logger.warning("source blocked or timeout: source=%s error=%s", source, message)
        else:
            active_logger.error("source collection error: source=%s error=%s", source, message)

    # Phase 3: Ingest into pgvector
    ingest_summary = ingest_jsonl_to_pgvector(table_name=settings.pgvector_table, incremental=True)
    active_logger.info(
        "ingest_pgvector completed: upserted=%s changed_records=%s deleted_records=%s status=%s",
        ingest_summary.get("upserted", 0),
        ingest_summary.get("changed_records", 0),
        ingest_summary.get("deleted_records", 0),
        ingest_summary.get("status", "unknown"),
    )

    # Phase 4: Auto-refresh stale embeddings
    refresh_summary = refresh_stale_embeddings(active_logger)

    # Phase 5: Log crawl result to database
    crawl_result = {
        "discovery": discovery_summary,
        "collect": collect_summary,
        "ingest": ingest_summary,
        "refresh": refresh_summary,
    }
    _save_crawl_log(crawl_result, active_logger)

    return crawl_result


def _save_crawl_log(result: dict[str, Any], logger: logging.Logger) -> None:
    """Save crawl result to PostgreSQL crawl_logs table."""
    import os
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return
    try:
        import psycopg
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS crawl_logs (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        discovery_approved INT DEFAULT 0,
                        chunks_total INT DEFAULT 0,
                        chunks_new INT DEFAULT 0,
                        errors_total INT DEFAULT 0,
                        summary JSONB
                    )
                """)
                import json
                cur.execute(
                    """INSERT INTO crawl_logs (discovery_approved, chunks_total, chunks_new, errors_total, summary)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (
                        len(result.get("discovery", {}).get("approved_urls", [])),
                        result.get("collect", {}).get("chunks_total", 0),
                        result.get("collect", {}).get("changed_urls", 0),
                        result.get("collect", {}).get("errors_total", 0),
                        json.dumps(result, default=str, ensure_ascii=False),
                    ),
                )
            conn.commit()
        logger.info("Crawl log saved to database.")
    except Exception as exc:
        logger.warning("Failed to save crawl log: %s", exc)


def refresh_stale_embeddings(logger: logging.Logger) -> dict[str, Any]:
    """Phase 5: Auto-detect and re-embed chunks that used the 'fake' embedding provider."""
    import os
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return {"status": "skipped", "reason": "no DATABASE_URL"}
    
    # Check if a live embedding provider is available
    from app.ingest_pgvector import _build_embedding_client
    client, provider_name = _build_embedding_client()
    if not client or provider_name == "fake":
        return {"status": "skipped", "reason": "no live embedding provider available"}
        
    try:
        import psycopg
        import json
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Find stale chunks
                cur.execute(f"SELECT chunk_id, content FROM {settings.pgvector_table} WHERE embedding_provider = 'fake' LIMIT 100")
                rows = cur.fetchall()
                if not rows:
                    return {"status": "ok", "refreshed_count": 0}
                
                logger.info("Found %d stale 'fake' embeddings. Re-embedding with %s...", len(rows), provider_name)
                
                texts = [row[1] for row in rows]
                chunk_ids = [row[0] for row in rows]
                vectors = client.embed_documents(texts)
                
                if vectors and len(vectors) == len(chunk_ids):
                    for chunk_id, vector in zip(chunk_ids, vectors):
                        cur.execute(
                            f"UPDATE {settings.pgvector_table} SET embedding = %s::vector, embedding_provider = %s WHERE chunk_id = %s",
                            (json.dumps(vector), provider_name, chunk_id)
                        )
                    conn.commit()
                    logger.info("Successfully refreshed %d embeddings.", len(vectors))
                    return {"status": "ok", "refreshed_count": len(vectors)}
                else:
                    return {"status": "error", "reason": "embedding count mismatch"}
    except Exception as exc:
        logger.warning("Failed to refresh stale embeddings: %s", exc)
        return {"status": "error", "reason": str(exc)}


def run_scheduled_blogs(logger: logging.Logger) -> dict[str, Any]:
    """Check blog_schedule table and trigger pipeline for due topics."""
    import os
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return {"status": "skipped", "reason": "no DATABASE_URL"}

    try:
        import psycopg
        from datetime import datetime, timezone

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS blog_schedule (
                        id SERIAL PRIMARY KEY,
                        topic TEXT NOT NULL,
                        language TEXT DEFAULT 'en',
                        cron_expression TEXT DEFAULT '0 9 * * MON',
                        is_active BOOLEAN DEFAULT TRUE,
                        last_run_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                conn.commit()

                # Find active schedules that haven't run today
                now = datetime.now(timezone.utc)
                cur.execute("""
                    SELECT id, topic, language, cron_expression, last_run_at
                    FROM blog_schedule
                    WHERE is_active = TRUE
                """)
                rows = cur.fetchall()

        generated_count = 0
        for row in rows:
            schedule_id, topic, language, cron_expr, last_run = row

            # Simple cron matching: check if it should run today
            if not _should_run_cron(cron_expr, now, last_run):
                continue

            logger.info("Scheduled blog trigger: topic='%s', lang='%s'", topic, language)
            try:
                from app.main import process_prompt
                from app.session_manager import SessionManager
                session = SessionManager()
                result = process_prompt(topic, session=session)
                generated_count += 1

                # Update last_run_at
                with psycopg.connect(db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE blog_schedule SET last_run_at = NOW() WHERE id = %s",
                            (schedule_id,),
                        )
                    conn.commit()

                logger.info("Scheduled blog generated for topic='%s'", topic)
            except Exception as exc:
                logger.error("Scheduled blog failed for topic='%s': %s", topic, exc)

        return {"status": "ok", "generated": generated_count}

    except Exception as exc:
        logger.error("run_scheduled_blogs failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def _should_run_cron(cron_expr: str, now, last_run) -> bool:
    """Simple cron matching for: minute hour day_of_month month day_of_week."""
    if last_run and last_run.date() == now.date():
        return False  # Already ran today

    parts = cron_expr.split()
    if len(parts) != 5:
        return False

    minute, hour, dom, month, dow = parts

    if minute != "*" and int(minute) != now.minute:
        return False
    if hour != "*" and int(hour) != now.hour:
        return False
    if dom != "*" and int(dom) != now.day:
        return False
    if month != "*" and int(month) != now.month:
        return False
    if dow != "*":
        # 0=MON in our system, Python weekday() 0=MON
        dow_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        target = dow_map.get(dow.upper(), None)
        if target is None:
            try:
                target = int(dow)
            except ValueError:
                return False
        if now.weekday() != target:
            return False

    return True


def mark_stale_knowledge(logger: logging.Logger, stale_months: int = 6) -> dict[str, Any]:
    """Mark knowledge chunks older than stale_months as stale."""
    import os
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return {"status": "skipped"}

    try:
        import psycopg
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_months * 30)).isoformat()

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Add is_stale column if not exists
                cur.execute(f"""
                    ALTER TABLE {settings.pgvector_table}
                    ADD COLUMN IF NOT EXISTS is_stale BOOLEAN DEFAULT FALSE
                """)
                # Mark old chunks
                cur.execute(f"""
                    UPDATE {settings.pgvector_table}
                    SET is_stale = TRUE
                    WHERE created_at < %s AND (is_stale IS NULL OR is_stale = FALSE)
                """, (cutoff,))
                stale_count = cur.rowcount
            conn.commit()

        logger.info("Marked %d chunks as stale (older than %d months)", stale_count, stale_months)
        return {"status": "ok", "stale_marked": stale_count}

    except Exception as exc:
        logger.warning("mark_stale_knowledge failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def run_worker_loop(poll_seconds: int = 30) -> None:
    logger = _build_logger()
    logger.info("worker started with poll_seconds=%s", poll_seconds)

    last_run: date | None = None
    last_schedule_check: date | None = None
    sleep_seconds = max(5, int(poll_seconds))

    while True:
        now = datetime.now()

        # Weekly ingestion job (Sunday 2am)
        if is_scheduled_window(now, last_run):
            try:
                run_ingestion_job(logger)
                mark_stale_knowledge(logger)
                last_run = now.date()
                logger.info("ingestion job completed successfully")
            except Exception as exc:  # pragma: no cover - runtime hardening
                logger.exception("ingestion job failed: %s", exc)
                last_run = now.date()

        # Hourly schedule check for blog generation
        if last_schedule_check != now.date() or now.minute == 0:
            try:
                result = run_scheduled_blogs(logger)
                if result.get("generated", 0) > 0:
                    logger.info("Scheduled blogs generated: %s", result)
                last_schedule_check = now.date()
            except Exception as exc:
                logger.warning("Scheduled blog check failed: %s", exc)

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_worker_loop()

