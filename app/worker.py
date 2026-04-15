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

    collect_summary = collect_sources(incremental=True)
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

    ingest_summary = ingest_jsonl_to_pgvector(table_name=settings.pgvector_table, incremental=True)
    active_logger.info(
        "ingest_pgvector completed: upserted=%s changed_records=%s deleted_records=%s status=%s",
        ingest_summary.get("upserted", 0),
        ingest_summary.get("changed_records", 0),
        ingest_summary.get("deleted_records", 0),
        ingest_summary.get("status", "unknown"),
    )

    return {
        "collect": collect_summary,
        "ingest": ingest_summary,
    }


def run_worker_loop(poll_seconds: int = 30) -> None:
    logger = _build_logger()
    logger.info("worker started with poll_seconds=%s", poll_seconds)

    last_run: date | None = None
    sleep_seconds = max(5, int(poll_seconds))

    while True:
        now = datetime.now()
        if is_scheduled_window(now, last_run):
            try:
                run_ingestion_job(logger)
                last_run = now.date()
                logger.info("ingestion job completed successfully")
            except Exception as exc:  # pragma: no cover - runtime hardening
                logger.exception("ingestion job failed: %s", exc)
                last_run = now.date()

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_worker_loop()
