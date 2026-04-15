from datetime import date
from datetime import datetime
import io
import logging

import app.worker as worker_module


def _build_test_logger() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.Logger("worker_test_logger", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, stream


def test_is_scheduled_window_true_for_sunday_2am():
    now = datetime(2026, 4, 19, 2, 0, 0)  # Sunday

    assert worker_module.is_scheduled_window(now, last_run_date=None)


def test_is_scheduled_window_false_if_already_ran_today():
    now = datetime(2026, 4, 19, 2, 30, 0)  # Sunday

    assert not worker_module.is_scheduled_window(now, last_run_date=date(2026, 4, 19))


def test_run_ingestion_job_logs_blocking_errors(monkeypatch):
    logger, stream = _build_test_logger()

    monkeypatch.setattr(
        worker_module,
        "collect_sources",
        lambda incremental=True: {
            "chunks_total": 3,
            "changed_urls": 1,
            "unchanged_urls": 0,
            "errors_total": 1,
            "errors": [
                {
                    "url": "https://oaic.gov.au/privacy",
                    "error": "403 forbidden from source",
                }
            ],
        },
    )
    monkeypatch.setattr(
        worker_module,
        "ingest_jsonl_to_pgvector",
        lambda table_name=None, incremental=True: {
            "status": "ok",
            "upserted": 3,
            "changed_records": 3,
            "deleted_records": 0,
        },
    )

    result = worker_module.run_ingestion_job(logger)

    output = stream.getvalue().lower()
    assert "source blocked or timeout" in output
    assert result["collect"]["chunks_total"] == 3
    assert result["ingest"]["status"] == "ok"
