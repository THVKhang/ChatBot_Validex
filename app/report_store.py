from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - optional dependency
    FileLock = None  # type: ignore[assignment,misc]

from app.config import settings


REPORT_STATUSES = ("Draft", "Reviewed", "Approved")
STATUS_TRANSITIONS = {
    "Draft": {"Draft", "Reviewed"},
    "Reviewed": {"Reviewed", "Approved"},
    "Approved": {"Approved"},
}


def _normalize_status(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    mapping = {
        "draft": "Draft",
        "reviewed": "Reviewed",
        "approved": "Approved",
    }
    return mapping.get(text)


def _coerce_report(raw_report: dict[str, Any]) -> dict[str, Any]:
    report = dict(raw_report)
    normalized_status = _normalize_status(str(report.get("status") or ""))
    report["status"] = normalized_status or "Draft"
    return report


def _read_reports(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    reports = raw.get("reports", []) if isinstance(raw, dict) else []
    if not isinstance(reports, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in reports:
        if isinstance(item, dict):
            normalized.append(_coerce_report(item))
    return normalized


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _write_reports(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def save_report(
    *,
    prompt: str,
    title: str,
    outline: list[str],
    draft: str,
    sources_used: list[str],
    sections: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    storage_path = Path(settings.reports_path)
    lock = FileLock(str(_lock_path(storage_path)), timeout=10) if FileLock is not None else None

    report = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "Draft",
        "session_id": session_id,
        "prompt": prompt,
        "title": title,
        "outline": outline,
        "draft": draft,
        "sources_used": sources_used,
        "sections": sections or [],
    }

    if lock is not None:
        with lock:
            reports = _read_reports(storage_path)
            reports.append(report)
            _write_reports(storage_path, reports)
    else:
        reports = _read_reports(storage_path)
        reports.append(report)
        _write_reports(storage_path, reports)
    return report


def list_reports(limit: int = 50) -> list[dict[str, Any]]:
    storage_path = Path(settings.reports_path)
    reports = _read_reports(storage_path)
    ordered = sorted(reports, key=lambda item: item.get("created_at", ""), reverse=True)

    summaries: list[dict[str, Any]] = []
    for item in ordered[: max(1, limit)]:
        draft = item.get("draft", "")
        summaries.append(
            {
                "id": item.get("id"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "status": item.get("status", "Draft"),
                "session_id": item.get("session_id"),
                "title": item.get("title", ""),
                "prompt": item.get("prompt", ""),
                "draft_preview": draft[:180],
            }
        )
    return summaries


def get_report(report_id: str) -> dict[str, Any] | None:
    storage_path = Path(settings.reports_path)
    reports = _read_reports(storage_path)
    for item in reports:
        if item.get("id") == report_id:
            return item
    return None


def update_report_status(report_id: str, status: str) -> tuple[dict[str, Any] | None, str | None]:
    target_status = _normalize_status(status)
    if target_status is None:
        return None, "invalid_status"

    storage_path = Path(settings.reports_path)
    lock = FileLock(str(_lock_path(storage_path)), timeout=10) if FileLock is not None else None
    now = datetime.now(timezone.utc).isoformat()

    def _do_update() -> tuple[dict[str, Any] | None, str | None]:
        reports = _read_reports(storage_path)
        for item in reports:
            if item.get("id") != report_id:
                continue
            current_status = _normalize_status(str(item.get("status", "Draft"))) or "Draft"
            allowed_next = STATUS_TRANSITIONS.get(current_status, {current_status})
            if target_status not in allowed_next:
                return None, "invalid_transition"
            item["status"] = target_status
            item["updated_at"] = now
            _write_reports(storage_path, reports)
            return item, None
        return None, "not_found"

    if lock is not None:
        with lock:
            return _do_update()
    return _do_update()


def delete_report(report_id: str) -> bool:
    storage_path = Path(settings.reports_path)
    lock = FileLock(str(_lock_path(storage_path)), timeout=10) if FileLock is not None else None

    def _do_delete() -> bool:
        reports = _read_reports(storage_path)
        kept = [item for item in reports if item.get("id") != report_id]
        if len(kept) == len(reports):
            return False
        _write_reports(storage_path, kept)
        return True

    if lock is not None:
        with lock:
            return _do_delete()
    return _do_delete()
