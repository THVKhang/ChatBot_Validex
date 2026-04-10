from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


def _read_reports(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    reports = raw.get("reports", []) if isinstance(raw, dict) else []
    if not isinstance(reports, list):
        return []
    return reports


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
    reports = _read_reports(storage_path)

    report = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "prompt": prompt,
        "title": title,
        "outline": outline,
        "draft": draft,
        "sources_used": sources_used,
        "sections": sections or [],
    }
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


def delete_report(report_id: str) -> bool:
    storage_path = Path(settings.reports_path)
    reports = _read_reports(storage_path)
    kept = [item for item in reports if item.get("id") != report_id]
    if len(kept) == len(reports):
        return False

    _write_reports(storage_path, kept)
    return True
