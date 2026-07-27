"""Delta Sync — Change Data Capture pipeline for continuous legal data freshness.

Legislation changes. Policies update. If you only scrape once and leave it,
after 6 months the system will advise based on outdated law.

━━━ STRATEGY: Change Data Capture ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Weekly: Re-scrape all target URLs
2. SHA-256: Compare new text hash vs previous hash in source_state.json
3. If unchanged → SKIP (save embedding costs)
4. If changed → Find old chunks in DB → set status='repealed'
5. Re-chunk & re-embed new content as status='in_force'
6. Log changes to validex_changelog table

Run: python -m app.delta_sync
Or via cron: 0 2 * * 0 (Sunday 2am)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "data/canonical/source_state.json"
DEFAULT_CHANGELOG_PATH = "data/canonical/delta_changelog.jsonl"
DEFAULT_JSONL_PATH = "data/canonical/au_blog_chunks.jsonl"


def _content_hash(text: str) -> str:
    """SHA-256 hash of content for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_state(path: Path) -> dict[str, str]:
    """Load previous crawl state (URL → content hash)."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    states = payload.get("states", {}) if isinstance(payload, dict) else {}
    return states if isinstance(states, dict) else {}


def _log_change(
    changelog_path: Path,
    chunk_id: str,
    action: str,
    source_url: str = "",
    old_status: str = "",
    new_status: str = "",
    reason: str = "",
) -> None:
    """Append a change record to the changelog file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_id": chunk_id,
        "action": action,
        "source_url": source_url,
        "old_status": old_status,
        "new_status": new_status,
        "reason": reason,
    }
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    with changelog_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _find_chunks_by_source_url(
    jsonl_path: Path, source_url: str
) -> list[str]:
    """Find all chunk_ids belonging to a source URL in the JSONL file."""
    chunk_ids = []
    if not jsonl_path.exists():
        return chunk_ids
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("source_url") == source_url:
            chunk_id = record.get("chunk_id", "")
            if chunk_id:
                chunk_ids.append(chunk_id)
    return chunk_ids


def delta_sync(
    dry_run: bool = False,
    state_path: str = DEFAULT_STATE_PATH,
    changelog_path: str = DEFAULT_CHANGELOG_PATH,
    jsonl_path: str = DEFAULT_JSONL_PATH,
) -> dict[str, Any]:
    """Run Delta Sync: detect changes, repeal old chunks, re-ingest new ones.

    Parameters
    ----------
    dry_run : bool
        If True, only report what WOULD change without modifying anything.
    state_path : str
        Path to the source state file (URL → content hash).
    changelog_path : str
        Path to the delta changelog file.
    jsonl_path : str
        Path to the canonical JSONL file.

    Returns
    -------
    dict
        Summary of changes detected and applied.
    """
    state_file = Path(state_path)
    changelog_file = Path(changelog_path)
    jsonl_file = Path(jsonl_path)

    previous_state = _load_state(state_file)

    if not previous_state:
        return {
            "status": "skip",
            "message": "No previous state found. Run collect_sources first.",
            "changes": [],
        }

    # Step 1: Re-fetch all URLs and compute new hashes
    from app.collect_au_sources import (
        DEFAULT_TARGETS,
        _fetch_url,
        _is_allowed,
        _content_hash as collector_hash,
    )

    changes: list[dict[str, Any]] = []
    unchanged = 0
    errors: list[dict[str, str]] = []

    for url in DEFAULT_TARGETS:
        if not _is_allowed(url):
            continue
        if url not in previous_state:
            continue  # New URL, not a delta

        try:
            text, _, _, _, last_modified = _fetch_url(url, timeout=15)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            continue

        if not text:
            continue

        new_hash = collector_hash(text)  # Use SHA-1 from collector for consistency
        old_hash = previous_state.get(url, "")

        if new_hash == old_hash:
            unchanged += 1
            continue

        # CHANGE DETECTED
        old_chunk_ids = _find_chunks_by_source_url(jsonl_file, url)

        change = {
            "url": url,
            "action": "amended",
            "old_hash": old_hash[:12],
            "new_hash": new_hash[:12],
            "affected_chunks": len(old_chunk_ids),
            "chunk_ids": old_chunk_ids,
            "last_modified": last_modified,
        }
        changes.append(change)

        if not dry_run:
            # Step 2: Mark old chunks as 'repealed' in DB
            try:
                from app.vector_repository import PGVectorRepository
                from app.config import settings

                repo = PGVectorRepository()
                repealed_count = repo.mark_repealed(
                    settings.pgvector_table,
                    old_chunk_ids,
                    superseded_by=f"delta_sync_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                )

                for chunk_id in old_chunk_ids:
                    _log_change(
                        changelog_file,
                        chunk_id=chunk_id,
                        action="repeal",
                        source_url=url,
                        old_status="in_force",
                        new_status="repealed",
                        reason=f"Content changed (delta sync). Old hash: {old_hash[:12]}, New: {new_hash[:12]}",
                    )

                logger.info(
                    "Delta Sync: Repealed %d chunks for %s (hash changed: %s → %s)",
                    repealed_count, url, old_hash[:12], new_hash[:12],
                )
            except Exception as exc:
                logger.error("Delta Sync: Failed to repeal chunks for %s: %s", url, exc)
                errors.append({"url": url, "error": f"repeal failed: {exc}"})

    summary = {
        "status": "dry_run" if dry_run else "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_urls_checked": len([u for u in DEFAULT_TARGETS if _is_allowed(u)]),
        "unchanged": unchanged,
        "changed": len(changes),
        "changes": changes,
        "errors": errors,
        "next_step": (
            "Run `python -m app.collect_au_sources` to re-ingest changed URLs with new content."
            if changes else "No changes detected. Database is up-to-date."
        ),
    }

    if not dry_run and changes:
        logger.info(
            "Delta Sync complete: %d URLs changed, %d unchanged, %d errors. "
            "Run collect_sources to re-ingest.",
            len(changes), unchanged, len(errors),
        )

    return summary


def main():
    """CLI entry point for Delta Sync."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Delta Sync — Change Data Capture for Validex")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without applying them")
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="Path to source state file")
    parser.add_argument("--changelog", default=DEFAULT_CHANGELOG_PATH, help="Path to changelog file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = delta_sync(
        dry_run=args.dry_run,
        state_path=args.state_path,
        changelog_path=args.changelog,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
