from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _title_from_stem(stem: str) -> str:
    tokens = [token for token in stem.replace("-", "_").split("_") if token and not token.isdigit()]
    if not tokens:
        return "Untitled"
    return " ".join(word.capitalize() for word in tokens)


def _topic_from_stem(stem: str) -> str:
    tokens = [token for token in stem.replace("-", "_").split("_") if token and not token.isdigit()]
    return " ".join(tokens[:4]) if tokens else "general"


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def ingest_raw_documents(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    metadata_path: str = "data/metadata/documents.json",
    source: str = "raw_import",
) -> dict[str, int]:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    metadata_file = Path(metadata_path)

    processed_path.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    metadata_items = _load_json(metadata_file)
    index_by_stem = {item.get("file_stem"): item for item in metadata_items if item.get("file_stem")}

    added = 0
    updated = 0

    for txt_file in sorted(raw_path.glob("*.txt")):
        cleaned_content = _clean_text(txt_file.read_text(encoding="utf-8"))
        if not cleaned_content:
            continue

        out_file = processed_path / txt_file.name
        out_file.write_text(cleaned_content, encoding="utf-8")

        stem = txt_file.stem
        record = {
            "id": f"doc_{stem}",
            "file_stem": stem,
            "title": _title_from_stem(stem),
            "source": source,
            "topic": _topic_from_stem(stem),
            "document_type": "note",
            "approved": True,
            "content": cleaned_content,
        }

        if stem in index_by_stem:
            index_by_stem[stem].update(record)
            updated += 1
        else:
            metadata_items.append(record)
            index_by_stem[stem] = record
            added += 1

    metadata_file.write_text(json.dumps(metadata_items, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"added": added, "updated": updated, "total": len(metadata_items)}


if __name__ == "__main__":
    result = ingest_raw_documents()
    print(json.dumps(result, indent=2))
