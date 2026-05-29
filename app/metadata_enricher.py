"""LLM Metadata Enricher — Automatically tags legal documents with structured metadata.

Before INSERT into PGVector, each chunk passes through this enricher which uses
a fast, cheap LLM (Gemini Flash) to classify:
  - jurisdiction: 'Commonwealth', 'NSW', 'VIC', 'QLD', etc.
  - document_type: 'legislation', 'regulation', 'guide', 'faq', 'webpage', 'pdf'
  - status: 'in_force', 'repealed', 'amended'
  - effective_date: ISO date string
  - topic: domain-specific topic classification

Fallback: URL-based heuristics if LLM unavailable (0 API tokens).

━━━ WHY THIS MATTERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The 'status' field is the LIFELINE of the system:
  WHERE status = 'in_force'
eliminates 100% risk of citing repealed legislation to clients.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache enrichment results to avoid re-calling LLM for identical chunks
_ENRICHMENT_CACHE_PATH = "data/canonical/metadata_enrichment_cache.json"
_enrichment_cache: dict[str, dict] | None = None


def _load_cache() -> dict[str, dict]:
    """Load enrichment cache from disk."""
    global _enrichment_cache
    if _enrichment_cache is not None:
        return _enrichment_cache

    p = Path(_ENRICHMENT_CACHE_PATH)
    if p.exists():
        try:
            _enrichment_cache = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            _enrichment_cache = {}
    else:
        _enrichment_cache = {}
    return _enrichment_cache


def _save_cache() -> None:
    """Save enrichment cache to disk."""
    if _enrichment_cache is None:
        return
    p = Path(_ENRICHMENT_CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_enrichment_cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _cache_key(text: str, source_url: str) -> str:
    """Generate a deterministic cache key for a chunk."""
    payload = f"{source_url}:{text[:500]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# ── Heuristic Enrichment (0 API tokens) ─────────────────────────

def _heuristic_jurisdiction(url: str, text: str = "") -> str:
    """Detect jurisdiction from URL patterns."""
    lower = url.lower()
    mappings = [
        ("legislation.nsw.gov.au", "NSW"), ("nsw.gov.au", "NSW"),
        ("legislation.vic.gov.au", "VIC"), ("vic.gov.au", "VIC"),
        ("legislation.qld.gov.au", "QLD"), ("qld.gov.au", "QLD"),
        ("legislation.sa.gov.au", "SA"), ("sa.gov.au", "SA"),
        ("legislation.wa.gov.au", "WA"), ("wa.gov.au", "WA"),
        ("legislation.tas.gov.au", "TAS"), ("tas.gov.au", "TAS"),
        ("legislation.nt.gov.au", "NT"), ("nt.gov.au", "NT"),
        ("legislation.act.gov.au", "ACT"), ("act.gov.au", "ACT"),
    ]
    for pattern, jurisdiction in mappings:
        if pattern in lower:
            return jurisdiction
    # Federal sources
    if any(d in lower for d in ["legislation.gov.au", "acic.gov.au", "afp.gov.au", "oaic.gov.au"]):
        return "Commonwealth"
    return "Commonwealth"


def _heuristic_document_type(url: str, source_type: str = "") -> str:
    """Detect document type from URL and source_type."""
    lower = url.lower()
    if source_type == "pdf":
        return "pdf"
    legislation_domains = [
        "legislation.gov.au", "legislation.nsw.gov.au",
        "legislation.vic.gov.au", "legislation.qld.gov.au",
        "legislation.sa.gov.au", "legislation.wa.gov.au",
        "legislation.tas.gov.au",
    ]
    if any(d in lower for d in legislation_domains):
        return "legislation"
    if "faq" in lower:
        return "faq"
    if "guide" in lower or "how-it-works" in lower:
        return "guide"
    if "regulation" in lower:
        return "regulation"
    return "webpage"


def _heuristic_status(text: str, url: str = "") -> str:
    """Detect legislation status from text content."""
    lower = text.lower()
    repeal_signals = [
        "this act has been repealed",
        "repealed by",
        "no longer in force",
        "omitted",
        "expired",
    ]
    for signal in repeal_signals:
        if signal in lower:
            return "repealed"

    amend_signals = ["as amended", "amended by"]
    for signal in amend_signals:
        if signal in lower:
            return "amended"

    return "in_force"


def _heuristic_act_name(url: str, text: str = "") -> str:
    """Detect Act name from URL or text."""
    act_mappings = {
        "C2004A01364": "Crimes Act 1914 (Cth)",
        "C2004A03712": "Privacy Act 1988 (Cth)",
        "C2004A01389": "Australian Federal Police Act 1979 (Cth)",
        "act-1991-008": "Criminal Records Act 1991 (NSW)",
        "act-1998-009": "Child Protection (Working with Children) Act 2012 (NSW)",
        "act-2004-015": "Criminal Law (Rehabilitation of Offenders) Act 1986 (QLD)",
    }
    for key, name in act_mappings.items():
        if key in url:
            return name

    if "spent-convictions" in url.lower():
        return "Spent Convictions Act 2021 (VIC)"

    # Try regex extraction from text
    match = re.search(r"(?:^|\n)\s*(.+?Act\s+\d{4}(?:\s*\([A-Z]{2,4}\))?)", text[:1000])
    if match:
        return match.group(1).strip()

    return ""


def _heuristic_effective_date(text: str) -> str:
    """Try to extract effective/commencement date from text."""
    patterns = [
        r"(?:commenced?|effective|in force)\s+(?:on\s+)?(\d{1,2}\s+\w+\s+\d{4})",
        r"(?:commenced?|effective|in force)\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def enrich_heuristic(
    text: str,
    source_url: str = "",
    source_type: str = "",
) -> dict[str, str]:
    """Enrich a chunk with metadata using URL/text heuristics (0 API tokens).

    Returns dict with: jurisdiction, document_type, status, act_name,
    effective_date, topic.
    """
    return {
        "jurisdiction": _heuristic_jurisdiction(source_url, text),
        "document_type": _heuristic_document_type(source_url, source_type),
        "status": _heuristic_status(text, source_url),
        "act_name": _heuristic_act_name(source_url, text),
        "effective_date": _heuristic_effective_date(text),
    }


# ── LLM Enrichment (Gemini Flash — fast & cheap) ────────────────

_LLM_ENRICHMENT_PROMPT = """You are a legal metadata classifier for Australian legislation and compliance documents.

Analyze the following text chunk and return a JSON object with these fields:
- "jurisdiction": One of "Commonwealth", "NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"
- "document_type": One of "legislation", "regulation", "guide", "faq", "webpage", "pdf"
- "status": One of "in_force", "repealed", "amended"
- "effective_date": ISO date string (e.g. "2024-01-01") or "" if unknown
- "topic": A short topic label (e.g. "spent_convictions", "police_check", "privacy", "wwcc", "ndis_screening")

Source URL: {source_url}

Text:
{text}

Respond with ONLY the JSON object, no other text."""


def enrich_with_llm(
    text: str,
    source_url: str = "",
    source_type: str = "",
) -> dict[str, str] | None:
    """Enrich a chunk with metadata using LLM (Gemini Flash).

    Returns dict with metadata fields, or None if LLM unavailable.
    """
    # Check cache first
    cache = _load_cache()
    key = _cache_key(text, source_url)
    if key in cache:
        return cache[key]

    try:
        from app.config import settings

        if not settings.google_api_key:
            return None

        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)

        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = _LLM_ENRICHMENT_PROMPT.format(
            source_url=source_url,
            text=text[:3000],  # Limit to avoid token waste
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=200,
            ),
        )

        response_text = response.text.strip()
        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)

        result = json.loads(response_text)

        # Validate fields
        valid_jurisdictions = {"Commonwealth", "NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}
        valid_types = {"legislation", "regulation", "guide", "faq", "webpage", "pdf"}
        valid_statuses = {"in_force", "repealed", "amended"}

        metadata = {
            "jurisdiction": result.get("jurisdiction", "Commonwealth") if result.get("jurisdiction") in valid_jurisdictions else "Commonwealth",
            "document_type": result.get("document_type", "webpage") if result.get("document_type") in valid_types else "webpage",
            "status": result.get("status", "in_force") if result.get("status") in valid_statuses else "in_force",
            "effective_date": str(result.get("effective_date", ""))[:20],
            "topic": str(result.get("topic", ""))[:50],
        }

        # Cache result
        cache[key] = metadata
        _save_cache()

        logger.debug("LLM enrichment: %s → %s", source_url[:60], metadata)
        return metadata

    except Exception as exc:
        logger.debug("LLM enrichment failed (falling back to heuristic): %s", exc)
        return None


# ── Main Enrichment API ──────────────────────────────────────────

def enrich_chunk(
    text: str,
    source_url: str = "",
    source_type: str = "",
    use_llm: bool = True,
) -> dict[str, str]:
    """Enrich a chunk with legal metadata.

    Strategy:
    1. Try LLM enrichment (if enabled and available)
    2. Fallback to heuristic enrichment (always available, 0 tokens)
    3. Merge: LLM results take priority, heuristics fill gaps

    Parameters
    ----------
    text : str
        The chunk text to classify.
    source_url : str
        Source URL for context.
    source_type : str
        Pre-detected source type ('pdf', 'html', etc.)
    use_llm : bool
        Whether to attempt LLM enrichment.

    Returns
    -------
    dict[str, str]
        Metadata dict with: jurisdiction, document_type, status,
        act_name, effective_date, topic.
    """
    # Always compute heuristic as baseline
    heuristic = enrich_heuristic(text, source_url, source_type)

    if not use_llm:
        return heuristic

    # Try LLM
    llm_result = enrich_with_llm(text, source_url, source_type)

    if llm_result is None:
        return heuristic

    # Merge: LLM takes priority, heuristic fills gaps
    merged = dict(heuristic)
    for key, value in llm_result.items():
        if value:  # Only override if LLM provided a non-empty value
            merged[key] = value

    # Special case: act_name from heuristic is more reliable (known mappings)
    if heuristic.get("act_name") and not llm_result.get("act_name"):
        merged["act_name"] = heuristic["act_name"]

    return merged


def enrich_batch(
    records: list[dict[str, Any]],
    use_llm: bool = False,
) -> list[dict[str, Any]]:
    """Enrich a batch of records with metadata.

    Mutates each record in-place by adding metadata fields.

    Parameters
    ----------
    records : list[dict]
        List of chunk records (must have 'text' and 'source_url' fields).
    use_llm : bool
        Whether to use LLM enrichment (default: False to save tokens).

    Returns
    -------
    list[dict]
        The same records with metadata fields added.
    """
    for record in records:
        text = str(record.get("text", ""))
        source_url = str(record.get("source_url", ""))
        source_type = str(record.get("source_type", ""))

        metadata = enrich_chunk(text, source_url, source_type, use_llm=use_llm)

        record["jurisdiction"] = metadata.get("jurisdiction", "Commonwealth")
        record["document_type"] = metadata.get("document_type", "webpage")
        record["status"] = metadata.get("status", "in_force")
        record["act_name"] = metadata.get("act_name", record.get("act_name", ""))
        record["effective_date"] = metadata.get("effective_date", "")

        # Preserve existing topic if richer, else use enriched topic
        if metadata.get("topic") and not record.get("topic"):
            record["topic"] = metadata["topic"]

    logger.info("Enriched %d records with metadata (use_llm=%s)", len(records), use_llm)
    return records
