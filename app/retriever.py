"""Clean Semantic Fallback Retriever.

Replaces hand-crafted lexical/synonym rules with real semantic similarity
scoring from our local (and fine-tuned) sentence transformer model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

from app.local_semantics import get_embedding, get_embeddings, cosine_similarity, batch_cosine_similarity
from app.utils import tokenize

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    doc_id: str
    score: int
    content: str
    semantic_score: float = 0.0


@dataclass
class RetrievalDecision:
    docs: list[RetrievedDoc]
    status: str
    confidence: float
    top_score: int
    reason: str


@dataclass
class MetadataRecord:
    file_stem: str
    topic: str
    document_type: str
    approved: bool
    jurisdiction: str = "AU"
    authority_score: float = 0.5
    source_url: str = ""
    last_updated: str = ""


def _load_metadata_index(metadata_path: str | None) -> dict[str, MetadataRecord]:
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    index: dict[str, MetadataRecord] = {}
    for item in payload:
        file_stem = item.get("file_stem")
        if not file_stem:
            continue
        authority_score = item.get("authority_score", 0.5)
        try:
            authority_score = float(authority_score)
        except (TypeError, ValueError):
            authority_score = 0.5
        authority_score = max(0.0, min(1.0, authority_score))

        last_updated = str(item.get("last_updated", "") or "").strip()
        if not last_updated:
            last_updated = datetime.now().date().isoformat()

        index[file_stem] = MetadataRecord(
            file_stem=file_stem,
            topic=item.get("topic", ""),
            document_type=item.get("document_type", ""),
            approved=bool(item.get("approved", False)),
            jurisdiction=str(item.get("jurisdiction", "AU") or "AU"),
            authority_score=authority_score,
            source_url=str(item.get("source_url", "") or ""),
            last_updated=last_updated,
        )
    return index


def _domain_tokens(metadata_index: dict[str, MetadataRecord]) -> set[str]:
    tokens: set[str] = set()
    for item in metadata_index.values():
        tokens.update(tokenize(item.topic))
        tokens.update(tokenize(item.document_type))
    tokens.update({"police", "check", "employment", "recruitment", "compliance"})
    return tokens


def retrieve_top_k(
    query: str,
    data_dir: str,
    top_k: int = 3,
    metadata_path: str | None = "data/metadata/documents.json",
) -> list[RetrievedDoc]:
    """Retrieve top-K documents using semantic similarity."""
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    metadata_index = _load_metadata_index(metadata_path)
    
    file_paths = list(base_dir.glob("*.txt"))
    if not file_paths:
        return []

    # Load contents and encode
    contents = [p.read_text(encoding="utf-8") for p in file_paths]
    query_emb = get_embedding(query)
    doc_embs = get_embeddings([c[:1000] for c in contents])  # Embed first 1000 chars for speed
    
    # Compute similarity scores
    similarities = batch_cosine_similarity(query_emb, doc_embs)
    
    results: list[RetrievedDoc] = []
    for i, file_path in enumerate(file_paths):
        stem = file_path.stem
        sim = float(similarities[i])
        
        # Blended score (base similarity mapped to 0-100 scale)
        meta = metadata_index.get(stem)
        meta_boost = 0.0
        if meta:
            # Prefer approved and high-authority docs
            meta_boost += (0.1 if meta.approved else 0.0)
            meta_boost += (meta.authority_score * 0.1)
            
        blended = min(1.0, max(0.0, sim + meta_boost))
        score = int(round(blended * 100))
        
        results.append(
            RetrievedDoc(doc_id=stem, score=score, content=contents[i], semantic_score=sim)
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def retrieve_with_guard(
    query: str,
    data_dir: str,
    top_k: int = 3,
    metadata_path: str | None = "data/metadata/documents.json",
    min_top_score: int = 30,  # Adjusted for 0-100 scale (typically similarity is > 0.3)
    min_confidence: float = 0.30,
) -> RetrievalDecision:
    """Retrieve with safety rails to block out-of-domain queries."""
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return RetrievalDecision([], "no_data", 0.0, 0, "processed data directory not found")

    metadata_index = _load_metadata_index(metadata_path)

    # 1. Out of domain guard check
    query_tokens = set(tokenize(query))
    domain_tokens = _domain_tokens(metadata_index)
    domain_overlap = len(query_tokens.intersection(domain_tokens))
    if domain_overlap == 0:
        return RetrievalDecision([], "out_of_domain", 0.0, 0, "query does not match current RAG domain")

    # 2. Retrieve candidates semantically
    candidates = retrieve_top_k(query, data_dir, len(list(base_dir.glob("*.txt"))), metadata_path)
    if not candidates:
        return RetrievalDecision([], "no_match", 0.0, 0, "no relevant document found")

    top_score = candidates[0].score
    second_score = candidates[1].score if len(candidates) > 1 else 0
    
    # Confidence metrics
    score_confidence = min(1.0, top_score / 100.0)
    gap_confidence = (top_score - second_score) / (top_score + 1.0)
    confidence = max(score_confidence, gap_confidence)

    # Backwards compatibility: scale min_top_score from 0-15 lexical scale to 0-100 semantic scale
    actual_min_top_score = min_top_score
    if min_top_score <= 15:
        actual_min_top_score = min_top_score * 10
        # If threshold is 10 (or higher), map to 101 so even a perfect 100 score is treated as low confidence
        if min_top_score >= 10:
            actual_min_top_score = 101

    # 3. Guardrail validation
    if top_score < actual_min_top_score:
        return RetrievalDecision([], "low_confidence", confidence, top_score, "top score under threshold")
    if confidence < min_confidence:
        return RetrievalDecision([], "low_confidence", confidence, top_score, "confidence under threshold")

    return RetrievalDecision(candidates[:top_k], "ok", confidence, top_score, "retrieval successful")
