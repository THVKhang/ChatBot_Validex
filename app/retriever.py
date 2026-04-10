from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from math import sqrt

from app.utils import tokenize


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


SYNONYM_GROUPS = {
    "police": {"police", "background", "screening"},
    "check": {"check", "verification", "verify"},
    "employment": {"employment", "employer", "recruitment", "onboarding", "hr"},
    "time": {"time", "long", "processing", "duration", "day", "days"},
    "documents": {"documents", "document", "required", "requirements", "paperwork", "forms"},
    "who": {"who", "needs", "need", "applicants", "applicant", "job", "seekers", "candidates"},
}

CONCEPT_GROUPS = {
    "safety_check": {"police", "check", "background", "screening", "verification"},
    "audience": {"applicants", "candidate", "candidates", "job", "seekers", "employer", "hr"},
    "time": {"time", "processing", "duration", "day", "days", "long", "timeline"},
    "requirements": {"required", "requirements", "documents", "document", "id", "passport", "proof"},
    "compliance": {"compliance", "policy", "regulation", "regulated", "risk"},
    "workflow": {"onboarding", "process", "steps", "guide", "checklist"},
}


def _expand_query_tokens(query_tokens: set[str]) -> set[str]:
    expanded = set(query_tokens)
    for token in list(query_tokens):
        for _, group in SYNONYM_GROUPS.items():
            if token in group:
                expanded.update(group)
    return expanded


def _score(query_tokens: set[str], content: str, doc_id: str, query_lower: str) -> int:
    content_tokens = set(tokenize(content))
    score = len(query_tokens.intersection(content_tokens))

    # Intent-aware boosts to improve top-k ordering.
    if any(k in query_lower for k in ["how long", "processing time", "take"]):
        if any(k in content_tokens for k in ["time", "processing", "day", "days", "1", "3"]):
            score += 3
        if "time" in doc_id or "processing" in doc_id:
            score += 2

    if any(k in query_lower for k in ["who needs", "for first-time", "for job seekers", "for employment"]):
        if any(k in content_tokens for k in ["employer", "recruitment", "candidate", "applicants", "job"]):
            score += 2

    if any(k in query_lower for k in ["documents", "required", "requirements"]):
        if any(k in content_tokens for k in ["documents", "required", "requirements", "id", "passport"]):
            score += 4
        if "document" in doc_id or "requirement" in doc_id:
            score += 2

    if "employment" in query_lower and any(k in content_tokens for k in ["employment", "employer", "recruitment"]):
        score += 2

    return score


def _concept_vector(tokens: set[str]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for concept, words in CONCEPT_GROUPS.items():
        hit_count = len(tokens.intersection(words))
        if hit_count > 0:
            vector[concept] = float(hit_count)
    return vector


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    shared_keys = set(vec_a.keys()).intersection(vec_b.keys())
    numerator = sum(vec_a[key] * vec_b[key] for key in shared_keys)
    norm_a = sqrt(sum(value * value for value in vec_a.values()))
    norm_b = sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _semantic_score(query_tokens: set[str], content: str) -> float:
    content_tokens = set(tokenize(content))
    query_vec = _concept_vector(query_tokens)
    content_vec = _concept_vector(content_tokens)
    return _cosine_similarity(query_vec, content_vec)


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


def _metadata_boost(
    query_tokens: set[str],
    query_lower: str,
    metadata: MetadataRecord | None,
) -> int:
    if not metadata:
        return 0

    boost = 0
    topic_tokens = set(tokenize(metadata.topic))
    overlap = len(query_tokens.intersection(topic_tokens))
    boost += overlap * 2

    if metadata.approved:
        boost += 1

    # Prefer higher-authority sources (e.g. government pages) when relevance is similar.
    boost += int(round(metadata.authority_score * 3))

    if "compliance" in query_lower and metadata.document_type in {"checklist", "requirements"}:
        boost += 3
    if any(k in query_lower for k in ["how long", "processing time", "take"]) and metadata.document_type == "faq":
        boost += 2
    if any(k in query_lower for k in ["documents", "required", "requirements"]) and metadata.document_type == "requirements":
        boost += 3

    return boost


def retrieve_top_k(
    query: str,
    data_dir: str,
    top_k: int = 3,
    metadata_path: str | None = "data/metadata/documents.json",
) -> list[RetrievedDoc]:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    query_lower = query.lower()
    query_tokens = _expand_query_tokens(set(tokenize(query)))
    metadata_index = _load_metadata_index(metadata_path)
    results: list[RetrievedDoc] = []

    for file_path in base_dir.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        lexical_score = _score(query_tokens, content, file_path.stem, query_lower)
        metadata_score = _metadata_boost(query_tokens, query_lower, metadata_index.get(file_path.stem))
        semantic = _semantic_score(query_tokens, content)
        score = lexical_score + metadata_score + int(semantic * 4)
        if score > 0:
            results.append(
                RetrievedDoc(doc_id=file_path.stem, score=score, content=content, semantic_score=semantic)
            )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def retrieve_with_guard(
    query: str,
    data_dir: str,
    top_k: int = 3,
    metadata_path: str | None = "data/metadata/documents.json",
    min_top_score: int = 3,
    min_confidence: float = 0.35,
) -> RetrievalDecision:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return RetrievalDecision([], "no_data", 0.0, 0, "processed data directory not found")

    query_lower = query.lower()
    query_tokens = _expand_query_tokens(set(tokenize(query)))
    metadata_index = _load_metadata_index(metadata_path)

    domain_tokens = _domain_tokens(metadata_index)
    domain_overlap = len(query_tokens.intersection(domain_tokens))
    if domain_overlap == 0:
        return RetrievalDecision([], "out_of_domain", 0.0, 0, "query does not match current RAG domain")

    all_results: list[RetrievedDoc] = []
    for file_path in base_dir.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        lexical_score = _score(query_tokens, content, file_path.stem, query_lower)
        metadata_score = _metadata_boost(query_tokens, query_lower, metadata_index.get(file_path.stem))
        semantic = _semantic_score(query_tokens, content)
        score = lexical_score + metadata_score + int(semantic * 4)
        if score > 0:
            all_results.append(
                RetrievedDoc(doc_id=file_path.stem, score=score, content=content, semantic_score=semantic)
            )

    all_results.sort(key=lambda item: item.score, reverse=True)
    if not all_results:
        return RetrievalDecision([], "no_match", 0.0, 0, "no relevant document found")

    top_score = all_results[0].score
    second_score = all_results[1].score if len(all_results) > 1 else 0
    confidence = top_score / (top_score + second_score + 1)

    if top_score < min_top_score:
        return RetrievalDecision([], "low_confidence", confidence, top_score, "top score under threshold")
    if confidence < min_confidence:
        return RetrievalDecision([], "low_confidence", confidence, top_score, "confidence under threshold")

    return RetrievalDecision(all_results[:top_k], "ok", confidence, top_score, "retrieval successful")
