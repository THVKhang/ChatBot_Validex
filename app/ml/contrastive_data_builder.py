"""Contrastive Data Builder — Generates training triplets for Embedding fine-tuning.

Creates (anchor_query, positive_document, negative_document) triplets by:
1. Mining real query-document pairs from ML training data + processed docs
2. Synthetic query augmentation via LLM paraphrasing  
3. Hard negative mining (same domain, wrong topic)

Output: data/ml/contrastive_triplets.jsonl
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "data/ml/contrastive_triplets.jsonl"
TRAINING_DATA_PATH = "data/ml/training_data.jsonl"
PROCESSED_DOCS_DIR = "data/processed"
METADATA_PATH = "data/metadata/documents.json"


# ── Document Loading ────────────────────────────────────────────────

def _load_processed_documents() -> list[dict[str, str]]:
    """Load all processed documents with their content."""
    docs_dir = Path(PROCESSED_DOCS_DIR)
    if not docs_dir.exists():
        logger.warning("Processed docs directory not found: %s", PROCESSED_DOCS_DIR)
        return []
    
    docs = []
    for file_path in sorted(docs_dir.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            docs.append({
                "doc_id": file_path.stem,
                "content": content,
                "file": str(file_path),
            })
    logger.info("Loaded %d processed documents", len(docs))
    return docs


def _load_metadata() -> dict[str, dict]:
    """Load document metadata for topic matching."""
    meta_path = Path(METADATA_PATH)
    if not meta_path.exists():
        return {}
    
    try:
        items = json.loads(meta_path.read_text(encoding="utf-8"))
        return {item["file_stem"]: item for item in items if "file_stem" in item}
    except (json.JSONDecodeError, KeyError):
        return {}


def _load_training_topics() -> list[dict[str, Any]]:
    """Load unique topics from training data."""
    path = Path(TRAINING_DATA_PATH)
    if not path.exists():
        return []
    
    topics = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            topic = record.get("metadata", {}).get("topic", "")
            if topic and topic not in seen:
                seen.add(topic)
                topics.append({
                    "topic": topic,
                    "intent": record.get("metadata", {}).get("intent", "create_blog"),
                    "retrieval_score_max": record.get("features", {}).get("retrieval_score_max", 0),
                    "retrieval_doc_count": record.get("features", {}).get("retrieval_doc_count", 0),
                })
        except json.JSONDecodeError:
            continue
    
    logger.info("Loaded %d unique topics from training data", len(topics))
    return topics


# ── Matching: Query -> Document ──────────────────────────────────────

# Hand-annotated golden pairs: the HIGHEST QUALITY training signal.
# Each tuple: (query, correct_doc_id, hard_negative_doc_id)
GOLDEN_ANNOTATED_TRIPLETS = [
    ("How long does a police check take in Australia",
     "doc_06_faq_processing_time", "doc_11_spent_convictions"),
    ("police check processing time",
     "doc_06_faq_processing_time", "doc_09_wwcc_guide"),
    ("how long does background check take",
     "doc_16_processing_times", "doc_13_employer_compliance"),
    ("What documents do I need for a police check",
     "doc_07_required_documents", "doc_06_faq_processing_time"),
    ("required ID for national police check",
     "doc_07_required_documents", "doc_08_who_needs_police_check"),
    ("police check document requirements checklist",
     "doc_07_required_documents", "doc_05_compliance_note"),
    ("Working with children check requirements",
     "doc_09_wwcc_guide", "doc_10_ndis_screening"),
    ("WWCC process and application",
     "doc_09_wwcc_guide", "doc_14_immigration_work"),
    ("working with children check NSW",
     "doc_09_wwcc_guide", "doc_15_aged_care_screening"),
    ("NDIS worker screening process",
     "doc_10_ndis_screening", "doc_09_wwcc_guide"),
    ("disability support worker police check",
     "doc_10_ndis_screening", "doc_13_employer_compliance"),
    ("What are spent convictions in Australia",
     "doc_11_spent_convictions", "doc_06_faq_processing_time"),
    ("do old criminal records show on police check",
     "doc_11_spent_convictions", "doc_01_police_check"),
    ("spent convictions rehabilitation period",
     "doc_11_spent_convictions", "doc_17_privacy_obligations"),
    ("Identity verification 100 point check",
     "doc_12_identity_verification", "doc_07_required_documents"),
    ("how to verify identity for background check",
     "doc_12_identity_verification", "doc_08_who_needs_police_check"),
    ("100 point ID check documents list",
     "doc_12_identity_verification", "doc_05_compliance_note"),
    ("Employer compliance obligations for background checks",
     "doc_13_employer_compliance", "doc_06_faq_processing_time"),
    ("what employers must do for police checks",
     "doc_13_employer_compliance", "doc_02_employer_guide"),
    ("employer obligations screening employees",
     "doc_13_employer_compliance", "doc_17_privacy_obligations"),
    ("Immigration work visa police check requirements",
     "doc_14_immigration_work", "doc_11_spent_convictions"),
    ("do I need police check for work visa",
     "doc_14_immigration_work", "doc_08_who_needs_police_check"),
    ("Aged care worker screening requirements Australia",
     "doc_15_aged_care_screening", "doc_10_ndis_screening"),
    ("nursing home staff police check",
     "doc_15_aged_care_screening", "doc_09_wwcc_guide"),
    ("aged care background check process",
     "doc_15_aged_care_screening", "doc_13_employer_compliance"),
    ("Police check processing times 2024 2025",
     "doc_16_processing_times", "doc_07_required_documents"),
    ("how long national police check takes currently",
     "doc_16_processing_times", "doc_11_spent_convictions"),
    ("Privacy Act employer obligations screening",
     "doc_17_privacy_obligations", "doc_13_employer_compliance"),
    ("can employer share police check results",
     "doc_17_privacy_obligations", "doc_05_compliance_note"),
    ("privacy law background check data retention",
     "doc_17_privacy_obligations", "doc_12_identity_verification"),
    ("what is a national police check",
     "doc_01_police_check", "doc_06_faq_processing_time"),
    ("police check definition and purpose",
     "doc_01_police_check", "doc_02_employer_guide"),
    ("Who needs a police check for employment",
     "doc_08_who_needs_police_check", "doc_07_required_documents"),
    ("which jobs require police checks Australia",
     "doc_08_who_needs_police_check", "doc_13_employer_compliance"),
    ("employer guide to background checks",
     "doc_02_employer_guide", "doc_05_compliance_note"),
    ("recruitment background screening guide",
     "doc_02_employer_guide", "doc_04_candidate_experience"),
]


def _find_positive_negative_semantic(
    query: str,
    documents: list[dict[str, str]],
    model,
) -> tuple[dict | None, dict | None]:
    """Find the best positive and hardest negative document using semantic similarity.
    
    Uses the BASE embedding model to compute real semantic similarity,
    ensuring the training signal is STRONGER than keyword overlap.
    """
    import numpy as np
    
    if len(documents) < 2:
        return None, None
    
    # Encode query and all docs
    doc_texts = [d["content"][:500] for d in documents]
    
    query_emb = model.encode([query], normalize_embeddings=True)
    doc_embs = model.encode(doc_texts, normalize_embeddings=True)
    
    # Compute similarities
    similarities = np.dot(doc_embs, query_emb.T).flatten()
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Positive = highest similarity doc
    best_idx = sorted_indices[0]
    best_sim = similarities[best_idx]
    
    if best_sim < 0.15:  # Too dissimilar, skip
        return None, None
    
    # Hard negative = medium-ranked doc (not the most dissimilar, which is trivial)
    # Pick from rank 30-70% for hard negatives
    n = len(sorted_indices)
    hard_neg_start = max(1, n // 3)
    hard_neg_end = max(hard_neg_start + 1, 2 * n // 3)
    hard_neg_idx = sorted_indices[random.randint(hard_neg_start, min(hard_neg_end, n - 1))]
    
    if best_idx == hard_neg_idx:
        return None, None
    
    return documents[best_idx], documents[hard_neg_idx]


# ── Synthetic Query Augmentation ────────────────────────────────────

# Hand-crafted paraphrase templates for domain queries
PARAPHRASE_TEMPLATES = [
    # Structure variations
    "What is {topic}",
    "Explain {topic}",
    "How does {topic} work",
    "Tell me about {topic}",
    "Guide to {topic}",
    "Requirements for {topic}",
    "{topic} explained simply",
    "Step by step {topic}",
    # Australian legal variations
    "{topic} in Australia",
    "Australian {topic} requirements",
    "{topic} for employers in Australia",
    "{topic} process and timeline",
    # Question variations
    "What do I need to know about {topic}",
    "How long does {topic} take",
    "Who needs {topic}",
    "Is {topic} mandatory",
    "Cost of {topic}",
]

# Domain-specific query seeds (cover the Validex domain)
DOMAIN_QUERY_SEEDS = [
    "national police check Australia",
    "criminal history check employment",
    "police check processing time",
    "police check required documents",
    "WWCC working with children check",
    "NDIS worker screening check",
    "spent convictions Australia law",
    "identity verification 100 point check",
    "employer compliance background screening",
    "immigration work visa police check",
    "aged care worker screening requirements",
    "privacy act employer obligations screening",
    "volunteer police check requirements",
    "police check vs working with children check difference",
    "how to apply national police check online",
    "police check for healthcare workers",
    "teacher registration police check Australia",
    "police check renewal how often",
    "criminal record disclosure employment rights",
    "background check fair work act",
]


def _generate_synthetic_queries(topics: list[dict[str, Any]]) -> list[str]:
    """Generate synthetic query variations from topics and seeds."""
    synthetic = []
    
    # From existing topics: create paraphrased versions
    for topic_info in topics:
        topic = topic_info["topic"]
        # Extract core concept (remove "How to...", "What is..." prefixes)
        core = re.sub(r'^(how to|what is|explain|guide to|the role of)\s+', '', topic.lower(), flags=re.IGNORECASE)
        
        for template in random.sample(PARAPHRASE_TEMPLATES, min(5, len(PARAPHRASE_TEMPLATES))):
            synthetic.append(template.format(topic=core))
    
    # From domain seeds: use as-is + add paraphrases
    for seed in DOMAIN_QUERY_SEEDS:
        synthetic.append(seed)
        for template in random.sample(PARAPHRASE_TEMPLATES, 3):
            synthetic.append(template.format(topic=seed))
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in synthetic:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)
    
    logger.info("Generated %d synthetic queries", len(unique))
    return unique


def _generate_llm_paraphrases(topics: list[str], n_per_topic: int = 5) -> list[str]:
    """Use LLM to generate high-quality paraphrases (optional, costs API tokens).
    
    Returns empty list if LLM is unavailable.
    """
    try:
        from app.langchain_pipeline import pipeline
        llm = getattr(pipeline, "_fast_llm", pipeline._llm)
        if not llm:
            return []
    except Exception:
        logger.info("LLM not available for paraphrase generation, using templates only")
        return []
    
    paraphrases = []
    for topic in topics[:20]:  # Cap at 20 to limit API cost
        prompt = (
            f"Generate {n_per_topic} different ways a user might search for information about: '{topic}'\n"
            "Each query should be a natural search question. Return ONLY the queries, one per line.\n"
            "Include variations with different words, different question formats, and Australian English."
        )
        try:
            response = llm.invoke(prompt)
            text = getattr(response, "content", str(response)).strip()
            lines = [l.strip().lstrip("0123456789.-) ") for l in text.split("\n") if l.strip()]
            paraphrases.extend(lines[:n_per_topic])
        except Exception as exc:
            logger.warning("LLM paraphrase failed for '%s': %s", topic, exc)
            continue
    
    logger.info("Generated %d LLM paraphrases", len(paraphrases))
    return paraphrases


# ── Main Builder ────────────────────────────────────────────────────

def build_contrastive_triplets(
    use_llm: bool = False,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> int:
    """Build contrastive triplets and save to JSONL.
    
    Strategy:
    1. Inject hand-annotated golden triplets (highest quality signal)
    2. Use base embedding model for semantic matching of synthetic queries
    3. Augment with sentence-level anchors
    """
    from sentence_transformers import SentenceTransformer
    
    documents = _load_processed_documents()
    metadata = _load_metadata()
    topics = _load_training_topics()
    
    if not documents:
        logger.error("No documents found. Cannot build triplets.")
        return 0
    
    # Build doc_id -> content index
    doc_index = {d["doc_id"]: d for d in documents}
    
    # Load base embedding model for semantic matching
    logger.info("Loading base embedding model for semantic matching...")
    base_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # Pre-encode all documents once
    doc_texts = [d["content"][:500] for d in documents]
    doc_embs = base_model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)
    
    triplets = []
    
    # ── Step 1: Inject golden annotated triplets (HIGHEST priority) ──
    golden_count = 0
    for query, pos_id, neg_id in GOLDEN_ANNOTATED_TRIPLETS:
        if pos_id in doc_index and neg_id in doc_index:
            triplets.append({
                "anchor": query,
                "positive": doc_index[pos_id]["content"][:1000],
                "negative": doc_index[neg_id]["content"][:1000],
                "positive_doc_id": pos_id,
                "negative_doc_id": neg_id,
            })
            golden_count += 1
    
    logger.info("Injected %d golden annotated triplets", golden_count)
    
    # ── Step 2: Collect synthetic queries ──
    all_queries = []
    for t in topics:
        all_queries.append(t["topic"])
    
    synthetic = _generate_synthetic_queries(topics)
    all_queries.extend(synthetic)
    
    if use_llm:
        topic_strings = [t["topic"] for t in topics]
        llm_queries = _generate_llm_paraphrases(topic_strings)
        all_queries.extend(llm_queries)
    
    # Deduplicate
    seen = set()
    unique_queries = []
    for q in all_queries:
        q_lower = q.lower().strip()
        if q_lower not in seen and len(q_lower) > 5:
            seen.add(q_lower)
            unique_queries.append(q)
    
    logger.info("Total unique queries: %d", len(unique_queries))
    
    # ── Step 3: Build triplets using semantic matching ──
    import numpy as np
    
    for query in unique_queries:
        query_emb = base_model.encode([query], normalize_embeddings=True)
        similarities = np.dot(doc_embs, query_emb.T).flatten()
        sorted_indices = np.argsort(similarities)[::-1]
        
        best_idx = sorted_indices[0]
        best_sim = float(similarities[best_idx])
        
        if best_sim < 0.15:
            continue
        
        # Hard negative: pick from middle ranks
        n = len(sorted_indices)
        hard_neg_start = max(1, n // 3)
        hard_neg_end = max(hard_neg_start + 1, 2 * n // 3)
        hard_neg_idx = sorted_indices[random.randint(hard_neg_start, min(hard_neg_end, n - 1))]
        
        if best_idx == hard_neg_idx:
            continue
        
        triplets.append({
            "anchor": query,
            "positive": documents[best_idx]["content"][:1000],
            "negative": documents[hard_neg_idx]["content"][:1000],
            "positive_doc_id": documents[best_idx]["doc_id"],
            "negative_doc_id": documents[hard_neg_idx]["doc_id"],
        })
    
    logger.info("Built %d triplets (golden + semantic)", len(triplets))
    
    # ── Step 4: Augmentation — multiple negatives per golden pair ──
    augmented = []
    for query, pos_id, _ in GOLDEN_ANNOTATED_TRIPLETS:
        if pos_id not in doc_index:
            continue
        # Create extra triplets with different negatives
        for neg_doc in documents:
            if neg_doc["doc_id"] != pos_id and random.random() < 0.3:
                augmented.append({
                    "anchor": query,
                    "positive": doc_index[pos_id]["content"][:1000],
                    "negative": neg_doc["content"][:1000],
                    "positive_doc_id": pos_id,
                    "negative_doc_id": neg_doc["doc_id"],
                })
    
    triplets.extend(augmented)
    logger.info("After augmentation: %d triplets", len(triplets))
    
    # ── Step 5: Shuffle and save ──
    random.shuffle(triplets)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
    
    logger.info("Saved %d triplets to %s", len(triplets), output_path)
    return len(triplets)


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    use_llm = "--llm" in sys.argv
    count = build_contrastive_triplets(use_llm=use_llm)
    print(f"\n{'='*60}")
    print(f"Generated {count} contrastive triplets")
    print(f"Output: {DEFAULT_OUTPUT_PATH}")
    print(f"{'='*60}")

