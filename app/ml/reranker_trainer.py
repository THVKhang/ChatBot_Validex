"""Reranker Trainer — Fine-tunes ms-marco-MiniLM-L-12-v2 on Validex domain data.

Uses binary relevance labels derived from:
  - Editor verdicts (ACCEPT=1, REJECT=0)
  - RAG Evaluator scores
  - Manual query-document pair annotations

After training:
  - Model saved to data/models/reranker-finetuned-validex/
  - Integration: reranker.py loads fine-tuned model automatically
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "data/models/reranker-finetuned-validex"
BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
TRAINING_DATA_PATH = "data/ml/training_data.jsonl"
PROCESSED_DOCS_DIR = "data/processed"
REPORT_PATH = "data/ml/reranker_training_report.json"


def _build_reranker_pairs() -> list[dict]:
    """Build (query, document, label) pairs for reranker training.
    
    Sources:
    1. From training_data.jsonl: topic + matched documents → positive pairs
    2. Cross-topic pairs → negative pairs (hard negatives)
    3. Hand-crafted domain pairs for precision
    """
    pairs = []
    
    # Load training topics
    training_path = Path(TRAINING_DATA_PATH)
    topics = []
    if training_path.exists():
        for line in training_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                    topic = record.get("metadata", {}).get("topic", "")
                    if topic:
                        topics.append(topic)
                except json.JSONDecodeError:
                    continue
    
    # Load all documents
    docs_dir = Path(PROCESSED_DOCS_DIR)
    documents = {}
    for f in sorted(docs_dir.glob("*.txt")):
        content = f.read_text(encoding="utf-8").strip()
        if content:
            documents[f.stem] = content
    
    if not documents:
        logger.warning("No documents found for reranker training")
        return []
    
    doc_list = list(documents.items())
    
    # ── Source 1: Hand-crafted positive pairs (high-confidence labels) ──
    positive_annotations = [
        ("police check processing time Australia", "doc_06_faq_processing_time", 1.0),
        ("how long does a background check take", "doc_06_faq_processing_time", 0.9),
        ("required documents for police check", "doc_07_required_documents", 1.0),
        ("what ID do I need for a police check", "doc_07_required_documents", 0.9),
        ("who needs a police check for work", "doc_08_who_needs_police_check", 1.0),
        ("working with children check NSW", "doc_09_wwcc_guide", 1.0),
        ("WWCC requirements and process", "doc_09_wwcc_guide", 0.9),
        ("NDIS worker screening check requirements", "doc_10_ndis_screening", 1.0),
        ("disability support worker police check", "doc_10_ndis_screening", 0.8),
        ("spent convictions Australia law", "doc_11_spent_convictions", 1.0),
        ("do old criminal records show on police check", "doc_11_spent_convictions", 0.9),
        ("identity verification 100 point check", "doc_12_identity_verification", 1.0),
        ("how to verify identity for background check", "doc_12_identity_verification", 0.8),
        ("employer compliance background screening", "doc_13_employer_compliance", 1.0),
        ("what employers must do for police checks", "doc_13_employer_compliance", 0.9),
        ("immigration visa police check Australia", "doc_14_immigration_work", 1.0),
        ("do I need police check for work visa", "doc_14_immigration_work", 0.9),
        ("aged care worker screening requirements", "doc_15_aged_care_screening", 1.0),
        ("nursing home staff police check", "doc_15_aged_care_screening", 0.8),
        ("police check processing times 2024 2025", "doc_16_processing_times", 1.0),
        ("privacy act employer obligations", "doc_17_privacy_obligations", 1.0),
        ("can employer share police check results", "doc_17_privacy_obligations", 0.8),
        ("what is a national police check", "doc_01_police_check", 1.0),
        ("employer guide to background checks", "doc_02_employer_guide", 0.9),
    ]
    
    for query, doc_id, label in positive_annotations:
        if doc_id in documents:
            pairs.append({
                "query": query,
                "document": documents[doc_id][:500],
                "label": label,
                "doc_id": doc_id,
                "source": "annotated_positive",
            })
    
    # ── Source 2: Hard negatives (related domain, wrong topic) ──
    hard_negative_pairs = [
        ("police check processing time", "doc_11_spent_convictions", 0.1),
        ("police check processing time", "doc_09_wwcc_guide", 0.15),
        ("working with children check", "doc_14_immigration_work", 0.1),
        ("spent convictions law", "doc_06_faq_processing_time", 0.05),
        ("NDIS worker screening", "doc_13_employer_compliance", 0.15),
        ("privacy act employer", "doc_15_aged_care_screening", 0.1),
        ("aged care screening", "doc_09_wwcc_guide", 0.15),
        ("identity verification", "doc_08_who_needs_police_check", 0.1),
        ("immigration police check", "doc_11_spent_convictions", 0.1),
        ("employer compliance", "doc_06_faq_processing_time", 0.1),
    ]
    
    for query, doc_id, label in hard_negative_pairs:
        if doc_id in documents:
            pairs.append({
                "query": query,
                "document": documents[doc_id][:500],
                "label": label,
                "doc_id": doc_id,
                "source": "annotated_negative",
            })
    
    # ── Source 3: Random negatives from training topics ──
    unique_topics = list(set(topics))
    for topic in unique_topics[:30]:
        # Pick 2 random documents as negatives
        for _ in range(2):
            rand_doc_id, rand_content = random.choice(doc_list)
            # Quick check: if topic words overlap heavily with doc, skip (might be positive)
            topic_words = set(re.findall(r'\w+', topic.lower()))
            doc_words = set(re.findall(r'\w+', rand_content.lower()))
            overlap = len(topic_words & doc_words) / max(len(topic_words), 1)
            if overlap < 0.3:  # Low overlap = likely negative
                pairs.append({
                    "query": topic,
                    "document": rand_content[:500],
                    "label": 0.1,
                    "doc_id": rand_doc_id,
                    "source": "random_negative",
                })
    
    random.shuffle(pairs)
    logger.info("Built %d reranker training pairs (pos/neg)", len(pairs))
    return pairs


def train_reranker(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_model: str = BASE_MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> dict[str, Any]:
    """Fine-tune cross-encoder reranker on domain data."""
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    
    pairs = _build_reranker_pairs()
    if len(pairs) < 10:
        raise ValueError(f"Too few training pairs ({len(pairs)}). Need annotated data.")
    
    # Split: 85% train, 15% dev
    split_idx = max(1, int(len(pairs) * 0.85))
    train_pairs = pairs[:split_idx]
    dev_pairs = pairs[split_idx:]
    
    logger.info("Reranker Training: %d pairs, Dev: %d pairs", len(train_pairs), len(dev_pairs))
    
    # Load base model
    logger.info("Loading base reranker: %s", base_model)
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    
    # Prepare training examples
    train_examples = [
        InputExample(texts=[p["query"], p["document"]], label=float(p["label"]))
        for p in train_pairs
    ]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    warmup_steps = max(1, int(len(train_dataloader) * epochs * warmup_ratio))
    
    # ── Before evaluation ──
    logger.info("Evaluating base reranker before fine-tuning...")
    before_metrics = _evaluate_reranker(model, dev_pairs)
    logger.info("BEFORE — Accuracy: %.2f%%, Avg Score Gap: %.4f",
                before_metrics["accuracy"] * 100, before_metrics["avg_score_gap"])
    
    # ── Train ──
    logger.info(
        "Starting reranker training: %d examples, %d epochs, batch=%d",
        len(train_examples), epochs, batch_size,
    )
    
    start_time = time.time()
    
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )
    
    training_time = time.time() - start_time
    logger.info("Reranker training completed in %.1f seconds", training_time)
    
    # Explicitly save model (some versions don't auto-save to output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    logger.info("Reranker model saved to %s", output_dir)
    
    # ── After evaluation (reuse trained model, don't reload) ──
    after_metrics = _evaluate_reranker(model, dev_pairs)
    logger.info("AFTER - Accuracy: %.2f%%, Avg Score Gap: %.4f",
                after_metrics["accuracy"] * 100, after_metrics["avg_score_gap"])
    
    # ── Report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model,
        "output_dir": output_dir,
        "training_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "epochs": epochs,
        "training_time_seconds": round(training_time, 1),
        "before": before_metrics,
        "after": after_metrics,
        "improvement": {
            "accuracy_delta": round(after_metrics["accuracy"] - before_metrics["accuracy"], 4),
            "score_gap_delta": round(after_metrics["avg_score_gap"] - before_metrics["avg_score_gap"], 4),
        },
    }
    
    report_path = Path(REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Reranker report saved to %s", REPORT_PATH)
    
    return report


def _evaluate_reranker(model, pairs: list[dict]) -> dict:
    """Evaluate reranker: does it score positives > negatives?"""
    if not pairs:
        return {"accuracy": 0.0, "avg_score_gap": 0.0}
    
    # Separate positives and negatives
    positives = [p for p in pairs if p["label"] > 0.5]
    negatives = [p for p in pairs if p["label"] <= 0.5]
    
    if not positives or not negatives:
        return {"accuracy": 0.5, "avg_score_gap": 0.0}
    
    # Score all pairs
    pos_scores = model.predict(
        [[p["query"], p["document"]] for p in positives]
    )
    neg_scores = model.predict(
        [[p["query"], p["document"]] for p in negatives]
    )
    
    avg_pos = float(sum(pos_scores)) / len(pos_scores)
    avg_neg = float(sum(neg_scores)) / len(neg_scores)
    
    # Accuracy: what fraction of positives score higher than average negative
    correct = sum(1 for s in pos_scores if float(s) > avg_neg)
    accuracy = correct / len(pos_scores)
    
    return {
        "accuracy": round(accuracy, 4),
        "avg_pos_score": round(avg_pos, 4),
        "avg_neg_score": round(avg_neg, 4),
        "avg_score_gap": round(avg_pos - avg_neg, 4),
    }


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("=" * 70)
    print("RERANKER FINE-TUNING - ms-marco-MiniLM -> Validex Domain")
    print("=" * 70)
    
    epochs = 3
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        if idx + 1 < len(sys.argv):
            epochs = int(sys.argv[idx + 1])
    
    try:
        report = train_reranker(epochs=epochs)
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"  Training:    {report['training_pairs']} pairs, {report['epochs']} epochs")
        print(f"  Time:        {report['training_time_seconds']:.1f}s")
        print(f"  Before:      Accuracy={report['before']['accuracy']*100:.1f}%, Gap={report['before']['avg_score_gap']:.4f}")
        print(f"  After:       Accuracy={report['after']['accuracy']*100:.1f}%, Gap={report['after']['avg_score_gap']:.4f}")
        print(f"  Improvement: Acc Δ={report['improvement']['accuracy_delta']*100:+.1f}%")
        print(f"  Model saved: {report['output_dir']}")
        print(f"{'='*70}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
