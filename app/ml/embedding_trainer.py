"""Embedding Trainer — Fine-tunes BAAI/bge-base-en-v1.5 on Validex domain data.

Uses MultipleNegativesRankingLoss (Contrastive Learning) to teach the
embedding model to understand Australian legal/compliance terminology.

After training:
  - Model saved to data/models/bge-base-finetuned-validex/
  - Evaluate with Recall@5, MRR@5 on golden test set
  - Integration: langchain_pipeline.py loads fine-tuned model automatically
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TRIPLETS_PATH = "data/ml/contrastive_triplets.jsonl"
DEFAULT_OUTPUT_DIR = "data/models/bge-base-finetuned-validex"
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EVALUATION_REPORT_PATH = "data/ml/embedding_training_report.json"


def _load_triplets(path: str = DEFAULT_TRIPLETS_PATH) -> list[dict]:
    """Load contrastive triplets from JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Triplets file not found: {path}\n"
            "Run contrastive_data_builder.py first to generate training data."
        )
    
    triplets = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                triplets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    logger.info("Loaded %d triplets from %s", len(triplets), path)
    return triplets


def _build_golden_test_set() -> list[dict]:
    """Build evaluation set from golden answers + domain knowledge.
    
    Returns list of {"query": str, "positive_doc_id": str, "positive_content": str}
    """
    eval_set = []
    
    # Load golden answers
    golden_path = Path("data/goldens/golden_answers.json")
    if golden_path.exists():
        try:
            goldens = json.loads(golden_path.read_text(encoding="utf-8"))
            for g in goldens:
                eval_set.append({
                    "query": g.get("title", ""),
                    "positive_content": g.get("golden_draft", ""),
                })
        except json.JSONDecodeError:
            pass
    
    # Add hand-crafted evaluation pairs (domain expert knowledge)
    eval_pairs = [
        {"query": "How long does a police check take in Australia",
         "expected_doc": "doc_06_faq_processing_time"},
        {"query": "What documents do I need for a police check",
         "expected_doc": "doc_07_required_documents"},
        {"query": "Working with children check requirements",
         "expected_doc": "doc_09_wwcc_guide"},
        {"query": "NDIS worker screening process",
         "expected_doc": "doc_10_ndis_screening"},
        {"query": "What are spent convictions in Australia",
         "expected_doc": "doc_11_spent_convictions"},
        {"query": "Identity verification 100 point check",
         "expected_doc": "doc_12_identity_verification"},
        {"query": "Employer compliance obligations for background checks",
         "expected_doc": "doc_13_employer_compliance"},
        {"query": "Immigration and work visa police check requirements",
         "expected_doc": "doc_14_immigration_work"},
        {"query": "Aged care worker screening requirements Australia",
         "expected_doc": "doc_15_aged_care_screening"},
        {"query": "Police check processing time 2024 2025",
         "expected_doc": "doc_16_processing_times"},
        {"query": "Privacy Act and employer screening obligations",
         "expected_doc": "doc_17_privacy_obligations"},
        {"query": "Who needs a police check for employment",
         "expected_doc": "doc_08_who_needs_police_check"},
    ]
    
    # Load actual document content for these pairs
    docs_dir = Path("data/processed")
    for pair in eval_pairs:
        doc_path = docs_dir / f"{pair['expected_doc']}.txt"
        if doc_path.exists():
            pair["positive_content"] = doc_path.read_text(encoding="utf-8").strip()
            eval_set.append(pair)
    
    logger.info("Built evaluation set with %d pairs", len(eval_set))
    return eval_set


def evaluate_model(model, eval_set: list[dict], all_docs: list[dict], top_k: int = 5) -> dict:
    """Evaluate embedding model on the golden test set.
    
    Metrics:
    - Recall@K: fraction of queries where the correct doc is in top-K results
    - MRR@K: Mean Reciprocal Rank (1/rank of first correct result)
    """
    if not eval_set or not all_docs:
        return {"recall_at_k": 0.0, "mrr_at_k": 0.0, "top_k": top_k}
    
    # Encode all documents once
    doc_contents = [d["content"][:500] for d in all_docs]
    doc_ids = [d.get("doc_id", d.get("expected_doc", "")) for d in all_docs]
    
    doc_embeddings = model.encode(doc_contents, show_progress_bar=False, normalize_embeddings=True)
    
    hits = 0
    reciprocal_ranks = []
    
    for item in eval_set:
        query = item.get("query", "")
        expected_doc = item.get("expected_doc", "")
        
        if not query:
            continue
        
        # Encode query
        query_emb = model.encode([query], normalize_embeddings=True)
        
        # Compute similarities
        import numpy as np
        similarities = np.dot(doc_embeddings, query_emb.T).flatten()
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_doc_ids = [doc_ids[i] for i in top_indices]
        
        # Check if expected doc is in top-K
        if expected_doc and expected_doc in top_doc_ids:
            hits += 1
            rank = top_doc_ids.index(expected_doc) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    total = len(eval_set)
    recall = hits / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    return {
        "recall_at_k": round(recall, 4),
        "mrr_at_k": round(mrr, 4),
        "top_k": top_k,
        "total_queries": total,
        "hits": hits,
    }


def train_embedding(
    triplets_path: str = DEFAULT_TRIPLETS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_model: str = BASE_MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> dict[str, Any]:
    """Fine-tune embedding model using contrastive learning.
    
    Returns training report with before/after metrics.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader
    
    # Load data
    triplets = _load_triplets(triplets_path)
    if len(triplets) < 10:
        raise ValueError(f"Too few triplets ({len(triplets)}). Need at least 10 for training.")
    
    # Split: 90% train, 10% dev
    split_idx = max(1, int(len(triplets) * 0.9))
    train_triplets = triplets[:split_idx]
    dev_triplets = triplets[split_idx:]
    
    logger.info("Training: %d triplets, Dev: %d triplets", len(train_triplets), len(dev_triplets))
    
    # Load base model
    logger.info("Loading base model: %s", base_model)
    model = SentenceTransformer(base_model)
    
    # Load documents for evaluation
    docs_dir = Path("data/processed")
    all_docs = []
    for f in sorted(docs_dir.glob("*.txt")):
        content = f.read_text(encoding="utf-8").strip()
        if content:
            all_docs.append({"doc_id": f.stem, "content": content})
    
    # Build golden eval set
    eval_set = _build_golden_test_set()
    
    # ── Before metrics ──
    logger.info("Evaluating BASE model (before fine-tuning)...")
    before_metrics = evaluate_model(model, eval_set, all_docs)
    logger.info("BEFORE — Recall@5: %.2f%%, MRR@5: %.4f",
                before_metrics["recall_at_k"] * 100, before_metrics["mrr_at_k"])
    
    # ── Prepare training data ──
    train_examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in train_triplets
    ]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # MultipleNegativesRankingLoss: the gold standard for contrastive embedding training
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Dev evaluator
    dev_evaluator = None
    if dev_triplets:
        dev_evaluator = TripletEvaluator(
            anchors=[t["anchor"] for t in dev_triplets],
            positives=[t["positive"] for t in dev_triplets],
            negatives=[t["negative"] for t in dev_triplets],
            name="validex-dev",
        )
    
    # ── Train ──
    warmup_steps = max(1, int(len(train_dataloader) * epochs * warmup_ratio))
    
    logger.info(
        "Starting training: %d examples, %d epochs, batch=%d, lr=%s, warmup=%d steps",
        len(train_examples), epochs, batch_size, learning_rate, warmup_steps,
    )
    
    start_time = time.time()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        evaluator=dev_evaluator,
        evaluation_steps=max(1, len(train_dataloader) // 2),
        output_path=output_dir,
        show_progress_bar=True,
    )
    
    training_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds", training_time)
    
    # ── After metrics ──
    logger.info("Evaluating FINE-TUNED model (after training)...")
    # Reload the saved model for fair evaluation
    finetuned_model = SentenceTransformer(output_dir)
    after_metrics = evaluate_model(finetuned_model, eval_set, all_docs)
    logger.info("AFTER — Recall@5: %.2f%%, MRR@5: %.4f",
                after_metrics["recall_at_k"] * 100, after_metrics["mrr_at_k"])
    
    # ── Report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model,
        "output_dir": output_dir,
        "training_triplets": len(train_triplets),
        "dev_triplets": len(dev_triplets),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": round(training_time, 1),
        "before": before_metrics,
        "after": after_metrics,
        "improvement": {
            "recall_delta": round(after_metrics["recall_at_k"] - before_metrics["recall_at_k"], 4),
            "mrr_delta": round(after_metrics["mrr_at_k"] - before_metrics["mrr_at_k"], 4),
        },
    }
    
    # Save report
    report_path = Path(EVALUATION_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Training report saved to %s", EVALUATION_REPORT_PATH)
    
    return report


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("=" * 70)
    print("EMBEDDING FINE-TUNING - BAAI/bge-base-en-v1.5 -> Validex Domain")
    print("=" * 70)
    
    epochs = 3
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        if idx + 1 < len(sys.argv):
            epochs = int(sys.argv[idx + 1])
    
    try:
        report = train_embedding(epochs=epochs)
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"  Training:      {report['training_triplets']} triplets, {report['epochs']} epochs")
        print(f"  Time:          {report['training_time_seconds']:.1f}s")
        print(f"  Before:        Recall@5={report['before']['recall_at_k']*100:.1f}%, MRR@5={report['before']['mrr_at_k']:.4f}")
        print(f"  After:         Recall@5={report['after']['recall_at_k']*100:.1f}%, MRR@5={report['after']['mrr_at_k']:.4f}")
        print(f"  Improvement:   Recall Δ={report['improvement']['recall_delta']*100:+.1f}%, MRR Δ={report['improvement']['mrr_delta']:+.4f}")
        print(f"  Model saved:   {report['output_dir']}")
        print("=" * 70)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
