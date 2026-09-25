"""Cross-Encoder Reranker for RAG retrieval results.

Uses the ms-marco-MiniLM-L-12-v2 ONNX model (already bundled in data/models/)
to rerank candidate documents by computing a precise relevance score for each
(query, document) pair.

This is the highest-ROI upgrade for RAG accuracy: the initial dense retriever
finds "probably relevant" passages, but the ordering is often wrong. The
reranker fixes this by using a cross-encoder that jointly attends to both
the query and the document.

Pipeline integration:
    Researcher → Reranker → RAG_Evaluator
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the bundled ONNX reranker model (pre-trained)
DEFAULT_MODEL_DIR = os.path.join("data", "models", "ms-marco-MiniLM-L-12-v2")

# Path to fine-tuned Validex domain reranker (trained by reranker_trainer.py)
FINETUNED_MODEL_DIR = os.path.join("data", "models", "reranker-finetuned-validex")

# Global singleton
_RERANKER = None
_RERANKER_LOCK = threading.Lock()


@dataclass
class RerankResult:
    """A document with its reranker score."""
    doc: dict[str, Any]
    rerank_score: float


class CrossEncoderReranker:
    """Reranker using FlashRank (ONNX-based, CPU-friendly, no GPU required).

    FlashRank is chosen because:
    - Uses the ONNX model already in data/models/ (no downloads)
    - Runs on CPU with good latency (~50ms for 7 docs)
    - No heavy torch/tensorflow dependency
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, max_length: int = 512):
        self.model_dir = model_dir
        self.max_length = max_length
        self._ranker = None
        self._use_crossencoder = False
        self._load_attempted = False
        self._load_lock = threading.Lock()

    def _load_model(self):
        """Lazy-load the reranker model.

        Priority:
        1. Fine-tuned Validex domain model (data/models/reranker-finetuned-validex/)
        2. FlashRank ONNX pre-trained model
        3. SentenceTransformers CrossEncoder fallback

        Every backend is tried defensively: a missing package, a missing model
        file or a failed download must degrade to the next option (and finally to
        "no reranking"), never propagate to the caller.
        """
        if self._ranker is not None or self._load_attempted:
            return

        # Serialise loading so concurrent requests don't each build a model.
        with self._load_lock:
            if self._ranker is not None or self._load_attempted:
                return
            self._load_attempted = True

            # Priority 1: Fine-tuned domain model
            finetuned_path = Path(FINETUNED_MODEL_DIR)
            if finetuned_path.exists() and (finetuned_path / "config.json").exists():
                try:
                    from sentence_transformers import CrossEncoder
                    self._ranker = CrossEncoder(str(finetuned_path), max_length=self.max_length)
                    self._use_crossencoder = True
                    logger.info("Loaded FINE-TUNED Validex reranker from %s", FINETUNED_MODEL_DIR)
                    return
                except Exception as exc:
                    logger.warning("Failed to load fine-tuned reranker: %s — trying pre-trained", exc)

            # Priority 2: FlashRank ONNX pre-trained
            model_path = Path(self.model_dir)
            if model_path.exists():
                try:
                    from flashrank import Ranker
                    self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(model_path.parent))
                    self._use_crossencoder = False
                    logger.info("CrossEncoder Reranker loaded from %s", self.model_dir)
                    return
                except Exception as exc:
                    logger.warning("Failed to load FlashRank reranker: %s — trying HF fallback", exc)

            # Priority 3: SentenceTransformers CrossEncoder (download from HF)
            try:
                from sentence_transformers import CrossEncoder
                self._ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=self.max_length)
                self._use_crossencoder = True
                logger.info("Loaded SentenceTransformers CrossEncoder (pre-trained) as fallback")
            except Exception as exc:
                logger.warning("No reranker backend available (%s) — reranking disabled", exc)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """Rerank documents by cross-encoder relevance to the query.

        Parameters
        ----------
        query : str
            The user's search query or topic.
        documents : list[dict]
            Candidate documents from the retriever. Each must have a
            ``content_key`` field with the text to score.
        top_k : int, optional
            If provided, return only the top-K reranked documents.
        content_key : str
            The key in each document dict that holds the text content.

        Returns
        -------
        list[dict]
            Documents sorted by reranker score (descending), with an added
            ``rerank_score`` field.
        """
        if not documents or not query:
            return documents

        try:
            self._load_model()
        except Exception as exc:
            logger.warning("Reranker model load failed: %s — returning original order", exc)
            return documents

        if self._ranker is None:
            logger.debug("Reranker not loaded — returning documents in original order")
            return documents

        try:
            if getattr(self, '_use_crossencoder', False):
                return self._rerank_with_crossencoder(query, documents, top_k, content_key)
            else:
                return self._rerank_with_flashrank(query, documents, top_k, content_key)
        except Exception as exc:
            logger.warning("Reranking failed: %s — returning original order", exc)
            return documents

    def _rerank_with_flashrank(
        self, query: str, documents: list[dict], top_k: int | None, content_key: str
    ) -> list[dict]:
        """Rerank using FlashRank library."""
        from flashrank import RerankRequest

        # FlashRank expects list of dicts with "text" key
        passages = []
        for doc in documents:
            passages.append({
                "id": doc.get("doc_id", ""),
                "text": str(doc.get(content_key, ""))[:self.max_length * 4],
                "meta": doc,
            })

        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)

        # Rebuild document list with rerank scores. Copy rather than write into
        # result["meta"], which is the caller's own dict.
        reranked = []
        for result in results:
            doc_copy = dict(result["meta"])
            doc_copy["rerank_score"] = float(result["score"])
            reranked.append(doc_copy)

        # Sort descending by rerank_score
        reranked.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        logger.info(
            "Reranker (FlashRank): reranked %d docs, top score=%.4f",
            len(reranked),
            reranked[0].get("rerank_score", 0) if reranked else 0,
        )
        return reranked

    def _rerank_with_crossencoder(
        self, query: str, documents: list[dict], top_k: int | None, content_key: str
    ) -> list[dict]:
        """Rerank using sentence-transformers CrossEncoder."""
        pairs = []
        for doc in documents:
            text = str(doc.get(content_key, ""))[:self.max_length * 4]
            pairs.append([query, text])

        scores = self._ranker.predict(pairs)

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)

        if top_k:
            scored_docs = scored_docs[:top_k]

        logger.info(
            "Reranker (CrossEncoder): reranked %d docs, top score=%.4f",
            len(scored_docs),
            scored_docs[0]["rerank_score"] if scored_docs else 0,
        )
        return scored_docs


# Module-level singleton
def get_reranker() -> CrossEncoderReranker:
    """Get or create the global reranker instance."""
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                _RERANKER = CrossEncoderReranker()
    return _RERANKER
