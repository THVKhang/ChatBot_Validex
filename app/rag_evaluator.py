"""RAG Evaluator — Multi-dimensional quality gate between Retrieval and Generation.

Evaluates retrieved documents across 4 dimensions:
  - Relevance: Semantic similarity to topic
  - Coverage: How many sub-aspects of the topic are covered
  - Diversity: How varied the source content is
  - Freshness: Metadata-based recency score

If the weighted score is below threshold, triggers Researcher retry with
specific feedback about WHAT is missing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.parser import ParsedPrompt

logger = logging.getLogger(__name__)

# Cosine-similarity calibration band for the local sentence encoder.
#
# BGE-class encoders compress their similarity range: unrelated text still
# scores ~0.35-0.42, so a band anchored near zero marks everything "relevant".
# These bounds were measured against the live corpus with the fine-tuned
# bge-base model — blended (0.7*max + 0.3*avg) similarity of the top-5 retrieved
# chunks over five on-topic and five off-topic queries:
#
#     on-topic   0.525 .. 0.813   (WWCC, spent convictions, NDIS, police checks)
#     off-topic  0.335 .. 0.414   (sourdough, hiking, asyncio, jazz, bicycles)
#
# Floor sits just above the off-topic ceiling, ceiling near the on-topic best.
# RE-MEASURE THESE IF THE ENCODER CHANGES — they are model-specific, not
# universal cosine thresholds.
RELEVANCE_FLOOR = 0.42
RELEVANCE_CEILING = 0.80

# A sub-aspect counts as covered once some document reaches this similarity.
# Sits inside the same measured gap, above off-topic noise.
COVERAGE_SIM_THRESHOLD = 0.55


class EvaluationResult(BaseModel):
    """Structured output for context evaluation."""
    score: float = Field(description="Weighted score from 0.0 to 1.0")
    relevance: float = Field(default=0.0, description="Semantic relevance 0-1")
    coverage: float = Field(default=0.0, description="Topic coverage 0-1")
    diversity: float = Field(default=0.0, description="Source diversity 0-1")
    freshness: float = Field(default=0.0, description="Recency score 0-1")
    reasoning: str = Field(description="Brief reasoning for the score")
    missing_aspects: list[str] = Field(default_factory=list, description="Sub-topics not covered")
    needs_enrichment: bool = Field(description="True if weighted score < threshold")


class RAGEvaluator:
    """Multi-dimensional RAG context evaluator."""

    # Weights for final score
    W_RELEVANCE = 0.40
    W_COVERAGE = 0.30
    W_DIVERSITY = 0.20
    W_FRESHNESS = 0.10

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    # ── Metric 1: Relevance (semantic similarity) ──────────────────
    def _score_relevance(self, topic: str, docs: list[Document]) -> float:
        """Cosine similarity between topic and doc contents."""
        try:
            from app.local_semantics import get_embedding, get_embeddings, batch_cosine_similarity

            topic_emb = get_embedding(topic)
            doc_texts = [doc.page_content[:500] for doc in docs[:5]]
            doc_embs = get_embeddings(doc_texts)

            sims = batch_cosine_similarity(topic_emb, doc_embs)
            max_sim = float(max(sims))
            avg_sim = float(sum(sims) / len(sims))

            # Blend best-match and overall similarity, then rescale the band that
            # actually discriminates for MiniLM/BGE-class encoders. The previous
            # formula ((max*2)+(avg*0.5)) saturated at 1.0 from max_sim >= 0.4,
            # which made this dimension a constant for anything on-topic.
            blended = (0.7 * max_sim) + (0.3 * avg_sim)
            normalized = (blended - RELEVANCE_FLOOR) / (RELEVANCE_CEILING - RELEVANCE_FLOOR)
            return min(1.0, max(0.0, normalized))
        except Exception as exc:
            logger.error("Relevance scoring failed: %s", exc)
            return 0.5

    # ── Metric 2: Coverage (sub-aspect check) ──────────────────────
    def _generate_sub_aspects(self, topic: str) -> list[str]:
        """Generate sub-aspects a good article about this topic should cover."""
        # Common article aspects
        base_aspects = [
            f"what is {topic}",
            f"how {topic} works",
            f"requirements for {topic}",
            f"benefits of {topic}",
            f"process and steps for {topic}",
        ]
        return base_aspects

    def _score_coverage(self, topic: str, docs: list[Document]) -> tuple[float, list[str]]:
        """Check how many sub-aspects are covered by at least one doc.
        Returns (score, list_of_missing_aspects).
        """
        try:
            from app.local_semantics import get_embedding, get_embeddings, batch_cosine_similarity

            sub_aspects = self._generate_sub_aspects(topic)
            if not sub_aspects:
                return 0.5, []

            doc_texts = [doc.page_content[:600] for doc in docs[:5]]
            if not doc_texts:
                return 0.0, sub_aspects

            doc_embs = get_embeddings(doc_texts)
            # One batched encode for every aspect instead of one call per aspect.
            aspect_embs = get_embeddings(sub_aspects)

            covered = 0
            missing = []
            for aspect, aspect_emb in zip(sub_aspects, aspect_embs):
                sims = batch_cosine_similarity(aspect_emb, doc_embs)
                best_sim = float(max(sims))
                if best_sim >= COVERAGE_SIM_THRESHOLD:
                    covered += 1
                else:
                    missing.append(aspect)

            score = covered / len(sub_aspects)
            return score, missing
        except Exception as exc:
            logger.error("Coverage scoring failed: %s", exc)
            return 0.5, []

    # ── Metric 3: Diversity (content variety) ──────────────────────
    def _score_diversity(self, docs: list[Document]) -> float:
        """Measure how diverse the retrieved docs are (low similarity = high diversity)."""
        if len(docs) <= 1:
            return 0.5

        try:
            from app.local_semantics import get_embeddings, batch_cosine_similarity

            doc_texts = [doc.page_content[:400] for doc in docs[:5]]
            doc_embs = get_embeddings(doc_texts)

            # Average pairwise cosine similarity. Must go through
            # batch_cosine_similarity: a raw dot product is only equal to cosine
            # when the encoder happens to emit unit-norm vectors, which is not
            # guaranteed across the fine-tuned and fallback models.
            total_sim = 0.0
            pair_count = 0
            for i in range(len(doc_embs)):
                for j in range(i + 1, len(doc_embs)):
                    sim = float(batch_cosine_similarity(doc_embs[i], doc_embs[j:j + 1])[0])
                    total_sim += sim
                    pair_count += 1

            if pair_count == 0:
                return 0.5

            avg_pairwise_sim = total_sim / pair_count
            # Low pairwise similarity = high diversity → invert
            diversity = 1.0 - min(1.0, max(0.0, avg_pairwise_sim))
            return diversity
        except Exception as exc:
            logger.error("Diversity scoring failed: %s", exc)
            return 0.5

    # ── Metric 4: Freshness (metadata-based) ──────────────────────
    def _score_freshness(self, docs: list[Document]) -> float:
        """Score based on document recency metadata."""
        from datetime import datetime

        scores = []
        now = datetime.now()

        for doc in docs[:5]:
            last_updated = doc.metadata.get("last_updated", "")
            if not last_updated:
                scores.append(0.5)  # Unknown = neutral
                continue
            try:
                doc_date = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
                days_old = (now - doc_date.replace(tzinfo=None)).days
                # < 30 days = 1.0, 30-180 = 0.7, 180-365 = 0.5, > 365 = 0.3
                if days_old < 30:
                    scores.append(1.0)
                elif days_old < 180:
                    scores.append(0.7)
                elif days_old < 365:
                    scores.append(0.5)
                else:
                    scores.append(0.3)
            except (ValueError, TypeError):
                scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.5

    # ── Main Evaluation ────────────────────────────────────────────
    def evaluate_context(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
    ) -> EvaluationResult:
        """Multi-dimensional evaluation of retrieved context."""
        if not docs:
            logger.warning("RAG Evaluator: No documents retrieved")
            return EvaluationResult(
                score=0.0,
                reasoning="No documents available in context.",
                missing_aspects=[f"everything about {parsed.topic}"],
                needs_enrichment=True,
            )

        if not parsed.topic:
            return EvaluationResult(score=0.5, reasoning="Empty topic", needs_enrichment=False)

        # Score each dimension
        relevance = self._score_relevance(parsed.topic, docs)
        coverage, missing_aspects = self._score_coverage(parsed.topic, docs)
        diversity = self._score_diversity(docs)
        freshness = self._score_freshness(docs)

        # Weighted final score
        weighted = (
            self.W_RELEVANCE * relevance
            + self.W_COVERAGE * coverage
            + self.W_DIVERSITY * diversity
            + self.W_FRESHNESS * freshness
        )

        needs_enrichment = weighted < self.threshold

        reasoning = (
            f"rel={relevance:.2f} cov={coverage:.2f} "
            f"div={diversity:.2f} fresh={freshness:.2f} → "
            f"weighted={weighted:.2f} (threshold={self.threshold})"
        )

        logger.info(
            "RAG Evaluator: %s | missing=%s | enrich=%s",
            reasoning, missing_aspects[:3], needs_enrichment,
        )

        return EvaluationResult(
            score=weighted,
            relevance=relevance,
            coverage=coverage,
            diversity=diversity,
            freshness=freshness,
            reasoning=reasoning,
            missing_aspects=missing_aspects,
            needs_enrichment=needs_enrichment,
        )

    def enrich_context(self, parsed: ParsedPrompt, current_docs: list[Document]) -> list[Document]:
        """Trigger web search to enrich context if original retrieval was poor."""
        logger.info("RAG Evaluator enriching context for topic: %s", parsed.topic)
        try:
            from app.agents.researcher_node import _web_search_with_scraping

            web_docs = _web_search_with_scraping(parsed.topic, max_results=3)

            # Combine without duplicates
            seen_urls = {d.metadata.get("source_url") for d in current_docs if d.metadata.get("source_url")}
            enriched_docs = list(current_docs)
            for d in web_docs:
                if d.metadata.get("source_url") not in seen_urls:
                    enriched_docs.append(d)

            logger.info("RAG Evaluator added %d new docs from enrichment", len(enriched_docs) - len(current_docs))
            return enriched_docs

        except Exception as exc:
            logger.error("RAG Evaluator context enrichment failed: %s", exc)
            return current_docs


rag_evaluator = RAGEvaluator(threshold=0.45)
