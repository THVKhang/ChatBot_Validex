"""RAG Evaluator — Acts as a quality gate between Retrieval and Generation.

Evaluates the relevance of retrieved documents to the parsed prompt.
If the context score is below a threshold, it triggers a fallback web search
to enrich the context before handing off to the Writer Node.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.langchain_pipeline import pipeline
from app.parser import ParsedPrompt

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Structured output for context evaluation."""
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief reasoning for the score")
    needs_enrichment: bool = Field(description="True if score < 0.3 or missing key info")


class RAGEvaluator:
    """Evaluates RAG context relevance and enriches if necessary."""

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def evaluate_context(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
    ) -> EvaluationResult:
        """Evaluate if the retrieved documents adequately cover the topic."""
        if not docs:
            logger.warning("RAG Evaluator: No documents retrieved")
            return EvaluationResult(
                score=0.0,
                reasoning="No documents available in context.",
                needs_enrichment=True,
            )

        if pipeline._llm is None:
            # Fallback heuristic: assume good if we have docs and no LLM to check
            return EvaluationResult(score=0.5, reasoning="LLM disabled, heuristic pass", needs_enrichment=False)

        doc_summary = "\n".join([f"- {d.metadata.get('title', 'Unknown')}: {d.page_content[:150]}" for d in docs[:5]])
        
        prompt = (
            f"You are evaluating the relevance of retrieved documents for a blog topic.\n"
            f"Topic: {parsed.topic}\n"
            f"Intent: {parsed.intent}\n\n"
            f"Retrieved Documents:\n{doc_summary}\n\n"
            "Score how well these documents cover the topic from 0.0 to 1.0. "
            "Return a JSON object with: 'score' (float), 'reasoning' (string), "
            "'needs_enrichment' (boolean - true if score < 0.3)."
        )

        try:
            # Use structured output for reliable JSON parsing
            structured_llm = pipeline._llm.with_structured_output(EvaluationResult)
            result = structured_llm.invoke(prompt)
            if isinstance(result, EvaluationResult):
                logger.info("RAG Evaluator: score=%.2f, needs_enrichment=%s", result.score, result.needs_enrichment)
                return result
        except Exception as exc:
            logger.warning("RAG Evaluator failed: %s", exc)

        # Fallback to heuristic
        return EvaluationResult(score=0.5, reasoning="Evaluation failed, using heuristic", needs_enrichment=False)

    def enrich_context(self, parsed: ParsedPrompt, current_docs: list[Document]) -> list[Document]:
        """Trigger web search to enrich context if original retrieval was poor."""
        logger.info("RAG Evaluator enriching context for topic: %s", parsed.topic)
        try:
            from app.agents.researcher_node import researcher_node
            from app.graph_state import GraphState
            
            # Create a dummy state just for the researcher to fetch more via web
            mock_state = GraphState(
                prompt=parsed.topic,
                parsed=parsed.model_dump(),
                retrieved_docs=[],  # Start fresh for web search
                session=None,       # type: ignore
            )
            # Temporarily force researcher to bypass DB and go straight to web
            result_state = researcher_node(mock_state)
            new_docs_data = result_state.get("retrieved_docs", [])
            
            new_docs = [
                Document(
                    page_content=d["content"],
                    metadata={
                        "doc_id": d["doc_id"],
                        "score": d["score"],
                        "source": d.get("source", "web_search"),
                        "title": d.get("title", ""),
                        "source_url": d.get("source_url", ""),
                    }
                )
                for d in new_docs_data
            ]
            
            # Combine without duplicates
            seen_urls = {d.metadata.get("source_url") for d in current_docs if d.metadata.get("source_url")}
            enriched_docs = list(current_docs)
            for d in new_docs:
                if d.metadata.get("source_url") not in seen_urls:
                    enriched_docs.append(d)
                    
            logger.info("RAG Evaluator added %d new docs from enrichment", len(enriched_docs) - len(current_docs))
            return enriched_docs
            
        except Exception as exc:
            logger.error("RAG Evaluator context enrichment failed: %s", exc)
            return current_docs

rag_evaluator = RAGEvaluator(threshold=0.3)
