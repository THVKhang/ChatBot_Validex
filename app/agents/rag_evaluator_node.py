"""RAG Evaluator Node — evaluates retrieved context quality and decides retry."""
import logging
from langchain_core.documents import Document
from app.graph_state import GraphState
from app.rag_evaluator import rag_evaluator

logger = logging.getLogger(__name__)

from app.agents.base import BaseAgentNode

class RagEvaluatorAgentNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Evaluate retrieved context with multi-dimensional scoring and decide if Researcher needs to retry."""
        logger.info("Executing RAG Evaluator Node")
        
        parsed_data = state.get("parsed", {})
        from app.parser import ParsedPrompt
        parsed_prompt = ParsedPrompt(
            raw_prompt=state.get("prompt", ""),
            intent=parsed_data.get("intent", ""),
            topic=parsed_data.get("topic", ""),
            audience=parsed_data.get("audience", ""),
            tone=parsed_data.get("tone", ""),
            length=parsed_data.get("length", ""),
            custom_instructions=parsed_data.get("context_note", "")
        )
        
        retrieved_docs_data = state.get("retrieved_docs", [])
        
        # Convert state dict back to Langchain Document objects for evaluation
        docs = []
        for d in retrieved_docs_data:
            docs.append(Document(
                page_content=d.get("content", ""),
                metadata={
                    "doc_id": d.get("doc_id", ""),
                    "score": d.get("score", 0.0),
                    "source": d.get("source", ""),
                    "title": d.get("title", ""),
                    "source_url": d.get("source_url", ""),
                }
            ))
            
        eval_result = rag_evaluator.evaluate_context(parsed_prompt, docs)
        
        if eval_result.needs_enrichment:
            # Build specific feedback so Researcher knows WHAT to look for
            feedback_parts = [
                f"Context insufficient (score {eval_result.score:.2f}).",
                f"Relevance={eval_result.relevance:.2f}, Coverage={eval_result.coverage:.2f}, "
                f"Diversity={eval_result.diversity:.2f}, Freshness={eval_result.freshness:.2f}."
            ]
            if eval_result.missing_aspects:
                missing_str = "; ".join(eval_result.missing_aspects[:3])
                feedback_parts.append(f"Missing aspects: {missing_str}")
            
            feedback = " ".join(feedback_parts)
            logger.warning(f"RAG Evaluator: {feedback}")
            return {"rag_feedback": feedback}
        
        logger.info(
            f"RAG Evaluator: Context approved (score {eval_result.score:.2f}, "
            f"rel={eval_result.relevance:.2f}, cov={eval_result.coverage:.2f})"
        )
        return {"rag_feedback": None}

rag_evaluator_node = RagEvaluatorAgentNode()

