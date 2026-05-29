"""ML Collector Node — LangGraph node that harvests training data after blog completion.

Runs as the FINAL node in the RAG pipeline (after Editor), collecting
features and labels from the completed GraphState and persisting them
to the ML training dataset.

This node is passive — it never blocks or rejects a draft. It only
observes and records.
"""

import logging
from app.graph_state import GraphState
from app.agents.base import BaseAgentNode

logger = logging.getLogger(__name__)


class MLCollectorNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Extract ML features and labels from the final pipeline state."""
        logger.info("Executing ML Collector Node")

        # Skip collection for edit-only requests (no RAG involved)
        if state.get("edit_instruction"):
            logger.info("ML Collector: Skipping (edit-only request)")
            return {"ml_features": {}, "ml_quality_prediction": None}

        try:
            from app.ml.ml_data_collector import collect_from_state

            result = collect_from_state(
                state,
                use_llm_judge=False,  # Use heuristic labels (0 API tokens)
            )

            logger.info(
                "ML Collector: recorded features (%d dims), quality=%s",
                len(result.get("ml_features", {})),
                result.get("ml_labels", {}).get("quality_class", "?"),
            )

            return {
                "ml_features": result.get("ml_features", {}),
                "ml_quality_prediction": None,  # No ML model prediction yet
            }
        except Exception as exc:
            logger.error("ML Collector failed (non-blocking): %s", exc)
            return {"ml_features": {}, "ml_quality_prediction": None}


ml_collector_node = MLCollectorNode()
