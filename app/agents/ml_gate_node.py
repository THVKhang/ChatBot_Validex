"""ML Quality Gate Node — LangGraph node that uses trained ML models
to predict blog quality and optionally block low-quality outputs.

Pipeline position:
    Editor → ML_Quality_Gate → ML_Collector → END
                   ↓ (if blocked in ENFORCE mode)
              Writer (generation issue) OR Researcher (retrieval issue)

Operating Modes:
  SHADOW (default): Logs predictions for human review. Never blocks the pipeline.
  ENFORCE: Auto-rejects when model has high confidence (>80%) that quality is low.

CRITICAL ANTI-PATTERN FIXED (Survivorship Bias):
  When blocking a draft, the gate STILL records the rejected sample to the
  training dataset with quality_class="low". This prevents the "echo chamber"
  where training data only contains passing samples and the model forgets
  what bad content looks like.
"""

import logging
from app.graph_state import GraphState
from app.agents.base import BaseAgentNode

logger = logging.getLogger(__name__)


class MLGateNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Run ML quality prediction on the current draft.

        In SHADOW mode: logs prediction, never blocks.
        In ENFORCE mode: blocks low-quality drafts, routes to appropriate node.

        IMPORTANT: Always records the sample to training data, whether
        passing or blocking. This prevents Survivorship Bias.
        """
        logger.info("Executing ML Quality Gate Node")

        # Skip for edit-only requests
        if state.get("edit_instruction"):
            logger.info("ML Gate: Skipping (edit-only request)")
            return {"ml_quality_prediction": None}

        # Skip if draft is empty
        draft = state.get("draft", "")
        if not draft:
            logger.info("ML Gate: Skipping (empty draft)")
            return {"ml_quality_prediction": None}

        try:
            from app.ml.ml_data_collector import extract_features
            from app.ml.ml_quality_gate import get_ml_quality_gate

            # Extract features from current state
            features = extract_features(state)

            # Get ML prediction
            gate = get_ml_quality_gate()
            prediction = gate.predict_quality(features)

            if not prediction.get("model_available", False):
                logger.info("ML Gate: No trained model available — pass-through")
                return {
                    "ml_features": features,
                    "ml_quality_prediction": prediction,
                }

            # Check if we should block (respects Shadow/Enforce mode)
            should_block, reason = gate.should_block(prediction, features)
            mode = prediction.get("gate_mode", "shadow")

            if should_block:
                # Only allow blocking once per pipeline run to prevent infinite loops
                revision_count = state.get("revision_count", 0)
                if revision_count >= 2:
                    logger.warning(
                        "ML Gate [%s]: Would block but revision_count=%d — allowing through",
                        mode.upper(), revision_count,
                    )
                    return {
                        "ml_features": features,
                        "ml_quality_prediction": prediction,
                    }

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # FIX: Survivorship Bias — Record the REJECTED sample
                # BEFORE routing back. The ML Collector at the END only
                # sees passing samples. Without this, training data
                # contains only "high quality" samples and the model
                # forgets what "low quality" looks like.
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                self._record_rejected_sample(state, features, prediction, reason)

                # Determine which node to route back to (dual-scope)
                route_target = gate.diagnose_route(prediction)
                failure_source = prediction.get("failure_source", "none")

                logger.warning(
                    "ML Gate [%s] BLOCKED → %s (failure=%s): %s",
                    mode.upper(), route_target, failure_source, reason,
                )

                feedback = f"[ML Quality Gate | {failure_source}] {reason}"

                result = {
                    "ml_features": features,
                    "ml_quality_prediction": prediction,
                }

                if route_target == "Researcher":
                    result["rag_feedback"] = feedback
                else:
                    result["editor_feedback"] = feedback

                return result

            logger.info("ML Gate [%s]: %s", mode.upper(), reason)
            return {
                "ml_features": features,
                "ml_quality_prediction": prediction,
            }

        except Exception as exc:
            logger.error("ML Gate failed (non-blocking): %s", exc)
            return {"ml_quality_prediction": None}

    def _record_rejected_sample(
        self,
        state: dict,
        features: dict,
        prediction: dict,
        reason: str,
    ) -> None:
        """Record a rejected/blocked draft as a NEGATIVE training sample.

        This is the critical fix for Survivorship Bias. Without this,
        training data only contains passing (high quality) samples.

        The rejected sample is saved with:
        - quality_class = "low" (forced — ML model said so)
        - label_source = "ml_gate_reject" (distinguishable from heuristic/llm)
        - faithfulness_score from ML prediction
        """
        try:
            from app.ml.ml_data_collector import save_training_record

            labels = {
                "faithfulness_score": prediction.get("faithfulness_pred", 0.2),
                "relevance_score": 0.3,  # Rejected → low relevance
                "quality_class": "low",  # Forced negative label
                "label_source": "ml_gate_reject",
            }

            parsed = state.get("parsed", {})
            metadata = {
                "topic": parsed.get("topic", ""),
                "intent": parsed.get("intent", ""),
                "complexity_level": state.get("complexity_level", "simple"),
                "request_id": state.get("request_id", ""),
                "rejection_reason": reason,
                "failure_source": prediction.get("failure_source", "none"),
                "gate_mode": prediction.get("gate_mode", "shadow"),
            }

            save_training_record(features, labels, metadata)
            logger.info(
                "ML Gate: Recorded REJECTED sample (Hard Negative) to training data"
            )
        except Exception as exc:
            logger.warning("ML Gate: Failed to record rejected sample: %s", exc)


ml_gate_node = MLGateNode()
