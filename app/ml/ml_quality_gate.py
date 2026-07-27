"""ML Quality Gate — Uses trained ML models to predict blog quality in real-time.

Loads the trained XGBoost models from data/ml/models/ and provides:
  - predict_quality(): Returns quality class + faithfulness prediction + confidence
  - should_block(): Decision function with Shadow Mode support
  - diagnose_route(): Determines if failure is retrieval-side or generation-side

Operating Modes:
  - SHADOW (Phase 1): Log predictions but NEVER block. For human review.
  - ENFORCE (Phase 2): Auto-reject when F1-score > 0.85 and confidence > 80%.

Red Team Mitigations:
  - Concept Drift Detection: Compares incoming features against training
    distribution. If >50% of features are >3σ out-of-distribution,
    auto-disables the gate to prevent false blocking from stale models.
  - Uses active_features from feature_meta.json (supports pruned feature sets).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "data/ml/models"
DEFAULT_REPORT_PATH = "data/ml/training_report.json"
SHADOW_LOG_PATH = "data/ml/shadow_mode_log.jsonl"

# ── Operating Mode ──────────────────────────────────────────────────
# SHADOW = log only (never block), ENFORCE = auto-reject low quality
ML_GATE_MODE = os.getenv("ML_GATE_MODE", "shadow")  # "shadow" or "enforce"

# F1-score threshold to auto-promote from SHADOW → ENFORCE
F1_PROMOTION_THRESHOLD = 0.85

# Confidence threshold: only block if model is >80% confident the quality is "low"
BLOCK_CONFIDENCE_THRESHOLD = 0.80
# Faithfulness threshold: also block if predicted faithfulness is very low
FAITHFULNESS_BLOCK_THRESHOLD = 0.25

# Feature columns (must match ml_trainer.FEATURE_COLUMNS exactly)
# NOTE: May be overridden at runtime by feature_meta.json (when features are pruned)
FEATURE_COLUMNS = [
    "retrieval_doc_count",
    "retrieval_score_mean",
    "retrieval_score_max",
    "retrieval_semantic_mean",
    "retrieval_relevance",
    "retrieval_coverage",
    "retrieval_diversity",
    "retrieval_freshness",
    "source_authority_mean",
    "draft_word_count",
    "draft_heading_count",
    "draft_fk_grade",
    "draft_reading_ease",
    "draft_keyword_density",
    "semantic_coherence",
    "nli_contradictions",
    "topic_complexity_score",
    "revision_count",
    "retrieval_attempts",
]

QUALITY_CLASS_INV = {0: "low", 1: "medium", 2: "high"}

# ── Concept Drift Detection ─────────────────────────────────────────
# If a feature value is > N standard deviations from training mean → OOD
DRIFT_SIGMA_THRESHOLD = 3.0
# If > X% of features are OOD → declare drift and disable gate
DRIFT_FEATURE_FRACTION = 0.50


def _get_model_f1_score() -> float:
    """Read the classifier's macro F1-score from the last training report."""
    report_path = Path(DEFAULT_REPORT_PATH)
    if not report_path.exists():
        return 0.0
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        clf_report = report.get("classifier", {}).get("classification_report", {})
        # macro avg F1 from sklearn classification_report
        return float(clf_report.get("macro avg", {}).get("f1-score", 0.0))
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0.0


def _resolve_mode() -> str:
    """Determine the effective operating mode.

    Priority:
    1. Explicit env var ML_GATE_MODE="enforce" → enforce
    2. Auto-promotion: if F1 > threshold → enforce
    3. Default → shadow
    """
    explicit = ML_GATE_MODE.lower().strip()
    if explicit == "enforce":
        return "enforce"

    # Auto-promotion check
    f1 = _get_model_f1_score()
    if f1 >= F1_PROMOTION_THRESHOLD:
        logger.info(
            "ML Gate: Auto-promoting to ENFORCE mode (F1=%.3f >= %.3f)",
            f1, F1_PROMOTION_THRESHOLD,
        )
        return "enforce"

    return "shadow"


def _log_shadow_prediction(
    prediction: dict[str, Any],
    would_block: bool,
    reason: str,
    features: dict[str, Any] | None = None,
) -> None:
    """Log prediction to shadow mode log for human review."""
    import time
    path = Path(SHADOW_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "shadow",
        "prediction": prediction,
        "would_block": would_block,
        "reason": reason,
        "features_summary": {
            "retrieval_relevance": features.get("retrieval_relevance", 0) if features else 0,
            "retrieval_coverage": features.get("retrieval_coverage", 0) if features else 0,
            "nli_contradictions": features.get("nli_contradictions", 0) if features else 0,
            "semantic_coherence": features.get("semantic_coherence", 0) if features else 0,
        } if features else {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MLQualityGate:
    """Real-time quality gate using trained ML models.

    Supports two modes:
    - SHADOW: Logs predictions for human review, never blocks
    - ENFORCE: Auto-rejects low quality when confidence is high
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self._classifier = None
        self._regressor = None
        self._scaler = None
        self._loaded = False
        self._active_features: list[str] | None = None
        self._training_dist: dict[str, dict] | None = None

    def _load_models(self) -> bool:
        """Lazy-load trained models from disk. Returns True if successful."""
        if self._loaded:
            return self._classifier is not None

        self._loaded = True
        model_path = Path(self.model_dir)

        clf_path = model_path / "quality_classifier.joblib"
        reg_path = model_path / "faithfulness_regressor.joblib"
        scaler_path = model_path / "feature_scaler.joblib"
        feature_meta_path = model_path / "feature_meta.json"
        dist_path = model_path / "training_distribution.json"

        if not clf_path.exists():
            logger.info("ML Quality Gate: No trained models found at %s — gate disabled", self.model_dir)
            return False

        try:
            import joblib
            self._classifier = joblib.load(clf_path)
            self._regressor = joblib.load(reg_path) if reg_path.exists() else None
            self._scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            # Load active features (may differ from FEATURE_COLUMNS if pruned)
            if feature_meta_path.exists():
                meta = json.loads(feature_meta_path.read_text(encoding="utf-8"))
                self._active_features = meta.get("active_features", FEATURE_COLUMNS)
                dropped = meta.get("dropped_features", [])
                if dropped:
                    logger.info(
                        "ML Quality Gate: Using %d active features (%d pruned: %s)",
                        len(self._active_features), len(dropped), dropped,
                    )
            else:
                self._active_features = list(FEATURE_COLUMNS)

            # Load training distribution for Concept Drift detection
            if dist_path.exists():
                self._training_dist = json.loads(dist_path.read_text(encoding="utf-8"))
                logger.info("ML Quality Gate: Loaded training distribution for drift detection")

            logger.info("ML Quality Gate: Models loaded from %s", self.model_dir)
            return True
        except Exception as exc:
            logger.error("ML Quality Gate: Failed to load models: %s", exc)
            return False

    def _get_active_features(self) -> list[str]:
        """Get the active feature list (may be pruned)."""
        return self._active_features or FEATURE_COLUMNS

    def _check_concept_drift(self, features: dict[str, Any]) -> dict[str, Any]:
        """Check if incoming features are out-of-distribution vs training data.

        Compares each feature against the training distribution (mean ± N*std).
        If too many features are OOD, the model may be stale and should NOT
        be trusted for blocking decisions.

        Returns
        -------
        dict
            - drift_detected: bool
            - ood_fraction: float (fraction of features that are OOD)
            - ood_features: list[str] (names of OOD features)
            - message: str
        """
        if not self._training_dist:
            return {"drift_detected": False, "ood_fraction": 0.0, "ood_features": [], "message": "No training distribution available"}

        active = self._get_active_features()
        ood_features = []

        for col in active:
            if col not in self._training_dist:
                continue
            dist = self._training_dist[col]
            val = float(features.get(col, 0.0))
            mean = dist.get("mean", 0.0)
            std = dist.get("std", 1.0)
            if std < 0.001:
                std = 0.001  # Prevent division by near-zero

            z_score = abs(val - mean) / std
            if z_score > DRIFT_SIGMA_THRESHOLD:
                ood_features.append(col)

        ood_fraction = len(ood_features) / len(active) if active else 0.0
        drift_detected = ood_fraction >= DRIFT_FEATURE_FRACTION

        if drift_detected:
            msg = (
                f"⚠️ CONCEPT DRIFT DETECTED: {len(ood_features)}/{len(active)} features "
                f"({ood_fraction:.0%}) are >{DRIFT_SIGMA_THRESHOLD}σ out-of-distribution. "
                f"OOD features: {ood_features}. Model predictions may be unreliable!"
            )
            logger.warning(msg)
        else:
            msg = f"Drift check OK: {len(ood_features)}/{len(active)} features OOD ({ood_fraction:.0%})"
            logger.debug(msg)

        return {
            "drift_detected": drift_detected,
            "ood_fraction": round(ood_fraction, 3),
            "ood_features": ood_features,
            "message": msg,
        }

    def predict_quality(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict blog quality using the trained ML model.

        Parameters
        ----------
        features : dict
            Feature vector from ml_data_collector.extract_features().

        Returns
        -------
        dict
            Prediction result with keys:
            - quality_class: "high" / "medium" / "low"
            - quality_confidence: float 0.0-1.0
            - faithfulness_pred: float 0.0-1.0
            - failure_source: "retrieval" / "generation" / "both" / "none"
            - model_available: bool
            - gate_mode: "shadow" / "enforce"
        """
        mode = _resolve_mode()

        if not self._load_models():
            return {
                "quality_class": "unknown",
                "quality_confidence": 0.0,
                "faithfulness_pred": 0.5,
                "failure_source": "none",
                "model_available": False,
                "gate_mode": mode,
            }

        try:
            active = self._get_active_features()

            # Build feature vector using ACTIVE features (may be pruned)
            x = np.array([[float(features.get(col, 0.0)) for col in active]])

            # Scale
            if self._scaler is not None:
                x = self._scaler.transform(x)

            # ── Concept Drift Check ──
            drift_result = self._check_concept_drift(features)
            drift_detected = drift_result["drift_detected"]

            # Classify
            quality_pred = int(self._classifier.predict(x)[0])
            quality_class = QUALITY_CLASS_INV.get(quality_pred, "medium")

            # Get classification probabilities (confidence)
            confidence = 0.0
            if hasattr(self._classifier, "predict_proba"):
                proba = self._classifier.predict_proba(x)[0]
                confidence = float(max(proba))

            # Predict faithfulness
            faithfulness_pred = 0.5
            if self._regressor is not None:
                faithfulness_pred = float(self._regressor.predict(x)[0])
                faithfulness_pred = min(1.0, max(0.0, faithfulness_pred))

            # Diagnose failure source for dual-scope routing
            from app.ml.ml_data_collector import diagnose_failure_source
            failure_source = diagnose_failure_source(features)

            # If Concept Drift detected, override confidence to 0
            # This prevents false blocking when the model is stale
            if drift_detected:
                logger.warning(
                    "ML Gate: Concept Drift detected — zeroing confidence "
                    "to prevent false blocking (was %.2f)", confidence,
                )
                confidence = 0.0  # Effectively disables blocking

            logger.info(
                "ML Quality Gate [%s]: class=%s (conf=%.2f), faith=%.3f, "
                "failure=%s, drift=%s",
                mode.upper(), quality_class, confidence, faithfulness_pred,
                failure_source, drift_detected,
            )

            return {
                "quality_class": quality_class,
                "quality_confidence": round(confidence, 4),
                "faithfulness_pred": round(faithfulness_pred, 4),
                "failure_source": failure_source,
                "model_available": True,
                "gate_mode": mode,
                "concept_drift": drift_result,
            }
        except Exception as exc:
            logger.error("ML Quality Gate prediction failed: %s", exc)
            return {
                "quality_class": "unknown",
                "quality_confidence": 0.0,
                "faithfulness_pred": 0.5,
                "failure_source": "none",
                "model_available": False,
                "gate_mode": _resolve_mode(),
            }

    def should_block(
        self, prediction: dict[str, Any], features: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """Decide whether to block the draft based on ML prediction.

        In SHADOW mode: always returns (False, reason) but logs what
        WOULD have happened for human review.

        In ENFORCE mode: blocks when confidence threshold is met.

        Returns
        -------
        tuple[bool, str]
            (should_block, reason)
        """
        mode = prediction.get("gate_mode", _resolve_mode())

        if not prediction.get("model_available", False):
            return False, "ML model not available — pass-through"

        quality_class = prediction.get("quality_class", "medium")
        confidence = prediction.get("quality_confidence", 0.0)
        faithfulness = prediction.get("faithfulness_pred", 0.5)
        failure_source = prediction.get("failure_source", "none")

        # Determine if we WOULD block
        would_block = False
        reason = f"ML gate passed (quality={quality_class}, conf={confidence:.2f})"

        if quality_class == "low" and confidence >= BLOCK_CONFIDENCE_THRESHOLD:
            would_block = True
            reason = (
                f"ML model predicts LOW quality (confidence={confidence:.2f}). "
                f"Faithfulness={faithfulness:.3f}. Failure source: {failure_source}."
            )
        elif faithfulness < FAITHFULNESS_BLOCK_THRESHOLD and confidence >= 0.6:
            would_block = True
            reason = (
                f"ML model predicts very low faithfulness ({faithfulness:.3f}). "
                f"Quality={quality_class}, confidence={confidence:.2f}. "
                f"Failure source: {failure_source}."
            )

        # Shadow Mode: log but never block
        if mode == "shadow":
            _log_shadow_prediction(prediction, would_block, reason, features)
            if would_block:
                logger.warning(
                    "ML Gate [SHADOW]: WOULD have blocked — %s (NOT blocking in shadow mode)",
                    reason,
                )
            return False, f"[SHADOW] {reason}"

        # Enforce Mode: actually block
        return would_block, reason

    def diagnose_route(self, prediction: dict[str, Any]) -> str:
        """Determine which node to route back to based on failure diagnosis.

        Returns
        -------
        str
            'Researcher' — retrieval issues (need more/better docs)
            'Writer' — generation issues (need better writing)
            'Writer' — default for 'both' or 'none'
        """
        failure_source = prediction.get("failure_source", "none")
        if failure_source == "retrieval":
            return "Researcher"
        # 'generation', 'both', or 'none' → route to Writer
        return "Writer"

    @property
    def is_ready(self) -> bool:
        """Check if ML models are loaded and ready for prediction."""
        return self._load_models()

    @property
    def current_mode(self) -> str:
        """Get the current operating mode."""
        return _resolve_mode()


# Module-level singleton
_GATE: MLQualityGate | None = None


def get_ml_quality_gate() -> MLQualityGate:
    """Get or create the global ML Quality Gate instance."""
    global _GATE
    if _GATE is None:
        _GATE = MLQualityGate()
    return _GATE
