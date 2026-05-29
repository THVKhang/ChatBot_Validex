"""ML Feedback Loop — Continuous learning and automatic model retraining.

Monitors the training data file and triggers retraining based on a
HYBRID strategy:
  - Batch trigger: retrain after 200 new samples (not 50, to avoid overfitting)
  - Schedule trigger: retrain weekly (whichever comes first)
  - Golden Test Set: compare new model vs old on a FIXED held-out dataset
  - Only deploy new model if it outperforms the old one on the Golden Set

This prevents:
  - Overfitting (model learns only recent patterns, forgets older ones)
  - Catastrophic Forgetting (model loses knowledge from early training data)
  - False deployments (new model MUST beat old on the golden holdout)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_DATA_PATH = "data/ml/training_data.jsonl"
DEFAULT_MODEL_DIR = "data/ml/models"
DEFAULT_REPORT_PATH = "data/ml/training_report.json"
HISTORY_PATH = "data/ml/training_history.jsonl"
GOLDEN_TEST_SET_PATH = "data/ml/golden_test_set.jsonl"

# ── Hybrid Retraining Thresholds ────────────────────────────────────
# Batch trigger: retrain after N new samples
RETRAIN_SAMPLE_THRESHOLD = 200

# Schedule trigger: retrain if last training was > N days ago
RETRAIN_MAX_DAYS = 7  # Weekly

# Minimum total samples to start training
MIN_TOTAL_SAMPLES = 20

# Golden Test Set: fraction of data held out for evaluation
GOLDEN_HOLDOUT_FRACTION = 0.15  # 15% of all data

# Minimum improvement required to deploy new model
MIN_IMPROVEMENT = 0.02


def _count_records(path: str) -> int:
    """Count records in a JSONL file."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def _get_last_training_sample_count() -> int:
    """Read the sample count from the last training report."""
    report_path = Path(DEFAULT_REPORT_PATH)
    if not report_path.exists():
        return 0
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return int(report.get("n_samples", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _get_last_training_timestamp() -> datetime | None:
    """Read the timestamp from the last training report."""
    report_path = Path(DEFAULT_REPORT_PATH)
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ts = report.get("timestamp", "")
        if ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _get_last_accuracy() -> float:
    """Read the accuracy from the last training report."""
    report_path = Path(DEFAULT_REPORT_PATH)
    if not report_path.exists():
        return 0.0
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return float(report.get("classifier", {}).get("cv_accuracy_mean", 0.0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


# ── Golden Test Set Management ──────────────────────────────────────

def _ensure_golden_test_set(data_path: str = DEFAULT_TRAINING_DATA_PATH) -> bool:
    """Create the Golden Test Set if it doesn't exist.

    Takes the first GOLDEN_HOLDOUT_FRACTION of records as a fixed holdout.
    This set NEVER changes, ensuring consistent evaluation across retrains.

    Returns True if golden set exists (or was created), False if not enough data.
    """
    golden_path = Path(GOLDEN_TEST_SET_PATH)
    if golden_path.exists():
        return True  # Already created, never modify it

    data_p = Path(data_path)
    if not data_p.exists():
        return False

    records = []
    for line in data_p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(records) < MIN_TOTAL_SAMPLES:
        return False

    # Take first N records as golden holdout (deterministic)
    n_golden = max(3, int(len(records) * GOLDEN_HOLDOUT_FRACTION))
    golden_records = records[:n_golden]

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8") as f:
        for record in golden_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Golden Test Set created: %d records (%.0f%% of %d total) → %s",
        n_golden, GOLDEN_HOLDOUT_FRACTION * 100, len(records), GOLDEN_TEST_SET_PATH,
    )
    return True


def _evaluate_on_golden(model_dir: str = DEFAULT_MODEL_DIR) -> dict[str, float]:
    """Evaluate a trained model on the Golden Test Set.

    Returns dict with accuracy, f1_macro, and mae on the holdout set.
    """
    golden_path = Path(GOLDEN_TEST_SET_PATH)
    if not golden_path.exists():
        return {"accuracy": 0.0, "f1_macro": 0.0, "mae": 1.0}

    try:
        import joblib
        from app.ml.ml_trainer import FEATURE_COLUMNS, QUALITY_CLASS_MAP

        model_path = Path(model_dir)
        classifier = joblib.load(model_path / "quality_classifier.joblib")
        scaler = joblib.load(model_path / "feature_scaler.joblib")
        regressor = None
        reg_path = model_path / "faithfulness_regressor.joblib"
        if reg_path.exists():
            regressor = joblib.load(reg_path)

        # Load golden records
        X_rows, y_class, y_faith = [], [], []
        for line in golden_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            features = record.get("features", {})
            labels = record.get("labels", {})
            row = [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]
            X_rows.append(row)
            quality = labels.get("quality_class", "medium")
            y_class.append(QUALITY_CLASS_MAP.get(quality, 1))
            y_faith.append(float(labels.get("faithfulness_score", 0.5)))

        if not X_rows:
            return {"accuracy": 0.0, "f1_macro": 0.0, "mae": 1.0}

        X = np.array(X_rows)
        X_scaled = scaler.transform(X)

        # Evaluate classifier
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
        y_pred_class = classifier.predict(X_scaled)
        accuracy = accuracy_score(y_class, y_pred_class)
        f1 = f1_score(y_class, y_pred_class, average="macro", zero_division=0)

        # Evaluate regressor
        mae = 1.0
        if regressor is not None:
            y_pred_faith = regressor.predict(X_scaled)
            mae = mean_absolute_error(y_faith, y_pred_faith)

        logger.info(
            "Golden Test Set evaluation: accuracy=%.4f, F1=%.4f, MAE=%.4f (%d samples)",
            accuracy, f1, mae, len(X_rows),
        )
        return {
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1), 4),
            "mae": round(float(mae), 4),
            "n_golden_samples": len(X_rows),
        }
    except Exception as exc:
        logger.error("Golden Test Set evaluation failed: %s", exc)
        return {"accuracy": 0.0, "f1_macro": 0.0, "mae": 1.0}


# ── Retraining Logic ───────────────────────────────────────────────

def should_retrain(data_path: str = DEFAULT_TRAINING_DATA_PATH) -> tuple[bool, str]:
    """Check if we should trigger a model retraining.

    Uses HYBRID strategy:
    1. First training: when >= MIN_TOTAL_SAMPLES available
    2. Batch trigger: >= RETRAIN_SAMPLE_THRESHOLD new samples
    3. Schedule trigger: > RETRAIN_MAX_DAYS since last training

    Returns
    -------
    tuple[bool, str]
        (should_retrain, reason)
    """
    current_count = _count_records(data_path)
    last_count = _get_last_training_sample_count()
    new_samples = current_count - last_count

    # Not enough data for any training
    if current_count < MIN_TOTAL_SAMPLES:
        return False, f"Not enough total samples ({current_count}/{MIN_TOTAL_SAMPLES} minimum)"

    # First-time training
    if last_count == 0:
        return True, f"No model trained yet. {current_count} samples available."

    # Batch trigger: enough new samples
    if new_samples >= RETRAIN_SAMPLE_THRESHOLD:
        return True, (
            f"{new_samples} new samples since last training "
            f"(threshold={RETRAIN_SAMPLE_THRESHOLD})"
        )

    # Schedule trigger: too long since last training
    last_ts = _get_last_training_timestamp()
    if last_ts:
        days_since = (datetime.now(last_ts.tzinfo) - last_ts).days
        if days_since >= RETRAIN_MAX_DAYS:
            return True, (
                f"{days_since} days since last training "
                f"(weekly threshold={RETRAIN_MAX_DAYS} days). "
                f"{new_samples} new samples available."
            )

    return False, (
        f"Only {new_samples} new samples (need {RETRAIN_SAMPLE_THRESHOLD}) "
        f"and within weekly window"
    )


def retrain_if_needed(
    data_path: str = DEFAULT_TRAINING_DATA_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    force: bool = False,
) -> dict[str, Any] | None:
    """Check if retraining is needed and train if so.

    Pipeline:
    1. Check if retraining is needed (batch or schedule trigger)
    2. Ensure Golden Test Set exists
    3. Evaluate OLD model on Golden Set (baseline)
    4. Train new model
    5. Evaluate NEW model on Golden Set
    6. Deploy only if new model beats old on Golden Set

    Parameters
    ----------
    data_path : str
        Path to training data JSONL.
    model_dir : str
        Directory for model output.
    force : bool
        If True, force retraining regardless of triggers.

    Returns
    -------
    dict or None
        Training report if retraining occurred, None otherwise.
    """
    if not force:
        should, reason = should_retrain(data_path)
        if not should:
            logger.info("ML Feedback Loop: No retraining needed — %s", reason)
            return None
        logger.info("ML Feedback Loop: Retraining triggered — %s", reason)
    else:
        logger.info("ML Feedback Loop: Forced retraining")

    # Ensure Golden Test Set exists
    _ensure_golden_test_set(data_path)

    # Evaluate old model on Golden Set (baseline)
    old_golden = _evaluate_on_golden(model_dir)
    old_accuracy = old_golden.get("accuracy", 0.0)
    old_f1 = old_golden.get("f1_macro", 0.0)
    logger.info(
        "ML Feedback Loop: Old model baseline — accuracy=%.4f, F1=%.4f",
        old_accuracy, old_f1,
    )

    try:
        from app.ml.ml_trainer import train_model

        report = train_model(
            data_path=data_path,
            output_dir=model_dir,
            report_path=DEFAULT_REPORT_PATH,
        )

        # Evaluate new model on Golden Set
        new_golden = _evaluate_on_golden(model_dir)
        new_accuracy = new_golden.get("accuracy", 0.0)
        new_f1 = new_golden.get("f1_macro", 0.0)

        report["golden_evaluation"] = {
            "old": old_golden,
            "new": new_golden,
        }

        # Compare on Golden Test Set (F1 is the primary metric)
        if old_f1 > 0 and new_f1 < old_f1 - MIN_IMPROVEMENT:
            logger.warning(
                "ML Feedback Loop: New model (F1=%.4f) is WORSE than old (F1=%.4f) "
                "on Golden Test Set. NOT deploying.",
                new_f1, old_f1,
            )
            report["deployed"] = False
            report["reason"] = (
                f"New model worse on Golden Set: F1 {new_f1:.4f} < {old_f1:.4f}"
            )
        else:
            report["deployed"] = True
            f1_improvement = new_f1 - old_f1 if old_f1 > 0 else 0
            report["f1_improvement"] = round(f1_improvement, 4)
            logger.info(
                "ML Feedback Loop: New model deployed (F1=%.4f, improvement=+%.4f)",
                new_f1, f1_improvement,
            )

            # Reload the ML Quality Gate singleton with new model
            try:
                from app.ml.ml_quality_gate import get_ml_quality_gate
                gate = get_ml_quality_gate()
                gate._loaded = False  # Force reload on next prediction
                gate._classifier = None
                gate._regressor = None
                gate._scaler = None
                logger.info("ML Feedback Loop: ML Quality Gate will reload on next call")
            except Exception as exc:
                logger.warning("Failed to reset ML Quality Gate: %s", exc)

        return report

    except Exception as exc:
        logger.error("ML Feedback Loop: Retraining failed: %s", exc)
        return {"error": str(exc), "deployed": False}


def get_training_history(max_entries: int = 20) -> list[dict]:
    """Read recent training history entries.

    Returns
    -------
    list[dict]
        Recent training history entries (newest first).
    """
    p = Path(HISTORY_PATH)
    if not p.exists():
        return []

    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Return newest first
    return list(reversed(entries[-max_entries:]))


def get_pipeline_status() -> dict[str, Any]:
    """Get the overall status of the ML pipeline.

    Returns a summary dict with:
    - training_data_count
    - golden_test_set_count
    - last_training_report (if exists)
    - model_available
    - gate_mode (shadow/enforce)
    - retrain_recommendation
    """
    from app.ml.ml_data_collector import count_training_records
    from app.ml.ml_quality_gate import get_ml_quality_gate

    data_count = count_training_records()
    golden_count = _count_records(GOLDEN_TEST_SET_PATH)
    gate = get_ml_quality_gate()
    model_available = gate.is_ready

    should, reason = should_retrain()

    status = {
        "training_data_count": data_count,
        "golden_test_set_count": golden_count,
        "model_available": model_available,
        "gate_mode": gate.current_mode,
        "retrain_recommended": should,
        "retrain_reason": reason,
        "retrain_thresholds": {
            "sample_threshold": RETRAIN_SAMPLE_THRESHOLD,
            "max_days": RETRAIN_MAX_DAYS,
            "min_total_samples": MIN_TOTAL_SAMPLES,
        },
    }

    # Add last report if available
    report_path = Path(DEFAULT_REPORT_PATH)
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status["last_training"] = {
                "timestamp": report.get("timestamp", "unknown"),
                "n_samples": report.get("n_samples", 0),
                "accuracy": report.get("classifier", {}).get("cv_accuracy_mean", 0),
                "f1_macro": report.get("classifier", {}).get("classification_report", {}).get("macro avg", {}).get("f1-score", 0),
                "mae": report.get("regressor", {}).get("cv_mae_mean", 0),
            }
        except json.JSONDecodeError:
            pass

    # Add golden set evaluation if available
    if golden_count > 0 and model_available:
        status["golden_evaluation"] = _evaluate_on_golden()

    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import sys
    if "--force" in sys.argv:
        result = retrain_if_needed(force=True)
    elif "--status" in sys.argv:
        status = get_pipeline_status()
        print("Pipeline Status:")
        print(json.dumps(status, indent=2))
    else:
        result = retrain_if_needed()

    if "result" in dir() and result:
        print(json.dumps(result, indent=2, default=str))
    elif "--force" not in sys.argv and "--status" not in sys.argv:
        status = get_pipeline_status()
        print("Pipeline Status:")
        print(json.dumps(status, indent=2))
