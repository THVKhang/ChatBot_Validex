"""ML Quality Model Trainer — Trains XGBoost/scikit-learn models from collected
RAG pipeline data.

Reads training records from data/ml/training_data.jsonl, trains:
  1. XGBClassifier  → quality_class prediction (high/medium/low)
  2. XGBRegressor   → faithfulness_score prediction (0.0 - 1.0)

Exports trained models + scaler to data/ml/models/ as joblib files.

━━━ RED TEAM MITIGATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SURFACE FEATURE AUDITING (Goodhart's Law / Shortcut Learning):
   After training, checks if surface features (word_count, heading_count)
   dominate >30% of total importance. If so, auto-drops them and retrains
   to force the model to learn from core signals (NLI, retrieval quality).

2. HUMAN-WEIGHTED SAMPLES (Echo Chamber / Model Collapse):
   Samples with label_source="human" get 10x sample_weight.
   This acts as an "anchor" that prevents the model from drifting into
   the LLM-as-a-judge's biases over multiple retraining cycles.

3. CLASS BALANCE MONITORING:
   Logs warnings if training data has extreme class imbalance
   (e.g., 95% "high" and 5% "low" — sign of Survivorship Bias).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Paths
DEFAULT_TRAINING_DATA_PATH = "data/ml/training_data.jsonl"
DEFAULT_MODEL_OUTPUT_DIR = "data/ml/models"
DEFAULT_REPORT_PATH = "data/ml/training_report.json"
HISTORY_PATH = "data/ml/training_history.jsonl"

# Feature columns in fixed order (must match ml_data_collector.extract_features)
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

QUALITY_CLASS_MAP = {"low": 0, "medium": 1, "high": 2}
QUALITY_CLASS_INV = {v: k for k, v in QUALITY_CLASS_MAP.items()}

# Minimum training samples required
MIN_TRAINING_SAMPLES = 20

# ── Red Team Thresholds ─────────────────────────────────────────
# Surface features that the model might abuse as shortcuts (Goodhart's Law)
SURFACE_FEATURES = {"draft_word_count", "draft_heading_count"}

# If ANY surface feature exceeds this fraction of total importance → auto-drop
SURFACE_IMPORTANCE_THRESHOLD = 0.30

# Human-labeled samples get this weight multiplier (Echo Chamber prevention)
HUMAN_SAMPLE_WEIGHT = 10.0

# Minimum class representation threshold (Survivorship Bias detection)
MIN_CLASS_FRACTION = 0.05  # Each class must be >= 5% of total


def load_training_data(
    path: str = DEFAULT_TRAINING_DATA_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training records from JSONL and return (X, y_class, y_faith, sample_weights).

    Parameters
    ----------
    path : str
        Path to the JSONL training data file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - X: feature matrix (n_samples, n_features)
        - y_class: quality class labels (n_samples,) encoded as int
        - y_faith: faithfulness scores (n_samples,) as float
        - sample_weights: per-sample weights (n_samples,) — human=10x, AI=1x
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            records.append(record)
        except json.JSONDecodeError:
            continue

    if len(records) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Insufficient training data: {len(records)} records "
            f"(need at least {MIN_TRAINING_SAMPLES})"
        )

    X_rows = []
    y_class = []
    y_faith = []
    weights = []

    for record in records:
        features = record.get("features", {})
        labels = record.get("labels", {})

        row = [float(features.get(col, 0.0)) for col in FEATURE_COLUMNS]
        X_rows.append(row)

        quality = labels.get("quality_class", "medium")
        y_class.append(QUALITY_CLASS_MAP.get(quality, 1))
        y_faith.append(float(labels.get("faithfulness_score", 0.5)))

        # ── Human Anchor: samples labeled by humans get 10x weight ──
        label_source = labels.get("label_source", "heuristic")
        if label_source == "human":
            weights.append(HUMAN_SAMPLE_WEIGHT)
        else:
            weights.append(1.0)

    # ── Class Balance Monitoring (Survivorship Bias detection) ──
    _check_class_balance(y_class, len(records))

    logger.info(
        "Loaded %d training samples (%d features, %d human-labeled)",
        len(X_rows), len(FEATURE_COLUMNS),
        sum(1 for w in weights if w > 1.0),
    )
    return np.array(X_rows), np.array(y_class), np.array(y_faith), np.array(weights)


def _check_class_balance(y_class: list[int], total: int) -> None:
    """Log warnings if class distribution shows Survivorship Bias."""
    from collections import Counter
    counts = Counter(y_class)

    for class_id, class_name in QUALITY_CLASS_INV.items():
        fraction = counts.get(class_id, 0) / total
        if fraction < MIN_CLASS_FRACTION:
            logger.warning(
                "⚠️ CLASS IMBALANCE: class '%s' has only %.1f%% of samples (%d/%d). "
                "This may indicate Survivorship Bias — the ML Gate may be blocking "
                "too aggressively, preventing 'low' samples from reaching training. "
                "Check ml_gate_node.py _record_rejected_sample() is working.",
                class_name, fraction * 100, counts.get(class_id, 0), total,
            )
        else:
            logger.info(
                "Class '%s': %.1f%% (%d samples)",
                class_name, fraction * 100, counts.get(class_id, 0),
            )


def _audit_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
) -> tuple[list[str], list[tuple[str, float]], list[str]]:
    """Audit feature importances for Goodhart's Law / Shortcut Learning.

    If surface features (word_count, heading_count) dominate > THRESHOLD
    of total importance, returns them as features to DROP.

    Returns
    -------
    tuple[list[str], list, list[str]]
        - features_to_drop: list of feature names to remove
        - importance_sorted: sorted (feature, importance) pairs
        - warnings: list of warning messages
    """
    total_imp = float(importances.sum())
    if total_imp == 0:
        return [], [], []

    importance_sorted = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    features_to_drop = []
    warnings = []

    for feat_name, feat_imp in importance_sorted:
        fraction = feat_imp / total_imp
        if feat_name in SURFACE_FEATURES and fraction > SURFACE_IMPORTANCE_THRESHOLD:
            features_to_drop.append(feat_name)
            warning_msg = (
                f"🚨 GOODHART ALERT: Surface feature '{feat_name}' has "
                f"{fraction:.1%} importance (>{SURFACE_IMPORTANCE_THRESHOLD:.0%} threshold). "
                f"Model may be taking shortcuts! Auto-dropping this feature."
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)

    if not features_to_drop:
        logger.info(
            "Feature audit PASSED: No surface features dominate. Top-3: %s",
            [(f, f"{v:.3f}") for f, v in importance_sorted[:3]],
        )

    return features_to_drop, importance_sorted, warnings


def train_model(
    data_path: str = DEFAULT_TRAINING_DATA_PATH,
    output_dir: str = DEFAULT_MODEL_OUTPUT_DIR,
    report_path: str = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Train ML quality models and export them.

    Includes Red Team mitigations:
    1. Human sample weighting (10x)
    2. Feature importance auditing (auto-drop shortcuts)
    3. Class balance monitoring

    Returns
    -------
    dict
        Training report with metrics, feature audit, and health warnings.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, mean_absolute_error
    import joblib

    try:
        from xgboost import XGBClassifier, XGBRegressor
        use_xgboost = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        use_xgboost = False
        logger.info("XGBoost not available, using scikit-learn GradientBoosting")

    # Load data with sample weights
    X, y_class, y_faith, sample_weights = load_training_data(data_path)
    n_samples, n_features = X.shape
    active_features = list(FEATURE_COLUMNS)  # May change after audit

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Initial training (to get feature importances for audit)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build classifier
    if use_xgboost:
        classifier = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
    else:
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )

    # Fit with sample weights (Human Anchor = 10x weight)
    classifier.fit(X_scaled, y_class, sample_weight=sample_weights)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Feature Importance Audit (Goodhart's Law protection)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    health_warnings = []
    features_dropped = []

    if hasattr(classifier, "feature_importances_"):
        features_to_drop, importance_sorted_raw, audit_warnings = _audit_feature_importance(
            classifier.feature_importances_, active_features
        )
        health_warnings.extend(audit_warnings)

        if features_to_drop:
            # ── AUTO-RETRAIN without shortcut features ──
            logger.warning(
                "Re-training with %d surface features DROPPED: %s",
                len(features_to_drop), features_to_drop,
            )
            features_dropped = features_to_drop
            drop_indices = [active_features.index(f) for f in features_to_drop]
            active_features = [f for f in active_features if f not in features_to_drop]

            # Remove columns from X
            X_pruned = np.delete(X, drop_indices, axis=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pruned)
            n_features = X_scaled.shape[1]

            # Rebuild and retrain classifier
            if use_xgboost:
                classifier = XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    objective="multi:softmax", num_class=3,
                    eval_metric="mlogloss", random_state=42, verbosity=0,
                )
            else:
                classifier = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42,
                )
            classifier.fit(X_scaled, y_class, sample_weight=sample_weights)
            logger.info("Re-training complete with %d features (dropped %d)", n_features, len(features_to_drop))
    else:
        importance_sorted_raw = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Cross-validation + Final metrics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    n_splits = min(5, n_samples)
    if n_splits < 2:
        n_splits = 2

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf_cv_scores = cross_val_score(classifier, X_scaled, y_class, cv=cv, scoring="accuracy")

    # Full-data predictions for report
    y_class_pred = classifier.predict(X_scaled)
    clf_report = classification_report(
        y_class, y_class_pred, target_names=["low", "medium", "high"], output_dict=True
    )

    # Final feature importance (after possible pruning)
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importance = sorted(
            zip(active_features, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        feature_importance = []

    # ── Train Regressor (faithfulness_score: 0.0-1.0) ──
    if use_xgboost:
        regressor = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        regressor = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )

    from sklearn.model_selection import KFold
    reg_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    reg_cv_scores = cross_val_score(
        regressor, X_scaled, y_faith, cv=reg_cv, scoring="neg_mean_absolute_error"
    )

    regressor.fit(X_scaled, y_faith, sample_weight=sample_weights)
    y_faith_pred = regressor.predict(X_scaled)
    faith_mae = mean_absolute_error(y_faith, y_faith_pred)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: Save models + Concept Drift baseline statistics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, out_dir / "quality_classifier.joblib")
    joblib.dump(regressor, out_dir / "faithfulness_regressor.joblib")
    joblib.dump(scaler, out_dir / "feature_scaler.joblib")

    # Save active feature list (may differ from FEATURE_COLUMNS if pruned)
    feature_meta = {
        "active_features": active_features,
        "dropped_features": features_dropped,
        "all_features": list(FEATURE_COLUMNS),
    }
    (out_dir / "feature_meta.json").write_text(
        json.dumps(feature_meta, indent=2), encoding="utf-8"
    )

    # Save feature distribution statistics for Concept Drift detection
    drift_stats = {}
    for i, col in enumerate(active_features):
        col_data = X_scaled[:, i]
        drift_stats[col] = {
            "mean": round(float(np.mean(col_data)), 4),
            "std": round(float(np.std(col_data)), 4),
            "min": round(float(np.min(col_data)), 4),
            "max": round(float(np.max(col_data)), 4),
            "p25": round(float(np.percentile(col_data, 25)), 4),
            "p75": round(float(np.percentile(col_data, 75)), 4),
        }
    (out_dir / "training_distribution.json").write_text(
        json.dumps(drift_stats, indent=2), encoding="utf-8"
    )

    logger.info("Models + drift stats saved to %s", output_dir)

    # ── Build report ──
    from collections import Counter
    class_counts = Counter(y_class.tolist())

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_columns": active_features,
        "model_type": "XGBoost" if use_xgboost else "GradientBoosting",
        "class_distribution": {
            QUALITY_CLASS_INV[k]: v for k, v in class_counts.items()
        },
        "human_labeled_count": int(sum(1 for w in sample_weights if w > 1.0)),
        "classifier": {
            "cv_accuracy_mean": round(float(clf_cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(clf_cv_scores.std()), 4),
            "classification_report": clf_report,
            "feature_importance_top10": feature_importance[:10],
        },
        "regressor": {
            "cv_mae_mean": round(float(-reg_cv_scores.mean()), 4),
            "cv_mae_std": round(float(reg_cv_scores.std()), 4),
            "train_mae": round(float(faith_mae), 4),
        },
        "model_files": {
            "classifier": str(out_dir / "quality_classifier.joblib"),
            "regressor": str(out_dir / "faithfulness_regressor.joblib"),
            "scaler": str(out_dir / "feature_scaler.joblib"),
        },
        "red_team_audit": {
            "features_dropped": features_dropped,
            "surface_importance_threshold": SURFACE_IMPORTANCE_THRESHOLD,
            "health_warnings": health_warnings,
            "human_sample_weight": HUMAN_SAMPLE_WEIGHT,
        },
    }

    # Save report
    report_p = Path(report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    report_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append to training history
    history_p = Path(HISTORY_PATH)
    history_p.parent.mkdir(parents=True, exist_ok=True)
    history_entry = {
        "timestamp": report["timestamp"],
        "n_samples": n_samples,
        "n_features": n_features,
        "cv_accuracy": report["classifier"]["cv_accuracy_mean"],
        "cv_mae": report["regressor"]["cv_mae_mean"],
        "model_type": report["model_type"],
        "features_dropped": features_dropped,
        "human_labeled": report["human_labeled_count"],
        "class_distribution": report["class_distribution"],
    }
    with history_p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")

    logger.info(
        "Training complete: %d samples (%.0f%% human-labeled), "
        "accuracy=%.4f (±%.4f), MAE=%.4f, %d features dropped",
        n_samples,
        (report["human_labeled_count"] / n_samples * 100) if n_samples else 0,
        clf_cv_scores.mean(),
        clf_cv_scores.std(),
        -reg_cv_scores.mean(),
        len(features_dropped),
    )

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        result = train_model()
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError) as e:
        print(f"Cannot train: {e}")
        print("Collect more data by running the RAG pipeline first.")
