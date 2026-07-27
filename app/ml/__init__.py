"""ML/DL Quality Control Pipeline — Package Init.

This package provides machine learning-based quality control for the RAG pipeline:
- ml_data_collector: Harvests training data from each RAG pipeline run
- ml_trainer: Trains XGBoost/scikit-learn quality models
- ml_quality_gate: Real-time quality prediction using trained models
- ml_feedback_loop: Continuous learning and model retraining
"""
