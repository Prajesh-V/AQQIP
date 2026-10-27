"""
AQQIP MLflow Tracking

Handles recording experiments, metrics, models, and artifacts to MLflow.
"""
from __future__ import annotations

import mlflow
from typing import Dict, Any, Optional

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger("mlflow_tracking")


class MLflowTracker:
    """Wrapper for MLflow logging operations."""

    def __init__(self, experiment_name: str):
        import os
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str = "model"):
        """Log a LightGBM model."""
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(model, artifact_path)
