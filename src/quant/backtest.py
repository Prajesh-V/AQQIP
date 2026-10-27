"""
AQQIP Backtest Evaluation

Evaluates the trained model over the chronologically separate backtest period.
"""
from __future__ import annotations

import pandas as pd
from typing import Any, Dict

from src.quant.baseline_models import BaselineModels
from src.utils.logging import get_logger

logger = get_logger("backtest")


class Backtester:
    """Evaluates models on the chronological holdout set."""

    @staticmethod
    def evaluate(
        model: Any, 
        X_test: pd.DataFrame, y_test: pd.Series, 
        features: list[str]
    ) -> Dict[str, float]:
        """
        Runs the model against the backtest split and calculates metrics.
        Returns a dictionary of metrics.
        """
        if len(X_test) == 0:
            logger.warning("Empty backtest set")
            return {}
            
        logger.info(f"Running backtest on {len(X_test)} observations")
        
        # LightGBM predict returns probability
        y_prob = model.predict(X_test[features])
        
        # Calculate same metrics as trainer
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score
        
        if len(y_test.unique()) <= 1:
            logger.warning("Backtest set contains only one class. AUC undefined.")
            return {}
            
        y_pred = (y_prob > 0.5).astype(int)
        
        metrics = {
            "test_roc_auc": roc_auc_score(y_test, y_prob),
            "test_pr_auc": average_precision_score(y_test, y_prob),
            "test_brier": brier_score_loss(y_test, y_prob),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_positive_rate": float(y_prob.mean())
        }
        
        logger.info("Backtest complete", roc_auc=metrics["test_roc_auc"], pr_auc=metrics["test_pr_auc"])
        return metrics
