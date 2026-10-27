"""
Regression tests for Evaluation Metrics and Brier Score consistency.
Ensures Brier score and probability evaluation errors cannot recur silently.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from src.quant.lightgbm_trainer import LightGBMTrainer


def test_brier_score_not_zero_when_predictions_differ_from_labels():
    trainer = LightGBMTrainer(features=["dummy"])
    
    # 10% positive rate (10 ones, 90 zeros)
    y_true = pd.Series([1] * 10 + [0] * 90)
    # Predicted probabilities around base rate (e.g. 0.05 to 0.30)
    rng = np.random.default_rng(42)
    y_prob = rng.uniform(0.04, 0.32, size=100)
    
    metrics = trainer._evaluate_metrics(y_true, y_prob)
    
    # Key naming check
    assert "brier_score" in metrics
    assert "brier" in metrics
    assert metrics["brier_score"] == metrics["brier"]
    
    # Mathematical sanity check
    expected_brier = brier_score_loss(y_true, y_prob)
    assert np.isclose(metrics["brier_score"], expected_brier)
    assert metrics["brier_score"] > 0.0, "Brier score must be strictly positive for non-deterministic predictions"
    assert metrics["brier_score"] < 1.0


def test_zero_precision_recall_at_05_threshold_explained():
    trainer = LightGBMTrainer(features=["dummy"])
    
    # If all probabilities are below 0.50, precision/recall/F1 must be 0.0
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_prob = np.full(100, 0.15)
    
    metrics = trainer._evaluate_metrics(y_true, y_prob)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["positive_rate"] == 0.15
    assert metrics["brier_score"] > 0.0
