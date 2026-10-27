"""
AQQIP Baseline Models

Implements baseline classifiers for comparison against LightGBM.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score
from typing import Dict, Any, Optional

class BaselineModels:
    """Provides baseline performance metrics."""

    @staticmethod
    def evaluate_majority_class(y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Always predicts the training base rate or majority class."""
        train_prevalence = float(y_train.mean())
        preds = np.full(len(y_test), train_prevalence)
        
        # If test set has only one class, AUC is undefined
        if len(y_test.unique()) > 1:
            roc = float(roc_auc_score(y_test, preds)) if not np.all(preds == preds[0]) else 0.5
            pr = float(average_precision_score(y_test, preds))
        else:
            roc, pr = np.nan, np.nan
            
        brier = float(brier_score_loss(y_test, preds))
            
        return {
            "model": "majority_class",
            "roc_auc": roc,
            "pr_auc": pr,
            "brier_score": brier
        }

    @staticmethod
    def evaluate_logistic_regression(
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        features: list[str],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Simple logistic regression on selected features."""
        X_tr = X_train[features].fillna(0) # Simple imputation for baseline
        X_te = X_test[features].fillna(0)
        
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_tr, y_train)
        
        preds = model.predict_proba(X_te)[:, 1]
        
        if len(y_test.unique()) > 1:
            roc = float(roc_auc_score(y_test, preds))
            pr = float(average_precision_score(y_test, preds))
        else:
            roc, pr = np.nan, np.nan
            
        brier = float(brier_score_loss(y_test, preds))
        y_pred = (preds >= threshold).astype(int)
        
        return {
            "model": "logistic_regression",
            "roc_auc": roc,
            "pr_auc": pr,
            "brier_score": brier,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "predicted_positive_rate": float(y_pred.mean())
        }

