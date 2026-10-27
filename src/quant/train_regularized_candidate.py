"""
AQQIP Regularized Model Candidate Trainer

Trains and evaluates the audited regularized candidate (M_Regularized):
- Cross-sectional relative return and relative volatility features.
- Heavy inductive regularization (max_depth=3, num_leaves=8, colsample_bytree=0.7, min_child_samples=80).
- Fitted strictly on train (2014-2023) with temporal boundary purging.
- Operational threshold calibrated strictly on validation (2024).
- Preserves all prior MLflow artifacts (M, MF, MFQ).
- Logs new candidate to MLflow with complete provenance and metrics.
"""
from __future__ import annotations

import os
import sys
import hashlib
import json
from datetime import date, datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)

from config.settings import get_settings
from src.quant.train_pipeline import (
    load_raw_data,
    DEFAULT_MARKET_FEATURES,
    DEFAULT_FUNDAMENTAL_FEATURES,
    DEFAULT_QUALITATIVE_FEATURES
)
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.benchmark_runner import compute_calibration_diagnostics, select_operational_threshold
from src.quant.rolling_origin_evaluator import REGULARIZED_MARKET_FEATURES
from src.utils.logging import get_logger

logger = get_logger("train_regularized_candidate")


def train_and_log_regularized_candidate() -> Dict[str, Any]:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment("aqqip_quant_models")

    # 1. Load data
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    
    # 2. Build dataset with conservative publication lag
    builder = HistoricalDatasetBuilder()
    panel = builder.build_dataset(market_df, fund_df, qual_df)
    
    # 3. Chronological split with temporal boundary purging
    train_df, val_df, test_df = builder.split_and_label(
        panel,
        val_start="2024-01-01",
        test_start="2025-01-01",
        alpha=0.10,
        enforce_temporal_purging=True
    )

    train_prev = float(train_df["target_label"].mean())
    val_prev = float(val_df["target_label"].mean())
    test_prev = float(test_df["target_label"].mean())

    features = list(REGULARIZED_MARKET_FEATURES)
    logger.info("Training regularized candidate M_Regularized", feature_count=len(features), features=features)

    X_train = train_df[features].copy()
    y_train = train_df["target_label"].values
    
    X_val = val_df[features].copy()
    y_val = val_df["target_label"].values

    X_test = test_df[features].copy()
    y_test = test_df["target_label"].values

    # 4. Fit heavily regularized shallow LightGBM
    params = {
        "n_estimators": 80,
        "max_depth": 3,
        "num_leaves": 8,
        "min_child_samples": 80,
        "learning_rate": 0.03,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbosity": -1,
        "objective": "binary",
        "metric": "binary_logloss"
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # 5. Validation calibration & operational threshold selection
    val_probs = model.predict_proba(X_val)[:, 1]
    op_thresh, val_f1 = select_operational_threshold(y_val, val_probs)
    val_roc = float(roc_auc_score(y_val, val_probs))
    val_pr = float(average_precision_score(y_val, val_probs))
    val_bs = float(brier_score_loss(y_val, val_probs))
    val_bss = float(1.0 - (val_bs / brier_score_loss(y_val, np.full(len(y_val), train_prev))))

    # 6. Diagnostic Test evaluation (2025-2026)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_roc = float(roc_auc_score(y_test, test_probs))
    test_pr = float(average_precision_score(y_test, test_probs))
    test_bs = float(brier_score_loss(y_test, test_probs))
    
    # Deployable Brier Skill Score vs Train Prevalence
    test_bss_deployable = float(1.0 - (test_bs / brier_score_loss(y_test, np.full(len(y_test), train_prev))))
    test_bss_insample = float(1.0 - (test_bs / brier_score_loss(y_test, np.full(len(y_test), test_prev))))
    
    test_ece, test_mce, cal_table = compute_calibration_diagnostics(y_test, test_probs, n_bins=10)

    # Operational metrics
    y_pred_op = (test_probs >= op_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_op)
    test_f1 = float(f1_score(y_test, y_pred_op, zero_division=0))
    test_prec = float(precision_score(y_test, y_pred_op, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred_op, zero_division=0))

    # 7. Log to MLflow
    with mlflow.start_run(run_name="M_Regularized_CrossSectional") as run:
        run_id = run.info.run_id
        
        # Tags
        mlflow.set_tags({
            "model_family": "LightGBM",
            "model_variant": "M_Regularized",
            "feature_set": "market_cross_sectional",
            "evaluation_status": "candidate_improved",
            "temporal_purging": "enforced",
            "conservative_lag": "enforced_1day"
        })

        # Params
        mlflow.log_params(params)
        mlflow.log_params({
            "val_operational_threshold": op_thresh,
            "train_prevalence": train_prev,
            "val_prevalence": val_prev,
            "test_prevalence": test_prev,
            "feature_count": len(features),
            "features_list": ",".join(features)
        })

        # Metrics
        mlflow.log_metrics({
            "val_roc_auc": val_roc,
            "val_pr_auc": val_pr,
            "val_brier_score": val_bs,
            "val_brier_skill": val_bss,
            "val_f1_at_op": val_f1,
            "test_roc_auc_diagnostic": test_roc,
            "test_pr_auc_diagnostic": test_pr,
            "test_brier_score_diagnostic": test_bs,
            "test_brier_skill_deployable": test_bss_deployable,
            "test_brier_skill_insample": test_bss_insample,
            "test_ece": test_ece,
            "test_mce": test_mce,
            "test_f1_at_op": test_f1,
            "test_precision_at_op": test_prec,
            "test_recall_at_op": test_rec
        })

        # Log model artifact
        mlflow.lightgbm.log_model(model.booster_, artifact_path="model")
        
        logger.info(
            "Logged M_Regularized to MLflow",
            run_id=run_id,
            test_roc_auc=test_roc,
            test_pr_auc=test_pr,
            test_bss=test_bss_deployable
        )

    return {
        "run_id": run_id,
        "features": features,
        "val_metrics": {
            "roc_auc": val_roc,
            "pr_auc": val_pr,
            "brier_score": val_bs,
            "brier_skill": val_bss,
            "op_threshold": op_thresh,
            "f1_at_op": val_f1
        },
        "test_metrics_diagnostic": {
            "roc_auc": test_roc,
            "pr_auc": test_pr,
            "brier_score": test_bs,
            "brier_skill_deployable": test_bss_deployable,
            "brier_skill_insample": test_bss_insample,
            "ece": test_ece,
            "mce": test_mce,
            "f1_at_op": test_f1,
            "precision_at_op": test_prec,
            "recall_at_op": test_rec,
            "confusion_matrix": cm.tolist()
        }
    }


if __name__ == "__main__":
    result = train_and_log_regularized_candidate()
    print("\n=== M_Regularized Training and MLflow Logging Result ===")
    print(f"MLflow Run ID: {result['run_id']}")
    print("Features:", result['features'])
    print("\nValidation Metrics (2024):")
    for k, v in result['val_metrics'].items():
        print(f"  {k}: {v}")
    print("\nDiagnostic Test Metrics (2025-2026):")
    for k, v in result['test_metrics_diagnostic'].items():
        print(f"  {k}: {v}")
