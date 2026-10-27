"""
Authoritative Versioned Evaluation Artifact Generator

Generates a single authoritative, versioned prediction evaluation record for
the diagnostic test holdout period (2025-2026) across M, MF, MFQ, and M_Regularized.
Enforces arithmetic reconciliation of all classification, calibration, and ranking metrics.
"""

from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)
import mlflow
import mlflow.lightgbm

from config.settings import get_settings
from src.quant.train_pipeline import (
    load_raw_data, DEFAULT_MARKET_FEATURES, DEFAULT_FUNDAMENTAL_FEATURES, DEFAULT_QUALITATIVE_FEATURES
)
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.benchmark_runner import compute_calibration_diagnostics, select_operational_threshold

ARTIFACT_DIR = Path("eval_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ARTIFACT_DIR / "authoritative_eval_predictions_2025_2026.csv"
PARQUET_PATH = ARTIFACT_DIR / "authoritative_eval_predictions_2025_2026.parquet"

REGULARIZED_FEATURES = DEFAULT_MARKET_FEATURES + ["return_20d_rel", "vol_ratio_20d"]

MODEL_RUNS = {
    "M": ("59a7f799181243c796b73ee3b03f5686", DEFAULT_MARKET_FEATURES),
    "MF": ("31ac56b4ff71411dad7cb13315459c30", DEFAULT_MARKET_FEATURES + DEFAULT_FUNDAMENTAL_FEATURES),
    "MFQ": ("54a2403adc824c179a20a75208325a10", DEFAULT_MARKET_FEATURES + DEFAULT_FUNDAMENTAL_FEATURES + DEFAULT_QUALITATIVE_FEATURES),
    "M_Regularized": ("62d5c881d4bb4922b3d93ccaad9b5438", REGULARIZED_FEATURES),
}


def build_and_save_authoritative_artifact() -> pd.DataFrame:
    """Builds and persists the versioned evaluation dataset with predictions and thresholds."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    builder = HistoricalDatasetBuilder()
    panel = builder.build_dataset(market_df, fund_df, qual_df)
    train_df, val_df, test_df = builder.split_and_label(
        panel,
        val_start="2024-01-01",
        test_start="2025-01-01",
        alpha=0.10,
        enforce_temporal_purging=True
    )

    data_fingerprint = hashlib.sha256(
        test_df[["ticker", "trade_date", "adj_close", "forward_return_pct"]].to_csv(index=False).encode()
    ).hexdigest()[:16]

    artifact_df = test_df[["trade_date", "ticker", "target_label"]].copy()
    artifact_df["data_fingerprint"] = data_fingerprint
    artifact_df["feature_pipeline_version"] = "2.1.0-cross-sectional-purged"
    artifact_df["train_prevalence"] = float(train_df["target_label"].mean())
    artifact_df["val_prevalence"] = float(val_df["target_label"].mean())
    artifact_df["test_prevalence"] = float(test_df["target_label"].mean())

    for model_name, (run_id, feat_cols) in MODEL_RUNS.items():
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        
        # Validation predictions to find operational threshold
        X_val = val_df[feat_cols]
        y_val = val_df["target_label"].values
        val_probs = model.predict(X_val)
        op_thresh, val_f1 = select_operational_threshold(y_val, val_probs)

        # Test predictions
        X_test = test_df[feat_cols]
        test_probs = model.predict(X_test)

        artifact_df[f"prob_{model_name}"] = test_probs
        artifact_df[f"op_thresh_{model_name}"] = op_thresh
        artifact_df[f"run_id_{model_name}"] = run_id
        artifact_df[f"val_f1_{model_name}"] = val_f1

    # Persist artifacts
    artifact_df.to_csv(CSV_PATH, index=False)
    artifact_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Persisted authoritative evaluation artifact to:\n  {CSV_PATH}\n  {PARQUET_PATH}")
    return artifact_df


def compute_metrics_from_artifact(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Computes mathematically reconciled metrics directly from the authoritative artifact."""
    if df is None:
        if PARQUET_PATH.exists():
            df = pd.read_parquet(PARQUET_PATH)
        elif CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
        else:
            df = build_and_save_authoritative_artifact()

    y_test = df["target_label"].values
    train_prev = float(df["train_prevalence"].iloc[0])
    test_prev = float(df["test_prevalence"].iloc[0])
    N = len(y_test)

    reconciled = {}
    for model_name in MODEL_RUNS.keys():
        probs = df[f"prob_{model_name}"].values
        op_t = float(df[f"op_thresh_{model_name}"].iloc[0])
        run_id = str(df[f"run_id_{model_name}"].iloc[0])

        # 1. Ranking & calibration (threshold-independent)
        roc = float(roc_auc_score(y_test, probs))
        pr = float(average_precision_score(y_test, probs))
        bs = float(brier_score_loss(y_test, probs))

        bs_ref_train = float(brier_score_loss(y_test, np.full(N, train_prev)))
        bs_ref_test = float(brier_score_loss(y_test, np.full(N, test_prev)))

        bss_deployable = float(1.0 - (bs / bs_ref_train))
        bss_insample = float(1.0 - (bs / bs_ref_test))

        ece, mce, cal_table = compute_calibration_diagnostics(y_test, probs, n_bins=10)

        # 2. Operating threshold metrics (strict arithmetic)
        preds_op = (probs >= op_t).astype(int)
        cm = confusion_matrix(y_test, preds_op)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        pos_alert_count = tp + fp
        actual_pos_count = tp + fn
        alert_rate = pos_alert_count / N
        precision = tp / pos_alert_count if pos_alert_count > 0 else 0.0
        recall = tp / actual_pos_count if actual_pos_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Fixed alert budgets (top 5%, 10%, 15%, 20%)
        budget_metrics = {}
        for pct in [5, 10, 15, 20]:
            k = int(N * (pct / 100.0))
            cutoff = np.partition(probs, -k)[-k]
            b_preds = (probs >= cutoff).astype(int)
            b_tp = int(np.sum((b_preds == 1) & (y_test == 1)))
            b_fp = int(np.sum((b_preds == 1) & (y_test == 0)))
            b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
            b_rec = b_tp / actual_pos_count if actual_pos_count > 0 else 0.0
            budget_metrics[f"top_{pct}pct"] = {
                "threshold_cutoff": round(float(cutoff), 4),
                "precision": round(b_prec, 4),
                "recall": round(b_rec, 4),
                "tp": b_tp,
                "fp": b_fp
            }

        reconciled[model_name] = {
            "run_id": run_id,
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "brier_score": round(bs, 4),
            "bss_deployable": round(bss_deployable, 4),
            "bss_insample": round(bss_insample, 4),
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "op_threshold": round(op_t, 4),
            "alert_rate": round(alert_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "fixed_budget_eval": budget_metrics,
            "calibration_table": cal_table
        }

    return reconciled


if __name__ == "__main__":
    df = build_and_save_authoritative_artifact()
    metrics = compute_metrics_from_artifact(df)
    import json
    print(json.dumps(metrics, indent=2))
