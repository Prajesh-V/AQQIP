"""
Rolling-Origin Pretrained Transformer Experiment and Ablation

Evaluates incremental predictive value of frozen SEC-BERT qualitative features versus:
1. M_Regularized (market + cross-sectional features alone).
2. M_Regularized + Existing qualitative features (TF-IDF / Jaccard).
3. M_Regularized + SEC-BERT qualitative features.
4. Qualitative ablation on the common matched sample.

Uses strict chronological rolling-origin evaluation with purged target windows (H=20),
point-in-time filing availability, validation-only operational thresholding, and
separate MLflow tracking.
"""

from __future__ import annotations
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)
import mlflow
import mlflow.lightgbm
import mysql.connector

from config.settings import get_settings
from src.quant.train_pipeline import (
    load_raw_data, DEFAULT_MARKET_FEATURES, DEFAULT_FUNDAMENTAL_FEATURES, DEFAULT_QUALITATIVE_FEATURES
)
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.point_in_time_join import PointInTimeJoiner
from src.quant.feature_builder import FeatureBuilder
from src.quant.benchmark_runner import compute_calibration_diagnostics, select_operational_threshold
from src.utils.logging import get_logger

logger = get_logger("transformer_experiment")

REGULARIZED_MARKET_FEATURES = DEFAULT_MARKET_FEATURES + ["return_20d_rel", "vol_ratio_20d"]

EXISTING_QUAL_FEATURES = [
    "rf_cosine_similarity",
    "rf_length_change_pct",
    "rf_added_sentences_ratio",
    "mda_cosine_similarity",
    "mda_length_change_pct"
]

SECBERT_FEATURES = [
    "secbert_rf_semantic_drift",
    "secbert_rf_norm_diff",
    "secbert_mda_semantic_drift",
    "secbert_rf_mda_divergence",
    "secbert_rf_available",
    "secbert_has_prior"
]

LGB_PARAMS_REGULARIZED = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 8,
    "max_depth": 3,
    "min_child_samples": 80,
    "colsample_bytree": 0.70,
    "subsample": 0.80,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "seed": 42
}


def load_transformer_qualitative_features() -> pd.DataFrame:
    """Loads extracted SEC-BERT features from MySQL."""
    settings = get_settings()
    conn = mysql.connector.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.user,
        password=settings.mysql.password,
        database=settings.mysql.database
    )
    query = """
    SELECT ticker, filing_date, accession_number, prior_accession_number,
           secbert_rf_available, secbert_mda_available, secbert_has_prior,
           secbert_rf_chunks, secbert_mda_chunks, secbert_rf_cosine_sim,
           secbert_rf_semantic_drift, secbert_rf_norm_diff,
           secbert_mda_cosine_sim, secbert_mda_semantic_drift,
           secbert_rf_mda_divergence
    FROM qualitative_transformer_features;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def merge_transformer_panel(
    base_panel: pd.DataFrame,
    transformer_df: pd.DataFrame,
    lag_days: int = 1
) -> pd.DataFrame:
    """Point-in-time forward-fill join of transformer features with conservative lag."""
    df_panel = base_panel.copy()
    df_panel["trade_date"] = pd.to_datetime(df_panel["trade_date"]).astype("datetime64[ns]")
    df_trans = transformer_df.copy()
    df_trans["filing_date"] = pd.to_datetime(df_trans["filing_date"]).astype("datetime64[ns]")

    # Apply conservative availability lag
    if lag_days > 0:
        df_trans["available_date"] = (df_trans["filing_date"] + pd.Timedelta(days=lag_days)).astype("datetime64[ns]")
    else:
        df_trans["available_date"] = df_trans["filing_date"].astype("datetime64[ns]")

    df_panel = df_panel.sort_values(["ticker", "trade_date"])
    df_trans = df_trans.sort_values(["ticker", "available_date"])

    merged_chunks = []
    for ticker, group in df_panel.groupby("ticker", group_keys=False):
        ticker_trans = df_trans[df_trans["ticker"] == ticker]
        if ticker_trans.empty:
            for col in SECBERT_FEATURES:
                group[col] = np.nan
            merged_chunks.append(group)
            continue

        merged = pd.merge_asof(
            group,
            ticker_trans[["available_date"] + SECBERT_FEATURES],
            left_on="trade_date",
            right_on="available_date",
            direction="backward"
        )
        merged.drop(columns=["available_date"], inplace=True, errors="ignore")
        merged_chunks.append(merged)

    result = pd.concat(merged_chunks, ignore_index=True)
    return result


class RollingOriginTransformerExperiment:
    """Runs rolling-origin paired comparisons across candidate feature sets."""

    def __init__(self):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        self.experiment_name = "aqqip_transformer_experiments"
        mlflow.set_experiment(self.experiment_name)

    def run_experiment(self) -> Dict[str, Any]:
        logger.info("Starting Rolling-Origin Transformer Experiment")
        market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
        transformer_df = load_transformer_qualitative_features()
        logger.info(f"Loaded transformer features: {len(transformer_df)} rows")

        builder = HistoricalDatasetBuilder()
        panel = builder.build_dataset(market_df, fund_df, qual_df)
        panel = merge_transformer_panel(panel, transformer_df, lag_days=1)

        folds = [
            ("Fold_1_2021", "2021-01-01", "2022-01-01"),
            ("Fold_2_2022", "2022-01-01", "2023-01-01"),
            ("Fold_3_2023", "2023-01-01", "2024-01-01"),
            ("Fold_4_2024", "2024-01-01", "2025-01-01"),
        ]

        models = {
            "M_Regularized": REGULARIZED_MARKET_FEATURES,
            "M_Reg_Plus_ExistingQual": REGULARIZED_MARKET_FEATURES + EXISTING_QUAL_FEATURES,
            "M_Reg_Plus_SecBert": REGULARIZED_MARKET_FEATURES + SECBERT_FEATURES,
            "ExistingQual_Ablation": EXISTING_QUAL_FEATURES,
            "SecBert_Ablation": SECBERT_FEATURES,
        }

        fold_results = {m: [] for m in models.keys()}
        matched_fold_results = {m: [] for m in models.keys()}

        for fold_name, val_start, test_start in folds:
            logger.info(f"Evaluating {fold_name}: Val {val_start} to {test_start}")
            train_df, val_df, _ = builder.split_and_label(
                panel,
                val_start=val_start,
                test_start=test_start,
                alpha=0.10,
                enforce_temporal_purging=True
            )

            train_prev = float(train_df["target_label"].mean())
            val_prev = float(val_df["target_label"].mean())
            y_train = train_df["target_label"].values
            y_val = val_df["target_label"].values

            # Matched sample mask (where secbert features are available)
            val_has_qual = val_df["secbert_has_prior"].fillna(0).astype(bool).values

            for model_name, feat_cols in models.items():
                X_train = train_df[feat_cols].copy()
                X_val = val_df[feat_cols].copy()

                # Train LightGBM model
                dtrain = lgb.Dataset(X_train, label=y_train, params={"feature_pre_filter": False})
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, params={"feature_pre_filter": False})

                model = lgb.train(
                    LGB_PARAMS_REGULARIZED,
                    dtrain,
                    valid_sets=[dval],
                    num_boost_round=300,
                    callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
                )

                val_probs = model.predict(X_val)

                # Broader population evaluation
                roc = float(roc_auc_score(y_val, val_probs))
                pr = float(average_precision_score(y_val, val_probs))
                bs = float(brier_score_loss(y_val, val_probs))
                bs_ref = float(brier_score_loss(y_val, np.full(len(y_val), train_prev)))
                bss = float(1.0 - (bs / bs_ref))

                op_thresh, _ = select_operational_threshold(y_val, val_probs)
                op_preds = (val_probs >= op_thresh).astype(int)
                prec = float(precision_score(y_val, op_preds, zero_division=0))
                rec = float(recall_score(y_val, op_preds, zero_division=0))
                f1 = float(f1_score(y_val, op_preds, zero_division=0))
                alert_rate = float(op_preds.mean())
                fp_share = float(np.sum((op_preds == 1) & (y_val == 0)) / max(1, np.sum(op_preds == 1)))

                ece, mce, _ = compute_calibration_diagnostics(y_val, val_probs, n_bins=10)

                # Fixed alert budgets: 10%, 20%, 30%
                budget_metrics = {}
                n_val = len(y_val)
                actual_val_pos = int(np.sum(y_val == 1))
                for pct in [10, 20, 30]:
                    k = int(n_val * (pct / 100.0))
                    cutoff = np.partition(val_probs, -k)[-k]
                    b_preds = (val_probs >= cutoff).astype(int)
                    b_tp = int(np.sum((b_preds == 1) & (y_val == 1)))
                    b_fp = int(np.sum((b_preds == 1) & (y_val == 0)))
                    b_prec = b_tp / max(1, b_tp + b_fp)
                    b_rec = b_tp / max(1, actual_val_pos)
                    b_fp_share = b_fp / max(1, b_tp + b_fp)
                    budget_metrics[f"budget_{pct}pct"] = {
                        "cutoff": round(float(cutoff), 4),
                        "precision": round(b_prec, 4),
                        "recall": round(b_rec, 4),
                        "fp_share": round(b_fp_share, 4),
                        "alerts": b_tp + b_fp,
                        "tp": b_tp,
                        "fp": b_fp
                    }

                fold_res = {
                    "fold": fold_name,
                    "roc_auc": round(roc, 4),
                    "pr_auc": round(pr, 4),
                    "pr_auc_rel_prev": round(pr / max(1e-4, val_prev), 4),
                    "brier_score": round(bs, 4),
                    "bss": round(bss, 4),
                    "ece": round(ece, 4),
                    "op_thresh": round(op_thresh, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "alert_rate": round(alert_rate, 4),
                    "fp_share": round(fp_share, 4),
                    "fixed_budgets": budget_metrics
                }
                fold_results[model_name].append(fold_res)

                # Matched sample evaluation (where qualitative filings exist)
                if np.sum(val_has_qual) > 50 and len(np.unique(y_val[val_has_qual])) > 1:
                    y_m = y_val[val_has_qual]
                    p_m = val_probs[val_has_qual]
                    roc_m = float(roc_auc_score(y_m, p_m))
                    pr_m = float(average_precision_score(y_m, p_m))
                    matched_fold_results[model_name].append({
                        "fold": fold_name,
                        "n_matched": int(np.sum(val_has_qual)),
                        "matched_prev": round(float(y_m.mean()), 4),
                        "roc_auc": round(roc_m, 4),
                        "pr_auc": round(pr_m, 4),
                    })

                # Log to MLflow
                with mlflow.start_run(run_name=f"{model_name}_{fold_name}") as run:
                    mlflow.log_params({
                        "model": model_name,
                        "fold": fold_name,
                        "val_start": val_start,
                        "test_start": test_start,
                        "feature_count": len(feat_cols),
                        "features": ",".join(feat_cols)
                    })
                    mlflow.log_metrics({
                        "val_roc_auc": roc,
                        "val_pr_auc": pr,
                        "val_bss": bss,
                        "val_f1": f1,
                        "val_alert_rate": alert_rate,
                        "val_precision": prec,
                        "val_recall": rec,
                        "val_ece": ece
                    })

        # Aggregate metrics across folds
        summary = {}
        for m, res_list in fold_results.items():
            rocs = [r["roc_auc"] for r in res_list]
            prs = [r["pr_auc"] for r in res_list]
            bsss = [r["bss"] for r in res_list]
            f1s = [r["f1"] for r in res_list]
            summary[m] = {
                "mean_roc_auc": round(float(np.mean(rocs)), 4),
                "std_roc_auc": round(float(np.std(rocs)), 4),
                "mean_pr_auc": round(float(np.mean(prs)), 4),
                "mean_bss": round(float(np.mean(bsss)), 4),
                "mean_f1": round(float(np.mean(f1s)), 4),
                "fold_details": res_list
            }

        return {
            "summary": summary,
            "matched_fold_results": matched_fold_results
        }


if __name__ == "__main__":
    exp = RollingOriginTransformerExperiment()
    results = exp.run_experiment()
    import json
    print("\n=== ROLLING-ORIGIN TRANSFORMER EXPERIMENT SUMMARY ===")
    print(json.dumps(results["summary"], indent=2))
    print("\n=== MATCHED SAMPLE COMPARISON ===")
    print(json.dumps(results["matched_fold_results"], indent=2))
