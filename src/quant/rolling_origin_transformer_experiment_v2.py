"""
Rolling-Origin Pretrained Transformer Experiment v2 & Precision-First Alert Policy

Evaluates:
1. M_Regularized (quantitative market baseline).
2. M_Reg_Plus_SecBert_v1 (existing 4-chunk SEC-BERT features).
3. M_Reg_Plus_SecBert_v2 (revised 8-chunk full-section + adverse-risk evidence + novelty features).
4. Evidence-Gated Policy:
   Alert fires if Quantitative Model flags drop (P >= T_op) AND (
     Adverse Risk Evidence confirmed (adverse_evidence > 0 OR semantic_drift > median)
     OR (if filing data missing, quantitative signal meets conservative hurdle P >= T_conservative)
   )

Evaluates on identical eligible observations across 4 pre-2025 chronological folds:
- Top 5%, 10%, 20% alert budgets.
- Precision, recall, false-positive share, alert counts, PR-AUC, Brier score, ECE.
- Common matched sample vs broader eligible population.
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
from src.quant.benchmark_runner import compute_calibration_diagnostics, select_operational_threshold
from src.utils.logging import get_logger

logger = get_logger("transformer_experiment_v2")

REGULARIZED_MARKET_FEATURES = DEFAULT_MARKET_FEATURES + ["return_20d_rel", "vol_ratio_20d"]

SECBERT_V1_FEATURES = [
    "secbert_rf_semantic_drift",
    "secbert_rf_norm_diff",
    "secbert_mda_semantic_drift",
    "secbert_rf_mda_divergence",
    "secbert_rf_available",
    "secbert_has_prior"
]

SECBERT_V2_FEATURES = [
    "secbert_v2_rf_semantic_drift",
    "secbert_v2_rf_novelty_score",
    "secbert_v2_adverse_evidence_score",
    "secbert_v2_rf_norm_diff",
    "secbert_v2_mda_semantic_drift",
    "secbert_v2_rf_mda_divergence",
    "secbert_v2_rf_available",
    "secbert_v2_has_prior"
]

LGB_PARAMS = {
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


def load_v1_and_v2_transformer_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both v1 and v2 extracted SEC-BERT features from MySQL."""
    settings = get_settings()
    conn = mysql.connector.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.user,
        password=settings.mysql.password,
        database=settings.mysql.database
    )
    v1_df = pd.read_sql("SELECT ticker, filing_date, " + ", ".join(SECBERT_V1_FEATURES) + " FROM qualitative_transformer_features;", conn)
    v2_df = pd.read_sql("SELECT ticker, filing_date, " + ", ".join(SECBERT_V2_FEATURES) + " FROM qualitative_transformer_features_v2;", conn)
    conn.close()

    v1_df["filing_date"] = pd.to_datetime(v1_df["filing_date"]).astype("datetime64[ns]")
    v2_df["filing_date"] = pd.to_datetime(v2_df["filing_date"]).astype("datetime64[ns]")
    return v1_df, v2_df


def merge_features_asof(base_panel: pd.DataFrame, feat_df: pd.DataFrame, feat_cols: List[str], lag_days: int = 1) -> pd.DataFrame:
    """Forward-fill joins features as-of trade_date with conservative availability lag."""
    df_panel = base_panel.copy()
    df_panel["trade_date"] = pd.to_datetime(df_panel["trade_date"]).astype("datetime64[ns]")
    df_feat = feat_df.copy()
    df_feat["available_date"] = (df_feat["filing_date"] + pd.Timedelta(days=lag_days)).astype("datetime64[ns]")

    df_panel = df_panel.sort_values(["ticker", "trade_date"])
    df_feat = df_feat.sort_values(["ticker", "available_date"])

    merged_chunks = []
    for ticker, group in df_panel.groupby("ticker", group_keys=False):
        t_feat = df_feat[df_feat["ticker"] == ticker]
        if t_feat.empty:
            for col in feat_cols:
                group[col] = np.nan
            merged_chunks.append(group)
            continue

        merged = pd.merge_asof(
            group,
            t_feat[["available_date"] + feat_cols],
            left_on="trade_date",
            right_on="available_date",
            direction="backward"
        )
        merged.drop(columns=["available_date"], inplace=True, errors="ignore")
        merged_chunks.append(merged)

    return pd.concat(merged_chunks, ignore_index=True)


class RollingOriginTransformerExperimentV2:
    """Executes Priority 3 precision-first comparisons and evidence-gated alert policies."""

    def __init__(self):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        mlflow.set_experiment("aqqip_transformer_experiments_v2")

    def run_experiment(self) -> Dict[str, Any]:
        logger.info("Starting Rolling-Origin Transformer Experiment v2 & Alert Policy Evaluation")
        market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
        v1_df, v2_df = load_v1_and_v2_transformer_features()
        logger.info(f"Loaded transformer features: v1={len(v1_df)} rows, v2={len(v2_df)} rows")

        builder = HistoricalDatasetBuilder()
        panel = builder.build_dataset(market_df, fund_df, qual_df)
        panel = merge_features_asof(panel, v1_df, SECBERT_V1_FEATURES, lag_days=1)
        panel = merge_features_asof(panel, v2_df, SECBERT_V2_FEATURES, lag_days=1)

        folds = [
            ("Fold_1_2021", "2021-01-01", "2022-01-01"),
            ("Fold_2_2022", "2022-01-01", "2023-01-01"),
            ("Fold_3_2023", "2023-01-01", "2024-01-01"),
            ("Fold_4_2024", "2024-01-01", "2025-01-01"),
        ]

        models = {
            "M_Regularized": REGULARIZED_MARKET_FEATURES,
            "M_Reg_Plus_SecBert_v1": REGULARIZED_MARKET_FEATURES + SECBERT_V1_FEATURES,
            "M_Reg_Plus_SecBert_v2": REGULARIZED_MARKET_FEATURES + SECBERT_V2_FEATURES,
        }

        fold_results = {m: [] for m in models.keys()}
        fold_results["Evidence_Gated_Policy"] = []
        matched_fold_results = {m: [] for m in list(models.keys()) + ["Evidence_Gated_Policy"]}

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
            val_has_qual = val_df["secbert_v2_has_prior"].fillna(0).astype(bool).values

            probs_dict = {}

            # Train models
            for model_name, feat_cols in models.items():
                X_train = train_df[feat_cols].copy()
                X_val = val_df[feat_cols].copy()

                dtrain = lgb.Dataset(X_train, label=y_train, params={"feature_pre_filter": False})
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, params={"feature_pre_filter": False})

                model = lgb.train(
                    LGB_PARAMS,
                    dtrain,
                    valid_sets=[dval],
                    num_boost_round=300,
                    callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
                )

                val_probs = model.predict(X_val)
                probs_dict[model_name] = val_probs

                # Evaluate standard metrics
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
                ece, _, _ = compute_calibration_diagnostics(y_val, val_probs, n_bins=10)

                # Fixed alert budgets: 5%, 10%, 20%
                budget_metrics = {}
                n_val = len(y_val)
                actual_val_pos = int(np.sum(y_val == 1))
                for pct in [5, 10, 20]:
                    k = int(n_val * (pct / 100.0))
                    cutoff = np.partition(val_probs, -k)[-k]
                    b_preds = (val_probs >= cutoff).astype(int)
                    b_tp = int(np.sum((b_preds == 1) & (y_val == 1)))
                    b_fp = int(np.sum((b_preds == 1) & (y_val == 0)))
                    b_prec = b_tp / max(1, b_tp + b_fp)
                    b_rec = b_tp / max(1, actual_val_pos)
                    b_fp_share = b_fp / max(1, b_tp + b_fp)
                    budget_metrics[f"budget_{pct}pct"] = {
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

                # Matched sample
                if np.sum(val_has_qual) > 50:
                    y_m = y_val[val_has_qual]
                    p_m = val_probs[val_has_qual]
                    matched_fold_results[model_name].append({
                        "fold": fold_name,
                        "n_matched": int(np.sum(val_has_qual)),
                        "roc_auc": round(float(roc_auc_score(y_m, p_m)), 4),
                        "pr_auc": round(float(average_precision_score(y_m, p_m)), 4),
                    })

            # -------------------------------------------------------------
            # Policy Candidate: Evidence-Gated Alert Policy
            # -------------------------------------------------------------
            p_quant = probs_dict["M_Regularized"]
            op_t_quant = fold_results["M_Regularized"][-1]["op_thresh"]

            # Gate condition:
            # If company has qualitative filing: require positive adverse evidence OR high semantic drift
            # If qualitative filing is missing: require higher quantitative confidence hurdle
            adverse_ev = val_df["secbert_v2_adverse_evidence_score"].fillna(0.0).values
            drift = val_df["secbert_v2_rf_semantic_drift"].fillna(0.0).values
            median_drift = float(np.nanmedian(train_df["secbert_v2_rf_semantic_drift"])) if not train_df["secbert_v2_rf_semantic_drift"].dropna().empty else 0.001

            has_evidence = (adverse_ev > 0.0) | (drift > median_drift)

            # Evidence gated alert prediction
            # Higher hurdle for missing: top 10% cutoff of quantitative model
            high_hurdle_missing = np.partition(p_quant, -int(len(p_quant) * 0.10))[-int(len(p_quant) * 0.10)]

            gated_preds = np.zeros(len(y_val), dtype=int)
            for i in range(len(y_val)):
                if val_has_qual[i]:
                    if p_quant[i] >= op_t_quant and has_evidence[i]:
                        gated_preds[i] = 1
                else:
                    if p_quant[i] >= high_hurdle_missing:
                        gated_preds[i] = 1

            g_tp = int(np.sum((gated_preds == 1) & (y_val == 1)))
            g_fp = int(np.sum((gated_preds == 1) & (y_val == 0)))
            g_prec = g_tp / max(1, g_tp + g_fp)
            g_rec = g_tp / max(1, np.sum(y_val == 1))
            g_f1 = (2 * g_prec * g_rec / (g_prec + g_rec)) if (g_prec + g_rec) > 0 else 0.0
            g_alert_rate = float(gated_preds.mean())
            g_fp_share = g_fp / max(1, g_tp + g_fp)

            # Continuous score for gated ranking: p_quant multiplied by evidence multiplier
            evidence_multiplier = np.where(val_has_qual, np.where(has_evidence, 1.25, 0.75), 1.0)
            gated_score = p_quant * evidence_multiplier

            # Gated fixed budgets
            gated_budgets = {}
            for pct in [5, 10, 20]:
                k = int(n_val * (pct / 100.0))
                cutoff = np.partition(gated_score, -k)[-k]
                b_preds = (gated_score >= cutoff).astype(int)
                b_tp = int(np.sum((b_preds == 1) & (y_val == 1)))
                b_fp = int(np.sum((b_preds == 1) & (y_val == 0)))
                gated_budgets[f"budget_{pct}pct"] = {
                    "precision": round(b_tp / max(1, b_tp + b_fp), 4),
                    "recall": round(b_tp / max(1, actual_val_pos), 4),
                    "fp_share": round(b_fp / max(1, b_tp + b_fp), 4),
                    "alerts": b_tp + b_fp,
                    "tp": b_tp,
                    "fp": b_fp
                }

            fold_results["Evidence_Gated_Policy"].append({
                "fold": fold_name,
                "roc_auc": round(float(roc_auc_score(y_val, gated_score)), 4),
                "pr_auc": round(float(average_precision_score(y_val, gated_score)), 4),
                "brier_score": round(float(brier_score_loss(y_val, np.clip(gated_score, 0, 1))), 4),
                "bss": round(float(1.0 - (brier_score_loss(y_val, np.clip(gated_score, 0, 1)) / bs_ref)), 4),
                "ece": round(float(compute_calibration_diagnostics(y_val, np.clip(gated_score, 0, 1))[0]), 4),
                "op_thresh": round(op_t_quant, 4),
                "precision": round(g_prec, 4),
                "recall": round(g_rec, 4),
                "f1": round(g_f1, 4),
                "alert_rate": round(g_alert_rate, 4),
                "fp_share": round(g_fp_share, 4),
                "fixed_budgets": gated_budgets
            })

            if np.sum(val_has_qual) > 50:
                y_m = y_val[val_has_qual]
                g_m = gated_score[val_has_qual]
                matched_fold_results["Evidence_Gated_Policy"].append({
                    "fold": fold_name,
                    "n_matched": int(np.sum(val_has_qual)),
                    "roc_auc": round(float(roc_auc_score(y_m, g_m)), 4),
                    "pr_auc": round(float(average_precision_score(y_m, g_m)), 4),
                })

        # Summary
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
    exp = RollingOriginTransformerExperimentV2()
    res = exp.run_experiment()
    import json
    print("\n=== ROLLING-ORIGIN EXPERIMENT V2 SUMMARY ===")
    print(json.dumps(res["summary"], indent=2))
    print("\n=== MATCHED SAMPLE COMPARISON V2 ===")
    print(json.dumps(res["matched_fold_results"], indent=2))
