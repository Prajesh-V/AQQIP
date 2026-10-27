"""
Rigorous Alert Budget Evaluator & Paired Block-Bootstrap Validation

Addresses Priority 2 and Priority 4:
1. Evaluates on a Common Matched Universe where all models are scored on identical ticker-date observations.
2. Performs daily cross-sectional ranking on each decision date t:
   k_t = round(N_t * budget_pct)
   Selects the exact top-k_t securities with deterministic tie-breaking.
   Guarantees identical daily alert counts across all fixed-budget models.
3. Separately evaluates the Evidence-Gated Policy when it abstains/filters (reporting realized alert rate).
4. Evaluates Broader Available Populations separately with explicit coverage reporting.
5. Performs Paired Block-Bootstrap (resampling trading dates) to compute robust standard errors,
   95% confidence intervals, and two-sided p-values preserving cross-sectional correlation.
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
    precision_score, recall_score, f1_score
)
import mysql.connector

from config.settings import get_settings
from src.quant.train_pipeline import (
    load_raw_data, DEFAULT_MARKET_FEATURES
)
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.benchmark_runner import compute_calibration_diagnostics
from src.utils.logging import get_logger

logger = get_logger("alert_budget_evaluator")

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


def load_v1_and_v2_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def evaluate_daily_budget(
    df: pd.DataFrame,
    score_col: str,
    budget_pct: float,
    tie_col: str = "ticker"
) -> Tuple[int, int, int, int]:
    """
    Selects top-k_t on each date t deterministically.
    Returns (TP, FP, Alerts, PositivesInUniverse).
    """
    total_tp = 0
    total_fp = 0
    total_alerts = 0
    total_pos = int(df["target_label"].sum())

    for date, group in df.groupby("trade_date"):
        n_t = len(group)
        if n_t == 0:
            continue
        k_t = max(1, int(round(n_t * budget_pct)))
        # Deterministic ranking: highest score first, then ticker ascending as tie-breaker
        sorted_group = group.sort_values(by=[score_col, tie_col], ascending=[False, True])
        selected = sorted_group.iloc[:k_t]

        tp = int(selected["target_label"].sum())
        fp = len(selected) - tp
        total_tp += tp
        total_fp += fp
        total_alerts += len(selected)

    return total_tp, total_fp, total_alerts, total_pos


def run_comprehensive_evaluation() -> Dict[str, Any]:
    logger.info("Starting Rigorous Alert Budget & Uncertainty Evaluation")
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    v1_df, v2_df = load_v1_and_v2_features()

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

    budgets = [0.05, 0.10, 0.20]

    # Containers for matched and broader results
    matched_results: Dict[str, List[Dict[str, Any]]] = {
        "M_Regularized": [],
        "M_Reg_Plus_SecBert_v1": [],
        "M_Reg_Plus_SecBert_v2": [],
        "Evidence_Gated_Ranked": [],
        "Evidence_Gated_Abstaining": []
    }

    broad_results: Dict[str, List[Dict[str, Any]]] = {
        "M_Regularized": [],
        "M_Reg_Plus_SecBert_v1": [],
        "M_Reg_Plus_SecBert_v2": [],
    }

    # Store fold prediction panels for bootstrap
    fold_panels = []

    for fold_name, val_start, test_start in folds:
        logger.info(f"Processing {fold_name} ({val_start} to {test_start})")
        train_df, val_df, _ = builder.split_and_label(
            panel,
            val_start=val_start,
            test_start=test_start,
            alpha=0.10,
            enforce_temporal_purging=True
        )

        y_train = train_df["target_label"].values
        y_val = val_df["target_label"].values
        val_df = val_df.copy()

        # Fit models
        for m_name, feat_cols in models.items():
            dtrain = lgb.Dataset(train_df[feat_cols], label=y_train, params={"feature_pre_filter": False})
            dval = lgb.Dataset(val_df[feat_cols], label=y_val, reference=dtrain, params={"feature_pre_filter": False})
            model = lgb.train(
                LGB_PARAMS,
                dtrain,
                valid_sets=[dval],
                num_boost_round=300,
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
            )
            val_df[f"prob_{m_name}"] = model.predict(val_df[feat_cols])

        # Define Evidence-Gated score
        # Multiplier: if qualitative evidence confirmed (adverse_evidence > 0 OR drift > median), 1.25, else 0.75
        median_drift = float(np.nanmedian(train_df["secbert_v2_rf_semantic_drift"])) if not train_df["secbert_v2_rf_semantic_drift"].dropna().empty else 0.001
        has_ev = (val_df["secbert_v2_adverse_evidence_score"].fillna(0.0) > 0.0) | (val_df["secbert_v2_rf_semantic_drift"].fillna(0.0) > median_drift)
        val_df["has_evidence"] = has_ev.astype(int)
        val_df["prob_Evidence_Gated_Ranked"] = val_df["prob_M_Regularized"] * np.where(has_ev, 1.25, 0.75)

        # -------------------------------------------------------------
        # Part A: Common Matched Universe (both v1 and v2 have prior)
        # -------------------------------------------------------------
        matched_mask = (val_df["secbert_has_prior"] == 1) & (val_df["secbert_v2_has_prior"] == 1)
        df_matched = val_df[matched_mask].copy()
        n_matched = len(df_matched)
        n_matched_pos = int(df_matched["target_label"].sum())

        for m_name in ["M_Regularized", "M_Reg_Plus_SecBert_v1", "M_Reg_Plus_SecBert_v2", "Evidence_Gated_Ranked"]:
            prob_col = f"prob_{m_name}"
            roc = float(roc_auc_score(df_matched["target_label"], df_matched[prob_col]))
            pr = float(average_precision_score(df_matched["target_label"], df_matched[prob_col]))
            bs = float(brier_score_loss(df_matched["target_label"], np.clip(df_matched[prob_col], 0, 1)))

            b_dict = {}
            for b in budgets:
                tp, fp, alerts, pos = evaluate_daily_budget(df_matched, prob_col, b)
                prec = tp / max(1, alerts)
                rec = tp / max(1, pos)
                fp_share = fp / max(1, alerts)
                b_dict[f"budget_{int(b*100)}pct"] = {
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "fp_share": round(fp_share, 4),
                    "alerts": alerts,
                    "tp": tp,
                    "fp": fp
                }

            matched_results[m_name].append({
                "fold": fold_name,
                "n_obs": n_matched,
                "n_pos": n_matched_pos,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4),
                "brier_score": round(bs, 4),
                "budgets": b_dict
            })

        # Evaluate Evidence_Gated_Abstaining on Matched
        # Takes top-k from M_Regularized, but suppresses alert if has_evidence == 0
        gated_abstain_budgets = {}
        for b in budgets:
            tot_tp, tot_fp, tot_alerts = 0, 0, 0
            for date, group in df_matched.groupby("trade_date"):
                n_t = len(group)
                k_t = max(1, int(round(n_t * b)))
                sorted_group = group.sort_values(by=["prob_M_Regularized", "ticker"], ascending=[False, True])
                selected = sorted_group.iloc[:k_t]
                # Filter: only keep if has_evidence == 1
                filtered = selected[selected["has_evidence"] == 1]
                tp = int(filtered["target_label"].sum())
                fp = len(filtered) - tp
                tot_tp += tp
                tot_fp += fp
                tot_alerts += len(filtered)

            gated_abstain_budgets[f"budget_{int(b*100)}pct"] = {
                "precision": round(tot_tp / max(1, tot_alerts), 4),
                "recall": round(tot_tp / max(1, n_matched_pos), 4),
                "fp_share": round(tot_fp / max(1, tot_alerts), 4),
                "realized_alerts": tot_alerts,
                "realized_alert_rate": round(tot_alerts / max(1, n_matched), 4),
                "tp": tot_tp,
                "fp": tot_fp
            }

        matched_results["Evidence_Gated_Abstaining"].append({
            "fold": fold_name,
            "n_obs": n_matched,
            "n_pos": n_matched_pos,
            "budgets": gated_abstain_budgets
        })

        # -------------------------------------------------------------
        # Part B: Broader Available Population (Model's actual coverage)
        # -------------------------------------------------------------
        for m_name in ["M_Regularized", "M_Reg_Plus_SecBert_v1", "M_Reg_Plus_SecBert_v2"]:
            # Population for this model: non-null predictions
            prob_col = f"prob_{m_name}"
            df_pop = val_df.dropna(subset=[prob_col]).copy()
            n_pop = len(df_pop)
            n_pop_pos = int(df_pop["target_label"].sum())

            roc = float(roc_auc_score(df_pop["target_label"], df_pop[prob_col]))
            pr = float(average_precision_score(df_pop["target_label"], df_pop[prob_col]))
            bs = float(brier_score_loss(df_pop["target_label"], df_pop[prob_col]))

            b_dict = {}
            for b in budgets:
                tp, fp, alerts, pos = evaluate_daily_budget(df_pop, prob_col, b)
                prec = tp / max(1, alerts)
                rec = tp / max(1, pos)
                fp_share = fp / max(1, alerts)
                b_dict[f"budget_{int(b*100)}pct"] = {
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "fp_share": round(fp_share, 4),
                    "alerts": alerts,
                    "tp": tp,
                    "fp": fp
                }

            broad_results[m_name].append({
                "fold": fold_name,
                "n_obs": n_pop,
                "n_pos": n_pop_pos,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4),
                "brier_score": round(bs, 4),
                "budgets": b_dict
            })

        val_df["fold"] = fold_name
        fold_panels.append(df_matched)

    # -------------------------------------------------------------
    # Paired Date-Level Block Bootstrap on Matched Sample
    # -------------------------------------------------------------
    all_matched = pd.concat(fold_panels, ignore_index=True)
    dates = all_matched["trade_date"].unique()
    n_bootstrap = 500
    rng = np.random.default_rng(42)

    delta_roc_v1 = []
    delta_roc_v2 = []
    delta_pr_v1 = []
    delta_pr_v2 = []

    for _ in range(n_bootstrap):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        # Select records for sampled dates
        date_df = pd.DataFrame({"trade_date": sampled_dates})
        sample = pd.merge(date_df, all_matched, on="trade_date", how="inner")

        y_s = sample["target_label"].values
        if len(np.unique(y_s)) < 2:
            continue

        p_m = sample["prob_M_Regularized"].values
        p_v1 = sample["prob_M_Reg_Plus_SecBert_v1"].values
        p_v2 = sample["prob_M_Reg_Plus_SecBert_v2"].values

        roc_m = roc_auc_score(y_s, p_m)
        roc_v1 = roc_auc_score(y_s, p_v1)
        roc_v2 = roc_auc_score(y_s, p_v2)

        pr_m = average_precision_score(y_s, p_m)
        pr_v1 = average_precision_score(y_s, p_v1)
        pr_v2 = average_precision_score(y_s, p_v2)

        delta_roc_v1.append(roc_v1 - roc_m)
        delta_roc_v2.append(roc_v2 - roc_m)
        delta_pr_v1.append(pr_v1 - pr_m)
        delta_pr_v2.append(pr_v2 - pr_m)

    # Bootstrap statistics
    def calc_boot_stats(deltas: List[float]) -> Dict[str, Any]:
        arr = np.array(deltas)
        mean_d = float(np.mean(arr))
        se_d = float(np.std(arr))
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        # Two-sided empirical p-value for H0: delta <= 0
        p_val = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
        return {
            "mean_delta": round(mean_d, 4),
            "se": round(se_d, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "p_value": round(p_val, 4),
            "is_significant_5pct": bool(p_val < 0.05)
        }

    bootstrap_summary = {
        "v1_vs_M_ROC": calc_boot_stats(delta_roc_v1),
        "v2_vs_M_ROC": calc_boot_stats(delta_roc_v2),
        "v1_vs_M_PR": calc_boot_stats(delta_pr_v1),
        "v2_vs_M_PR": calc_boot_stats(delta_pr_v2),
    }

    return {
        "matched_results": matched_results,
        "broad_results": broad_results,
        "bootstrap_summary": bootstrap_summary
    }


if __name__ == "__main__":
    res = run_comprehensive_evaluation()
    import json
    with open("scratch/alert_budget_evaluation_output.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved evaluation results to scratch/alert_budget_evaluation_output.json")
