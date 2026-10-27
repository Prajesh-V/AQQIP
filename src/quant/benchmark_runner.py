"""
AQQIP Model Benchmark Runner

Evaluates and compares model baselines on the expanded multi-sector dataset:
1. Majority Class Baseline
2. Momentum Baseline (trailing 20d return drop rule)
3. Logistic Regression (balanced)
4. LightGBM Market-Only
5. LightGBM Quant-Integrated (Market + SEC Fundamentals)

Enforces:
- Strict temporal boundary purging.
- Operational threshold calibration strictly on validation partition (frozen for test).
- Complete evaluation metrics: ROC-AUC, PR-AUC, Brier score, Brier skill score,
  ECE, MCE, Precision, Recall, F1, Confusion Matrix, Prevalence, Predicted-Positive Rate.
- MLflow logging for each experiment.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)

from config.settings import get_settings
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.leakage_validator import LeakageValidator
from src.quant.mlflow_tracking import MLflowTracker
from src.quant.train_pipeline import load_raw_data, DEFAULT_MARKET_FEATURES, DEFAULT_FUNDAMENTAL_FEATURES
from src.utils.logging import get_logger

logger = get_logger("benchmark_runner")


def compute_calibration_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE), MCE, and reliability table."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    mce = 0.0
    table = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            count = int(np.sum(mask))
            mean_p = float(np.mean(y_prob[mask]))
            true_p = float(np.mean(y_true[mask]))
            err = abs(mean_p - true_p)
            ece += (count / len(y_prob)) * err
            mce = max(mce, err)
            table.append({
                "bin": f"[{bins[i]:.2f}, {bins[i+1]:.2f})",
                "count": count,
                "mean_pred": round(mean_p, 4),
                "true_rate": round(true_p, 4),
                "abs_error": round(err, 4)
            })
    return round(float(ece), 4), round(float(mce), 4), table


def select_operational_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> Tuple[float, float]:
    """Selects operational decision threshold maximizing F1 on validation partition."""
    thresholds = np.linspace(0.02, 0.80, 79)
    best_thresh = 0.50
    best_f1 = -1.0
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
            
    return round(best_thresh, 4), round(best_f1, 4)


def evaluate_predictions_authoritative(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    operational_threshold: float,
    train_prevalence: float
) -> Dict[str, Any]:
    """Calculates all required evaluation metrics on identical holdout rows."""
    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 and not np.all(y_prob == y_prob[0]) else 0.50
    pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(y_true.mean())
    bs = float(brier_score_loss(y_true, y_prob))
    
    # Reference Brier score (naive constant prevalence baseline)
    bs_ref_test = float(brier_score_loss(y_true, np.full(len(y_true), y_true.mean())))
    bs_ref_train = float(brier_score_loss(y_true, np.full(len(y_true), train_prevalence)))
    bss_test = round(1.0 - (bs / bs_ref_test), 4) if bs_ref_test > 0 else 0.0
    bss_train = round(1.0 - (bs / bs_ref_train), 4) if bs_ref_train > 0 else 0.0
    
    ece, mce, cal_table = compute_calibration_diagnostics(y_true, y_prob)
    
    # At frozen operational threshold
    op_preds = (y_prob >= operational_threshold).astype(int)
    cm_op = confusion_matrix(y_true, op_preds)
    tn_op, fp_op, fn_op, tp_op = int(cm_op[0, 0]), int(cm_op[0, 1]), int(cm_op[1, 0]), int(cm_op[1, 1])
    
    # At reference 0.50 threshold
    ref_preds = (y_prob >= 0.50).astype(int)
    cm_ref = confusion_matrix(y_true, ref_preds)
    tn_ref, fp_ref, fn_ref, tp_ref = int(cm_ref[0, 0]), int(cm_ref[0, 1]), int(cm_ref[1, 0]), int(cm_ref[1, 1])
    
    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "brier_score": round(bs, 5),
        "brier_skill_score_test_prev": bss_test,
        "brier_skill_score_train_prev": bss_train,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "mean_predicted_prob": round(float(y_prob.mean()), 4),
        "true_class_prevalence": round(float(y_true.mean()), 4),
        "operational_threshold": operational_threshold,
        "precision_at_op": round(float(precision_score(y_true, op_preds, zero_division=0)), 4),
        "recall_at_op": round(float(recall_score(y_true, op_preds, zero_division=0)), 4),
        "f1_at_op": round(float(f1_score(y_true, op_preds, zero_division=0)), 4),
        "predicted_pos_rate_at_op": round(float(op_preds.mean()), 4),
        "confusion_matrix_at_op": {"tn": tn_op, "fp": fp_op, "fn": fn_op, "tp": tp_op},
        "precision_at_50": round(float(precision_score(y_true, ref_preds, zero_division=0)), 4),
        "recall_at_50": round(float(recall_score(y_true, ref_preds, zero_division=0)), 4),
        "f1_at_50": round(float(f1_score(y_true, ref_preds, zero_division=0)), 4),
        "predicted_pos_rate_at_50": round(float(ref_preds.mean()), 4),
        "confusion_matrix_at_50": {"tn": tn_ref, "fp": fp_ref, "fn": fn_ref, "tp": tp_ref},
        "calibration_table": cal_table
    }


def run_benchmark_suite(
    val_start: str = "2023-01-01",
    test_start: str = "2025-01-01",
    alpha: float = 0.10
) -> Dict[str, Any]:
    """Runs benchmark comparison across all models on expanded dataset."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracker = MLflowTracker(experiment_name="AQQIP_Phase3_Quantitative")
    
    # 1. Load data & build leak-free panel
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    builder = HistoricalDatasetBuilder()
    panel = builder.build_dataset(market_df, fund_df, qual_df)
    
    # 2. Split with temporal purging
    train_df, val_df, test_df = builder.split_and_label(
        panel,
        val_start=val_start,
        test_start=test_start,
        alpha=alpha,
        enforce_temporal_purging=True
    )
    
    train_prev = float(train_df["target_label"].mean())
    y_train = train_df["target_label"].values
    y_val = val_df["target_label"].values
    y_test = test_df["target_label"].values
    
    market_features = list(DEFAULT_MARKET_FEATURES)
    integrated_features = list(DEFAULT_MARKET_FEATURES) + list(DEFAULT_FUNDAMENTAL_FEATURES)
    from src.quant.train_pipeline import DEFAULT_QUALITATIVE_FEATURES
    mfq_features = list(DEFAULT_MARKET_FEATURES) + list(DEFAULT_FUNDAMENTAL_FEATURES) + list(DEFAULT_QUALITATIVE_FEATURES)
    
    # Dataset metadata
    dataset_fingerprint = hashlib.sha256(train_df[["ticker", "trade_date", "adj_close", "forward_return_pct"]].to_csv(index=False).encode()).hexdigest()[:16]
    fitted_thresh = float(train_df["target_threshold_used"].iloc[0])
    
    benchmark_results = {}
    
    # ---------------------------------------------------------
    # Baseline 1: Majority Class (Prevalence)
    # ---------------------------------------------------------
    val_probs_maj = np.full(len(y_val), train_prev)
    test_probs_maj = np.full(len(y_test), train_prev)
    t_maj, _ = select_operational_threshold(y_val, val_probs_maj)
    res_maj = evaluate_predictions_authoritative(y_test, test_probs_maj, t_maj, train_prev)
    benchmark_results["majority_class"] = res_maj
    
    # ---------------------------------------------------------
    # Baseline 2: Momentum Baseline
    # Uses trailing 20-day return to predict forward 20-day drop
    # ---------------------------------------------------------
    mom_lr = LogisticRegression(max_iter=1000)
    mom_lr.fit(train_df[["return_20d", "return_60d"]].fillna(0), y_train)
    val_probs_mom = mom_lr.predict_proba(val_df[["return_20d", "return_60d"]].fillna(0))[:, 1]
    test_probs_mom = mom_lr.predict_proba(test_df[["return_20d", "return_60d"]].fillna(0))[:, 1]
    t_mom, _ = select_operational_threshold(y_val, val_probs_mom)
    res_mom = evaluate_predictions_authoritative(y_test, test_probs_mom, t_mom, train_prev)
    benchmark_results["momentum_baseline"] = res_mom
    
    # ---------------------------------------------------------
    # Baseline 3: Logistic Regression (Balanced)
    # ---------------------------------------------------------
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(train_df[market_features].fillna(0), y_train)
    val_probs_lr = lr.predict_proba(val_df[market_features].fillna(0))[:, 1]
    test_probs_lr = lr.predict_proba(test_df[market_features].fillna(0))[:, 1]
    t_lr, _ = select_operational_threshold(y_val, val_probs_lr)
    res_lr = evaluate_predictions_authoritative(y_test, test_probs_lr, t_lr, train_prev)
    benchmark_results["logistic_regression"] = res_lr
    
    # Common LightGBM params
    params_lgb = {
        "objective": "binary",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 50,
        "subsample": 0.8,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": 42
    }
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

    # ---------------------------------------------------------
    # Configuration M: LightGBM Market-Only
    # ---------------------------------------------------------
    dtrain_m = lgb.Dataset(train_df[market_features], label=y_train, params={"feature_pre_filter": False})
    dval_m = lgb.Dataset(val_df[market_features], label=y_val, reference=dtrain_m, params={"feature_pre_filter": False})
    
    model_m = lgb.train(params_lgb, dtrain_m, valid_sets=[dval_m], num_boost_round=500, callbacks=callbacks)
    val_probs_m = model_m.predict(val_df[market_features])
    test_probs_m = model_m.predict(test_df[market_features])
    t_lgb_m, _ = select_operational_threshold(y_val, val_probs_m)
    res_lgb_m = evaluate_predictions_authoritative(y_test, test_probs_m, t_lgb_m, train_prev)
    benchmark_results["lightgbm_market_only_M"] = res_lgb_m

    with tracker.start_run(run_name="LightGBM_Market_Only_M") as run_m:
        tracker.log_params({
            "config": "M",
            "features_count": len(market_features),
            "features": ",".join(market_features),
            "operational_threshold": t_lgb_m,
            "fitted_threshold": fitted_thresh
        })
        tracker.log_metrics({
            "test_roc_auc": res_lgb_m["roc_auc"],
            "test_pr_auc": res_lgb_m["pr_auc"],
            "test_brier_score": res_lgb_m["brier_score"],
            "test_bss_test_prev": res_lgb_m["brier_skill_score_test_prev"],
            "test_ece": res_lgb_m["expected_calibration_error"],
            "test_f1_at_op": res_lgb_m["f1_at_op"],
        })
        tracker.log_model(model_m)
        benchmark_results["run_id_M"] = run_m.info.run_id
    
    # ---------------------------------------------------------
    # Configuration MF: LightGBM Market + Fundamentals
    # ---------------------------------------------------------
    dtrain_mf = lgb.Dataset(train_df[integrated_features], label=y_train, params={"feature_pre_filter": False})
    dval_mf = lgb.Dataset(val_df[integrated_features], label=y_val, reference=dtrain_mf, params={"feature_pre_filter": False})
    
    model_mf = lgb.train(params_lgb, dtrain_mf, valid_sets=[dval_mf], num_boost_round=500, callbacks=callbacks)
    val_probs_mf = model_mf.predict(val_df[integrated_features])
    test_probs_mf = model_mf.predict(test_df[integrated_features])
    t_lgb_mf, _ = select_operational_threshold(y_val, val_probs_mf)
    res_lgb_mf = evaluate_predictions_authoritative(y_test, test_probs_mf, t_lgb_mf, train_prev)
    benchmark_results["lightgbm_quant_integrated_MF"] = res_lgb_mf

    with tracker.start_run(run_name="LightGBM_Quant_Integrated_MF") as run_mf:
        tracker.log_params({
            "config": "MF",
            "features_count": len(integrated_features),
            "features": ",".join(integrated_features),
            "operational_threshold": t_lgb_mf,
            "fitted_threshold": fitted_thresh
        })
        tracker.log_metrics({
            "test_roc_auc": res_lgb_mf["roc_auc"],
            "test_pr_auc": res_lgb_mf["pr_auc"],
            "test_brier_score": res_lgb_mf["brier_score"],
            "test_bss_test_prev": res_lgb_mf["brier_skill_score_test_prev"],
            "test_ece": res_lgb_mf["expected_calibration_error"],
            "test_f1_at_op": res_lgb_mf["f1_at_op"],
        })
        tracker.log_model(model_mf)
        benchmark_results["run_id_MF"] = run_mf.info.run_id

    # ---------------------------------------------------------
    # Configuration MFQ: LightGBM Market + Fundamentals + Qualitative Filing Changes
    # ---------------------------------------------------------
    dtrain_mfq = lgb.Dataset(train_df[mfq_features], label=y_train, params={"feature_pre_filter": False})
    dval_mfq = lgb.Dataset(val_df[mfq_features], label=y_val, reference=dtrain_mfq, params={"feature_pre_filter": False})

    model_mfq = lgb.train(params_lgb, dtrain_mfq, valid_sets=[dval_mfq], num_boost_round=500, callbacks=callbacks)
    val_probs_mfq = model_mfq.predict(val_df[mfq_features])
    test_probs_mfq = model_mfq.predict(test_df[mfq_features])
    t_lgb_mfq, _ = select_operational_threshold(y_val, val_probs_mfq)
    res_lgb_mfq = evaluate_predictions_authoritative(y_test, test_probs_mfq, t_lgb_mfq, train_prev)
    benchmark_results["lightgbm_qual_quant_integrated_MFQ"] = res_lgb_mfq

    with tracker.start_run(run_name="LightGBM_Qual_Quant_Integrated_MFQ") as run_mfq:
        tracker.log_params({
            "config": "MFQ",
            "features_count": len(mfq_features),
            "features": ",".join(mfq_features),
            "operational_threshold": t_lgb_mfq,
            "fitted_threshold": fitted_thresh
        })
        tracker.log_metrics({
            "test_roc_auc": res_lgb_mfq["roc_auc"],
            "test_pr_auc": res_lgb_mfq["pr_auc"],
            "test_brier_score": res_lgb_mfq["brier_score"],
            "test_bss_test_prev": res_lgb_mfq["brier_skill_score_test_prev"],
            "test_ece": res_lgb_mfq["expected_calibration_error"],
            "test_f1_at_op": res_lgb_mfq["f1_at_op"],
        })
        tracker.log_model(model_mfq)
        benchmark_results["run_id_MFQ"] = run_mfq.info.run_id

    # ---------------------------------------------------------
    # Matched-Sample Analysis on Identical Eligible Rows
    # Filter rows where both fundamental & qualitative data are valid
    # ---------------------------------------------------------
    matched_test_mask = test_df["altman_z_score"].notna() & test_df["rf_change_magnitude"].notna()
    test_matched = test_df[matched_test_mask]
    y_test_matched = test_matched["target_label"].values

    if len(test_matched) > 0:
        matched_results = {}
        for cfg_name, model_obj, f_list, op_t in [
            ("M_matched", model_m, market_features, t_lgb_m),
            ("MF_matched", model_mf, integrated_features, t_lgb_mf),
            ("MFQ_matched", model_mfq, mfq_features, t_lgb_mfq)
        ]:
            probs_sub = model_obj.predict(test_matched[f_list])
            sub_res = evaluate_predictions_authoritative(y_test_matched, probs_sub, op_t, train_prev)
            matched_results[cfg_name] = sub_res
        benchmark_results["matched_sample_analysis"] = {
            "matched_rows_count": len(test_matched),
            "matched_prevalence": round(float(y_test_matched.mean()), 4),
            "results": matched_results
        }

        
    return {
        "dataset_meta": {
            "fingerprint": dataset_fingerprint,
            "tickers": sorted(panel["ticker"].unique().tolist()),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_dates": (str(train_df["trade_date"].min()), str(train_df["trade_date"].max())),
            "val_dates": (str(val_df["trade_date"].min()), str(val_df["trade_date"].max())),
            "test_dates": (str(test_df["trade_date"].min()), str(test_df["trade_date"].max())),
            "fitted_threshold": fitted_thresh,
            "train_pos_rate": train_prev,
            "val_pos_rate": float(val_df["target_label"].mean()),
            "test_pos_rate": float(test_df["target_label"].mean()),
        },
        "results": benchmark_results
    }


def main():
    res = run_benchmark_suite()
    meta = res["dataset_meta"]
    results = res["results"]
    
    print("\n" + "=" * 80)
    print("      AQQIP EXPANDED DATASET BENCHMARK & EVALUATION INTEGRITY REPORT")
    print("=" * 80)
    print(f"Dataset Fingerprint:       {meta['fingerprint']}")
    print(f"Universe Tickers ({len(meta['tickers'])}):   {', '.join(meta['tickers'])}")
    print(f"Chronological Partitions:")
    print(f"  Train:      {meta['train_dates'][0]} to {meta['train_dates'][1]} | {meta['train_rows']:,} rows | Drop rate: {meta['train_pos_rate']*100:.2f}%")
    print(f"  Validation: {meta['val_dates'][0]} to {meta['val_dates'][1]} | {meta['val_rows']:,} rows | Drop rate: {meta['val_pos_rate']*100:.2f}%")
    print(f"  Holdout:    {meta['test_dates'][0]} to {meta['test_dates'][1]} | {meta['test_rows']:,} rows | Drop rate: {meta['test_pos_rate']*100:.2f}%")
    print(f"Fitted Target Threshold (Train alpha=0.10): {meta['fitted_threshold']:.4f}%")
    print(f"MLflow Run IDs: M={results.get('run_id_M')} | MF={results.get('run_id_MF')} | MFQ={results.get('run_id_MFQ')}")
    print("-" * 80)
    print(f"{'MODEL':<34} {'ROC-AUC':<8} {'PR-AUC':<8} {'Brier':<8} {'BSS(test)':<10} {'ECE':<8} {'Op Thresh':<10} {'F1(op)':<8} {'Prec(op)':<9} {'Rec(op)':<8}")
    print("-" * 80)
    
    for name, r in results.items():
        if name in ("run_id_M", "run_id_MF", "run_id_MFQ", "matched_sample_analysis") or not isinstance(r, dict) or "roc_auc" not in r:
            continue
        print(f"{name:<34} {r['roc_auc']:<8.4f} {r['pr_auc']:<8.4f} {r['brier_score']:<8.4f} {r['brier_skill_score_test_prev']:<+10.4f} {r['expected_calibration_error']:<8.4f} {r['operational_threshold']:<10.4f} {r['f1_at_op']:<8.4f} {r['precision_at_op']:<9.4f} {r['recall_at_op']:<8.4f}")
        cm = r["confusion_matrix_at_op"]
        print(f"  -> Confusion Matrix at p >= {r['operational_threshold']:.4f}: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,} (Pred Pos Rate: {r['predicted_pos_rate_at_op']*100:.2f}%)")

    matched_info = results.get("matched_sample_analysis")
    if matched_info:
        print("\n" + "=" * 80)
        print(f"CONTROLLED MATCHED-SAMPLE EVALUATION ({matched_info['matched_rows_count']:,} identical eligible holdout rows, Prev={matched_info['matched_prevalence']*100:.2f}%):")
        print("-" * 80)
        print(f"{'CONFIGURATION':<34} {'ROC-AUC':<8} {'PR-AUC':<8} {'Brier':<8} {'BSS(test)':<10} {'ECE':<8} {'F1(op)':<8} {'Prec(op)':<9} {'Rec(op)':<8}")
        print("-" * 80)
        for sub_name, sr in matched_info["results"].items():
            print(f"{sub_name:<34} {sr['roc_auc']:<8.4f} {sr['pr_auc']:<8.4f} {sr['brier_score']:<8.4f} {sr['brier_skill_score_test_prev']:<+10.4f} {sr['expected_calibration_error']:<8.4f} {sr['f1_at_op']:<8.4f} {sr['precision_at_op']:<9.4f} {sr['recall_at_op']:<8.4f}")
            cm = sr["confusion_matrix_at_op"]
            print(f"  -> CM: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

