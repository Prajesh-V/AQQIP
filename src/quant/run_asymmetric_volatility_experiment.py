"""
AQQIP — Authorized Rolling-Origin Experiment Execution Runner (Specification v4.7)

Executes the preregistered four-fold rolling-origin experiment (2021–2024)
evaluating Candidates A, B, C, and D:
- Primary Inferential Hypothesis: Candidate C vs. Baseline A.
- Exploratory Descriptive Benchmarks: Candidates B and D.
- Strict isolation: Results, models, and predictions saved exclusively in an isolated directory.
- Read-only data access: No writes to MySQL, MongoDB, ChromaDB, or MLflow.
- Protected artifacts: Authoritative holdout predictions (2025–2026) untouched.
"""
from __future__ import annotations

import sys
import os
import json
import time
import math
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.database.mysql_client import get_mysql_connection
from src.quant.asymmetric_volatility_experiment import (
    CANDIDATE_A_FEATURES,
    CANDIDATE_B_FEATURES,
    CANDIDATE_C_FEATURES,
    CANDIDATE_D_FEATURES,
    CANDIDATE_SCHEMAS,
    BASE_LGBM_HYPERPARAMS,
    TRAINING_POLICY,
    FOLD_DEFINITIONS,
    build_candidate_panel,
    split_experiment_fold,
    assign_daily_alerts,
    compute_daily_alert_budget,
    deterministic_rank,
    run_moving_block_bootstrap,
    evaluate_decision_matrix,
    proportion_to_pp,
    pp_to_proportion,
    validate_gate_g1,
    validate_gate_g2,
    validate_gate_g3,
    validate_gate_g4,
    validate_gate_g5,
    validate_gate_g6,
    validate_gate_g7,
)

FROZEN_SPEC_PATH = r"C:\Users\prajesh\.gemini\antigravity-ide\brain\6cd62b1a-81e0-4795-a91a-7069a3fa5900\aqqip_volatility_spec_v4_7.md"
EXPECTED_SPEC_SHA256 = "eabd5e448d038dd26ce5168c4c1bb652d97980e508da53805710d5cfd0d69ec1"

PROTECTED_FILES = [
    r"C:\dev\AQQIP\eval_artifacts\authoritative_eval_predictions_2025_2026.parquet",
    r"C:\dev\AQQIP\eval_artifacts\authoritative_eval_predictions_2025_2026.csv",
    r"C:\dev\AQQIP\mlruns\125769975230458810\62d5c881d4bb4922b3d93ccaad9b5438\meta.yaml",
]


def compute_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_preflight_integrity() -> Dict[str, str]:
    """Verifies that the frozen specification and protected artifacts are intact."""
    print("=" * 80)
    print("STEP 1: PRE-FLIGHT INTEGRITY & SPECIFICATION VERIFICATION")
    print("=" * 80)

    # 1. Check specification v4.7
    if not os.path.exists(FROZEN_SPEC_PATH):
        raise FileNotFoundError(f"Specification v4.7 missing at {FROZEN_SPEC_PATH}")
    spec_hash = compute_sha256(FROZEN_SPEC_PATH)
    if spec_hash != EXPECTED_SPEC_SHA256:
        raise ValueError(f"Specification v4.7 hash mismatch! Expected {EXPECTED_SPEC_SHA256}, got {spec_hash}")
    print(f"[PASS] Specification v4.7 verified intact (SHA-256: {spec_hash[:16]}...)")

    # 2. Record initial hashes of protected files
    initial_hashes = {}
    for p in PROTECTED_FILES:
        if os.path.exists(p):
            h = compute_sha256(p)
            initial_hashes[p] = h
            print(f"[PASS] Protected artifact verified: {os.path.basename(p)} (SHA-256: {h[:16]}...)")
        else:
            raise FileNotFoundError(f"Protected artifact missing: {p}")

    return initial_hashes


def run_experiment(output_dir: Path) -> Dict[str, Any]:
    """Executes the complete four-fold rolling-origin experiment."""
    start_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()

    initial_protected_hashes = verify_preflight_integrity()

    # Create isolated output directory hierarchy
    models_dir = output_dir / "models"
    preds_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"

    for d in [models_dir, preds_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"STEP 2: ISOLATED OUTPUT DIRECTORY INITIALIZED: {output_dir}")
    print("=" * 80)

    # Load OHLCV price history from MySQL (Read-Only)
    print("\n[Step 2.1] Loading historical OHLCV price panel from MySQL (2014-01-02 to 2024-12-31)...")
    with get_mysql_connection() as conn:
        df_raw = pd.read_sql(
            """
            SELECT ticker, trade_date, open, high, low, close, volume, adj_close
            FROM ohlcv_prices
            WHERE trade_date >= '2014-01-02' AND trade_date <= '2024-12-31'
            ORDER BY ticker, trade_date ASC
            """,
            conn
        )
    print(f"Loaded {len(df_raw)} records across {df_raw['ticker'].nunique()} tickers and {df_raw['trade_date'].nunique()} trading sessions.")

    # Build canonical panel
    print("\n[Step 2.2] Constructing canonical candidate panel via build_candidate_panel()...")
    panel = build_candidate_panel(df_raw)
    print(f"Constructed panel with {len(panel)} rows and {len(panel.columns)} columns.")

    # Validate Gate G1 through G7
    print("\n[Step 2.3] Validating hard pre-execution gates G1–G7...")
    eval_sessions = []
    for f in FOLD_DEFINITIONS:
        s_start, s_end = f["eval_retained"]
        eval_sessions.extend(range(s_start, s_end + 1))
    eval_panel_check = panel[panel["session"].isin(eval_sessions)].copy()

    g1_ok, g1_msg = validate_gate_g1(eval_panel_check)
    g2_ok, g2_msg = validate_gate_g2(eval_panel_check)
    g3_ok, g3_msg = validate_gate_g3(eval_panel_check)
    g4_ok, g4_msg = validate_gate_g4(eval_panel_check)
    g5_ok, g5_msg = validate_gate_g5(panel)
    g6_ok, g6_msg = validate_gate_g6(panel)
    g7_ok, g7_msg = validate_gate_g7(panel)

    for g_id, ok, msg in [
        ("G1", g1_ok, g1_msg), ("G2", g2_ok, g2_msg), ("G3", g3_ok, g3_msg),
        ("G4", g4_ok, g4_msg), ("G5", g5_ok, g5_msg), ("G6", g6_ok, g6_msg),
        ("G7", g7_ok, g7_msg)
    ]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {g_id}: {msg}")
        if not ok:
            raise RuntimeError(f"Gate {g_id} failed: {msg}")

    print("\n" + "=" * 80)
    print("STEP 3: EXECUTING ROLLING-ORIGIN TRAINING AND OUT-OF-SAMPLE EVALUATION")
    print("=" * 80)

    candidates = ["A", "B", "C", "D"]
    fold_eval_dfs: List[pd.DataFrame] = []
    fold_records: List[Dict[str, Any]] = []
    selected_rounds_record: Dict[str, Dict[int, int]] = {c: {} for c in candidates}

    # Iterate over Folds 1 to 4
    for fold_def in FOLD_DEFINITIONS:
        k = fold_def["fold"]
        year = fold_def["eval_year"]
        print(f"\n>>> EXECUTING FOLD {k} (Evaluation Year: {year}) <<<")

        # Split fold with exact session geometry and independent threshold calibration
        split_data = split_experiment_fold(panel, fold_num=k, alpha=0.10)
        tau_inner = split_data["tau_inner"]
        tau_outer = split_data["tau_outer"]
        print(f"  Thresholds: tau_inner = {tau_inner:.4f}%, tau_outer = {tau_outer:.4f}%")
        print(f"  Partitions: Inner-Train={len(split_data['inner_train'])}, Inner-Val={len(split_data['inner_val'])}, Outer-Refit={len(split_data['outer_refit'])}, Retained-Eval={len(split_data['outer_eval'])}")

        fold_eval_df = split_data["outer_eval"].copy()

        # Train and evaluate all 4 candidates
        for cand_name in candidates:
            features = list(CANDIDATE_SCHEMAS[cand_name])
            print(f"  --- Training Candidate_{cand_name} ({len(features)} features) ---")

            # Stage 1: Inner Train with early stopping on Inner Val
            X_in_tr = split_data["inner_train"][features]
            y_in_tr = split_data["inner_train"]["target_label"].values
            X_in_val = split_data["inner_val"][features]
            y_in_val = split_data["inner_val"]["target_label"].values

            dtrain = lgb.Dataset(X_in_tr, label=y_in_tr)
            dval = lgb.Dataset(X_in_val, label=y_in_val, reference=dtrain)

            params = dict(BASE_LGBM_HYPERPARAMS)
            params["verbosity"] = -1

            # Early stopping with max 300 rounds, patience 25, metric binary_logloss
            bst_inner = lgb.train(
                params,
                dtrain,
                num_boost_round=TRAINING_POLICY["max_boosting_rounds"],
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=TRAINING_POLICY["early_stopping_patience"], verbose=False)],
            )

            b_star = int(bst_inner.best_iteration)
            selected_rounds_record[cand_name][k] = b_star
            val_loss = float(bst_inner.best_score["valid_0"]["binary_logloss"])
            print(f"    Stage 1: Selected B* = {b_star} rounds (Inner-Val Logloss: {val_loss:.6f})")

            # Stage 2: Outer Refit on independently reconstructed outer_refit partition
            X_refit = split_data["outer_refit"][features]
            y_refit = split_data["outer_refit"]["target_label"].values
            drefit = lgb.Dataset(X_refit, label=y_refit)

            # Train for exactly B* rounds (NO early stopping)
            bst_refit = lgb.train(
                params,
                drefit,
                num_boost_round=b_star,
            )

            # Save model booster
            model_path = models_dir / f"Candidate_{cand_name}_fold_{k}.txt"
            bst_refit.save_model(str(model_path))

            # Generate out-of-fold predictions on retained evaluation partition
            X_eval = fold_eval_df[features]
            preds = bst_refit.predict(X_eval)
            fold_eval_df[f"prob_{cand_name}"] = preds

        # Apply deterministic daily ranking rule and alert budget per candidate
        for cand_name in candidates:
            prob_col = f"prob_{cand_name}"
            alert_col = f"alert_{cand_name}"

            # Apply deterministic ranking and top-5% budget (k_t=2 for N=32)
            fold_eval_df = assign_daily_alerts(
                fold_eval_df,
                prob_col=prob_col,
                ticker_col="ticker",
                date_col="trade_date",
                alert_col=alert_col,
                fraction=0.05,
            )

            # Compute fold confusion matrix counts
            y_true = fold_eval_df["target_label"].values
            alerts = fold_eval_df[alert_col].values

            tp = int(np.sum((alerts == 1) & (y_true == 1)))
            fp = int(np.sum((alerts == 1) & (y_true == 0)))
            tn = int(np.sum((alerts == 0) & (y_true == 0)))
            fn = int(np.sum((alerts == 0) & (y_true == 1)))

            total_alerts = tp + fp
            denom_rec = tp + fn
            fp_share = (fp / total_alerts) if total_alerts > 0 else 0.0
            recall = (tp / denom_rec) if denom_rec > 0 else 0.0
            precision = (tp / total_alerts) if total_alerts > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            fold_records.append({
                "fold": k,
                "eval_year": year,
                "candidate": f"Candidate_{cand_name}",
                "role": "Primary" if cand_name in ["A", "C"] else "Exploratory",
                "features_count": len(CANDIDATE_SCHEMAS[cand_name]),
                "selected_rounds": selected_rounds_record[cand_name][k],
                "tau_inner": tau_inner,
                "tau_outer": tau_outer,
                "eval_rows": len(fold_eval_df),
                "eval_dates": fold_eval_df["trade_date"].nunique(),
                "total_alerts": total_alerts,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "fp_share_prop": fp_share,
                "fp_share_pct": fp_share * 100.0,
                "recall_prop": recall,
                "recall_pct": recall * 100.0,
                "precision_pct": precision * 100.0,
                "f1_score": f1,
            })

            print(f"    Alert Metrics (Candidate_{cand_name}): Alerts={total_alerts}, TP={tp}, FP={fp}, FP_Share={fp_share*100:.2f}%, Recall={recall*100:.2f}%")

        fold_eval_dfs.append(fold_eval_df)

    # Save fold-level metrics table
    df_fold_metrics = pd.DataFrame(fold_records)
    df_fold_metrics.to_csv(metrics_dir / "fold_level_experiment_results.csv", index=False)
    with open(metrics_dir / "fold_level_experiment_results.json", "w") as f:
        json.dump(fold_records, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 4: POOLED EVALUATION & ALERT ATTRIBUTION TAXONOMY")
    print("=" * 80)

    # Concatenate all evaluation partitions
    pooled_eval_df = pd.concat(fold_eval_dfs, ignore_index=True)
    if len(pooled_eval_df) != 29600:
        raise ValueError(f"Expected 29,600 pooled evaluation rows, got {len(pooled_eval_df)}")

    # Save complete out-of-fold prediction panel
    pooled_eval_df.to_parquet(preds_dir / "eval_predictions_2021_2024.parquet", index=False)
    pooled_eval_df.to_csv(preds_dir / "eval_predictions_2021_2024.csv", index=False)
    print(f"Saved complete out-of-fold predictions ({len(pooled_eval_df)} rows) to {preds_dir}")

    # Compute pooled metrics for all candidates
    pooled_records = []
    y_all = pooled_eval_df["target_label"].values
    n_pos_all = int(np.sum(y_all == 1))
    print(f"Total hazard positives across 29,600 observations: {n_pos_all} ({n_pos_all/len(y_all)*100:.2f}% prevalence)")

    for cand_name in candidates:
        alert_col = f"alert_{cand_name}"
        alerts = pooled_eval_df[alert_col].values

        tp = int(np.sum((alerts == 1) & (y_all == 1)))
        fp = int(np.sum((alerts == 1) & (y_all == 0)))
        tn = int(np.sum((alerts == 0) & (y_all == 0)))
        fn = int(np.sum((alerts == 0) & (y_all == 1)))

        total_alerts = tp + fp
        fp_share = fp / total_alerts if total_alerts > 0 else 0.0
        recall = tp / n_pos_all if n_pos_all > 0 else 0.0
        precision = tp / total_alerts if total_alerts > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        pooled_records.append({
            "candidate": f"Candidate_{cand_name}",
            "role": "Primary (Formal)" if cand_name in ["A", "C"] else "Exploratory (Informational Only)",
            "features_count": len(CANDIDATE_SCHEMAS[cand_name]),
            "mean_selected_rounds": float(np.mean(list(selected_rounds_record[cand_name].values()))),
            "eval_observations": len(pooled_eval_df),
            "total_alerts": total_alerts,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "fp_share_prop": fp_share,
            "fp_share_pp": fp_share * 100.0,
            "recall_prop": recall,
            "recall_pp": recall * 100.0,
            "precision_pp": precision * 100.0,
            "f1_score": f1,
        })

    df_pooled_metrics = pd.DataFrame(pooled_records)
    df_pooled_metrics.to_csv(metrics_dir / "pooled_candidate_comparison.csv", index=False)
    with open(metrics_dir / "pooled_candidate_comparison.json", "w") as f:
        json.dump(pooled_records, f, indent=2)

    # Compute Section 9 Alert Transition Taxonomy between Candidate A and Candidate C
    a_A = pooled_eval_df["alert_A"].values
    a_C = pooled_eval_df["alert_C"].values
    fwd_ret = pooled_eval_df["forward_return_pct"].values

    tp_retained = int(np.sum((a_A == 1) & (a_C == 1) & (y_all == 1)))
    fp_retained = int(np.sum((a_A == 1) & (a_C == 1) & (y_all == 0)))
    d_fp_mask = (a_A == 1) & (a_C == 0) & (y_all == 0)
    d_tp_mask = (a_A == 1) & (a_C == 0) & (y_all == 1)
    i_tp_mask = (a_A == 0) & (a_C == 1) & (y_all == 1)
    i_fp_mask = (a_A == 0) & (a_C == 1) & (y_all == 0)
    tn_retained = int(np.sum((a_A == 0) & (a_C == 0) & (y_all == 0)))
    fn_retained = int(np.sum((a_A == 0) & (a_C == 0) & (y_all == 1)))

    d_fp = int(np.sum(d_fp_mask))
    d_tp = int(np.sum(d_tp_mask))
    i_tp = int(np.sum(i_tp_mask))
    i_fp = int(np.sum(i_fp_mask))

    # Conservation identity check
    conservation_lhs = d_fp + d_tp
    conservation_rhs = i_fp + i_tp
    assert conservation_lhs == conservation_rhs, f"Alert conservation violated: {conservation_lhs} != {conservation_rhs}"

    # Mechanistic attribution: fraction of displaced false positives with forward_return_pct > 0.0%
    d_fp_upside_count = int(np.sum(d_fp_mask & (fwd_ret > 0.0)))
    rho_upside_prop = (d_fp_upside_count / d_fp) if d_fp > 0 else 0.0
    rho_upside_pct = rho_upside_prop * 100.0

    attribution_results = {
        "TP_retained": tp_retained,
        "FP_retained": fp_retained,
        "D_FP": d_fp,
        "D_TP": d_tp,
        "I_TP": i_tp,
        "I_FP": i_fp,
        "TN_retained": tn_retained,
        "FN_retained": fn_retained,
        "conservation_check": f"|D_FP| + |D_TP| = {conservation_lhs} == |I_FP| + |I_TP| = {conservation_rhs}",
        "D_FP_upside_count": d_fp_upside_count,
        "rho_upside_pct": rho_upside_pct,
        "rho_upside_prop": rho_upside_prop,
        "diagnostic_pass": (rho_upside_pct > 60.0) and (d_fp > 0),
    }

    with open(metrics_dir / "alert_attribution_taxonomy.json", "w") as f:
        json.dump(attribution_results, f, indent=2)

    print(f"Alert Transitions: D_FP={d_fp}, D_TP={d_tp}, I_TP={i_tp}, I_FP={i_fp}")
    print(f"Mechanistic Attribution: {d_fp_upside_count}/{d_fp} displaced false positives had positive forward returns (rho_upside = {rho_upside_pct:.2f}%)")

    print("\n" + "=" * 80)
    print("STEP 5: PREREGISTERED BOOTSTRAP INFERENCE (C vs. A)")
    print("=" * 80)

    # Point estimates on pooled evaluation panel
    cand_A_record = [r for r in pooled_records if r["candidate"] == "Candidate_A"][0]
    cand_C_record = [r for r in pooled_records if r["candidate"] == "Candidate_C"][0]

    point_delta_fp_share_prop = cand_C_record["fp_share_prop"] - cand_A_record["fp_share_prop"]
    point_delta_fp_share_pp = point_delta_fp_share_prop * 100.0

    point_delta_recall_prop = cand_C_record["recall_prop"] - cand_A_record["recall_prop"]
    point_delta_recall_pp = point_delta_recall_prop * 100.0

    print(f"Point Estimates (Candidate C minus Baseline A):")
    print(f"  Delta FP Share: {point_delta_fp_share_pp:+.4f} pp ({point_delta_fp_share_prop:+.6f} prop)")
    print(f"  Delta Recall:   {point_delta_recall_pp:+.4f} pp ({point_delta_recall_prop:+.6f} prop)")

    # Execute canonical moving-block bootstrap
    print(f"\nRunning 20-session within-fold moving-block bootstrap (B=1000, L=20, Seed=20240101)...")
    bootstrap_res = run_moving_block_bootstrap(
        fold_eval_dfs=fold_eval_dfs,
        block_length=20,
        n_replicates=1000,
        seed=20240101,
        max_retries=10000,
    )

    ci_fp_pp = bootstrap_res["ci_fp_share_pp"]
    ci_fp_prop = bootstrap_res["ci_fp_share_prop"]
    lb_rec_pp = bootstrap_res["lb_recall_pp"]
    lb_rec_prop = bootstrap_res["lb_recall_prop"]

    print(f"Bootstrap Replicate Diagnostics:")
    print(f"  Accepted: {bootstrap_res['n_accepted']}, Rejected: {bootstrap_res['n_rejected']}, Attempted: {bootstrap_res['n_attempted']}")
    print(f"  Rejection breakdown: {bootstrap_res['rejection_breakdown']}")
    print(f"Inferential Bounds:")
    print(f"  95% CI for Delta FP Share: [{ci_fp_pp[0]:+.4f} pp, {ci_fp_pp[1]:+.4f} pp] (prop: [{ci_fp_prop[0]:+.6f}, {ci_fp_prop[1]:+.6f}])")
    print(f"  5th Percentile LB for Delta Recall: {lb_rec_pp:+.4f} pp (prop: {lb_rec_prop:+.6f})")

    bootstrap_summary = {
        "point_delta_fp_share_pp": point_delta_fp_share_pp,
        "point_delta_fp_share_prop": point_delta_fp_share_prop,
        "point_delta_recall_pp": point_delta_recall_pp,
        "point_delta_recall_prop": point_delta_recall_prop,
        "ci_fp_share_95_pp": ci_fp_pp,
        "ci_fp_share_95_prop": ci_fp_prop,
        "lb_recall_5th_percentile_pp": lb_rec_pp,
        "lb_recall_5th_percentile_prop": lb_rec_prop,
        "n_accepted": bootstrap_res["n_accepted"],
        "n_rejected": bootstrap_res["n_rejected"],
        "n_attempted": bootstrap_res["n_attempted"],
        "rejection_breakdown": bootstrap_res["rejection_breakdown"],
        "block_length": 20,
        "n_replicates": 1000,
        "random_seed": 20240101,
    }

    with open(metrics_dir / "formal_bootstrap_inference_results.json", "w") as f:
        json.dump(bootstrap_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 6: PREREGISTERED DECISION MATRIX EVALUATION")
    print("=" * 80)

    decision_res = evaluate_decision_matrix(
        delta_fp_share_pp=point_delta_fp_share_pp,
        ci_fp_upper_pp=ci_fp_pp[1],
        lb_recall_pp=lb_rec_pp,
        rho_upside=rho_upside_pct,
        d_fp_count=d_fp,
    )

    with open(metrics_dir / "decision_matrix_evaluation.json", "w") as f:
        json.dump(decision_res, f, indent=2)

    print(f"Outcome ID:      {decision_res['outcome_id']}")
    print(f"Tier:            {decision_res['tier']}")
    print(f"Recall NI:       {decision_res['recall_ni']} (LB = {lb_rec_pp:+.4f} pp >= -1.0 pp)")
    print(f"CI Criterion:    {decision_res['ci_criterion']} (Upper = {ci_fp_pp[1]:+.4f} pp < 0.0 pp)")
    print(f"Mechanism:       {decision_res['mechanism']} (rho = {rho_upside_pct:.2f}% > 60.0%)")
    print(f"Status:          {decision_res['status']}")
    print(f"Interpretation:  {decision_res['interpretation']}")

    print("\n" + "=" * 80)
    print("STEP 7: POST-EXECUTION INTEGRITY VERIFICATION")
    print("=" * 80)

    # Check protected files
    for p, initial_h in initial_protected_hashes.items():
        current_h = compute_sha256(p)
        if current_h != initial_h:
            raise RuntimeError(f"PROTECTED ARTIFACT MUTATED! {p} changed from {initial_h} to {current_h}")
        print(f"[VERIFIED UNTOUCHED] {os.path.basename(p)} (SHA-256: {current_h[:16]}...)")

    # Check specification
    final_spec_hash = compute_sha256(FROZEN_SPEC_PATH)
    if final_spec_hash != EXPECTED_SPEC_SHA256:
        raise RuntimeError("FROZEN SPECIFICATION MUTATED!")
    print(f"[VERIFIED UNTOUCHED] Specification v4.7 (SHA-256: {final_spec_hash[:16]}...)")

    elapsed_sec = time.time() - t0
    end_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    execution_manifest = {
        "execution_status": "COMPLETED_SUCCESS",
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "elapsed_seconds": elapsed_sec,
        "spec_hash": final_spec_hash,
        "output_directory": str(output_dir),
        "decision_outcome": decision_res,
        "pooled_metrics": pooled_records,
        "bootstrap_summary": bootstrap_summary,
        "attribution_results": attribution_results,
        "selected_rounds": selected_rounds_record,
    }

    with open(output_dir / "execution_summary.json", "w") as f:
        json.dump(execution_manifest, f, indent=2)

    print(f"\nExperiment execution successfully completed in {elapsed_sec:.2f} seconds.")
    return execution_manifest


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "experiments" / f"asymmetric_volatility_phase2_{timestamp}"
    run_experiment(out_dir)
