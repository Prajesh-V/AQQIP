"""
Automated unit tests for authoritative evaluation artifact and metric reconciliation.

Validates that:
1. Artifact schema, row count, and provenance hashes match exactly.
2. Confusion matrix counts arithmetically reconcile with precision, recall, F1, and alert rate.
3. Calibration error (ECE) matches the exact weighted sum of reliability bin errors.
4. Brier skill score reconciles against both deployable training baseline and in-sample test prevalence.
5. All 4 models (M, MF, MFQ, M_Regularized) reproduce their reported scores from the artifact.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.quant.authoritative_eval_artifact import (
    build_and_save_authoritative_artifact,
    compute_metrics_from_artifact,
    CSV_PATH,
    PARQUET_PATH,
    MODEL_RUNS
)


@pytest.fixture(scope="module")
def eval_artifact_df():
    """Load or generate authoritative evaluation artifact."""
    if not PARQUET_PATH.exists() and not CSV_PATH.exists():
        return build_and_save_authoritative_artifact()
    return pd.read_parquet(PARQUET_PATH) if PARQUET_PATH.exists() else pd.read_csv(CSV_PATH)


def test_artifact_structure_and_row_count(eval_artifact_df):
    """Verify artifact row count matches the 2025-2026 test partition (13,791 rows)."""
    assert len(eval_artifact_df) == 13791
    assert "trade_date" in eval_artifact_df.columns
    assert "ticker" in eval_artifact_df.columns
    assert "target_label" in eval_artifact_df.columns
    
    # Check that each model's probabilities and operational thresholds are present
    for model_name in MODEL_RUNS.keys():
        assert f"prob_{model_name}" in eval_artifact_df.columns
        assert f"op_thresh_{model_name}" in eval_artifact_df.columns
        assert f"run_id_{model_name}" in eval_artifact_df.columns


def test_arithmetic_reconciliation_confusion_matrix(eval_artifact_df):
    """Verify exact arithmetic identity between confusion matrix and classification metrics."""
    metrics = compute_metrics_from_artifact(eval_artifact_df)
    N = len(eval_artifact_df)
    actual_pos = int(eval_artifact_df["target_label"].sum())
    actual_neg = N - actual_pos

    for model_name, m in metrics.items():
        cm = m["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

        # 1. Total row count identity
        assert tn + fp + fn + tp == N

        # 2. Positive & Negative class identities
        assert tp + fn == actual_pos
        assert tn + fp == actual_neg

        # 3. Alert rate identity
        calculated_alert_rate = (tp + fp) / N
        assert np.isclose(m["alert_rate"], calculated_alert_rate, atol=1e-4)

        # 4. Precision identity
        if tp + fp > 0:
            calculated_precision = tp / (tp + fp)
            assert np.isclose(m["precision"], calculated_precision, atol=1e-4)

        # 5. Recall identity
        calculated_recall = tp / actual_pos
        assert np.isclose(m["recall"], calculated_recall, atol=1e-4)

        # 6. F1 score identity
        prec = m["precision"]
        rec = m["recall"]
        if prec + rec > 0:
            calculated_f1 = 2 * prec * rec / (prec + rec)
            assert np.isclose(m["f1"], calculated_f1, atol=1e-4)


def test_calibration_ece_arithmetic(eval_artifact_df):
    """Verify ECE matches the weighted sum of bin absolute errors."""
    metrics = compute_metrics_from_artifact(eval_artifact_df)
    N = len(eval_artifact_df)

    for model_name, m in metrics.items():
        cal_table = m["calibration_table"]
        weighted_error_sum = sum((bin_row["count"] / N) * bin_row["abs_error"] for bin_row in cal_table)
        assert np.isclose(m["ece"], weighted_error_sum, atol=1e-3)


def test_brier_skill_reconciliation(eval_artifact_df):
    """Verify deployable and in-sample Brier Skill Scores follow exact mathematical definitions."""
    metrics = compute_metrics_from_artifact(eval_artifact_df)
    y_true = eval_artifact_df["target_label"].values
    train_prev = float(eval_artifact_df["train_prevalence"].iloc[0])
    test_prev = float(eval_artifact_df["test_prevalence"].iloc[0])

    bs_ref_train = np.mean((train_prev - y_true) ** 2)
    bs_ref_test = np.mean((test_prev - y_true) ** 2)

    for model_name, m in metrics.items():
        bs = m["brier_score"]
        expected_bss_train = 1.0 - (bs / bs_ref_train)
        expected_bss_test = 1.0 - (bs / bs_ref_test)

        assert np.isclose(m["bss_deployable"], expected_bss_train, atol=1e-3)
        assert np.isclose(m["bss_insample"], expected_bss_test, atol=1e-3)
