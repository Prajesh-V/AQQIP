"""
AQQIP LightGBM Training & Evaluation Pipeline

Orchestrates the complete dataset build, point-in-time join, chronological
splitting, Optuna hyperparameter optimization, LightGBM training, holdout
evaluation, baseline comparison, and MLflow logging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm

from config.settings import get_settings
from src.database.mysql_client import get_mysql_connection
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.leakage_validator import LeakageValidator
from src.quant.lightgbm_trainer import LightGBMTrainer
from src.quant.baseline_models import BaselineModels
from src.utils.logging import get_logger

logger = get_logger("train_pipeline")

DEFAULT_MARKET_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "daily_range_pct",
    "avg_range_20d",
    "volume_z_score_20d",
    "drawdown_60d",
]

DEFAULT_FUNDAMENTAL_FEATURES = [
    "altman_z_score",
    "beneish_m_score",
]

DEFAULT_QUALITATIVE_FEATURES = [
    "rf_length_change_pct",
    "rf_cosine_similarity",
    "rf_change_magnitude",
    "rf_added_sentences_count",
    "rf_deleted_sentences_count",
    "rf_added_sentences_ratio",
    "mda_length_change_pct",
    "mda_cosine_similarity",
    "mda_change_magnitude",
]


def load_raw_data(include_qualitative: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load raw OHLCV prices, fundamental ratios, and qualitative filing features from MySQL."""
    logger.info("Loading raw data from MySQL", include_qualitative=include_qualitative)
    with get_mysql_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT ticker, trade_date, open, high, low, close, adj_close, volume
            FROM ohlcv_prices
            ORDER BY ticker, trade_date ASC
        """)
        market_rows = cursor.fetchall()
        
        cursor.execute("""
            SELECT ticker, filing_date, altman_z_score, beneish_m_score, feature_completeness
            FROM fundamental_ratios
            WHERE filing_date IS NOT NULL
            ORDER BY ticker, filing_date ASC
        """)
        fund_rows = cursor.fetchall()

        qual_rows = []
        if include_qualitative:
            cursor.execute("""
                SELECT ticker, filing_date, rf_char_count, rf_length_change_pct,
                       rf_jaccard_similarity, rf_cosine_similarity, rf_change_magnitude,
                       rf_added_sentences_count, rf_deleted_sentences_count,
                       rf_added_sentences_ratio, rf_deleted_sentences_ratio,
                       mda_char_count, mda_length_change_pct, mda_jaccard_similarity,
                       mda_cosine_similarity, mda_change_magnitude
                FROM qualitative_filing_features
                WHERE filing_date IS NOT NULL
                ORDER BY ticker, filing_date ASC
            """)
            qual_rows = cursor.fetchall()

    market_df = pd.DataFrame(market_rows)
    fund_df = pd.DataFrame(fund_rows)
    qual_df = pd.DataFrame(qual_rows) if include_qualitative else None
    
    # Cast decimals and numeric types
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in market_df.columns:
            market_df[col] = pd.to_numeric(market_df[col], errors="coerce")
            
    for col in ["altman_z_score", "beneish_m_score", "feature_completeness"]:
        if col in fund_df.columns:
            fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")

    if qual_df is not None and not qual_df.empty:
        for col in qual_df.columns:
            if col not in ("ticker", "filing_date"):
                qual_df[col] = pd.to_numeric(qual_df[col], errors="coerce")

    logger.info(
        "Loaded raw data",
        market_records=len(market_df),
        fundamental_records=len(fund_df),
        qualitative_records=len(qual_df) if qual_df is not None else 0
    )
    if include_qualitative:
        return market_df, fund_df, qual_df
    return market_df, fund_df



def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Computes a deterministic SHA-256 fingerprint of the labeled dataset."""
    summary_cols = ["ticker", "trade_date", "adj_close", "forward_return_pct", "target_label"]
    available_cols = [c for c in summary_cols if c in df.columns]
    data_str = df[available_cols].to_csv(index=False)
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()[:16]


def run_training_pipeline(
    val_start: str = "2024-01-01",
    test_start: str = "2025-01-01",
    alpha: float = 0.10,
    n_trials: int = 20,
    market_only: bool = False
) -> Dict[str, Any]:
    """Executes the end-to-end training and evaluation workflow."""
    logger.info(
        "Starting LightGBM training pipeline",
        val_start=val_start,
        test_start=test_start,
        alpha=alpha,
        n_trials=n_trials,
        market_only=market_only
    )

    # 1. Load data
    market_df, fund_df = load_raw_data()
    if market_df.empty:
        raise ValueError("No market data found in MySQL ohlcv_prices table. Run ingestion first.")

    # 2. Build leak-free panel
    dataset_builder = HistoricalDatasetBuilder()
    if market_only:
        logger.info("Training configured as market-only baseline")
        fund_df = pd.DataFrame(columns=["ticker", "filing_date", "altman_z_score", "beneish_m_score"])

    panel = dataset_builder.build_dataset(market_df, fund_df)
    
    # 3. Chronological split & train-fitted labeling (with temporal boundary purging)
    train_df, val_df, test_df = dataset_builder.split_and_label(
        panel,
        val_start=val_start,
        test_start=test_start,
        alpha=alpha,
        enforce_temporal_purging=True
    )

    # 4. Define features
    features = list(DEFAULT_MARKET_FEATURES)
    if not market_only:
        features.extend(DEFAULT_FUNDAMENTAL_FEATURES)

    # 5. Dataset metadata and fingerprint
    fingerprint = compute_dataset_fingerprint(train_df)
    dataset_meta = {
        "dataset_fingerprint": fingerprint,
        "split_val_start": val_start,
        "split_test_start": test_start,
        "enforce_temporal_purging": True,
        "prediction_horizon_trading_days": 20,
        "target_alpha": alpha,
        "target_threshold_fitted": float(train_df["target_threshold_used"].iloc[0]),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_pos_rate": float(train_df["target_label"].mean()),
        "val_pos_rate": float(val_df["target_label"].mean()),
        "test_pos_rate": float(test_df["target_label"].mean()),
        "features": ",".join(features),
        "feature_count": len(features),
        "model_type": "market_only" if market_only else "quant_integrated",
    }

    # 6. Train LightGBM with Optuna & log to MLflow
    trainer = LightGBMTrainer(features=features)
    
    X_train, y_train = train_df[features], train_df["target_label"]
    X_val, y_val = val_df[features], val_df["target_label"]
    X_test, y_test = test_df[features], test_df["target_label"]

    best_model, metrics = trainer.train_with_optuna(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=n_trials,
        X_test=X_test,
        y_test=y_test,
        dataset_meta=dataset_meta
    )

    run_id = metrics.get("run_id")

    # 7. Verify model artifact in MLflow
    model_uri = f"runs:/{run_id}/model"
    logger.info("Verifying saved model artifact", model_uri=model_uri)
    loaded_model = mlflow.lightgbm.load_model(model_uri)
    sample_preds = loaded_model.predict(X_test.iloc[:5])
    assert len(sample_preds) == 5, "Model artifact verification failed"
    logger.info("Successfully verified model artifact", sample_preds=sample_preds.tolist())

    return {
        "dataset_meta": dataset_meta,
        "features": features,
        "metrics": metrics,
        "run_id": run_id,
        "model_uri": model_uri,
        "tickers": market_df["ticker"].unique().tolist(),
        "total_observations": len(train_df) + len(val_df) + len(test_df),
        "date_ranges": {
            "train": (str(train_df["trade_date"].min()), str(train_df["trade_date"].max())),
            "val": (str(val_df["trade_date"].min()), str(val_df["trade_date"].max())),
            "test": (str(test_df["trade_date"].min()), str(test_df["trade_date"].max())),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="AQQIP LightGBM Training & Evaluation")
    parser.add_argument("--val-start", type=str, default="2024-01-01", help="Validation split start YYYY-MM-DD")
    parser.add_argument("--test-start", type=str, default="2025-01-01", help="Holdout test split start YYYY-MM-DD")
    parser.add_argument("--alpha", type=float, default=0.10, help="Target drop quantile threshold")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trial count")
    parser.add_argument("--market-only", action="store_true", help="Train market-only baseline")
    args = parser.parse_args()

    result = run_training_pipeline(
        val_start=args.val_start,
        test_start=args.test_start,
        alpha=args.alpha,
        n_trials=args.n_trials,
        market_only=args.market_only
    )

    meta = result["dataset_meta"]
    metrics = result["metrics"]
    dates = result["date_ranges"]

    print("\n" + "=" * 65)
    print("      AQQIP LIGHTGBM TRAINING & EVALUATION REPORT")
    print("=" * 65)
    print(f"MLflow Run ID:          {result['run_id']}")
    print(f"Artifact URI:           {result['model_uri']}")
    print(f"Dataset Fingerprint:    {meta['dataset_fingerprint']}")
    print(f"Model Configuration:    {meta['model_type'].upper()}")
    print(f"Active Features ({meta['feature_count']}):      {', '.join(result['features'])}")
    print(f"Universe Tickers ({len(result['tickers'])}):    {', '.join(result['tickers'])}")
    print("-" * 65)
    print("CHRONOLOGICAL PARTITIONS & CLASS BALANCE:")
    print(f"  Train:      {dates['train'][0]} to {dates['train'][1]} | {meta['train_rows']:,} rows | Drop rate: {meta['train_pos_rate']*100:.2f}%")
    print(f"  Validation: {dates['val'][0]} to {dates['val'][1]} | {meta['val_rows']:,} rows | Drop rate: {meta['val_pos_rate']*100:.2f}%")
    print(f"  Holdout:    {dates['test'][0]} to {dates['test'][1]} | {meta['test_rows']:,} rows | Drop rate: {meta['test_pos_rate']*100:.2f}%")
    print(f"  Fitted Drop Threshold (Train alpha={meta['target_alpha']}): {meta['target_threshold_fitted']:.4f}%")
    print("-" * 65)
    print("HOLDOUT TEST EVALUATION (Chronologically Separate Test Set):")
    brier_val = metrics.get('test_brier_score', metrics.get('test_brier', None))
    brier_str = f"{brier_val:.4f}" if brier_val is not None else "N/A"
    print(f"  LightGBM ROC-AUC:          {metrics.get('test_roc_auc', 0.0):.4f}")
    print(f"  LightGBM PR-AUC:           {metrics.get('test_pr_auc', 0.0):.4f}")
    print(f"  LightGBM Brier Score:      {brier_str}")
    print(f"  LightGBM Mean Pred Prob:   {metrics.get('test_positive_rate', 0.0)*100:.2f}% (true base rate: {meta['test_pos_rate']*100:.2f}%)")
    print(f"  Class Prevalence:          {metrics.get('test_class_prevalence', meta['test_pos_rate'])*100:.2f}%")
    
    op_thresh = metrics.get('operational_threshold', 0.50)
    print("-" * 65)
    print(f"OPERATIONAL THRESHOLD EVALUATION (Calibrated on Val, Frozen for Test):")
    print(f"  Selection Criterion:       {metrics.get('operational_threshold_criterion', 'max_f1_on_validation')}")
    print(f"  Validation Selected Thresh:{op_thresh:.4f}")
    print(f"  Val F1 at Threshold:       {metrics.get('val_f1_at_threshold', 0.0):.4f}")
    print(f"  Val Precision / Recall:    {metrics.get('val_precision_at_threshold', 0.0):.4f} / {metrics.get('val_recall_at_threshold', 0.0):.4f}")
    print(f"  Test Holdout Results at p >= {op_thresh:.4f}:")
    print(f"    Precision:               {metrics.get('test_precision_at_op_threshold', 0.0):.4f}")
    print(f"    Recall:                  {metrics.get('test_recall_at_op_threshold', 0.0):.4f}")
    print(f"    F1 Score:                {metrics.get('test_f1_at_op_threshold', 0.0):.4f}")
    print(f"    Predicted Positive Rate: {metrics.get('test_predicted_positive_rate_at_op_threshold', 0.0)*100:.2f}%")
    cm = metrics.get('test_confusion_matrix', [[int(metrics.get('test_tn',0)), int(metrics.get('test_fp',0))], [int(metrics.get('test_fn',0)), int(metrics.get('test_tp',0))]])
    print(f"    Confusion Matrix:        TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"  Standard 0.50 Threshold Reference (uncalibrated for 10% base rate):")
    print(f"    Precision:               {metrics.get('test_precision', 0.0):.4f}")
    print(f"    Recall:                  {metrics.get('test_recall', 0.0):.4f}")
    print(f"    F1 Score:                {metrics.get('test_f1', 0.0):.4f}")
    print("-" * 65)
    print("BASELINE COMPARISON ON IDENTICAL TEST SET:")
    print(f"  Majority Class ROC-AUC:    {metrics.get('baseline_majority_roc_auc', 0.5):.4f}")
    print(f"  Majority Class PR-AUC:     {metrics.get('baseline_majority_pr_auc', 0.0):.4f}")
    print(f"  Majority Class Brier:      {metrics.get('baseline_majority_brier', 0.0):.4f}")
    print(f"  Logistic Regr ROC-AUC:     {metrics.get('baseline_lr_roc_auc', 0.0):.4f}")
    print(f"  Logistic Regr PR-AUC:      {metrics.get('baseline_lr_pr_auc', 0.0):.4f}")
    print(f"  Logistic Regr Brier:       {metrics.get('baseline_lr_brier', 0.0):.4f}")
    print(f"  Logistic Regr F1 (p>={op_thresh:.2f}): {metrics.get('baseline_lr_f1_at_op_thresh', 0.0):.4f}")
    print("-" * 65)
    print("MODEL VERIFICATION:")
    print("  Model loaded back from MLflow artifact store: SUCCESS")
    print("  Inference sanity check:                       PASSED")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
