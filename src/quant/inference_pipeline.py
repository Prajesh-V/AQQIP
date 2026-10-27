"""
AQQIP Real MLOps Inference Pipeline

Loads a verified LightGBM model artifact from MLflow, constructs features
from MySQL, validates feature schema against the logged artifact, generates
real probabilities, and persists prediction records to MySQL.

Distinguishes between retrospective evaluation and prospective predictions.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm

from config.settings import get_settings
from src.database.mysql_client import get_mysql_connection
from src.mlops.models import PredictionRecord
from src.mlops.repository import MLOpsRepository
from src.mlops.resolver import MLOpsResolver, MySQLMarketDataStore
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.predictor import Predictor, compute_feature_hash
from src.quant.train_pipeline import load_raw_data
from src.utils.logging import get_logger

logger = get_logger("inference_pipeline")


def load_mlflow_model(model_uri: str) -> Tuple[Any, List[str], str]:
    """
    Loads LightGBM booster from MLflow artifact repository and extracts feature schema.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    
    logger.info("Loading LightGBM model from MLflow", model_uri=model_uri)
    model = mlflow.lightgbm.load_model(model_uri)
    
    # Extract run_id from model_uri: "runs:/<run_id>/model"
    if model_uri.startswith("runs:/"):
        parts = model_uri.replace("runs:/", "").split("/")
        run_id = parts[0]
    else:
        run_id = "external_run"
        
    # Extract ordered feature names expected by the model
    expected_features = model.feature_name()
    logger.info(
        "Successfully loaded MLflow model",
        run_id=run_id,
        feature_count=len(expected_features),
        features=expected_features
    )
    return model, expected_features, run_id


def build_inference_panel(
    as_of_date: Optional[str] = None,
    tickers: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw market and fundamental data from MySQL and builds leak-free panel.
    Returns (full_panel, inference_slice).
    """
    market_df, fund_df = load_raw_data()
    if market_df.empty:
        raise ValueError("No market data found in MySQL ohlcv_prices table.")

    if tickers:
        market_df = market_df[market_df["ticker"].isin(tickers)].copy()
        fund_df = fund_df[fund_df["ticker"].isin(tickers)].copy()

    builder = HistoricalDatasetBuilder()
    panel = builder.build_dataset(market_df, fund_df)
    
    # Ensure trade_date is date object for comparison
    panel["trade_date_obj"] = pd.to_datetime(panel["trade_date"]).dt.date

    if as_of_date:
        target_dt = pd.to_datetime(as_of_date).date()
        slice_df = panel[panel["trade_date_obj"] == target_dt].copy()
    else:
        # Latest available trading day for each ticker
        idx_latest = panel.groupby("ticker")["trade_date_obj"].idxmax()
        slice_df = panel.loc[idx_latest].copy()

    return panel, slice_df


def run_inference(
    model_uri: str = "runs:/df161d0bc48349debc7cdb47810111d7/model",
    as_of_date: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    mode: str = "prospective",
    resolve: bool = False
) -> Dict[str, Any]:
    """
    Executes inference using saved model and real MySQL data.
    """
    # 1. Load model and feature schema
    model, expected_features, run_id = load_mlflow_model(model_uri)

    # 2. Build feature panel
    panel, slice_df = build_inference_panel(as_of_date=as_of_date, tickers=tickers)
    if slice_df.empty:
        raise ValueError(f"No feature records available for as_of_date={as_of_date}")

    # 3. Determine target dates and prospective vs retrospective classification
    target_dates = {}
    is_prospective = {}
    
    # Query all observed trading dates in MySQL to map exact 20-trading-day forward horizons
    market_dates = sorted(panel["trade_date_obj"].unique())
    market_date_set = set(market_dates)

    for _, row in slice_df.iterrows():
        t = row["ticker"]
        t_date = row["trade_date_obj"]
        
        # Find position in trading calendar
        if t_date in market_date_set:
            pos = market_dates.index(t_date)
            if pos + 20 < len(market_dates):
                # 20 trading days forward exists in historical data (retrospective)
                target_dates[t] = market_dates[pos + 20]
                is_prospective[t] = False
            else:
                # 20 trading days forward is in the future beyond available data (prospective)
                # Project forward by approximately 28 calendar days (~20 trading days)
                target_dates[t] = t_date + timedelta(days=28)
                is_prospective[t] = True
        else:
            target_dates[t] = t_date + timedelta(days=28)
            is_prospective[t] = True

    # 4. Initialize Predictor with exact features
    predictor = Predictor(
        model=model,
        features=expected_features,
        model_run_id=run_id,
        model_version="quant_integrated_v1",
        drop_threshold_pct=10.0,
        prediction_horizon_days=20
    )

    # 5. Generate predictions and persist to MySQL
    records = predictor.predict_panel(slice_df, target_dates=target_dates)
    repo = MLOpsRepository()
    saved_ids = []
    for r in records:
        pid = repo.create_prediction(r)
        saved_ids.append(pid)

    logger.info("Persisted predictions to MySQL", count=len(saved_ids), ids=saved_ids)

    # 6. Optional resolution if requested and data exists
    resolved_records = []
    if resolve:
        resolver = MLOpsResolver(market_store=MySQLMarketDataStore())
        resolved_records = resolver.resolve_predictions(records, current_date=date.today())
        for r in resolved_records:
            repo.update_resolved(r)
        logger.info("Resolved predictions against MySQL OHLCV", count=len(resolved_records))

    return {
        "model_uri": model_uri,
        "run_id": run_id,
        "mode": mode,
        "expected_features": expected_features,
        "prediction_count": len(records),
        "prediction_ids": saved_ids,
        "records": records,
        "resolved_records": resolved_records,
        "is_prospective": is_prospective,
    }


def main():
    parser = argparse.ArgumentParser(description="AQQIP Real MLOps Inference Entry Point")
    parser.add_argument(
        "--model-uri",
        type=str,
        default="runs:/df161d0bc48349debc7cdb47810111d7/model",
        help="MLflow model URI (e.g., runs:/df161d0bc48349debc7cdb47810111d7/model)"
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Inference trade date YYYY-MM-DD (defaults to latest available in MySQL)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (defaults to all universe tickers)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prospective", "retrospective"],
        default="prospective",
        help="Inference mode: 'prospective' (open outcome) or 'retrospective' (historical holdout)"
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Attempt immediate resolution against MySQL ohlcv_prices"
    )
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    result = run_inference(
        model_uri=args.model_uri,
        as_of_date=args.as_of_date,
        tickers=ticker_list,
        mode=args.mode,
        resolve=args.resolve
    )

    print("\n" + "=" * 70)
    print("         AQQIP REAL MLOPS INFERENCE EXECUTION REPORT")
    print("=" * 70)
    print(f"Model URI:            {result['model_uri']}")
    print(f"MLflow Run ID:        {result['run_id']}")
    print(f"Mode:                 {result['mode'].upper()}")
    print(f"Features ({len(result['expected_features'])}):        {', '.join(result['expected_features'])}")
    print(f"Generated Records:    {result['prediction_count']}")
    print("-" * 70)
    print("PREDICTION RECORDS GENERATED & PERSISTED TO MYSQL:")
    print(f"{'ID':<6} {'Ticker':<8} {'Trade Date':<12} {'Target Date':<12} {'Drop Prob':<11} {'Status':<10} {'Classification'}")
    print("-" * 70)
    
    for r in result["records"]:
        is_prosp = result["is_prospective"].get(r.ticker, True)
        cls_str = "PROSPECTIVE (Open)" if is_prosp else "RETROSPECTIVE"
        t_date_str = str(r.predicted_at.date() if isinstance(r.predicted_at, datetime) else r.predicted_at)
        print(f"{str(r.prediction_id):<6} {r.ticker:<8} {t_date_str:<12} {str(r.target_date):<12} {r.predicted_drop_probability:<11.4f} {r.status:<10} {cls_str}")
        print(f"       Feature Hash: {r.input_feature_hash}")
        if r.status == "resolved":
            print(f"       Resolved Price Change: {r.actual_price_change_pct:+.4f}% | Drop Observed: {r.actual_drop_observed}")

    print("-" * 70)
    if result["resolved_records"]:
        print(f"Resolved {len(result['resolved_records'])} records immediately using real MySQL market data.")
    else:
        print("Note: Prospective predictions remain OPEN until real market data for target_date exists in MySQL.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
