"""
End-to-end test suite for MLOps Prediction Lifecycle:
Model Load -> Feature Validation -> Prediction Persistence -> Actual-Outcome Resolution.
"""
import pytest
from datetime import date, datetime
import pandas as pd

from src.mlops.repository import MLOpsRepository
from src.mlops.resolver import MLOpsResolver, MySQLMarketDataStore
from src.quant.inference_pipeline import load_mlflow_model, build_inference_panel, run_inference


def test_model_load_and_feature_schema():
    """Verify loading the persisted LightGBM artifact and validating its ordered schema."""
    model_uri = "runs:/df161d0bc48349debc7cdb47810111d7/model"
    model, features, run_id = load_mlflow_model(model_uri)
    
    assert model is not None
    assert run_id == "df161d0bc48349debc7cdb47810111d7"
    assert len(features) == 11
    assert features == [
        "return_1d", "return_5d", "return_20d", "return_60d",
        "volatility_20d", "daily_range_pct", "avg_range_20d",
        "volume_z_score_20d", "drawdown_60d", "altman_z_score",
        "beneish_m_score"
    ]


def test_end_to_end_retrospective_prediction_lifecycle():
    """
    Test complete lifecycle on a historical holdout date:
    1. Feature construction from real MySQL data.
    2. Prediction generation via MLflow model.
    3. Persistence to MySQL predictions table.
    4. Successful resolution against observed MySQL OHLCV prices.
    """
    result = run_inference(
        model_uri="runs:/df161d0bc48349debc7cdb47810111d7/model",
        as_of_date="2025-01-02",
        tickers=["AAPL", "NVDA"],
        mode="retrospective",
        resolve=True
    )
    
    assert result["prediction_count"] == 2
    assert len(result["prediction_ids"]) == 2
    assert len(result["resolved_records"]) == 2
    
    repo = MLOpsRepository()
    for pid in result["prediction_ids"]:
        record = repo.get_prediction(pid)
        assert record is not None
        assert record.status == "resolved"
        assert record.actual_price_change_pct is not None
        assert record.actual_drop_observed is not None
        assert record.resolved_at is not None
        assert record.input_feature_hash is not None
        assert len(record.input_feature_hash) == 16


def test_end_to_end_prospective_prediction_remains_open():
    """
    Test that predictions on the latest dataset edge (current date)
    remain open because 20-trading-day future data has not yet occurred.
    Verifies that future prices are NEVER fabricated or substituted.
    """
    result = run_inference(
        model_uri="runs:/df161d0bc48349debc7cdb47810111d7/model",
        as_of_date=None,
        tickers=["AAPL"],
        mode="prospective",
        resolve=False
    )
    
    assert result["prediction_count"] == 1
    pid = result["prediction_ids"][0]
    
    repo = MLOpsRepository()
    record = repo.get_prediction(pid)
    assert record is not None
    assert record.status == "open"
    assert record.actual_drop_observed is None
    assert record.actual_price_change_pct is None
    pred_date = record.predicted_at.date() if isinstance(record.predicted_at, datetime) else record.predicted_at
    assert record.target_date > pred_date
    
    # Attempt resolution with MySQLMarketDataStore as of today
    # Target date market data for November 2026 does not exist in MySQL!
    resolver = MLOpsResolver(market_store=MySQLMarketDataStore())
    resolved = resolver.resolve_predictions([record], current_date=date.today())
    
    # Must leave open!
    assert len(resolved) == 0
    assert record.status == "open"
