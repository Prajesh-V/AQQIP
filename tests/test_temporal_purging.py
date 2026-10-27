"""
Tests for Temporal Validation Integrity and Purging.
Demonstrates that boundary-crossing forward windows are rejected or purged.
"""
import pytest
import datetime
import pandas as pd
import numpy as np

from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.leakage_validator import LeakageValidator


def test_temporal_purging_excludes_boundary_crossing_outcomes():
    builder = HistoricalDatasetBuilder()
    
    # 100 trading days
    dates = pd.date_range(start="2023-11-01", periods=100, freq="B")
    df = pd.DataFrame({
        "ticker": ["AAPL"] * 100,
        "trade_date": dates.date,
        "adj_close": np.linspace(100, 120, 100)
    })
    
    # Compute forward returns (H=20)
    df = builder.target_builder.compute_forward_returns(df)
    
    # Val start on day 40 (2023-12-27 approx), Test start on day 70
    val_date_str = str(dates[40].date())
    test_date_str = str(dates[70].date())
    
    # 1. With purging disabled, LeakageValidator must catch the leakage!
    train_unpurged = df[df['trade_date'] < dates[40].date()]
    val_unpurged = df[(df['trade_date'] >= dates[40].date()) & (df['trade_date'] < dates[70].date())]
    test_unpurged = df[df['trade_date'] >= dates[70].date()]
    
    errs = LeakageValidator.validate_split_boundaries(train_unpurged, val_unpurged, test_unpurged)
    assert len(errs) > 0, "Validator should detect boundary-crossing outcome window"
    assert any("Forward outcome leakage in Train" in e for e in errs)
    
    # 2. With purging enabled through split_and_label, affected rows must be removed!
    train_p, val_p, test_p = builder.split_and_label(
        df,
        val_start=val_date_str,
        test_start=test_date_str,
        alpha=0.10,
        enforce_temporal_purging=True
    )
    
    # The max target_date in train must be strictly before val_start
    train_max_target = pd.to_datetime(train_p['target_date']).max()
    assert train_max_target < pd.to_datetime(val_date_str)
    
    # The max target_date in val must be strictly before test_start
    val_max_target = pd.to_datetime(val_p['target_date']).max()
    assert val_max_target < pd.to_datetime(test_date_str)
    
    # Zero leakage errors after purging
    clean_errs = LeakageValidator.validate_split_boundaries(train_p, val_p, test_p)
    assert len(clean_errs) == 0


def test_multi_ticker_purging_exact_counts():
    """Verify that multi-ticker datasets purge exactly H rows per ticker at each boundary."""
    builder = HistoricalDatasetBuilder()
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT"]
    
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            rows.append({"ticker": t, "trade_date": d.date(), "adj_close": 100.0 + i})
    df = pd.DataFrame(rows)
    df = builder.target_builder.compute_forward_returns(df)
    
    val_date_str = str(dates[40].date())
    test_date_str = str(dates[70].date())
    
    train_p, val_p, test_p = builder.split_and_label(
        df,
        val_start=val_date_str,
        test_start=test_date_str,
        alpha=0.10,
        enforce_temporal_purging=True
    )
    
    # Train raw has 40 days * 2 tickers = 80 rows.
    # Exactly 20 days * 2 tickers = 40 rows cross boundary into val.
    # Purged train should have exactly 20 days * 2 tickers = 40 rows.
    assert len(train_p) == 40
    for t in tickers:
        ticker_train = train_p[train_p["ticker"] == t]
        assert len(ticker_train) == 20
        assert pd.to_datetime(ticker_train["target_date"]).max() < pd.to_datetime(val_date_str)
    
    # Val raw has 30 days * 2 tickers = 60 rows.
    # Exactly 20 days * 2 tickers = 40 rows cross boundary into test.
    # Purged val should have exactly 10 days * 2 tickers = 20 rows.
    assert len(val_p) == 20
    for t in tickers:
        ticker_val = val_p[val_p["ticker"] == t]
        assert len(ticker_val) == 10
        assert pd.to_datetime(ticker_val["target_date"]).max() < pd.to_datetime(test_date_str)

