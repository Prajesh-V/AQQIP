"""
Unit and Integration Tests for MySQLMarketDataStore and MLOpsResolver.

Verifies:
1. Successful resolution with exact target contract price calculation.
2. Missing future market data (leaves prediction open).
3. Missing entry price (leaves prediction open).
4. Incomplete target window (future target date relative to current_date leaves open).
5. Duplicate runs idempotency (already resolved predictions are never reprocessed).
6. Live MySQL integration against actual ohlcv_prices table.
"""
import pytest
from datetime import date, datetime
from src.mlops.models import PredictionRecord
from src.mlops.resolver import MockMarketDataStore, MySQLMarketDataStore, MLOpsResolver
from src.mlops.repository import MLOpsRepository


def test_successful_resolution_unit():
    """Valid entry and exit prices result in successful resolution and exact return."""
    entry_date = date(2024, 1, 2)
    target_date = date(2024, 1, 31)
    
    store = MockMarketDataStore(prices={
        ("AAPL", entry_date): 100.0,
        ("AAPL", target_date): 88.0,   # -12.0% drop
    })
    resolver = MLOpsResolver(store)
    
    pred = PredictionRecord(
        prediction_id=101,
        ticker="AAPL",
        model_run_id="test_run",
        predicted_drop_probability=0.75,
        predicted_at=datetime(2024, 1, 2, 9, 30),
        target_date=target_date,
        drop_threshold_pct=10.00,
        status="open"
    )
    
    resolved = resolver.resolve_predictions([pred], current_date=date(2024, 2, 1))
    
    assert len(resolved) == 1
    assert resolved[0].status == "resolved"
    assert resolved[0].actual_price_change_pct == -12.0000
    assert resolved[0].actual_drop_observed is True
    assert resolved[0].resolved_at is not None


def test_missing_future_data():
    """When exit price observation does not exist, prediction remains open."""
    entry_date = date(2024, 1, 2)
    target_date = date(2024, 1, 31)
    
    store = MockMarketDataStore(prices={
        ("AAPL", entry_date): 100.0,
        # ("AAPL", target_date) is missing
    })
    resolver = MLOpsResolver(store)
    
    pred = PredictionRecord(
        prediction_id=102,
        ticker="AAPL",
        model_run_id="test_run",
        predicted_drop_probability=0.75,
        predicted_at=datetime(2024, 1, 2, 9, 30),
        target_date=target_date,
        drop_threshold_pct=10.00,
        status="open"
    )
    
    resolved = resolver.resolve_predictions([pred], current_date=date(2024, 2, 1))
    assert len(resolved) == 0
    assert pred.status == "open"
    assert pred.actual_drop_observed is None


def test_missing_entry_price():
    """When entry price observation does not exist, prediction remains open."""
    target_date = date(2024, 1, 31)
    
    store = MockMarketDataStore(prices={
        # entry date missing
        ("AAPL", target_date): 88.0,
    })
    resolver = MLOpsResolver(store)
    
    pred = PredictionRecord(
        prediction_id=103,
        ticker="AAPL",
        model_run_id="test_run",
        predicted_drop_probability=0.75,
        predicted_at=datetime(2024, 1, 2, 9, 30),
        target_date=target_date,
        drop_threshold_pct=10.00,
        status="open"
    )
    
    resolved = resolver.resolve_predictions([pred], current_date=date(2024, 2, 1))
    assert len(resolved) == 0
    assert pred.status == "open"


def test_incomplete_target_window():
    """When target date has not arrived yet relative to current_date, prediction remains open."""
    entry_date = date(2025, 12, 15)
    target_date = date(2026, 1, 15)
    
    store = MockMarketDataStore(prices={
        ("AAPL", entry_date): 150.0,
        ("AAPL", target_date): 140.0,
    })
    resolver = MLOpsResolver(store)
    
    pred = PredictionRecord(
        prediction_id=104,
        ticker="AAPL",
        model_run_id="test_run",
        predicted_drop_probability=0.60,
        predicted_at=datetime(2025, 12, 15, 9, 30),
        target_date=target_date,
        drop_threshold_pct=10.00,
        status="open"
    )
    
    # As of 2025-12-31, the target date (2026-01-15) is in the future
    resolved = resolver.resolve_predictions([pred], current_date=date(2025, 12, 31))
    assert len(resolved) == 0
    assert pred.status == "open"


def test_duplicate_runs_idempotent():
    """Running resolver repeatedly is idempotent and never re-resolves closed records."""
    entry_date = date(2024, 1, 2)
    target_date = date(2024, 1, 31)
    
    store = MockMarketDataStore(prices={
        ("AAPL", entry_date): 100.0,
        ("AAPL", target_date): 95.0, # -5.0% change (not a sharp drop)
    })
    resolver = MLOpsResolver(store)
    
    pred = PredictionRecord(
        prediction_id=105,
        ticker="AAPL",
        model_run_id="test_run",
        predicted_drop_probability=0.30,
        predicted_at=datetime(2024, 1, 2, 9, 30),
        target_date=target_date,
        drop_threshold_pct=10.00,
        status="open"
    )
    
    # First run
    resolved_1 = resolver.resolve_predictions([pred], current_date=date(2024, 2, 1))
    assert len(resolved_1) == 1
    assert pred.status == "resolved"
    assert pred.actual_drop_observed is False
    assert pred.actual_price_change_pct == -5.0000
    first_resolved_at = pred.resolved_at
    
    # Second run with same prediction object (now status='resolved')
    resolved_2 = resolver.resolve_predictions([pred], current_date=date(2024, 2, 1))
    assert len(resolved_2) == 0
    assert pred.status == "resolved"
    assert pred.resolved_at == first_resolved_at


def test_mysql_market_data_store_live():
    """Integration test: MySQLMarketDataStore queries real ohlcv_prices."""
    store = MySQLMarketDataStore()
    
    # Known date in MySQL
    price = store.get_price("AAPL", date(2024, 1, 2))
    assert price is not None
    assert price > 0.0
    assert isinstance(price, float)
    
    # Non-existent date returns None
    assert store.get_price("AAPL", date(1990, 1, 1)) is None
    # Non-existent ticker returns None
    assert store.get_price("NONEXISTENT", date(2024, 1, 2)) is None


def test_resolver_live_mysql_integration():
    """End-to-end integration: Create open prediction in MySQL, resolve against real OHLCV data."""
    repo = MLOpsRepository()
    resolver = MLOpsResolver(market_store=MySQLMarketDataStore())
    
    # Create prediction record for historical date where real prices exist in MySQL
    # AAPL on 2024-01-02: adj_close ~ 183.4040
    # AAPL on 2024-01-31: adj_close ~ 182.1789
    entry_date = date(2024, 1, 2)
    target_date = date(2024, 1, 31)
    
    pred_record = PredictionRecord(
        ticker="AAPL",
        model_run_id="test_live_resolver_run",
        model_version="v1.0.0",
        predicted_drop_probability=0.25,
        predicted_at=datetime(2024, 1, 2, 9, 30),
        target_date=target_date,
        prediction_horizon_days=20,
        drop_threshold_pct=10.00,
        status="open"
    )
    
    pred_id = repo.create_prediction(pred_record)
    assert pred_id is not None
    
    # Verify it exists as 'open'
    saved = repo.get_prediction(pred_id)
    assert saved is not None
    assert saved.status == "open"
    
    # Resolve using live MySQL store
    resolved = resolver.resolve_predictions([saved], current_date=date(2024, 2, 1))
    assert len(resolved) == 1
    assert resolved[0].status == "resolved"
    # -0.6680% return
    assert resolved[0].actual_price_change_pct is not None
    assert abs(resolved[0].actual_price_change_pct - (-0.6680)) < 0.01
    assert resolved[0].actual_drop_observed is False
    
    # Persist resolved status to MySQL
    repo.update_resolved(resolved[0])
    
    # Verify in DB
    updated = repo.get_prediction(pred_id)
    assert updated.status == "resolved"
    assert updated.actual_drop_observed is False
    assert abs(updated.actual_price_change_pct - (-0.6680)) < 0.01
    assert updated.resolved_at is not None
    
    # Repeated resolution check: already resolved prediction is ignored
    rerun = resolver.resolve_predictions([updated], current_date=date(2024, 2, 1))
    assert len(rerun) == 0
