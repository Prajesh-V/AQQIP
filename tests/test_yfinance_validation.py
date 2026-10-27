"""
Tests for yfinance validation logic.
"""
import datetime
import pytest
from pydantic import ValidationError
from src.ingestion.ingestion_models import OHLCVRow

def test_valid_ohlcv_row():
    row = OHLCVRow(
        ticker="AAPL",
        trade_date=datetime.date(2023, 1, 1),
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        adj_close=105.0,
        volume=10000
    )
    assert row.ticker == "AAPL"

def test_invalid_volume():
    with pytest.raises(ValidationError) as exc:
        OHLCVRow(
            ticker="AAPL",
            trade_date=datetime.date(2023, 1, 1),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            adj_close=105.0,
            volume=-500
        )
    assert "Volume cannot be negative" in str(exc.value)

def test_high_lower_than_open():
    with pytest.raises(ValidationError) as exc:
        OHLCVRow(
            ticker="AAPL",
            trade_date=datetime.date(2023, 1, 1),
            open=100.0,
            high=95.0,
            low=90.0,
            close=92.0,
            adj_close=92.0,
            volume=1000
        )
    assert "High must be >= Open" in str(exc.value)

def test_low_higher_than_high():
    with pytest.raises(ValidationError) as exc:
        OHLCVRow(
            ticker="AAPL",
            trade_date=datetime.date(2023, 1, 1),
            open=100.0,
            high=110.0,
            low=120.0, # Impossible
            close=105.0,
            adj_close=105.0,
            volume=1000
        )
    assert "Low must be <= High" in str(exc.value)
