"""
Tests for Ticker/CIK resolution.
"""
from unittest.mock import patch, MagicMock
from src.ingestion.ticker_cik_resolver import TickerResolver
from src.ingestion.ingestion_models import TickerInfo


@patch("src.ingestion.ticker_cik_resolver.requests.get")
def test_resolve_valid_ticker(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"}
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    resolver = TickerResolver()
    info = resolver.resolve_ticker("aapl")
    
    assert info is not None
    assert info.ticker == "AAPL"
    assert info.cik == "0000320193"  # zero padded to 10
    assert info.company_name == "Apple Inc."

@patch("src.ingestion.ticker_cik_resolver.requests.get")
def test_resolve_unknown_ticker(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }
    mock_get.return_value = mock_response

    resolver = TickerResolver()
    info = resolver.resolve_ticker("UNKNOWN")
    assert info is None
