"""
Tests for data readiness querying.
"""
from unittest.mock import patch, MagicMock
from src.ingestion.ticker_universe import TickerUniverse

@patch("src.ingestion.ticker_universe.get_mongo_db")
def test_data_readiness(mock_db_func):
    mock_db = MagicMock()
    mock_db.ticker_universe.find.return_value = [
        {"ticker": "MSFT", "cik": "0000789019", "company_name": "Microsoft"}
    ]
    mock_db_func.return_value = mock_db
    
    universe = TickerUniverse()
    all_tickers = universe.get_all()
    
    assert len(all_tickers) == 1
    assert all_tickers[0].ticker == "MSFT"
