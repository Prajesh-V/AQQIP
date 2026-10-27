"""
Tests for yfinance fetcher idempotency and mock fetching.
"""
from unittest.mock import patch, MagicMock
import datetime
import pandas as pd
from src.ingestion.yfinance_fetcher import YFinanceFetcher

@patch("src.ingestion.yfinance_fetcher.yf.download")
@patch("src.ingestion.yfinance_fetcher.get_mysql_connection")
def test_yfinance_idempotency(mock_db, mock_download):
    mock_df = pd.DataFrame({
        "Open": [100.0],
        "High": [105.0],
        "Low": [95.0],
        "Close": [102.0],
        "Adj Close": [102.0],
        "Volume": [1000]
    }, index=[datetime.date(2023, 1, 1)])
    mock_download.return_value = mock_df
    
    mock_conn = MagicMock()
    mock_db.return_value.__enter__.return_value = mock_conn

    fetcher = YFinanceFetcher()
    rows_inserted = fetcher.fetch_and_store("AAPL", datetime.date(2023, 1, 1), datetime.date(2023, 1, 1))
    
    assert rows_inserted == 1
    # Verify the executemany was called to do the ON DUPLICATE KEY UPDATE
    mock_conn.cursor.return_value.executemany.assert_called_once()
    
    query = mock_conn.cursor.return_value.executemany.call_args[0][0]
    assert "ON DUPLICATE KEY UPDATE" in query
