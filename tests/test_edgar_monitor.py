"""
Tests for EDGAR Monitor logic.
"""
from unittest.mock import patch, MagicMock
from src.ingestion.edgar_monitor import EdgarMonitor
from src.ingestion.ingestion_models import TickerInfo, SECFilingMetadata
import datetime

@patch("src.ingestion.edgar_monitor.TickerUniverse")
@patch("src.ingestion.edgar_monitor.EdgarFetcher")
def test_edgar_monitor_polling_cycle(mock_fetcher_cls, mock_universe_cls):
    mock_universe = MagicMock()
    mock_universe.get_all.return_value = [
        TickerInfo(ticker="AAPL", cik="0000320193", company_name="Apple")
    ]
    mock_universe_cls.return_value = mock_universe
    
    mock_fetcher = MagicMock()
    # Return one discovered filing
    metadata = SECFilingMetadata(
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-23-000106",
        filing_type="10-K",
        filing_date=datetime.date(2023, 11, 3),
        source_url="http://example.com/filing.htm"
    )
    mock_fetcher.discover_filings.return_value = [metadata]
    mock_fetcher_cls.return_value = mock_fetcher

    # Mock DB so it says the filing is NOT processed yet
    mock_universe.db.sec_filings.find_one.return_value = None

    monitor = EdgarMonitor()
    monitor.run_polling_cycle()

    mock_universe.get_all.assert_called_once()
    mock_fetcher.discover_filings.assert_called_once_with("AAPL", "0000320193")
    mock_fetcher.process_filing.assert_called_once_with(metadata)
