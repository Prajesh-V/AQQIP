"""
Tests for EDGAR fetcher idempotency.
"""
from unittest.mock import patch, MagicMock
import datetime
from src.ingestion.edgar_fetcher import EdgarFetcher
from src.ingestion.ingestion_models import SECFilingMetadata, FilingState

@patch("src.ingestion.edgar_fetcher.requests.get")
@patch("src.ingestion.edgar_fetcher.get_mongo_db")
def test_process_filing_idempotency(mock_db_func, mock_get):
    # Mock DB
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.sec_filings = mock_collection
    mock_db_func.return_value = mock_db
    
    # Simulate that it already exists in DB with raw_content
    mock_collection.find_one.return_value = {
        "accession_number": "0000320193-23-000106",
        "raw_content": "<html><body>Existing content</body></html>"
    }

    metadata = SECFilingMetadata(
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-23-000106",
        filing_type="10-K",
        filing_date=datetime.date(2023, 11, 3),
        source_url="http://example.com/filing.htm"
    )

    fetcher = EdgarFetcher()
    fetcher.process_filing(metadata)
    
    # Should NOT have made a network request because it was already stored
    mock_get.assert_not_called()
    mock_collection.update_one.assert_not_called()
