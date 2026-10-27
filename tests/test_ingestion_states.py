"""
Tests for ingestion states.
"""
from src.ingestion.ingestion_models import SECFilingMetadata, FilingState
import datetime

def test_filing_states_flow():
    meta = SECFilingMetadata(
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-23-000106",
        filing_type="10-K",
        filing_date=datetime.date(2023, 11, 3),
        source_url="http://example.com/filing.htm"
    )
    
    assert meta.processing_status == FilingState.DISCOVERED
    meta.processing_status = FilingState.FETCHING
    assert meta.processing_status == FilingState.FETCHING
    meta.processing_status = FilingState.RAW_STORED
    assert meta.processing_status == FilingState.RAW_STORED
