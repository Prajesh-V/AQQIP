"""
Tests for SEC EDGAR metadata preserving logic.
"""
from src.ingestion.ingestion_models import SECFilingMetadata
import datetime

def test_metadata_fields():
    meta = SECFilingMetadata(
        ticker="MSFT",
        cik="0000789019",
        accession_number="0000789019-23-000054",
        filing_type="10-K",
        filing_date=datetime.date(2023, 7, 27),
        report_date=datetime.date(2023, 6, 30),
        fiscal_year=2023,
        source_url="https://www.sec.gov/..."
    )
    assert meta.ticker == "MSFT"
    assert meta.cik == "0000789019"
    assert meta.fiscal_year == 2023
    assert meta.report_date.month == 6
