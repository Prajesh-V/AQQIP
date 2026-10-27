"""
AQQIP SEC EDGAR Fetcher

Polls SEC EDGAR for filings, discovers new 10-K/10-Q accessions,
and persists raw HTML filings into MongoDB.
"""
from __future__ import annotations

import datetime
from typing import List

import requests

from config.settings import get_settings
from src.database.mongo_client import get_mongo_db
from src.ingestion.ingestion_models import SECFilingMetadata, FilingState
from src.utils.logging import get_logger
from src.utils.retry import with_http_retry

logger = get_logger("edgar_fetcher")


class EdgarFetcher:
    """Discovers and fetches SEC filings."""

    def __init__(self):
        self.settings = get_settings()
        self.headers = {"User-Agent": self.settings.edgar.user_agent}
        self.db = get_mongo_db()
        self.collection = self.db.sec_filings

    @with_http_retry
    def _fetch_json(self, url: str) -> dict:
        """Fetch JSON with retry."""
        resp = requests.get(url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    @with_http_retry
    def _fetch_text(self, url: str) -> str:
        """Fetch raw text (HTML/TXT) with retry."""
        resp = requests.get(url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.text

    def discover_filings(self, ticker: str, cik: str) -> List[SECFilingMetadata]:
        """
        Poll SEC submissions API for a CIK to discover 10-K/10-Q filings.
        Returns a list of discovered metadata (does not fetch the full documents yet).
        """
        logger.info("Discovering filings", ticker=ticker, cik=cik)
        # CIK must be exactly 10 digits for the submissions endpoint
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        
        try:
            data = self._fetch_json(url)
        except requests.exceptions.RequestException as e:
            logger.error("Failed to discover filings", ticker=ticker, error=str(e))
            return []

        filings_data = data.get("filings", {}).get("recent", {})
        if not filings_data:
            logger.warning("No recent filings found in submissions data", ticker=ticker)
            return []

        discovered: List[SECFilingMetadata] = []
        
        accession_numbers = filings_data.get("accessionNumber", [])
        forms = filings_data.get("form", [])
        filing_dates = filings_data.get("filingDate", [])
        report_dates = filings_data.get("reportDate", [])
        primary_docs = filings_data.get("primaryDocument", [])

        # The lists in 'recent' are all parallel arrays
        for idx in range(len(accession_numbers)):
            form = forms[idx]
            if form not in ("10-K", "10-Q"):
                continue
            
            acc_num = accession_numbers[idx]
            
            # The filing path is /Archives/edgar/data/{cik_no_zeros}/{acc_num_no_dashes}/{primary_doc}
            cik_int_str = str(int(cik_padded)) # remove leading zeros
            acc_no_dashes = acc_num.replace("-", "")
            primary_doc = primary_docs[idx]
            
            # Form the source URL to the actual document
            source_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int_str}/{acc_no_dashes}/{primary_doc}"

            try:
                filing_date = datetime.date.fromisoformat(filing_dates[idx])
                report_date_str = report_dates[idx]
                report_date = datetime.date.fromisoformat(report_date_str) if report_date_str else None
                
                # Derive fiscal year from report date if available
                fiscal_year = report_date.year if report_date else None
            except ValueError:
                logger.warning("Invalid date format in submission data", ticker=ticker, acc_num=acc_num)
                continue

            metadata = SECFilingMetadata(
                ticker=ticker,
                cik=cik,
                accession_number=acc_num,
                filing_type=form,
                filing_date=filing_date,
                report_date=report_date,
                fiscal_year=fiscal_year,
                source_url=source_url,
                processing_status=FilingState.DISCOVERED
            )
            discovered.append(metadata)

        logger.info("Discovered relevant filings", ticker=ticker, count=len(discovered))
        return discovered

    def process_filing(self, metadata: SECFilingMetadata) -> None:
        """
        Idempotently fetch and store a raw filing document.
        """
        # Check if we already have it in MongoDB with raw_content
        existing = self.collection.find_one({"accession_number": metadata.accession_number})
        if existing and existing.get("raw_content"):
            logger.debug("Filing already stored", accession=metadata.accession_number)
            return

        logger.info("Fetching raw filing document", accession=metadata.accession_number, url=metadata.source_url)
        
        try:
            # Update state to fetching
            metadata.processing_status = FilingState.FETCHING
            self._upsert_metadata(metadata)

            # Download document
            raw_text = self._fetch_text(metadata.source_url)
            metadata.raw_content = raw_text
            metadata.processing_status = FilingState.RAW_STORED
            
            self._upsert_metadata(metadata)
            logger.info("Successfully persisted raw filing", accession=metadata.accession_number)
            
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch filing document", accession=metadata.accession_number, error=str(e))
            metadata.processing_status = FilingState.FETCH_FAILED
            self._upsert_metadata(metadata)

    def _upsert_metadata(self, metadata: SECFilingMetadata) -> None:
        """Upsert the metadata document into MongoDB."""
        doc = metadata.model_dump()
        # Convert dates to datetime for MongoDB
        doc["filing_date"] = datetime.datetime.combine(doc["filing_date"], datetime.time.min)
        if doc["report_date"]:
            doc["report_date"] = datetime.datetime.combine(doc["report_date"], datetime.time.min)
            
        self.collection.update_one(
            {"accession_number": metadata.accession_number},
            {"$set": doc},
            upsert=True
        )
