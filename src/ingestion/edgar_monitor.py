"""
AQQIP EDGAR Near-Real-Time Monitor

Polls SEC EDGAR for new filings for the tracked ticker universe.
"""
from __future__ import annotations

import time

from src.ingestion.edgar_fetcher import EdgarFetcher
from src.ingestion.ticker_universe import TickerUniverse
from src.utils.logging import get_logger

logger = get_logger("edgar_monitor")


class EdgarMonitor:
    """Monitors the tracked universe for new filings."""

    def __init__(self):
        self.universe = TickerUniverse()
        self.fetcher = EdgarFetcher()
        self.db = self.universe.db

    def run_polling_cycle(self) -> None:
        """Run a single polling cycle across all tracked tickers."""
        tickers = self.universe.get_all()
        logger.info("Starting EDGAR polling cycle", tracked_count=len(tickers))

        for ticker_info in tickers:
            try:
                # 1. Discover all recent filings via SEC submissions API
                discovered_filings = self.fetcher.discover_filings(
                    ticker_info.ticker, ticker_info.cik
                )

                # 2. Process each relevant filing idempotently
                for metadata in discovered_filings:
                    # check if we already have it fully processed to avoid re-fetching
                    existing = self.db.sec_filings.find_one({
                        "accession_number": metadata.accession_number,
                        "raw_content": {"$exists": True, "$ne": None}
                    })
                    
                    if not existing:
                        logger.info(
                            "New filing detected",
                            ticker=ticker_info.ticker,
                            accession=metadata.accession_number,
                            form=metadata.filing_type
                        )
                        self.fetcher.process_filing(metadata)

                # Sleep briefly between companies to be ultra-conservative on SEC rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error("Error monitoring ticker", ticker=ticker_info.ticker, error=str(e))

        logger.info("EDGAR polling cycle complete")
