"""
AQQIP Ticker/CIK Resolver

Resolves company names or tickers to CIKs using SEC's company_tickers.json.
"""
from __future__ import annotations

import requests

from config.settings import get_settings
from src.ingestion.ingestion_models import TickerInfo
from src.utils.logging import get_logger
from src.utils.retry import network_retry

logger = get_logger("ticker_resolver")

# The SEC provides a mapping of all active tickers/CIKs
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class TickerResolver:
    """Resolves tickers to CIKs."""

    def __init__(self):
        self._cache: dict[str, TickerInfo] = {}
        self.settings = get_settings()

    @network_retry
    def _fetch_sec_mapping(self) -> dict:
        """Fetch the master JSON mapping from the SEC."""
        headers = {"User-Agent": self.settings.edgar.user_agent}
        response = requests.get(SEC_TICKERS_URL, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()

    def load_cache(self) -> None:
        """Load and cache the SEC ticker mapping."""
        if self._cache:
            return

        logger.info("Fetching SEC ticker mapping for CIK resolution")
        data = self._fetch_sec_mapping()
        
        # Structure is { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
        for entry in data.values():
            ticker = str(entry["ticker"]).upper()
            # CIKs must be zero-padded to 10 digits
            cik = str(entry["cik_str"]).zfill(10)
            title = str(entry["title"])
            
            self._cache[ticker] = TickerInfo(
                ticker=ticker,
                cik=cik,
                company_name=title
            )
        logger.info("Loaded SEC ticker mapping", count=len(self._cache))

    def resolve_ticker(self, ticker: str) -> TickerInfo | None:
        """Resolve a ticker symbol to its CIK and company name."""
        self.load_cache()
        ticker = ticker.strip().upper()
        return self._cache.get(ticker)
