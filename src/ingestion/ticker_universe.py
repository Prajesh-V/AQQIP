"""
AQQIP Ticker Universe

Manages the tracked universe of tickers in MongoDB.
"""
from __future__ import annotations

from typing import List

from pymongo.errors import DuplicateKeyError

from src.database.mongo_client import get_mongo_db
from src.ingestion.ingestion_models import TickerInfo
from src.ingestion.ticker_cik_resolver import TickerResolver
from src.utils.logging import get_logger

logger = get_logger("ticker_universe")


class TickerUniverse:
    """Manages the tracked universe of companies."""

    def __init__(self):
        self.db = get_mongo_db()
        self.collection = self.db.ticker_universe
        self.resolver = TickerResolver()

    def add_ticker(self, ticker: str) -> TickerInfo:
        """Add a ticker to the tracked universe, resolving its CIK automatically."""
        info = self.resolver.resolve_ticker(ticker)
        if not info:
            raise ValueError(f"Could not resolve CIK for ticker: {ticker}")

        doc = {
            "ticker": info.ticker,
            "cik": info.cik,
            "company_name": info.company_name
        }

        try:
            self.collection.insert_one(doc)
            logger.info("Added ticker to universe", ticker=info.ticker, cik=info.cik)
        except DuplicateKeyError:
            logger.debug("Ticker already in universe", ticker=info.ticker)
            # Fetch existing to ensure we return correct state if it was already there
            existing = self.collection.find_one({"ticker": info.ticker})
            if existing:
                info = TickerInfo(
                    ticker=existing["ticker"],
                    cik=existing["cik"],
                    company_name=existing.get("company_name")
                )

        return info

    def get_all(self) -> List[TickerInfo]:
        """Return all tracked tickers."""
        cursor = self.collection.find({})
        return [
            TickerInfo(
                ticker=doc["ticker"],
                cik=doc["cik"],
                company_name=doc.get("company_name")
            )
            for doc in cursor
        ]
