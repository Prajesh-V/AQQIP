"""
AQQIP Market Data Fetcher

Downloads and persists historical OHLCV data using yfinance.
Ensures point-in-time integrity and idempotent updates.
"""
from __future__ import annotations

import datetime
from typing import List

import pandas as pd
import yfinance as yf
from pydantic import ValidationError

from src.database.mysql_client import get_mysql_connection
from src.ingestion.ingestion_models import OHLCVRow
from src.utils.logging import get_logger
from src.utils.retry import with_network_retry

logger = get_logger("yfinance_fetcher")


class YFinanceFetcher:
    """Fetches market data from Yahoo Finance."""

    @with_network_retry
    def _download_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download raw data with retry logic."""
        # yf.download returns a MultiIndex column dataframe in recent versions if passing list, 
        # but single ticker should be flat. We use single ticker.
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df

    def fetch_and_store(self, ticker: str, start_date: datetime.date, end_date: datetime.date) -> int:
        """
        Fetch OHLCV data for the given date range and upsert into MySQL.
        Returns the number of rows successfully inserted/updated.
        """
        logger.info(
            "Fetching market data",
            ticker=ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat()
        )

        try:
            # yfinance expects YYYY-MM-DD
            start_str = start_date.isoformat()
            # yfinance end date is exclusive, so add 1 day to include end_date
            end_exclusive = end_date + datetime.timedelta(days=1)
            end_str = end_exclusive.isoformat()

            df = self._download_data(ticker, start_str, end_str)
        except Exception as e:
            logger.error("Failed to fetch data from yfinance", ticker=ticker, error=str(e))
            return 0

        if df.empty:
            logger.info("No market data returned", ticker=ticker)
            return 0

        # yfinance 0.2.x+ column names: Open, High, Low, Close, Adj Close, Volume
        # For single ticker, columns are flat. If multi-ticker was used (we didn't), they'd be multi-level.
        # Ensure flat columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1) # Drop the ticker level

        valid_rows: List[OHLCVRow] = []
        for index, row in df.iterrows():
            try:
                # Handle potential NaN values
                if pd.isna(row.get('Open')) or pd.isna(row.get('Volume')):
                    continue

                trade_date = index.date() if isinstance(index, pd.Timestamp) else index
                
                ohlcv = OHLCVRow(
                    ticker=ticker.upper(),
                    trade_date=trade_date,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    adj_close=float(row['Adj Close']) if 'Adj Close' in df.columns else float(row['Close']),
                    volume=int(row['Volume'])
                )
                valid_rows.append(ohlcv)
            except ValidationError as ve:
                logger.warning(
                    "Invalid OHLCV row skipped",
                    ticker=ticker,
                    date=str(index),
                    error=str(ve)
                )
            except Exception as e:
                 logger.warning(
                    "Unexpected error parsing row",
                    ticker=ticker,
                    date=str(index),
                    error=str(e)
                )

        if not valid_rows:
            logger.info("No valid rows to insert", ticker=ticker)
            return 0

        inserted_count = self._upsert_to_mysql(valid_rows)
        logger.info("Successfully ingested market data", ticker=ticker, rows=inserted_count)
        return inserted_count

    def _upsert_to_mysql(self, rows: List[OHLCVRow]) -> int:
        """Upsert rows to MySQL idempotently."""
        query = """
            INSERT INTO ohlcv_prices (
                ticker, trade_date, open, high, low, close, adj_close, volume, source
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                adj_close = VALUES(adj_close),
                volume = VALUES(volume),
                source = VALUES(source)
        """
        
        data = [
            (
                r.ticker,
                r.trade_date,
                r.open_price,
                r.high_price,
                r.low_price,
                r.close_price,
                r.adj_close_price,
                r.volume,
                r.source
            )
            for r in rows
        ]

        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            # execute_many rowcount is not perfectly exact for UPSERT in all MySQL versions,
            # but it is usually rows_affected (1 for insert, 2 for update)
            return len(rows)
