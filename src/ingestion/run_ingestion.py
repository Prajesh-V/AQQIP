"""
AQQIP Real-Data Ingestion Entrypoint

Ingests real historical market data (yfinance) and SEC EDGAR XBRL fundamentals
for the expanded multi-sector ticker universe into MySQL and MongoDB.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from typing import List, Dict

from config.settings import get_settings
from src.database.mysql_client import get_mysql_connection
from src.ingestion.edgar_company_facts import EdgarCompanyFactsExtractor
from src.ingestion.ticker_universe import TickerUniverse
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.utils.logging import get_logger

logger = get_logger("run_ingestion")

SECTOR_MAP: Dict[str, List[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "INTC", "CSCO"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "SBUX"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "Financials": ["JPM", "BAC", "WFC", "GS"],
    "Health Care": ["JNJ", "PFE", "UNH", "MRK"],
    "Energy": ["XOM", "CVX"],
    "Industrials": ["CAT", "BA", "GE"],
}

DEFAULT_TICKERS: List[str] = [ticker for group in SECTOR_MAP.values() for ticker in group]


def run_ingestion(
    tickers: List[str] = DEFAULT_TICKERS,
    start_date: datetime.date = datetime.date(2014, 1, 1),
    end_date: datetime.date = None
) -> dict:
    """
    Executes full ingestion pass for specified tickers:
    1. Resolve & register in ticker universe.
    2. Ingest 10+ years of OHLCV prices into MySQL ohlcv_prices.
    3. Ingest fundamental facts and compute Altman/Beneish scores into fundamental_ratios.
    """
    if end_date is None:
        end_date = datetime.date.today()

    logger.info("Starting AQQIP real-data ingestion", tickers=tickers, start=str(start_date), end=str(end_date))
    universe = TickerUniverse()
    y_fetcher = YFinanceFetcher()
    edgar_extractor = EdgarCompanyFactsExtractor()

    results = {
        "tickers": {},
        "market_rows_total": 0,
        "fundamental_rows_total": 0,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }

    for ticker in tickers:
        ticker = ticker.upper()
        logger.info(f"--- Processing {ticker} ---")
        
        # 1. Ticker resolution & CIK registration
        try:
            info = universe.add_ticker(ticker)
            logger.info("Resolved ticker", ticker=info.ticker, cik=info.cik, name=info.company_name)
            cik = info.cik
            company_name = info.company_name
        except Exception as e:
            logger.error("Failed to resolve ticker", ticker=ticker, error=str(e))
            continue

        # 2. Market data ingestion (10+ years OHLCV)
        market_rows = y_fetcher.fetch_and_store(ticker, start_date, end_date)
        
        # 3. Fundamentals ingestion (10+ years SEC XBRL facts)
        fund_rows = edgar_extractor.extract_and_store(ticker, cik)

        results["tickers"][ticker] = {
            "cik": cik,
            "name": company_name,
            "market_rows": market_rows,
            "fundamental_rows": fund_rows
        }
        results["market_rows_total"] += market_rows
        results["fundamental_rows_total"] += fund_rows

    # Verify against MySQL directly
    with get_mysql_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT ticker, COUNT(*) as rows_count, MIN(trade_date) as min_date, MAX(trade_date) as max_date 
            FROM ohlcv_prices 
            GROUP BY ticker 
            ORDER BY ticker ASC
        """)
        market_summary = cursor.fetchall()
        
        cursor.execute("""
            SELECT ticker, COUNT(*) as rows_count, COUNT(altman_z_score) as altman_count, 
                   COUNT(beneish_m_score) as beneish_count, MIN(filing_date) as min_filing, MAX(filing_date) as max_filing 
            FROM fundamental_ratios 
            GROUP BY ticker 
            ORDER BY ticker ASC
        """)
        fund_summary = cursor.fetchall()

    results["mysql_market_summary"] = market_summary
    results["mysql_fund_summary"] = fund_summary
    return results


def main():
    parser = argparse.ArgumentParser(description="AQQIP Real-Data Ingestion Pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of stock tickers")
    parser.add_argument("--start", type=str, default="2014-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=datetime.date.today().isoformat(), help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start_d = datetime.date.fromisoformat(args.start)
    end_d = datetime.date.fromisoformat(args.end)

    res = run_ingestion(args.tickers, start_d, end_d)
    
    print("\n================ INGESTION SUMMARY ================")
    print(f"Total Market Rows Ingested/Updated: {res['market_rows_total']}")
    print(f"Total Fundamental Rows Ingested:    {res['fundamental_rows_total']}")
    print("\nMarket Data Coverage in MySQL:")
    for m in res["mysql_market_summary"]:
        print(f"  {m['ticker']:<6}: {m['rows_count']:>5} rows | {m['min_date']} to {m['max_date']}")
    print("\nFundamental Ratios Coverage in MySQL:")
    for f in res["mysql_fund_summary"]:
        print(f"  {f['ticker']:<6}: {f['rows_count']:>3} statements | Altman={f['altman_count']:>3}, Beneish={f['beneish_count']:>3} | {f['min_filing']} to {f['max_filing']}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
