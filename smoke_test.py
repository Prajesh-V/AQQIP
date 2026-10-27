"""
Smoke test for Phase 2 Market Data Ingestion.
Runs a real ingestion pass against Yahoo Finance and SEC EDGAR.
"""
import sys
import datetime
from pathlib import Path
from src.ingestion.ticker_universe import TickerUniverse
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.ingestion.edgar_fetcher import EdgarFetcher
from src.database.mysql_client import get_mysql_connection

def run_smoke_test():
    print("\n--- AQQIP Phase 2 Smoke Test ---")
    
    try:
        # 1. Resolve Ticker -> CIK
        universe = TickerUniverse()
        print("Adding MSFT to universe...")
        info = universe.add_ticker("MSFT")
        print(f"Resolved: Ticker={info.ticker}, CIK={info.cik}, Name={info.company_name}")
        
        # 2. Fetch yfinance Data
        yfetcher = YFinanceFetcher()
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=10)
        print(f"Fetching market data from {start_date} to {end_date}...")
        
        inserted = yfetcher.fetch_and_store("MSFT", start_date, end_date)
        print(f"Inserted/Upserted {inserted} price rows into MySQL.")
        
        # Verify idempotency
        inserted_again = yfetcher.fetch_and_store("MSFT", start_date, end_date)
        print(f"Idempotent retry returned {inserted_again} rows affected (should be same as above).")
        
        # Check MySQL directly
        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ohlcv_prices WHERE ticker='MSFT'")
            count = cursor.fetchone()[0]
            print(f"Total OHLCV rows for MSFT in DB: {count}")
            assert count >= inserted, "Missing rows in DB!"
            
        # 3. Discover SEC Filings
        efetcher = EdgarFetcher()
        print("Discovering EDGAR filings...")
        filings = efetcher.discover_filings("MSFT", info.cik)
        print(f"Discovered {len(filings)} 10-K/10-Q filings.")
        
        if filings:
            # Fetch just the most recent one to prove it works without downloading 100 docs
            recent = filings[0]
            print(f"Fetching raw document for {recent.accession_number}...")
            efetcher.process_filing(recent)
            
            # Check MongoDB
            db = efetcher.collection.database
            stored = db.sec_filings.find_one({"accession_number": recent.accession_number})
            print("Successfully stored raw filing in MongoDB." if stored and stored.get('raw_content') else "Failed to store filing!")
            
            # Verify idempotency
            print("Testing EDGAR idempotency...")
            efetcher.process_filing(recent)
            print("Idempotent EDGAR process finished.")
            
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("Smoke test completed successfully.\n")

if __name__ == "__main__":
    run_smoke_test()
