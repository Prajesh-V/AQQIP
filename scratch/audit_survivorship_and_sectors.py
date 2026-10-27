import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
import mysql.connector
import pandas as pd
from config.settings import get_settings

def audit_survivorship():
    settings = get_settings()
    conn = mysql.connector.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.user,
        password=settings.mysql.password,
        database=settings.mysql.database
    )
    
    # 1. Ticker coverage and dates
    df_ohlcv = pd.read_sql("SELECT ticker, MIN(trade_date) as start_date, MAX(trade_date) as end_date, COUNT(*) as trading_days FROM ohlcv_prices GROUP BY ticker ORDER BY ticker;", conn)
    print("=== TICKER UNIVERSE DATE RANGES AND COVERAGE ===")
    print(df_ohlcv.to_string(index=False))
    
    # 2. Sector mapping from config or ticker_universe in Mongo
    from pymongo import MongoClient
    mongo = MongoClient(settings.mongo.uri)
    db = mongo[settings.mongo.database]
    tickers_meta = list(db["ticker_universe"].find({}, {"_id": 0}))
    df_meta = pd.DataFrame(tickers_meta)
    print(f"\n=== TICKER UNIVERSE METADATA ({len(df_meta)} tickers) ===")
    print(df_meta.head(10).to_string(index=False))
    
    # 3. Check sector distribution
    from src.ingestion.ticker_universe import DEFAULT_TICKERS, TICKER_SECTOR_MAP
    print(f"\nConfigured TICKER_SECTOR_MAP: {len(TICKER_SECTOR_MAP)} tickers")
    sector_counts = pd.Series(TICKER_SECTOR_MAP).value_counts()
    print("Sector distribution:")
    print(sector_counts)
    
    # Check date coverage anomalies
    print(f"\nMin start date across universe: {df_ohlcv['start_date'].min()}")
    print(f"Max start date across universe: {df_ohlcv['start_date'].max()}")
    print(f"Min end date across universe: {df_ohlcv['end_date'].min()}")
    print(f"Max end date across universe: {df_ohlcv['end_date'].max()}")
    
    # Survivorship bias assessment
    # Are all 32 tickers present from 2014-01-02 to 2026?
    non_standard_start = df_ohlcv[df_ohlcv['start_date'] > '2014-01-05']
    print(f"\nTickers starting after 2014-01-05: {len(non_standard_start)}")
    if not non_standard_start.empty:
        print(non_standard_start)

if __name__ == "__main__":
    audit_survivorship()
