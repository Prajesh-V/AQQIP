import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
from pymongo import MongoClient
import mysql.connector
import pandas as pd
from config.settings import get_settings

def inspect_filings_detail():
    settings = get_settings()
    mongo = MongoClient(settings.mongo.uri)
    db = mongo[settings.mongo.database]
    coll = db["sec_filings"]
    
    docs = list(coll.find({}, {"ticker": 1, "fiscal_year": 1, "filing_date": 1, "report_date": 1, "accession_number": 1, "raw_content": {"$slice": 200}}))
    df = pd.DataFrame(docs)
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['year'] = df['filing_date'].dt.year
    
    print("=== SEC_FILINGS COLLECTION AUDIT ===")
    print(f"Total documents: {len(df)}")
    print(f"Distinct tickers: {df['ticker'].nunique()}")
    print("Filing counts by calendar year:")
    print(df['year'].value_counts().sort_index())
    
    print("\nFiling counts by fiscal year:")
    print(df['fiscal_year'].value_counts().sort_index())
    
    # Check sample raw_content structure
    sample = coll.find_one()
    print(f"\nSample doc keys: {list(sample.keys())}")
    print(f"Sample raw_content type: {type(sample.get('raw_content'))}")
    if isinstance(sample.get('raw_content'), dict):
        print(f"Sample raw_content keys: {list(sample['raw_content'].keys())}")
        for k, v in sample['raw_content'].items():
            print(f"  Section {k}: length {len(str(v))} chars")
    elif isinstance(sample.get('raw_content'), str):
        print(f"Sample raw_content length: {len(sample['raw_content'])} chars")
        print(f"Sample raw_content snippet: {sample['raw_content'][:200]}")
        
    # Check MySQL qualitative_filing_features
    conn = mysql.connector.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.user,
        password=settings.mysql.password,
        database=settings.mysql.database
    )
    q_df = pd.read_sql("SELECT ticker, fiscal_year, filing_date, accession_number, prior_accession_number, rf_char_count, mda_char_count, rf_cosine_similarity FROM qualitative_filing_features", conn)
    print("\n=== MYSQL QUALITATIVE_FILING_FEATURES AUDIT ===")
    print(f"Total rows: {len(q_df)}")
    print(f"Distinct tickers: {q_df['ticker'].nunique()}")
    print(f"Valid prior pairs (prior_accession_number IS NOT NULL): {q_df['prior_accession_number'].notna().sum()}")
    print(f"Rows with non-zero rf_char_count: {(q_df['rf_char_count'] > 0).sum()}")
    print(f"Rows with non-zero mda_char_count: {(q_df['mda_char_count'] > 0).sum()}")
    print(f"Rows with valid rf_cosine_similarity: {q_df['rf_cosine_similarity'].notna().sum()}")
    
    print("\nQualitative rows by year:")
    q_df['filing_year'] = pd.to_datetime(q_df['filing_date']).dt.year
    print(q_df['filing_year'].value_counts().sort_index())
    
    # Check missing tickers
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM ohlcv_prices")
    all_tickers = set(r[0] for r in cursor.fetchall())
    filing_tickers = set(q_df['ticker'].unique())
    missing_tickers = all_tickers - filing_tickers
    print(f"\nTickers in OHLCV: {len(all_tickers)}, Tickers with filings: {len(filing_tickers)}")
    print(f"Tickers missing filings entirely: {missing_tickers}")

if __name__ == "__main__":
    inspect_filings_detail()
