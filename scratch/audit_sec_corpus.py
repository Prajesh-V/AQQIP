import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
import mysql.connector
from pymongo import MongoClient
import pandas as pd
from config.settings import get_settings

def audit_corpus():
    settings = get_settings()
    print("=== AUDIT OF SEC FILING CORPUS AND STORAGE ===")
    
    # 1. MySQL inspection
    conn = mysql.connector.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.user,
        password=settings.mysql.password,
        database=settings.mysql.database
    )
    cursor = conn.cursor(dictionary=True)
    
    # Tables in MySQL
    cursor.execute("SHOW TABLES;")
    tables = [list(r.values())[0] for r in cursor.fetchall()]
    print(f"MySQL Tables: {tables}")
    
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM `{t}`;")
        cnt = cursor.fetchone()['cnt']
        cursor.execute(f"DESCRIBE `{t}`;")
        cols = [r['Field'] for r in cursor.fetchall()]
        print(f"Table '{t}': {cnt} rows, Columns: {cols}")
    
    # Check qualitative_features table details
    if "qualitative_features" in tables:
        cursor.execute("SELECT ticker, COUNT(*) as cnt, MIN(filing_date) as min_d, MAX(filing_date) as max_d FROM qualitative_features GROUP BY ticker;")
        q_by_ticker = cursor.fetchall()
        print(f"\nqualitative_features by ticker ({len(q_by_ticker)} tickers):")
        for r in q_by_ticker[:10]:
            print(f"  {r['ticker']}: {r['cnt']} rows, {r['min_d']} to {r['max_d']}")
        
        cursor.execute("SELECT YEAR(filing_date) as yr, COUNT(*) as cnt FROM qualitative_features GROUP BY YEAR(filing_date) ORDER BY yr;")
        q_by_yr = cursor.fetchall()
        print("\nqualitative_features by year:")
        for r in q_by_yr:
            print(f"  Year {r['yr']}: {r['cnt']} rows")

    # 2. MongoDB inspection
    try:
        mongo_client = MongoClient(settings.mongo.uri, serverSelectionTimeoutMS=2000)
        db = mongo_client[settings.mongo.database]
        collections = db.list_collection_names()
        print(f"\nMongoDB Collections in '{settings.mongo.database}': {collections}")
        for c in collections:
            cnt = db[c].count_documents({})
            sample = db[c].find_one()
            sample_keys = list(sample.keys()) if sample else []
            print(f"Collection '{c}': {cnt} documents, sample keys: {sample_keys}")
            
            distinct_tickers = db[c].distinct("ticker")
            print(f"  Distinct tickers ({len(distinct_tickers)}): {distinct_tickers[:10]}")
            for field in ["form_type", "filing_type", "form", "type"]:
                if field in sample_keys:
                    print(f"  Distinct {field}: {db[c].distinct(field)}")
            
            # Check sections
            for sec in ["item_1a", "item_7", "item1a", "item7", "risk_factors", "mda"]:
                cnt_sec = db[c].count_documents({sec: {"$exists": True, "$ne": ""}})
                if cnt_sec > 0:
                    print(f"  Has {sec}: {cnt_sec}")
                cnt_nested = db[c].count_documents({f"sections.{sec}": {"$exists": True, "$ne": ""}})
                if cnt_nested > 0:
                    print(f"  Has sections.{sec}: {cnt_nested}")
    except Exception as e:
        print(f"MongoDB connection or query failed: {e}")

if __name__ == "__main__":
    audit_corpus()
