import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
from pymongo import MongoClient
import pandas as pd
from config.settings import get_settings
from src.qualitative.html_cleaner import clean_html, preprocess_text
from src.qualitative.section_extractor import extract_risk_factors, extract_mda

def test_extraction():
    settings = get_settings()
    mongo = MongoClient(settings.mongo.uri)
    db = mongo[settings.mongo.database]
    coll = db["sec_filings"]
    
    docs = list(coll.find({}, {"ticker": 1, "accession_number": 1, "fiscal_year": 1, "filing_date": 1, "raw_content": 1}).limit(5))
    print(f"Testing extraction on {len(docs)} sample filings...")
    
    for d in docs:
        raw_html = d["raw_content"]
        cleaned = preprocess_text(clean_html(raw_html))
        rf = extract_risk_factors(d["ticker"], d["accession_number"], cleaned)
        mda = extract_mda(d["ticker"], d["accession_number"], cleaned)
        print(f"Ticker: {d['ticker']}, FY: {d['fiscal_year']}, Accession: {d['accession_number']}")
        print(f"  Cleaned text length: {len(cleaned)} chars")
        print(f"  Item 1A status: {rf.state}, length: {len(rf.raw_text)} chars, words: {len(rf.raw_text.split())}")
        print(f"  Item 7 status: {mda.state}, length: {len(mda.raw_text)} chars, words: {len(mda.raw_text.split())}")

if __name__ == "__main__":
    test_extraction()
