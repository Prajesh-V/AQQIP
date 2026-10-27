"""
AQQIP Historical Filing-Change Feature Extractor

Extracts interpretable, point-in-time textual change features across consecutive 
SEC 10-K filings (Item 1A Risk Factors and Item 7 MD&A) with complete provenance:
- Security identifier (ticker)
- Source accession numbers (current and prior)
- Verified availability dates (filing dates)
- Textual lengths and length change percentages
- Token Jaccard and TF-IDF Cosine similarities
- Added/deleted sentence counts and proportions
- Strict historical ordering and zero forward-looking leakage.
"""
from __future__ import annotations

import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database.mysql_client import get_mysql_connection
from src.ingestion.edgar_fetcher import EdgarFetcher
from src.ingestion.ingestion_models import SECFilingMetadata
from src.qualitative.html_cleaner import clean_html, preprocess_text
from src.qualitative.section_extractor import extract_risk_factors, extract_mda
from src.qualitative.chunker import chunk_text
from src.qualitative.models import ExtractionState
from src.utils.logging import get_logger

logger = get_logger("filing_feature_extractor")


def compute_token_jaccard(s1: str, s2: str) -> float:
    """Computes Jaccard similarity over word tokens."""
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 and not tokens2:
        return 1.0
    union = tokens1 | tokens2
    if not union:
        return 1.0
    return float(len(tokens1 & tokens2) / len(union))


def compute_tfidf_cosine_similarity(s1: str, s2: str) -> float:
    """Computes TF-IDF cosine similarity between two texts."""
    if not s1.strip() or not s2.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vec.fit_transform([s1, s2])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        logger.warning("TF-IDF cosine similarity computation failed", error=str(e))
        return 0.0


class FilingFeatureExtractor:
    """Extracts and persists reproducible historical filing-change features."""

    def __init__(self):
        self.fetcher = EdgarFetcher()

    def process_ticker_filings(
        self,
        ticker: str,
        cik: str,
        start_year: int = 2014,
        max_filings: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Discovers, orders, and extracts consecutive YoY change features for ticker 10-Ks.
        Ensures strict chronological ordering (earlier filing t-1 compared against t).
        """
        logger.info("Discovering filings for feature extraction", ticker=ticker, cik=cik)
        discovered = self.fetcher.discover_filings(ticker, cik)
        
        # Filter for 10-K and 10-K/A filings
        ten_ks = [
            f for f in discovered
            if f.filing_type in ("10-K", "10-K/A") and (f.fiscal_year is None or f.fiscal_year >= start_year)
        ]
        
        # Sort chronologically by filing_date ascending to maintain strict temporal ordering
        ten_ks.sort(key=lambda x: (x.filing_date, x.accession_number))
        
        if max_filings:
            ten_ks = ten_ks[-max_filings:]
            
        logger.info(
            "Ordered historical 10-K filings",
            ticker=ticker,
            count=len(ten_ks),
            date_range=f"{ten_ks[0].filing_date} to {ten_ks[-1].filing_date}" if ten_ks else "None"
        )

        extracted_records: List[Dict[str, Any]] = []
        prior_filing_info: Optional[Dict[str, Any]] = None

        for filing_meta in ten_ks:
            # Idempotently fetch raw HTML if not stored
            self.fetcher.process_filing(filing_meta)
            
            # Fetch from MongoDB
            doc = self.fetcher.collection.find_one({"accession_number": filing_meta.accession_number})
            if not doc or not doc.get("raw_content"):
                logger.warning("Filing content missing in MongoDB", accession=filing_meta.accession_number)
                continue

            raw_html = doc["raw_content"]
            cleaned = preprocess_text(clean_html(raw_html))
            
            # Extract sections
            rf_res = extract_risk_factors(ticker, filing_meta.accession_number, cleaned)
            mda_res = extract_mda(ticker, filing_meta.accession_number, cleaned)
            
            rf_text = rf_res.raw_text
            mda_text = mda_res.raw_text
            
            rf_words = len(rf_text.split()) if rf_text else 0
            rf_chars = len(rf_text)
            rf_sentences = chunk_text(rf_text)
            rf_sent_count = len(rf_sentences)
            
            mda_words = len(mda_text.split()) if mda_text else 0
            mda_chars = len(mda_text)
            
            # Base record
            record: Dict[str, Any] = {
                "ticker": ticker,
                "cik": cik,
                "fiscal_year": filing_meta.fiscal_year,
                "form": filing_meta.filing_type,
                "filing_date": filing_meta.filing_date,
                "report_date": filing_meta.report_date,
                "accession_number": filing_meta.accession_number,
                "prior_accession_number": None,
                "prior_filing_date": None,
                "extraction_status": "SUCCESS" if (rf_res.state != ExtractionState.EXTRACTION_FAILED and mda_res.state != ExtractionState.EXTRACTION_FAILED) else "PARTIAL",
                "rf_char_count": rf_chars,
                "rf_word_count": rf_words,
                "rf_sentence_count": rf_sent_count,
                "rf_length_change_pct": 0.0,
                "rf_jaccard_similarity": 1.0,
                "rf_cosine_similarity": 1.0,
                "rf_change_magnitude": 0.0,
                "rf_added_sentences_count": 0,
                "rf_deleted_sentences_count": 0,
                "rf_added_sentences_ratio": 0.0,
                "rf_deleted_sentences_ratio": 0.0,
                "mda_char_count": mda_chars,
                "mda_word_count": mda_words,
                "mda_length_change_pct": 0.0,
                "mda_jaccard_similarity": 1.0,
                "mda_cosine_similarity": 1.0,
                "mda_change_magnitude": 0.0,
            }

            # If there is a prior filing, compute year-over-year change metrics
            if prior_filing_info is not None:
                record["prior_accession_number"] = prior_filing_info["accession_number"]
                record["prior_filing_date"] = prior_filing_info["filing_date"]

                # Risk Factors diffs
                prior_rf = prior_filing_info["rf_text"]
                if rf_text and prior_rf:
                    prior_chars = len(prior_rf)
                    record["rf_length_change_pct"] = round(float((rf_chars - prior_chars) / max(prior_chars, 1) * 100.0), 4)
                    jacc = compute_token_jaccard(prior_rf, rf_text)
                    cos = compute_tfidf_cosine_similarity(prior_rf, rf_text)
                    record["rf_jaccard_similarity"] = round(jacc, 4)
                    record["rf_cosine_similarity"] = round(cos, 4)
                    record["rf_change_magnitude"] = round(1.0 - cos, 4)

                    # Sentence adds / deletes
                    prior_sentences = set(prior_filing_info["rf_sentences"])
                    curr_sentences = set(rf_sentences)
                    added = curr_sentences - prior_sentences
                    deleted = prior_sentences - curr_sentences

                    record["rf_added_sentences_count"] = len(added)
                    record["rf_deleted_sentences_count"] = len(deleted)
                    record["rf_added_sentences_ratio"] = round(len(added) / max(len(curr_sentences), 1), 4)
                    record["rf_deleted_sentences_ratio"] = round(len(deleted) / max(len(prior_sentences), 1), 4)

                # MD&A diffs
                prior_mda = prior_filing_info["mda_text"]
                if mda_text and prior_mda:
                    prior_mda_chars = len(prior_mda)
                    record["mda_length_change_pct"] = round(float((mda_chars - prior_mda_chars) / max(prior_mda_chars, 1) * 100.0), 4)
                    mda_jacc = compute_token_jaccard(prior_mda, mda_text)
                    mda_cos = compute_tfidf_cosine_similarity(prior_mda, mda_text)
                    record["mda_jaccard_similarity"] = round(mda_jacc, 4)
                    record["mda_cosine_similarity"] = round(mda_cos, 4)
                    record["mda_change_magnitude"] = round(1.0 - mda_cos, 4)

            # Persist to MySQL
            self._upsert_mysql_feature(record)
            extracted_records.append(record)

            # Update prior filing info for next iteration
            prior_filing_info = {
                "accession_number": filing_meta.accession_number,
                "filing_date": filing_meta.filing_date,
                "rf_text": rf_text,
                "rf_sentences": rf_sentences,
                "mda_text": mda_text,
            }

        return extracted_records

    def _upsert_mysql_feature(self, record: Dict[str, Any]) -> None:
        """Upsert feature observation into MySQL qualitative_filing_features."""
        sql = """
            INSERT INTO qualitative_filing_features (
                ticker, cik, fiscal_year, form, filing_date, report_date,
                accession_number, prior_accession_number, prior_filing_date, extraction_status,
                rf_char_count, rf_word_count, rf_sentence_count, rf_length_change_pct,
                rf_jaccard_similarity, rf_cosine_similarity, rf_change_magnitude,
                rf_added_sentences_count, rf_deleted_sentences_count,
                rf_added_sentences_ratio, rf_deleted_sentences_ratio,
                mda_char_count, mda_word_count, mda_length_change_pct,
                mda_jaccard_similarity, mda_cosine_similarity, mda_change_magnitude
            ) VALUES (
                %(ticker)s, %(cik)s, %(fiscal_year)s, %(form)s, %(filing_date)s, %(report_date)s,
                %(accession_number)s, %(prior_accession_number)s, %(prior_filing_date)s, %(extraction_status)s,
                %(rf_char_count)s, %(rf_word_count)s, %(rf_sentence_count)s, %(rf_length_change_pct)s,
                %(rf_jaccard_similarity)s, %(rf_cosine_similarity)s, %(rf_change_magnitude)s,
                %(rf_added_sentences_count)s, %(rf_deleted_sentences_count)s,
                %(rf_added_sentences_ratio)s, %(rf_deleted_sentences_ratio)s,
                %(mda_char_count)s, %(mda_word_count)s, %(mda_length_change_pct)s,
                %(mda_jaccard_similarity)s, %(mda_cosine_similarity)s, %(mda_change_magnitude)s
            ) ON DUPLICATE KEY UPDATE
                prior_accession_number = VALUES(prior_accession_number),
                prior_filing_date = VALUES(prior_filing_date),
                extraction_status = VALUES(extraction_status),
                rf_char_count = VALUES(rf_char_count),
                rf_word_count = VALUES(rf_word_count),
                rf_sentence_count = VALUES(rf_sentence_count),
                rf_length_change_pct = VALUES(rf_length_change_pct),
                rf_jaccard_similarity = VALUES(rf_jaccard_similarity),
                rf_cosine_similarity = VALUES(rf_cosine_similarity),
                rf_change_magnitude = VALUES(rf_change_magnitude),
                rf_added_sentences_count = VALUES(rf_added_sentences_count),
                rf_deleted_sentences_count = VALUES(rf_deleted_sentences_count),
                rf_added_sentences_ratio = VALUES(rf_added_sentences_ratio),
                rf_deleted_sentences_ratio = VALUES(rf_deleted_sentences_ratio),
                mda_char_count = VALUES(mda_char_count),
                mda_word_count = VALUES(mda_word_count),
                mda_length_change_pct = VALUES(mda_length_change_pct),
                mda_jaccard_similarity = VALUES(mda_jaccard_similarity),
                mda_cosine_similarity = VALUES(mda_cosine_similarity),
                mda_change_magnitude = VALUES(mda_change_magnitude)
        """
        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, record)
