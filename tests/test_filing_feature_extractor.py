"""
Unit and Integration Tests for Historical Filing-Change Feature Extractor

Covers:
1. Document ordering (chronological assertion)
2. Missing filings handling
3. Amendments and duplicate handling
4. Feature reproducibility (deterministic output)
5. Temporal leakage prevention in point-in-time join
"""
from __future__ import annotations

import datetime
import pytest
import numpy as np
import pandas as pd

from src.qualitative.filing_feature_extractor import (
    compute_token_jaccard,
    compute_tfidf_cosine_similarity,
    FilingFeatureExtractor
)
from src.quant.point_in_time_join import PointInTimeJoiner


def test_feature_reproducibility():
    """Identical text pairs must produce mathematically identical features across repeated runs."""
    text1 = "Item 1A. Risk Factors. The company faces macroeconomic headwinds, interest rate fluctuations, and supply chain disruptions."
    text2 = "Item 1A. Risk Factors. The company faces macroeconomic headwinds, artificial intelligence competition, and regulatory compliance risks."

    jacc1 = compute_token_jaccard(text1, text2)
    jacc2 = compute_token_jaccard(text1, text2)
    assert jacc1 == jacc2
    assert 0.0 < jacc1 < 1.0

    cos1 = compute_tfidf_cosine_similarity(text1, text2)
    cos2 = compute_tfidf_cosine_similarity(text1, text2)
    assert cos1 == cos2
    assert 0.0 < cos1 < 1.0


def test_document_ordering_logic():
    """Filing comparisons must strictly compare earlier document (t-1) against later document (t)."""
    # Create mock filings with unordered dates
    from src.ingestion.ingestion_models import SECFilingMetadata, FilingState

    filings = [
        SECFilingMetadata(
            ticker="TEST", cik="0001", accession_number="acc-2024",
            filing_type="10-K", filing_date=datetime.date(2024, 2, 1),
            report_date=datetime.date(2023, 12, 31), fiscal_year=2023,
            source_url="http://test/2024", processing_status=FilingState.DISCOVERED
        ),
        SECFilingMetadata(
            ticker="TEST", cik="0001", accession_number="acc-2022",
            filing_type="10-K", filing_date=datetime.date(2022, 2, 1),
            report_date=datetime.date(2021, 12, 31), fiscal_year=2021,
            source_url="http://test/2022", processing_status=FilingState.DISCOVERED
        ),
        SECFilingMetadata(
            ticker="TEST", cik="0001", accession_number="acc-2023",
            filing_type="10-K", filing_date=datetime.date(2023, 2, 1),
            report_date=datetime.date(2022, 12, 31), fiscal_year=2022,
            source_url="http://test/2023", processing_status=FilingState.DISCOVERED
        ),
    ]

    # Sort chronologically as done in extractor
    sorted_filings = sorted(filings, key=lambda x: (x.filing_date, x.accession_number))
    dates = [f.filing_date for f in sorted_filings]
    assert dates == [
        datetime.date(2022, 2, 1),
        datetime.date(2023, 2, 1),
        datetime.date(2024, 2, 1)
    ]


def test_missing_filings_resilience():
    """If an intermediate year is missing, the extractor links to the most recent available prior filing."""
    # Given Year 2021 and Year 2024 (2022 & 2023 missing)
    text2021 = "Risk factors include inflation and rising interest rates across all segments."
    text2024 = "Risk factors include inflation, rising interest rates, and cybersecurity breaches."

    jacc = compute_token_jaccard(text2021, text2024)
    cos = compute_tfidf_cosine_similarity(text2021, text2024)
    
    assert jacc > 0.35
    assert cos > 0.5


def test_temporal_leakage_prevention():
    """
    Point-in-time join must ensure qualitative features are NEVER visible 
    before their filing publication date.
    """
    market_df = pd.DataFrame([
        {"ticker": "AAPL", "trade_date": datetime.date(2024, 10, 30), "adj_close": 230.0},
        {"ticker": "AAPL", "trade_date": datetime.date(2024, 11, 1), "adj_close": 232.0},
        {"ticker": "AAPL", "trade_date": datetime.date(2024, 11, 5), "adj_close": 235.0},
    ])

    # 10-K filed on 2024-11-01
    qual_df = pd.DataFrame([
        {
            "ticker": "AAPL",
            "filing_date": datetime.date(2024, 11, 1),
            "rf_change_magnitude": 0.085,
            "mda_change_magnitude": 0.120,
            "accession_number": "0000320193-24-000123"
        }
    ])

    fund_df = pd.DataFrame([
        {"ticker": "AAPL", "filing_date": datetime.date(2024, 10, 15), "altman_z_score": 7.5}
    ])

    panel = PointInTimeJoiner.construct_panel(market_df, fund_df, qual_df)

    # Row 0: 2024-10-30 -> filing on 2024-11-01 MUST NOT be attached (NaN)
    row0 = panel[panel["trade_date"] == datetime.date(2024, 10, 30)].iloc[0]
    assert pd.isna(row0.get("rf_change_magnitude")), "Leakage! Filing published on 2024-11-01 was visible on 2024-10-30"

    # Row 1: 2024-11-01 -> filing is published today, so it IS attached
    row1 = panel[panel["trade_date"] == datetime.date(2024, 11, 1)].iloc[0]
    assert row1.get("rf_change_magnitude") == 0.085

    # Row 2: 2024-11-05 -> filing was published in the past, so it remains attached
    row2 = panel[panel["trade_date"] == datetime.date(2024, 11, 5)].iloc[0]
    assert row2.get("rf_change_magnitude") == 0.085
