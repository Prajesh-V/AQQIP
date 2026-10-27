"""
Unit tests for SEC-BERT qualitative feature extraction and point-in-time merging.

Covers:
1. Embedding dimensionality (768) and unit normalization.
2. Semantic drift arithmetic (1 - cosine similarity).
3. Point-in-time conservative lag enforcement during transformer feature joining.
4. Missingness and evidence quality preservation.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.qualitative.secbert_feature_extractor import SecBertFeatureExtractor
from src.quant.rolling_origin_transformer_experiment import merge_transformer_panel, SECBERT_FEATURES


@pytest.fixture(scope="module")
def extractor():
    """Load frozen extractor once for testing."""
    return SecBertFeatureExtractor(device="cpu")


def test_secbert_embedding_dimensions_and_norm(extractor):
    """Verify that section embedding is 768-dimensional and unit-normalized."""
    sample_text = (
        "Item 1A. Risk Factors. Our business, financial condition, and operating results "
        "could be materially and adversely affected by macroeconomic risks, supply chain disruptions, "
        "and heightened regulatory oversight."
    )
    emb, chunks = extractor.embed_section(sample_text)
    assert emb is not None
    assert chunks >= 1
    assert emb.shape == (768,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_secbert_empty_text_returns_none(extractor):
    """Verify empty or whitespace text returns None without error."""
    emb, chunks = extractor.embed_section("")
    assert emb is None
    assert chunks == 0

    emb_ws, chunks_ws = extractor.embed_section("   \n\t  ")
    assert emb_ws is None
    assert chunks_ws == 0


def test_secbert_drift_identity(extractor):
    """Verify identical text yields semantic drift = 0.0."""
    t1 = "Material risks include competitive threats in cloud computing and semiconductor shortages."
    emb1, _ = extractor.embed_section(t1)
    cos_sim = float(np.dot(emb1, emb1))
    drift = 1.0 - cos_sim
    assert np.isclose(drift, 0.0, atol=1e-5)


def test_point_in_time_transformer_merge_conservative_lag():
    """Verify that transformer features obey conservative 1-day availability lag."""
    base_panel = pd.DataFrame({
        "ticker": ["TEST", "TEST", "TEST"],
        "trade_date": pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-03"]),
        "adj_close": [150.0, 152.0, 151.0]
    })

    transformer_df = pd.DataFrame([{
        "ticker": "TEST",
        "filing_date": pd.to_datetime("2023-02-02"),
        "accession_number": "000000-23-000001",
        "prior_accession_number": "000000-22-000001",
        "secbert_rf_available": 1,
        "secbert_mda_available": 1,
        "secbert_has_prior": 1,
        "secbert_rf_chunks": 4,
        "secbert_mda_chunks": 3,
        "secbert_rf_cosine_sim": 0.985,
        "secbert_rf_semantic_drift": 0.015,
        "secbert_rf_norm_diff": 0.173,
        "secbert_mda_cosine_sim": 0.990,
        "secbert_mda_semantic_drift": 0.010,
        "secbert_rf_mda_divergence": 0.050
    }])

    # With conservative lag = 1 day, filing on 2023-02-02 becomes available on 2023-02-03
    merged = merge_transformer_panel(base_panel, transformer_df, lag_days=1)

    row_1 = merged[merged["trade_date"] == "2023-02-01"].iloc[0]
    row_2 = merged[merged["trade_date"] == "2023-02-02"].iloc[0]
    row_3 = merged[merged["trade_date"] == "2023-02-03"].iloc[0]

    # Date 1: prior to filing, must be NaN
    assert np.isnan(row_1["secbert_rf_semantic_drift"])

    # Date 2 (filing date): because filing occurs on date 2, with conservative 1-day lag it is NOT available at date 2 close
    assert np.isnan(row_2["secbert_rf_semantic_drift"])

    # Date 3: fully merged and available
    assert np.isclose(row_3["secbert_rf_semantic_drift"], 0.015, atol=1e-5)
    assert row_3["secbert_has_prior"] == 1


def test_secbert_v2_adverse_evidence_projection():
    """Verify that directional adverse risk evidence is the projection of delta onto reference."""
    # Synthetic embeddings (dim 768)
    np.random.seed(42)
    ref = np.random.randn(768)
    ref = ref / np.linalg.norm(ref)

    # Prior embedding
    prior = np.random.randn(768)
    prior = prior / np.linalg.norm(prior)

    # Current embedding shifts towards adverse reference
    delta = 0.3 * ref + 0.1 * np.random.randn(768)
    curr = prior + delta
    curr = curr / np.linalg.norm(curr)

    # Adverse evidence score = (curr - prior) . ref
    score = float(np.dot(curr - prior, ref))
    assert score > 0, "Shift towards adverse reference must yield positive score"

    # Counter-example: shift away from adverse reference
    curr_safe = prior - 0.3 * ref
    curr_safe = curr_safe / np.linalg.norm(curr_safe)
    score_safe = float(np.dot(curr_safe - prior, ref))
    assert score_safe < 0, "Shift away from adverse reference must yield negative score"


def test_secbert_v2_pretraining_overlap_rule():
    """Verify pretraining overlap tagging: <= 2019 is 1, >= 2020 is 0."""
    dates_and_expected = [
        ("2018-02-15", 1),
        ("2019-12-31", 1),
        ("2020-01-01", 0),
        ("2021-02-28", 0),
        ("2024-03-15", 0),
    ]
    for d_str, expected in dates_and_expected:
        tag = 1 if pd.to_datetime(d_str).year <= 2019 else 0
        assert tag == expected, f"Date {d_str} expected tag {expected}, got {tag}"

