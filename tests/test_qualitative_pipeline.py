import pytest
import numpy as np
from src.qualitative.html_cleaner import clean_html, preprocess_text
from src.qualitative.section_extractor import extract_risk_factors, MIN_RISK_FACTORS_LENGTH
from src.qualitative.models import ExtractionState
from src.qualitative.chunker import chunk_text
from src.qualitative.semantic_diff import compute_semantic_diff
from src.qualitative.llm_provider import QualitativeInterpretation

def test_html_cleaning():
    raw_html = "<html><body><p>Risk factors include market volatility.</p><table><tr><td>Noise</td></tr></table></body></html>"
    cleaned = clean_html(raw_html)
    assert "Risk factors include market volatility." in cleaned
    assert "Noise" not in cleaned

def test_preprocessing():
    text = "This   is \xa0 a test."
    processed = preprocess_text(text)
    assert processed == "This is a test."

def test_section_extraction_primary():
    long_text = "x" * (MIN_RISK_FACTORS_LENGTH + 1)
    text = f"Item 1A. Risk Factors\n{long_text}\nItem 1B. Unresolved Staff Comments"
    result = extract_risk_factors("TEST", "0001", text)
    assert result.state == ExtractionState.SUCCESS
    assert len(result.raw_text) >= MIN_RISK_FACTORS_LENGTH

def test_section_extraction_fallback():
    long_text = "x" * (MIN_RISK_FACTORS_LENGTH + 1)
    text = f"Risk Factors\n{long_text}\nProperties"
    result = extract_risk_factors("TEST", "0001", text)
    assert result.state == ExtractionState.FALLBACK_USED
    assert len(result.raw_text) >= MIN_RISK_FACTORS_LENGTH

def test_section_extraction_failure_state():
    text = "No risk factors here, just a short text."
    result = extract_risk_factors("TEST", "0001", text)
    assert result.state == ExtractionState.EXTRACTION_FAILED
    assert result.raw_text == ""

def test_chunking():
    text = "This is a sentence. This is another long enough sentence."
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert chunks[0] == "This is a sentence."

def test_deterministic_semantic_diff():
    # Mock embeddings: perfectly orthogonal means sim=0, same means sim=1
    emb_n = np.array([[1.0, 0.0], [0.0, 1.0]])
    sentences_n = ["S1", "S2"]
    
    emb_n_1 = np.array([[1.0, 0.0], [0.707, 0.707]]) # S1 is identical, S2 is similar (sim~0.7)
    sentences_n_1 = ["S1", "S3"]
    
    # default threshold 0.8
    diff = compute_semantic_diff(sentences_n, emb_n, sentences_n_1, emb_n_1, threshold_added=0.8, threshold_identical=0.95)
    
    assert "S2" in diff["added"]
    assert "S3" in diff["deleted"]
    assert "S1" not in diff["added"] and "S1" not in diff["deleted"] and "S1" not in diff["shifted"]

def test_threshold_sensitivity():
    emb_n = np.array([[0.8, 0.6]]) # Vector length 1
    sentences_n = ["New risk"]
    
    emb_n_1 = np.array([[1.0, 0.0]]) # Vector length 1
    sentences_n_1 = ["Old risk"]
    
    # Cosine similarity is 0.8
    for t in [0.65, 0.70, 0.75]:
        # sim (0.8) > t, so it is NOT added/deleted
        diff = compute_semantic_diff(sentences_n, emb_n, sentences_n_1, emb_n_1, threshold_added=t)
        assert len(diff["added"]) == 0
        
    for t in [0.85]:
        # sim (0.8) < t, so it IS added/deleted
        diff = compute_semantic_diff(sentences_n, emb_n, sentences_n_1, emb_n_1, threshold_added=t)
        assert len(diff["added"]) == 1

def test_structured_llm_validation():
    # Test pydantic model directly
    interp = QualitativeInterpretation(
        summary="Risk increased.",
        risk_level_change="INCREASED",
        key_themes=["Cybersecurity"]
    )
    assert interp.risk_level_change == "INCREASED"
