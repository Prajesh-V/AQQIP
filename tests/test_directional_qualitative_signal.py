"""
Unit tests for Directional Qualitative Risk Signal Analyzer

Verifies:
1. Large textual change does NOT automatically generate downside risk (unsigned change intensity != adverse risk).
2. Genuine adverse disclosures (material weakness, default, investigation) produce elevated directional risk.
3. Missing or ungrounded interpretations are treated as missing, not as negative evidence.
4. Grounded evidence snippets and provenance fields are preserved.
"""
import pytest
from src.qualitative.directional_signal import (
    DirectionalQualitativeRiskAnalyzer,
    DirectionalRiskResult
)


def test_benign_large_textual_change_does_not_produce_high_risk():
    """
    Proves that a company with a high textual change intensity (e.g. 0.85)
    due to routine expansion, website updates, or marketing restructuring
    does NOT receive an elevated downside risk score.
    """
    analyzer = DirectionalQualitativeRiskAnalyzer()
    
    benign_sentences = [
        "We expanded into our new European markets with positive reception.",
        "We launched new commercial initiatives and updated our website address for customer convenience.",
        "A routine update was made to general administrative contact procedures across regional offices.",
        "We are continuing our strategic growth plan by increasing commercialization efforts."
    ]
    
    result = analyzer.analyze(
        added_sentences=benign_sentences,
        unsigned_change_magnitude=0.85,  # High unsigned change intensity
        is_available=True
    )
    
    assert result.unsigned_change_intensity == 0.85
    assert result.directional_risk_score == 0.0
    assert result.risk_direction == "BENIGN_OR_UNCHANGED"
    assert result.severe_adverse_count == 0
    assert result.moderate_adverse_count == 0
    assert result.benign_count >= 3
    assert result.is_missing is False
    assert result.is_grounded is True


def test_adverse_disclosures_produce_elevated_risk():
    """
    Proves that genuine adverse events (material weakness, formal investigation, default)
    produce a severe adverse directional risk score, along with grounded evidence snippets.
    """
    analyzer = DirectionalQualitativeRiskAnalyzer()
    
    adverse_sentences = [
        "Management identified material weaknesses in our internal control over financial reporting.",
        "The company has received a subpoena from the DOJ regarding a formal investigation into revenue recognition.",
        "We have defaulted under our primary credit agreement resulting in covenant violations."
    ]
    
    result = analyzer.analyze(
        added_sentences=adverse_sentences,
        unsigned_change_magnitude=0.15,  # Moderate/small change magnitude
        is_available=True
    )
    
    assert result.unsigned_change_intensity == 0.15
    assert result.directional_risk_score >= 0.8
    assert result.risk_direction == "SEVERE_ADVERSE"
    assert result.severe_adverse_count == 3
    assert len(result.evidence_snippets) >= 2
    assert any("material weakness" in s.lower() for s in result.evidence_snippets)
    assert any("subpoena" in s.lower() for s in result.evidence_snippets)


def test_unavailable_evidence_is_treated_as_missing_not_negative():
    """
    Proves that when qualitative filing features are missing or ungrounded,
    the score is 0.0 and flagged as missing, NEVER treated as negative evidence.
    """
    analyzer = DirectionalQualitativeRiskAnalyzer()
    
    result = analyzer.analyze(
        added_sentences=None,
        raw_text_change=None,
        unsigned_change_magnitude=0.0,
        is_available=False
    )
    
    assert result.is_missing is True
    assert result.directional_risk_score == 0.0
    assert result.risk_direction == "MISSING"
    assert result.extraction_status == "MISSING_EVIDENCE"
    assert len(result.evidence_snippets) == 0


def test_empty_added_sentences_with_zero_changes():
    """
    Proves that unchanged filings receive benign status and 0.0 risk score.
    """
    analyzer = DirectionalQualitativeRiskAnalyzer()
    
    result = analyzer.analyze(
        added_sentences=[],
        raw_text_change="",
        unsigned_change_magnitude=0.0,
        is_available=True
    )
    
    assert result.is_missing is False
    assert result.directional_risk_score == 0.0
    assert result.risk_direction == "BENIGN_OR_UNCHANGED"
    assert result.extraction_status == "SUCCESS"
