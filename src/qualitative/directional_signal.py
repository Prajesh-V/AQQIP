"""
AQQIP Directional Qualitative Risk Signal Module

Disentangles unsigned textual change intensity (magnitude of filing change)
from directional financial-adversity risk (evidence of deteriorating risk profile,
litigation, material weaknesses, default, or regulatory enforcement).

Ensures:
1. Filing-change magnitude is preserved as an unsigned change-intensity feature.
2. Directional risk is grounded strictly in textual evidence from added/modified disclosures.
3. Benign expansion or administrative rewriting does NOT produce a high adverse risk score.
4. Unavailable or ungrounded data is treated as missing, never as adverse evidence.
5. Zero paid API dependencies ($0 cost).
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

logger = get_logger("directional_signal")


# Loughran-McDonald inspired financial adversity patterns
SEVERE_ADVERSE_PATTERNS = [
    r"\bmaterial\s+weakness(es)?\b",
    r"\bdefault(ed|ing)?\b",
    r"\bbreach(ed|ing)?\s+of\s+covenant\b",
    r"\bcovenant\s+violation(s)?\b",
    r"\brestatement(s)?\b",
    r"\bgoing\s+concern\b",
    r"\binsolven(t|cy)\b",
    r"\bdelist(ed|ing)?\b",
    r"\bsubpoena(ed|s)?\b",
    r"\bformal\s+investigation(s)?\b",
    r"\bsec\s+investigation(s)?\b",
    r"\bdoj\s+investigation(s)?\b",
    r"\bcease\s+and\s+desist\b",
    r"\bimpairment\s+charge(s)?\b",
    r"\bliquidity\s+shortfall(s)?\b",
    r"\breceivership\b",
    r"\bchapter\s+11\b",
    r"\bbankrupt(cy)?\b",
]

MODERATE_ADVERSE_PATTERNS = [
    r"\bmaterial\s+adverse\s+(effect|impact)\b",
    r"\blitigat(ion|ed|ing)\b",
    r"\blawsuit(s)?\b",
    r"\bregulatory\s+(sanction|action|proceeding)(s)?\b",
    r"\bcybersecurity\s+(breach|incident|attack)(s)?\b",
    r"\bsupply\s+chain\s+disruption(s)?\b",
    r"\bpenalt(y|ies)\b",
    r"\baccelerat(ed|ion)\s+of\s+(indebtedness|debt)\b",
    r"\bcredit\s+rating\s+downgrade(s)?\b",
]

BENIGN_EXPANSION_PATTERNS = [
    r"\bexpand(ed|ing|sion)\s+(into|our|product|market|footprint)\b",
    r"\bnew\s+market(s)?\b",
    r"\blaunch(ed|ing)?\s+new\b",
    r"\bincreas(ed|ing)\s+commercialization\b",
    r"\bstrategic\s+growth\b",
    r"\broutine\s+update(s)?\b",
    r"\bwebsite\s+(address|url)\b",
    r"\breorganiz(ed|ation)\s+of\s+marketing\b",
    r"\bgeneral\s+administrative\b",
]


class DirectionalRiskResult(BaseModel):
    """Result of directional qualitative risk analysis."""
    directional_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Directional adverse risk score in [0.0, 1.0]. High = adverse risk."
    )
    unsigned_change_intensity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Unsigned change intensity / magnitude in [0.0, 1.0]."
    )
    risk_direction: str = Field(
        ...,
        description="'BENIGN_OR_UNCHANGED', 'MODERATE_ADVERSE', 'SEVERE_ADVERSE', or 'MISSING'"
    )
    evidence_snippets: List[str] = Field(
        default_factory=list,
        description="Grounded sentence quotes supporting the directional assessment."
    )
    severe_adverse_count: int = 0
    moderate_adverse_count: int = 0
    benign_count: int = 0
    model_provider: str = "lexicon_grounded_baseline"
    model_version: str = "v1.0.0"
    prompt_version: Optional[str] = None
    is_grounded: bool = True
    is_missing: bool = False
    extraction_status: str = "SUCCESS"


class DirectionalQualitativeRiskAnalyzer:
    """
    Interprets SEC filing textual changes to evaluate directional risk
    without equating mere text churn with adverse deterioration.
    """

    def __init__(self, prompt_version: str = "aqqip_risk_prompt_v1"):
        self.prompt_version = prompt_version
        self.severe_regex = [re.compile(p, re.IGNORECASE) for p in SEVERE_ADVERSE_PATTERNS]
        self.moderate_regex = [re.compile(p, re.IGNORECASE) for p in MODERATE_ADVERSE_PATTERNS]
        self.benign_regex = [re.compile(p, re.IGNORECASE) for p in BENIGN_EXPANSION_PATTERNS]

    def analyze(
        self,
        added_sentences: Optional[List[str]] = None,
        raw_text_change: Optional[str] = None,
        unsigned_change_magnitude: float = 0.0,
        is_available: bool = True
    ) -> DirectionalRiskResult:
        """
        Computes directional risk score and extracts supporting evidence.
        """
        # If qualitative evidence is unavailable or missing:
        if not is_available or (added_sentences is None and raw_text_change is None):
            return DirectionalRiskResult(
                directional_risk_score=0.0,
                unsigned_change_intensity=float(min(1.0, max(0.0, unsigned_change_magnitude))),
                risk_direction="MISSING",
                evidence_snippets=[],
                severe_adverse_count=0,
                moderate_adverse_count=0,
                benign_count=0,
                model_provider="lexicon_grounded_baseline",
                model_version="v1.0.0",
                is_grounded=False,
                is_missing=True,
                extraction_status="MISSING_EVIDENCE"
            )

        # Assemble sentences
        sentences: List[str] = []
        if added_sentences:
            sentences.extend(added_sentences)
        if raw_text_change:
            # Split raw text into rough sentence units
            split_s = re.split(r"(?<=[.!?])\s+", raw_text_change)
            sentences.extend([s.strip() for s in split_s if s.strip()])

        if not sentences:
            return DirectionalRiskResult(
                directional_risk_score=0.0,
                unsigned_change_intensity=float(min(1.0, max(0.0, unsigned_change_magnitude))),
                risk_direction="BENIGN_OR_UNCHANGED",
                evidence_snippets=[],
                is_grounded=True,
                is_missing=False,
                extraction_status="SUCCESS"
            )

        severe_hits: List[str] = []
        moderate_hits: List[str] = []
        benign_hits: List[str] = []

        for s in sentences:
            is_sev = any(rgx.search(s) for rgx in self.severe_regex)
            is_mod = any(rgx.search(s) for rgx in self.moderate_regex)
            is_ben = any(rgx.search(s) for rgx in self.benign_regex)

            if is_sev:
                severe_hits.append(s[:250])
            elif is_mod:
                moderate_hits.append(s[:250])
            if is_ben:
                benign_hits.append(s[:250])

        n_sev = len(severe_hits)
        n_mod = len(moderate_hits)
        n_ben = len(benign_hits)

        # Compute directional adverse risk score
        # Severe hits have high weight, moderate moderate weight, benign terms suppress false positives
        raw_score = (n_sev * 0.40) + (n_mod * 0.15) - (n_ben * 0.05)
        directional_score = float(min(1.0, max(0.0, raw_score)))

        if n_sev > 0 or directional_score >= 0.5:
            risk_direction = "SEVERE_ADVERSE"
        elif n_mod > 0 or directional_score >= 0.15:
            risk_direction = "MODERATE_ADVERSE"
        else:
            risk_direction = "BENIGN_OR_UNCHANGED"

        evidence = (severe_hits[:3] + moderate_hits[:3])

        return DirectionalRiskResult(
            directional_risk_score=round(directional_score, 4),
            unsigned_change_intensity=float(min(1.0, max(0.0, unsigned_change_magnitude))),
            risk_direction=risk_direction,
            evidence_snippets=evidence,
            severe_adverse_count=n_sev,
            moderate_adverse_count=n_mod,
            benign_count=n_ben,
            model_provider="lexicon_grounded_baseline",
            model_version="v1.0.0",
            prompt_version=self.prompt_version,
            is_grounded=True,
            is_missing=False,
            extraction_status="SUCCESS"
        )
