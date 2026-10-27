from __future__ import annotations

import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QualitativeSignal(BaseModel):
    """
    Qualitative signal representing semantic shift between two SEC filings.
    """
    ticker: str
    cik: str
    section: str = "risk_factors"
    year_from: int
    year_to: int
    accession_from: str
    accession_to: str
    cosine_similarity: float
    semantic_shift_score: float
    added_sentences: List[str]
    deleted_sentences: List[str]
    shifted_sentences: List[str]
    llm_summary: str
    llm_provider: str
    llm_model: str
    generated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    # SignalBundle adapter properties
    @property
    def added_count(self) -> int:
        return len(self.added_sentences)
        
    @property
    def deleted_count(self) -> int:
        return len(self.deleted_sentences)
        
    @property
    def shifted_count(self) -> int:
        return len(self.shifted_sentences)
        
    @property
    def filing_year_from(self) -> int:
        return self.year_from
        
    @property
    def filing_year_to(self) -> int:
        return self.year_to
        
    @property
    def source_accession(self) -> str:
        return self.accession_to

class ExtractionState(str, Enum):
    """Lifecycle states for extraction."""
    SUCCESS = "SUCCESS"
    FALLBACK_USED = "FALLBACK_USED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"

class ExtractedSection(BaseModel):
    """Result of section extraction."""
    ticker: str
    accession_number: str
    section_name: str
    raw_text: str
    state: ExtractionState
    chunk_ids: List[str] = Field(default_factory=list)
