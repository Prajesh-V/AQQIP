"""
AQQIP Ingestion Models

Pydantic schemas for data validation during ingestion.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TickerInfo(BaseModel):
    """Represents a tracked company."""
    ticker: str
    cik: str
    company_name: Optional[str] = None


class FilingState(str, Enum):
    """Lifecycle states for a filing."""
    DISCOVERED = "DISCOVERED"
    FETCHING = "FETCHING"
    RAW_STORED = "RAW_STORED"
    READY_FOR_PROCESSING = "READY_FOR_PROCESSING"
    FETCH_FAILED = "FETCH_FAILED"
    INVALID_METADATA = "INVALID_METADATA"
    DUPLICATE = "DUPLICATE"
    PROCESSING_BLOCKED = "PROCESSING_BLOCKED"


class SECFilingMetadata(BaseModel):
    """Metadata for an SEC filing."""
    ticker: str
    cik: str
    accession_number: str
    filing_type: str
    filing_date: datetime.date
    report_date: Optional[datetime.date] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    source_url: str
    processing_status: FilingState = FilingState.DISCOVERED
    ingested_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    raw_content: Optional[str] = None


class OHLCVRow(BaseModel):
    """Validated OHLCV market data row."""
    ticker: str
    trade_date: datetime.date
    open_price: float = Field(alias="open")
    high_price: float = Field(alias="high")
    low_price: float = Field(alias="low")
    close_price: float = Field(alias="close")
    adj_close_price: float = Field(alias="adj_close")
    volume: int
    source: str = "yfinance"

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Volume cannot be negative")
        return v

    @field_validator("high_price")
    @classmethod
    def validate_high(cls, v: float, info) -> float:
        open_p = info.data.get("open_price")
        if open_p is not None and v < open_p:
            raise ValueError("High must be >= Open")
        return v

    @field_validator("low_price")
    @classmethod
    def validate_low(cls, v: float, info) -> float:
        high_p = info.data.get("high_price")
        open_p = info.data.get("open_price")
        if high_p is not None and v > high_p:
            raise ValueError("Low must be <= High")
        if open_p is not None and v > open_p:
            raise ValueError("Low must be <= Open")
        return v
