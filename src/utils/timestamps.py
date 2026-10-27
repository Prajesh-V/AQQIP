"""
AQQIP Timestamp Utilities

Standardizes time handling. All datetimes should be timezone-aware UTC.
All dates should be standard `datetime.date`.
"""
from __future__ import annotations

import datetime


def now_utc() -> datetime.datetime:
    """Return the current UTC datetime, timezone-aware."""
    return datetime.datetime.now(datetime.timezone.utc)


def today_utc() -> datetime.date:
    """Return the current UTC date."""
    return now_utc().date()


def normalize_date(date_obj: datetime.date | datetime.datetime | str | None) -> datetime.date | None:
    """Convert various date representations into a standard datetime.date."""
    if date_obj is None:
        return None
    if isinstance(date_obj, datetime.datetime):
        return date_obj.date()
    if isinstance(date_obj, datetime.date):
        return date_obj
    if isinstance(date_obj, str):
        # Try ISO format
        try:
            return datetime.date.fromisoformat(date_obj[:10])
        except ValueError:
            pass
    return None
