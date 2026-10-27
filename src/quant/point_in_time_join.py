"""
AQQIP Point-in-Time Joiner

Constructs features for a given prediction timestamp `t` without leaking future information.
Strictly enforces `publication_date <= t`.
"""
from __future__ import annotations

import datetime
from typing import Dict, Any, List
import pandas as pd

class PointInTimeJoiner:
    """
    Joins market data and fundamental data ensuring no forward-looking leakage.
    """

    @staticmethod
    def get_latest_available_fundamentals(
        fundamentals_df: pd.DataFrame, 
        as_of_date: datetime.date
    ) -> pd.Series | None:
        """
        Given a dataframe of fundamental records (must contain 'filing_date' representing publication),
        returns the most recently published record as of `as_of_date`.
        """
        if fundamentals_df.empty or 'filing_date' not in fundamentals_df.columns:
            return None

        # Filter out anything published after as_of_date
        # Ensure we are comparing dates to dates
        fundamentals_df = fundamentals_df.copy()
        if pd.api.types.is_datetime64_any_dtype(fundamentals_df['filing_date']):
            fundamentals_df['filing_date'] = fundamentals_df['filing_date'].dt.date

        valid_records = fundamentals_df[fundamentals_df['filing_date'] <= as_of_date]

        if valid_records.empty:
            return None

        # Sort by filing date descending and pick the top one
        valid_records = valid_records.sort_values(by='filing_date', ascending=False)
        return valid_records.iloc[0]

    @staticmethod
    def construct_panel(
        market_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        qualitative_df: Optional[pd.DataFrame] = None,
        conservative_availability_lag_days: int = 1
    ) -> pd.DataFrame:
        """
        Constructs a point-in-time panel.
        For each trade_date in market_df, joins the fundamental and qualitative
        features available on or strictly before that trade date.
        
        conservative_availability_lag_days (default 1):
        Applies a conservative publication lag to ensure that filings submitted
        during or after trading hours on filing_date are only accessible starting
        on the subsequent trading session.
        """
        if market_df.empty:
            return pd.DataFrame()

        panel = market_df.copy()
        panel['trade_date_ts'] = pd.to_datetime(panel['trade_date']).astype("datetime64[ns]")
        panel = panel.sort_values('trade_date_ts')

        lag = pd.Timedelta(days=conservative_availability_lag_days) if conservative_availability_lag_days > 0 else pd.Timedelta(0)

        # 1. Join Fundamentals (backward point-in-time)
        if fundamentals_df is not None and not fundamentals_df.empty:
            funds_df = fundamentals_df.copy()
            funds_df['filing_date_ts'] = (pd.to_datetime(funds_df['filing_date']) + lag).astype("datetime64[ns]")
            funds_df = funds_df.sort_values('filing_date_ts')

            panel = pd.merge_asof(
                panel,
                funds_df,
                left_on='trade_date_ts',
                right_on='filing_date_ts',
                by='ticker',
                direction='backward' # This ensures filing_date_ts <= trade_date_ts
            )
            if 'filing_date_ts' in panel.columns:
                panel = panel.drop(columns=['filing_date_ts'])

        # 2. Join Qualitative Filing Features (backward point-in-time)
        if qualitative_df is not None and not qualitative_df.empty:
            qual_df = qualitative_df.copy()
            qual_df['filing_date_qual_ts'] = (pd.to_datetime(qual_df['filing_date']) + lag).astype("datetime64[ns]")
            qual_df = qual_df.sort_values('filing_date_qual_ts')

            # Drop overlapping columns if already in panel (e.g. CIK, fiscal_year) to prevent suffixes
            overlap_cols = [c for c in qual_df.columns if c in panel.columns and c not in ('ticker', 'filing_date_qual_ts')]
            if overlap_cols:
                qual_df = qual_df.drop(columns=overlap_cols)

            panel = pd.merge_asof(
                panel,
                qual_df,
                left_on='trade_date_ts',
                right_on='filing_date_qual_ts',
                by='ticker',
                direction='backward' # This ensures filing_date <= trade_date
            )
            if 'filing_date_qual_ts' in panel.columns:
                panel = panel.drop(columns=['filing_date_qual_ts'])

        # Drop the temporary TS column
        if 'trade_date_ts' in panel.columns:
            panel = panel.drop(columns=['trade_date_ts'])
            
        return panel

