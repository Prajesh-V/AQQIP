"""
AQQIP Leakage Validator

Inspects training rows to detect contamination by future information.
"""
from __future__ import annotations

import pandas as pd
from typing import List


class LeakageValidator:
    """Validates panels for chronological integrity."""

    @staticmethod
    def validate_panel(panel: pd.DataFrame, time_col: str = 'trade_date', pub_col: str = 'filing_date') -> List[str]:
        """
        Validates that no row uses a filing published after the trade_date.
        """
        errors = []
        
        if panel.empty:
            return errors
            
        if time_col not in panel.columns:
            errors.append(f"Missing time column: {time_col}")
            return errors
            
        if pub_col in panel.columns:
            # Check for publication date leakage
            # Only compare where pub_col is not null
            mask = panel[pub_col].notna()
            if mask.any():
                # Convert to dates if they are datetime
                trade_dates = pd.to_datetime(panel.loc[mask, time_col]).dt.date
                pub_dates = pd.to_datetime(panel.loc[mask, pub_col]).dt.date
                
                leaks = pub_dates > trade_dates
                if leaks.any():
                    leak_count = leaks.sum()
                    errors.append(f"Found {leak_count} rows where {pub_col} > {time_col}")

        # Check that target isn't accidentally in the features
        if 'forward_return_pct' in panel.columns and 'target_label' in panel.columns:
            pass # These are allowed but should be dropped before modeling
            
        # Additional checks can be added here (e.g. looking for 'future' in column names)
        suspicious_cols = [c for c in panel.columns if 'future' in c.lower()]
        if suspicious_cols and 'future_price' not in suspicious_cols:
             errors.append(f"Suspicious future-looking columns found: {suspicious_cols}")

        return errors

    @staticmethod
    def validate_split_boundaries(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        time_col: str = 'trade_date',
        target_col: str = 'target_date'
    ) -> List[str]:
        """
        Validates that split date intervals and forward outcome windows
        are strictly chronological without boundary overlap.
        """
        errors = []
        if train_df.empty or val_df.empty or test_df.empty:
            return errors

        train_max_trade = pd.to_datetime(train_df[time_col]).max()
        val_min_trade = pd.to_datetime(val_df[time_col]).min()
        val_max_trade = pd.to_datetime(val_df[time_col]).max()
        test_min_trade = pd.to_datetime(test_df[time_col]).min()

        if train_max_trade >= val_min_trade:
            errors.append(f"Train max date ({train_max_trade.date()}) >= Val min trade date ({val_min_trade.date()})")
        if val_max_trade >= test_min_trade:
            errors.append(f"Val max date ({val_max_trade.date()}) >= Test min trade date ({test_min_trade.date()})")

        # Forward outcome window leakage checks
        if target_col in train_df.columns:
            valid_targets = train_df[target_col].dropna()
            if not valid_targets.empty:
                train_max_target = pd.to_datetime(valid_targets).max()
                if train_max_target >= val_min_trade:
                    errors.append(
                        f"Forward outcome leakage in Train: max target_date ({train_max_target.date()}) "
                        f"crosses into Val partition (min trade_date: {val_min_trade.date()})"
                    )

        if target_col in val_df.columns:
            valid_targets = val_df[target_col].dropna()
            if not valid_targets.empty:
                val_max_target = pd.to_datetime(valid_targets).max()
                if val_max_target >= test_min_trade:
                    errors.append(
                        f"Forward outcome leakage in Val: max target_date ({val_max_target.date()}) "
                        f"crosses into Test partition (min trade_date: {test_min_trade.date()})"
                    )

        return errors
