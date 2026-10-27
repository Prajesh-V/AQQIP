"""
AQQIP Target Builder

Constructs the prediction target based on future returns.
Implements the configured specification (H days, alpha quantile threshold).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config.settings import get_settings


class TargetBuilder:
    """Builds targets without leaking test set thresholds into training."""

    def __init__(self):
        self.settings = get_settings()
        self.horizon = self.settings.prediction_target.horizon_trading_days
        self.price_field = self.settings.prediction_target.price_field
        # We will use the configured threshold percentage directly for now
        # However, the user request says: "y_t = 1[R(t->t+H) <= q_alpha(train)]"
        # We will compute the rolling H-day return and use a threshold.
        self.default_threshold_pct = self.settings.prediction_target.sharp_drop_threshold_pct

    def compute_forward_returns(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes forward return over H days for each row.
        market_df must be sorted by trade_date ascending.
        Requires ['ticker', 'trade_date', price_field]
        """
        df = market_df.copy()
        # Ensure sorted
        df = df.sort_values(['ticker', 'trade_date'])
        
        # Calculate future price and future date: shift by -horizon grouped by ticker
        # e.g., if horizon is 20, the price and date 20 days in the future is shifted back to the current row
        df['future_price'] = df.groupby('ticker')[self.price_field].shift(-self.horizon)
        df['target_date'] = df.groupby('ticker')['trade_date'].shift(-self.horizon)
        
        # Calculate percentage return
        df['forward_return_pct'] = (df['future_price'] - df[self.price_field]) / df[self.price_field] * 100.0
        
        return df

    def fit_threshold(self, train_returns: pd.Series, alpha: float) -> float:
        """
        Fits the threshold on the training set distribution using quantile `alpha`.
        e.g., if alpha = 0.05, finds the 5th percentile return (a sharp drop).
        """
        valid_returns = train_returns.dropna()
        if valid_returns.empty:
            # Fallback to configured default if no data (e.g., in unit tests without enough rows)
            return -self.default_threshold_pct
        return float(np.quantile(valid_returns, alpha))

    def apply_labels(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Assigns binary target label.
        1 if forward_return_pct <= threshold (i.e. it dropped harder than the threshold)
        0 otherwise
        Drops rows where future_price is NaN (cannot be evaluated).
        """
        result = df.copy()
        
        # Cannot label records without a known future outcome
        result = result.dropna(subset=['forward_return_pct'])
        
        result['target_label'] = (result['forward_return_pct'] <= threshold).astype(int)
        result['target_threshold_used'] = threshold
        
        return result
