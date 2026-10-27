"""
AQQIP Feature Builder

Constructs market and accounting features.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


class FeatureBuilder:
    """Builds quantitative features from raw market and fundamental data."""

    @staticmethod
    def build_market_features(market_df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """
        Computes 1-day, 5-day, 20-day, 60-day returns, volatility, range, etc.
        market_df must be sorted by trade_date ascending.
        """
        df = market_df.copy()
        
        # Sort properly
        df = df.sort_values(['ticker', 'trade_date'])
        
        # Returns
        for window in [1, 5, 20, 60]:
            df[f'return_{window}d'] = df.groupby('ticker')[price_col].pct_change(periods=window)
            
        # Volatility (20-day rolling standard deviation of daily returns)
        df['volatility_20d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(20).std())
        
        # Range (High - Low) / Close
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
            df['avg_range_20d'] = df.groupby('ticker')['daily_range_pct'].transform(lambda x: x.rolling(20).mean())
            
        # Volume z-score (20-day)
        if 'volume' in df.columns:
            rolling_mean = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
            rolling_std = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).std())
            df['volume_z_score_20d'] = (df['volume'] - rolling_mean) / rolling_std

        # Drawdown (from 60-day high)
        rolling_max = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(60).max())
        df['drawdown_60d'] = (df[price_col] - rolling_max) / rolling_max

        return df

    @staticmethod
    def build_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Computes point-in-time cross-sectional relative features across all tickers for each trade_date:
        - return_20d_rel: return_20d minus universe mean return_20d
        - vol_ratio_20d: volatility_20d divided by universe mean volatility_20d
        Zero forward leakage: computed strictly cross-sectionally per trade_date.
        """
        df = panel.copy()
        if 'return_20d' in df.columns:
            cs_mean_ret = df.groupby('trade_date')['return_20d'].transform('mean')
            df['return_20d_rel'] = df['return_20d'] - cs_mean_ret
        if 'volatility_20d' in df.columns:
            cs_mean_vol = df.groupby('trade_date')['volatility_20d'].transform('mean')
            df['vol_ratio_20d'] = (df['volatility_20d'] / cs_mean_vol.replace(0, np.nan)).fillna(1.0).clip(0.1, 10.0)
        return df

    @staticmethod
    def extract_fundamental_features(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts pre-computed Beneish and Altman ratios into flat features.
        """
        df = fundamentals_df.copy()
        # In a real scenario we'd parse the JSON or columns for Beneish components
        # For now, ensure Beneish M-Score and Altman Z-Score are columns.
        expected = ['beneish_m_score', 'altman_z_score']
        for col in expected:
            if col not in df.columns:
                df[col] = np.nan
        return df

    @staticmethod
    def extract_qualitative_features(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts pre-computed qualitative filing change features into flat numeric features.
        """
        df = panel.copy()
        expected = [
            'rf_length_change_pct',
            'rf_cosine_similarity',
            'rf_change_magnitude',
            'rf_added_sentences_count',
            'rf_deleted_sentences_count',
            'rf_added_sentences_ratio',
            'mda_length_change_pct',
            'mda_cosine_similarity',
            'mda_change_magnitude'
        ]
        for col in expected:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

