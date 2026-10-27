"""
Tests for Feature Builder.
"""
import pandas as pd
import datetime
import numpy as np
from src.quant.feature_builder import FeatureBuilder

def test_build_market_features():
    df = pd.DataFrame({
        'ticker': ['AAPL'] * 5,
        'trade_date': [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 4),
            datetime.date(2023, 1, 5)
        ],
        'adj_close': [100.0, 105.0, 102.0, 110.0, 115.0],
        'close': [100.0, 105.0, 102.0, 110.0, 115.0],
        'high': [102.0, 106.0, 104.0, 112.0, 118.0],
        'low': [98.0, 101.0, 100.0, 108.0, 114.0],
        'volume': [100, 200, 150, 300, 250]
    })
    
    features = FeatureBuilder.build_market_features(df)
    
    assert 'return_1d' in features.columns
    # day 2 return_1d should be 0.05
    assert np.isclose(features.loc[1, 'return_1d'], 0.05)
    
    assert 'daily_range_pct' in features.columns
    # day 1 range: (102 - 98) / 100 = 0.04
    assert np.isclose(features.loc[0, 'daily_range_pct'], 0.04)
