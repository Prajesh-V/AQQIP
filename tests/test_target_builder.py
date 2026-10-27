"""
Tests for Target Builder.
"""
import pandas as pd
import datetime
import numpy as np
from src.quant.target_builder import TargetBuilder
from unittest.mock import patch, MagicMock

def test_compute_forward_returns():
    builder = TargetBuilder()
    builder.horizon = 2
    builder.price_field = 'close'
    
    df = pd.DataFrame({
        'ticker': ['MSFT'] * 4,
        'trade_date': [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 4)
        ],
        'close': [100.0, 105.0, 90.0, 85.0]
    })
    
    returns_df = builder.compute_forward_returns(df)
    
    # Day 1 -> 2 days forward is Day 3 (90.0)
    # Return = (90 - 100) / 100 = -10%
    day1_ret = returns_df.loc[returns_df['trade_date'] == datetime.date(2023, 1, 1), 'forward_return_pct'].iloc[0]
    assert np.isclose(day1_ret, -10.0)
    
    # Day 3 and Day 4 will have NaN future prices because horizon > remaining days
    assert pd.isna(returns_df.loc[returns_df['trade_date'] == datetime.date(2023, 1, 3), 'forward_return_pct'].iloc[0])

def test_target_leakage_prevention():
    builder = TargetBuilder()
    
    train_returns = pd.Series([-5.0, 0.0, 5.0, -15.0])
    # numpy.quantile of [-15, -5, 0, 5] at 0.25 is -7.5
    threshold = builder.fit_threshold(train_returns, 0.25)
    assert np.isclose(threshold, -7.5)
    
    # Apply to a test set
    test_df = pd.DataFrame({
        'forward_return_pct': [-2.0, -10.0, 1.0]
    })
    
    labeled = builder.apply_labels(test_df, threshold)
    assert labeled.iloc[0]['target_label'] == 0 # -2.0 > -5.0
    assert labeled.iloc[1]['target_label'] == 1 # -10.0 <= -5.0
    assert labeled.iloc[2]['target_label'] == 0 # 1.0 > -5.0
