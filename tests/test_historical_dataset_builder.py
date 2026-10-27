"""
Tests for Historical Dataset Builder.
"""
import pandas as pd
import datetime
import numpy as np
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder

def test_split_and_label():
    builder = HistoricalDatasetBuilder()
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    df = pd.DataFrame({
        'ticker': ['AAPL'] * len(dates),
        'trade_date': dates.date,
        'forward_return_pct': np.random.normal(0, 5, len(dates))
    })
    
    # Manually insert a known extreme drop in the train set to verify threshold fitting
    df.loc[10, 'forward_return_pct'] = -20.0
    
    train, val, test = builder.split_and_label(df, val_start='2023-08-01', test_start='2023-11-01', alpha=0.1)
    
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    
    assert train['trade_date'].max() < datetime.date(2023, 8, 1)
    assert val['trade_date'].min() >= datetime.date(2023, 8, 1)
    assert test['trade_date'].min() >= datetime.date(2023, 11, 1)
    
    # All should have the exact same target_threshold_used
    train_thresh = train['target_threshold_used'].iloc[0]
    assert np.isclose(train_thresh, val['target_threshold_used'].iloc[0])
    assert np.isclose(train_thresh, test['target_threshold_used'].iloc[0])
