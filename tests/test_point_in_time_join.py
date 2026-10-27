"""
Tests for Point-In-Time Joiner.
"""
import pandas as pd
import datetime
from src.quant.point_in_time_join import PointInTimeJoiner

def test_point_in_time_join():
    market = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL', 'AAPL'],
        'trade_date': [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1), datetime.date(2023, 3, 1)],
        'price': [100, 110, 120]
    })
    
    funds = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL'],
        'filing_date': [datetime.date(2023, 1, 15), datetime.date(2023, 2, 15)],
        'eps': [1.5, 2.0]
    })
    
    panel = PointInTimeJoiner.construct_panel(market, funds)
    
    assert len(panel) == 3
    # Jan 1: no filing yet (earliest is Jan 15) -> eps is NaN
    assert pd.isna(panel.loc[panel['trade_date'] == datetime.date(2023, 1, 1), 'eps'].iloc[0])
    
    # Feb 1: filing from Jan 15 is available -> eps is 1.5
    assert panel.loc[panel['trade_date'] == datetime.date(2023, 2, 1), 'eps'].iloc[0] == 1.5
    
    # Mar 1: filing from Feb 15 is available -> eps is 2.0
    assert panel.loc[panel['trade_date'] == datetime.date(2023, 3, 1), 'eps'].iloc[0] == 2.0
