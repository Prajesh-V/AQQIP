"""
Tests for Leakage Validator.
"""
import pandas as pd
import datetime
from src.quant.leakage_validator import LeakageValidator

def test_leakage_validator_catches_future_publication():
    panel = pd.DataFrame({
        'trade_date': [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)],
        'filing_date': [datetime.date(2022, 12, 1), datetime.date(2023, 2, 15)] # Leakage on row 2!
    })
    
    errors = LeakageValidator.validate_panel(panel)
    assert len(errors) > 0
    assert "Found 1 rows where filing_date > trade_date" in errors[0]

def test_leakage_validator_clean():
    panel = pd.DataFrame({
        'trade_date': [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)],
        'filing_date': [datetime.date(2022, 12, 1), datetime.date(2023, 1, 15)]
    })
    
    errors = LeakageValidator.validate_panel(panel)
    assert len(errors) == 0


def test_validate_split_boundaries():
    train = pd.DataFrame({'trade_date': [datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)]})
    val = pd.DataFrame({'trade_date': [datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)]})
    test = pd.DataFrame({'trade_date': [datetime.date(2025, 1, 1), datetime.date(2025, 12, 31)]})
    
    # Clean chronological splits
    assert len(LeakageValidator.validate_split_boundaries(train, val, test)) == 0
    
    # Leaky overlapping split
    val_leaky = pd.DataFrame({'trade_date': [datetime.date(2023, 11, 1), datetime.date(2024, 12, 31)]})
    errs = LeakageValidator.validate_split_boundaries(train, val_leaky, test)
    assert len(errs) > 0
    assert "Train max date" in errs[0]
