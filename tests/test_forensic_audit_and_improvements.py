"""
Unit and regression tests for forecasting forensic audit and model improvements.

Covers:
1. Target builder independent calculation and zero lookahead.
2. Conservative availability lag enforcement in PointInTimeJoiner.
3. Cross-sectional market feature builder zero lookahead and correctness.
4. Deployable Brier skill score computation vs train prevalence in evaluate_predictions_authoritative.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date

from src.quant.target_builder import TargetBuilder
from src.quant.point_in_time_join import PointInTimeJoiner
from src.quant.feature_builder import FeatureBuilder
from src.quant.benchmark_runner import evaluate_predictions_authoritative


def test_target_builder_forward_returns_exact_contract():
    """Verify that TargetBuilder computes exact 20-session forward returns without lookahead."""
    dates = pd.bdate_range(start="2020-01-01", periods=30)
    prices = [100.0 * (1.01 ** i) for i in range(30)]
    df = pd.DataFrame({
        "ticker": ["TEST"] * 30,
        "trade_date": dates,
        "adj_close": prices
    })
    
    tb = TargetBuilder()
    forward_df = tb.compute_forward_returns(df)
    
    # First row: return should be (prices[20] - prices[0]) / prices[0] * 100.0
    expected_ret_0 = (prices[20] - prices[0]) / prices[0] * 100.0
    actual_ret_0 = forward_df.iloc[0]["forward_return_pct"]
    assert np.isclose(expected_ret_0, actual_ret_0, atol=1e-5)
    
    # Last 20 rows must have NaN future_price and forward_return_pct
    assert forward_df.iloc[-20:]["future_price"].isna().all()
    assert forward_df.iloc[-20:]["forward_return_pct"].isna().all()
    
    # Check labeling
    threshold = tb.fit_threshold(forward_df["forward_return_pct"], alpha=0.10)
    labeled = tb.apply_labels(forward_df, threshold)
    # Labeled rows should equal 30 - 20 = 10 rows
    assert len(labeled) == 10
    assert "target_label" in labeled.columns


def test_point_in_time_conservative_availability_lag():
    """Verify that filings made on trade date D are NOT visible on date D when conservative lag is enabled."""
    market_df = pd.DataFrame({
        "ticker": ["TEST", "TEST", "TEST"],
        "trade_date": pd.to_datetime(["2020-01-10", "2020-01-13", "2020-01-14"]),
        "adj_close": [100.0, 102.0, 103.0]
    })
    
    # Fundamental filing occurs on 2020-01-13
    fund_df = pd.DataFrame({
        "ticker": ["TEST"],
        "filing_date": pd.to_datetime(["2020-01-13"]),
        "altman_z_score": [3.5],
        "beneish_m_score": [-2.5]
    })
    
    joiner = PointInTimeJoiner()
    panel = joiner.construct_panel(market_df, fund_df, conservative_availability_lag_days=1)
    
    row_10 = panel[panel["trade_date"] == "2020-01-10"].iloc[0]
    row_13 = panel[panel["trade_date"] == "2020-01-13"].iloc[0]
    row_14 = panel[panel["trade_date"] == "2020-01-14"].iloc[0]
    
    # On 2020-01-10: prior to filing, must be NaN
    assert np.isnan(row_10["altman_z_score"])
    
    # On 2020-01-13 (filing day): with conservative 1-day lag, filing is not yet usable at market close
    assert np.isnan(row_13["altman_z_score"])
    
    # On 2020-01-14: fully available and merged
    assert row_14["altman_z_score"] == 3.5


def test_cross_sectional_features_zero_lookahead():
    """Verify cross-sectional features only use cross-sectional peers on the same trade date."""
    dates = pd.date_range("2020-01-01", periods=5)
    records = []
    for d in dates:
        records.append({"ticker": "A", "trade_date": d, "return_20d": 0.05, "volatility_20d": 0.02})
        records.append({"ticker": "B", "trade_date": d, "return_20d": 0.15, "volatility_20d": 0.04})
        
    df = pd.DataFrame(records)
    res = FeatureBuilder.build_cross_sectional_features(df)
    
    # For day 0: mean return is (0.05 + 0.15)/2 = 0.10
    # A relative return = 0.05 - 0.10 = -0.05
    # B relative return = 0.15 - 0.10 = +0.05
    row_a = res[(res["ticker"] == "A") & (res["trade_date"] == dates[0])].iloc[0]
    row_b = res[(res["ticker"] == "B") & (res["trade_date"] == dates[0])].iloc[0]
    
    assert np.isclose(row_a["return_20d_rel"], -0.05, atol=1e-6)
    assert np.isclose(row_b["return_20d_rel"], 0.05, atol=1e-6)
    
    # vol_ratio: mean vol = 0.03. A vol ratio = 0.02 / 0.03 = 2/3. B vol ratio = 0.04 / 0.03 = 4/3.
    assert np.isclose(row_a["vol_ratio_20d"], 2.0 / 3.0, atol=1e-6)
    assert np.isclose(row_b["vol_ratio_20d"], 4.0 / 3.0, atol=1e-6)


def test_deployable_brier_skill_score():
    """Verify that deployable Brier skill score uses training prevalence, not test prevalence."""
    y_test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 20% test prevalence
    y_prob = np.array([0.1] * 10)  # Constant prediction of 10% (the train prevalence)
    
    train_prev = 0.10
    
    res = evaluate_predictions_authoritative(
        y_true=y_test,
        y_prob=y_prob,
        operational_threshold=0.50,
        train_prevalence=train_prev
    )
    
    # Since y_prob matches train_prev exactly, BSS vs train_prev must be exactly 0.0
    assert np.isclose(res["brier_skill_score_train_prev"], 0.0, atol=1e-4)
    # But BSS vs test prevalence is negative because test prevalence is 0.20
    assert res["brier_skill_score_test_prev"] < 0.0
