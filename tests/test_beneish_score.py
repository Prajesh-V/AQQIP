"""
Tests for Beneish M-Score.
"""
from src.quant.beneish_score import compute_beneish_m_score
import math

def test_beneish_normal_values():
    result = compute_beneish_m_score(
        current_sales=110, prior_sales=100,
        current_cogs=50, prior_cogs=45,
        current_receivables=20, prior_receivables=15,
        current_current_assets=50, prior_current_assets=45,
        current_ppe=100, prior_ppe=95,
        current_depreciation=10, prior_depreciation=10,
        current_total_assets=200, prior_total_assets=180,
        current_sg_a=15, prior_sg_a=15,
        current_net_income=20, current_cash_flow_op=25,
        current_current_liabilities=30, prior_current_liabilities=25,
        current_long_term_debt=40, prior_long_term_debt=40
    )
    
    assert result["M_SCORE"] is not None
    assert isinstance(result["M_SCORE"], float)
    assert not math.isnan(result["M_SCORE"])
    assert result["SGI"] == 1.10  # 110 / 100

def test_beneish_zero_denominator():
    # If prior sales is 0, SGI should fallback to default 1.0 or safe division
    result = compute_beneish_m_score(
        current_sales=110, prior_sales=0,
        current_cogs=50, prior_cogs=45,
        current_receivables=20, prior_receivables=15,
        current_current_assets=50, prior_current_assets=45,
        current_ppe=100, prior_ppe=95,
        current_depreciation=10, prior_depreciation=10,
        current_total_assets=200, prior_total_assets=180,
        current_sg_a=15, prior_sg_a=15,
        current_net_income=20, current_cash_flow_op=25,
        current_current_liabilities=30, prior_current_liabilities=25,
        current_long_term_debt=40, prior_long_term_debt=40
    )
    assert result["SGI"] == 1.0
    assert not math.isnan(result["M_SCORE"])
