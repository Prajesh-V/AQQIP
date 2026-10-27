"""
Tests for Altman Z-Score.
"""
from src.quant.altman_score import compute_altman_z_double_prime_score
import math

def test_altman_normal_values():
    result = compute_altman_z_double_prime_score(
        current_assets=100, current_liabilities=50,
        retained_earnings=40, ebit=20,
        book_value_equity=150, total_liabilities=80,
        total_assets=300
    )
    assert result["Z_DOUBLE_PRIME"] is not None
    assert result["X1"] == (100 - 50) / 300
    assert result["X2"] == 40 / 300
    assert result["X3"] == 20 / 300
    assert result["X4"] == 150 / 80

def test_altman_zero_assets():
    result = compute_altman_z_double_prime_score(
        current_assets=100, current_liabilities=50,
        retained_earnings=40, ebit=20,
        book_value_equity=150, total_liabilities=80,
        total_assets=0
    )
    assert result["Z_DOUBLE_PRIME"] is None
