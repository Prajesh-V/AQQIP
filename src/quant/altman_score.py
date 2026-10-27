"""
AQQIP Altman Z-Score Calculation

Computes the Altman Z-Score to predict probability of bankruptcy.
Specifically uses the Double Prime (Z'') model, which is applicable 
to non-manufacturing and emerging market companies, making it more robust 
across a broader universe than the original manufacturer-only Z-Score.
"""
from __future__ import annotations

from typing import Dict, Optional
import math


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division avoiding division by zero. Default 0 for additive terms."""
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def compute_altman_z_double_prime_score(
    current_assets: float, current_liabilities: float,
    retained_earnings: float, ebit: float,
    book_value_equity: float, total_liabilities: float,
    total_assets: float
) -> Dict[str, Optional[float]]:
    """
    Computes Altman Z''-Score (Double Prime) model components.
    Formula: Z'' = 6.56(X1) + 3.26(X2) + 6.72(X3) + 1.05(X4)
    
    X1: Working Capital / Total Assets
    X2: Retained Earnings / Total Assets
    X3: EBIT / Total Assets
    X4: Book Value of Equity / Total Liabilities
    
    Note: The Sales/Total Assets term (X5) is omitted in Z'' to minimize 
    industry effects.
    """
    if total_assets == 0:
        return {
            "X1": 0.0, "X2": 0.0, "X3": 0.0, "X4": 0.0, "Z_DOUBLE_PRIME": None
        }

    working_capital = current_assets - current_liabilities
    
    x1 = safe_divide(working_capital, total_assets)
    x2 = safe_divide(retained_earnings, total_assets)
    x3 = safe_divide(ebit, total_assets)
    x4 = safe_divide(book_value_equity, total_liabilities)

    z_score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    return {
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "X4": x4,
        "Z_DOUBLE_PRIME": z_score
    }
