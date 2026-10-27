"""
AQQIP Beneish M-Score Calculation

Computes the Beneish M-Score using 8 variables to detect earnings manipulation.
"""
from __future__ import annotations

from typing import Optional, Dict
import math


def safe_divide(numerator: float, denominator: float, default: float = 1.0) -> float:
    """Safe division avoiding division by zero."""
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def compute_beneish_m_score(
    current_sales: float, prior_sales: float,
    current_cogs: float, prior_cogs: float,
    current_receivables: float, prior_receivables: float,
    current_current_assets: float, prior_current_assets: float,
    current_ppe: float, prior_ppe: float,
    current_depreciation: float, prior_depreciation: float,
    current_total_assets: float, prior_total_assets: float,
    current_sg_a: float, prior_sg_a: float,
    current_net_income: float, current_cash_flow_op: float,
    current_current_liabilities: float, prior_current_liabilities: float,
    current_long_term_debt: float, prior_long_term_debt: float
) -> Dict[str, Optional[float]]:
    """
    Computes Beneish M-Score components and the final score.
    Requires exactly two periods of data (current and prior).
    """
    # 1. DSRI (Days Sales in Receivables Index)
    current_dsr = safe_divide(current_receivables, current_sales)
    prior_dsr = safe_divide(prior_receivables, prior_sales)
    dsri = safe_divide(current_dsr, prior_dsr)

    # 2. GMI (Gross Margin Index)
    current_gm = safe_divide(current_sales - current_cogs, current_sales)
    prior_gm = safe_divide(prior_sales - prior_cogs, prior_sales)
    gmi = safe_divide(prior_gm, current_gm)

    # 3. AQI (Asset Quality Index)
    current_aq = safe_divide(current_total_assets - current_current_assets - current_ppe, current_total_assets)
    prior_aq = safe_divide(prior_total_assets - prior_current_assets - prior_ppe, prior_total_assets)
    aqi = safe_divide(current_aq, prior_aq)

    # 4. SGI (Sales Growth Index)
    sgi = safe_divide(current_sales, prior_sales)

    # 5. DEPI (Depreciation Index)
    current_dep_rate = safe_divide(current_depreciation, current_depreciation + current_ppe)
    prior_dep_rate = safe_divide(prior_depreciation, prior_depreciation + prior_ppe)
    depi = safe_divide(prior_dep_rate, current_dep_rate)

    # 6. SGAI (Sales General and Administrative Expenses Index)
    current_sga_sales = safe_divide(current_sg_a, current_sales)
    prior_sga_sales = safe_divide(prior_sg_a, prior_sales)
    sgai = safe_divide(current_sga_sales, prior_sga_sales)

    # 7. LVGI (Leverage Index)
    current_leverage = safe_divide(current_current_liabilities + current_long_term_debt, current_total_assets)
    prior_leverage = safe_divide(prior_current_liabilities + prior_long_term_debt, prior_total_assets)
    lvgi = safe_divide(current_leverage, prior_leverage)

    # 8. TATA (Total Accruals to Total Assets)
    tata = safe_divide(current_net_income - current_cash_flow_op, current_total_assets)

    # Calculate M-Score (8-variable model)
    m_score = (
        -4.84 
        + 0.920 * dsri 
        + 0.528 * gmi 
        + 0.404 * aqi 
        + 0.892 * sgi 
        + 0.115 * depi 
        - 0.172 * sgai 
        - 0.327 * lvgi 
        + 4.679 * tata
    )

    return {
        "DSRI": dsri,
        "GMI": gmi,
        "AQI": aqi,
        "SGI": sgi,
        "DEPI": depi,
        "SGAI": sgai,
        "LVGI": lvgi,
        "TATA": tata,
        "M_SCORE": m_score
    }
