"""
Exploratory Analysis of Maximum Adverse Price Movement and Market-Relative Downside

Examines whether false-positive alerts under the strict 20-day endpoint return definition
experienced substantial temporary intra-horizon drawdowns or market-relative underperformance.
Preserves the primary target contract unchanged.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.quant.train_pipeline import load_raw_data, DEFAULT_MARKET_FEATURES
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.authoritative_eval_artifact import CSV_PATH, PARQUET_PATH


def compute_maximum_adverse_drawdowns(market_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Computes intra-horizon maximum adverse drawdown over H subsequent trading sessions."""
    df = market_df.copy().sort_values(["ticker", "trade_date"])
    
    # Compute forward prices for each horizon 1..20
    forward_cols = []
    for h in range(1, horizon + 1):
        col_name = f"future_price_{h}d"
        df[col_name] = df.groupby("ticker")["adj_close"].shift(-h)
        forward_cols.append(col_name)
        
    # Minimum future price over the window
    df["min_future_price_20d"] = df[forward_cols].min(axis=1)
    df["max_adverse_drawdown_pct"] = (df["min_future_price_20d"] - df["adj_close"]) / df["adj_close"] * 100.0
    
    # Drop temp forward columns
    df.drop(columns=forward_cols, inplace=True)
    return df


def analyze_target_and_drawdowns() -> Dict[str, Any]:
    print("=== PRIORITY 4: MAXIMUM ADVERSE PRICE MOVEMENT & TARGET EXPLORATORY ANALYSIS ===")
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    
    df_drawdowns = compute_maximum_adverse_drawdowns(market_df, horizon=20)
    
    # Load authoritative eval predictions (Test 2025-2026, N=13,791)
    if PARQUET_PATH.exists():
        eval_df = pd.read_parquet(PARQUET_PATH)
    else:
        eval_df = pd.read_csv(CSV_PATH)
        
    eval_df["trade_date"] = pd.to_datetime(eval_df["trade_date"])
    df_drawdowns["trade_date"] = pd.to_datetime(df_drawdowns["trade_date"])
    
    merged = pd.merge(
        eval_df,
        df_drawdowns[["ticker", "trade_date", "max_adverse_drawdown_pct"]],
        on=["ticker", "trade_date"],
        how="inner"
    )
    
    print(f"Merged evaluation records with intra-horizon drawdowns: N={len(merged)}")
    
    threshold_used = -7.4293
    merged["endpoint_positive"] = (merged["target_label"] == 1).astype(int)
    merged["drawdown_positive"] = (merged["max_adverse_drawdown_pct"] <= threshold_used).astype(int)
    
    total_pos = merged["endpoint_positive"].sum()
    total_dd_pos = merged["drawdown_positive"].sum()
    print(f"Primary Target Endpoint Positives (R_20d <= {threshold_used}%): {total_pos} ({total_pos/len(merged)*100:.2f}%)")
    print(f"Intra-Horizon Drawdown Positives (Min_R_1..20d <= {threshold_used}%): {total_dd_pos} ({total_dd_pos/len(merged)*100:.2f}%)")
    
    # For Model M_Regularized
    op_t = float(merged["op_thresh_M_Regularized"].iloc[0])
    merged["alert_M_Reg"] = (merged["prob_M_Regularized"] >= op_t).astype(int)
    
    alerts = merged[merged["alert_M_Reg"] == 1]
    non_alerts = merged[merged["alert_M_Reg"] == 0]
    
    tp_endpoint = alerts[alerts["endpoint_positive"] == 1]
    fp_endpoint = alerts[alerts["endpoint_positive"] == 0]
    tn_endpoint = non_alerts[non_alerts["endpoint_positive"] == 0]
    fn_endpoint = non_alerts[non_alerts["endpoint_positive"] == 1]
    
    print(f"\nModel M_Regularized Alerts: {len(alerts)} (Alert Rate: {len(alerts)/len(merged)*100:.2f}%)")
    print(f"  True Positives (Endpoint Drop): {len(tp_endpoint)}")
    print(f"  False Positives (Endpoint Rebound/No Drop): {len(fp_endpoint)}")
    
    # Inspect False Positives
    fp_experienced_severe_dd = fp_endpoint[fp_endpoint["drawdown_positive"] == 1]
    fp_experienced_deep_plunge = fp_endpoint[fp_endpoint["max_adverse_drawdown_pct"] <= -10.0]
    
    print("\n--- FALSE POSITIVE FORENSIC ANATOMY ---")
    print(f"Total Nominal 'False Positives': {len(fp_endpoint)}")
    print(f"  Suffered interim drop <= {threshold_used}% before day 20: {len(fp_experienced_severe_dd)} ({len(fp_experienced_severe_dd)/len(fp_endpoint)*100:.2f}%)")
    print(f"  Suffered interim deep plunge <= -10.0% before day 20: {len(fp_experienced_deep_plunge)} ({len(fp_experienced_deep_plunge)/len(fp_endpoint)*100:.2f}%)")
    print(f"  Median Maximum Adverse Drawdown for False Positives: {fp_endpoint['max_adverse_drawdown_pct'].median():.2f}%")
    print(f"  Median Maximum Adverse Drawdown for True Negatives:  {tn_endpoint['max_adverse_drawdown_pct'].median():.2f}%")
    print(f"  Median Maximum Adverse Drawdown for True Positives:  {tp_endpoint['max_adverse_drawdown_pct'].median():.2f}%")
    
    results = {
        "n_samples": len(merged),
        "endpoint_prevalence": float(total_pos / len(merged)),
        "intra_horizon_drawdown_prevalence": float(total_dd_pos / len(merged)),
        "fp_count": len(fp_endpoint),
        "fp_suffered_interim_target_drop_pct": float(len(fp_experienced_severe_dd) / len(fp_endpoint)),
        "fp_suffered_interim_deep_plunge_pct": float(len(fp_experienced_deep_plunge) / len(fp_endpoint)),
        "median_drawdown_tp": float(tp_endpoint["max_adverse_drawdown_pct"].median()),
        "median_drawdown_fp": float(fp_endpoint["max_adverse_drawdown_pct"].median()),
        "median_drawdown_tn": float(tn_endpoint["max_adverse_drawdown_pct"].median())
    }
    return results


if __name__ == "__main__":
    analyze_target_and_drawdowns()
