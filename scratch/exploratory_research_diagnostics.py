"""
Exploratory Research Diagnostic Script

Extracts empirical evidence across:
1. Target dynamics: endpoint return vs intra-horizon drawdown, recovery rates, market beta vs idiosyncratic.
2. Regime analysis: Bull (2021), Bear (2022), Rebound (2023), Expansion (2024).
3. Sector breakdown: prevalence, predictability, and error distribution.
4. Feature family diagnostics: coverage, stability across folds, correlation with target.
5. Specific error case studies: high-confidence false positives, severe false negatives, clean true positives, evidence-gated rejections.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import mysql.connector

from config.settings import get_settings
from src.quant.train_pipeline import load_raw_data, DEFAULT_MARKET_FEATURES
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.exploratory_target_drawdown import compute_maximum_adverse_drawdowns

# Sector mapping for the 32 tickers
TICKER_SECTOR = {
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology", "CSCO": "Information Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "GOOGL": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services", "DIS": "Communication Services",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "JNJ": "Health Care", "UNH": "Health Care", "PFE": "Health Care", "MRK": "Health Care",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples", "COST": "Consumer Staples", "WMT": "Consumer Staples",
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "CVX": "Energy", "XOM": "Energy"
}

def run_diagnostics():
    print("Loading data...")
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    builder = HistoricalDatasetBuilder()
    panel = builder.build_dataset(market_df, fund_df, qual_df)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    
    # Compute drawdowns
    panel = compute_maximum_adverse_drawdowns(panel, horizon=20)
    panel["sector"] = panel["ticker"].map(TICKER_SECTOR).fillna("Other")
    
    # Compute forward 20d return
    panel["future_price_20d"] = panel.groupby("ticker")["adj_close"].shift(-20)
    panel["return_20d_pct"] = (panel["future_price_20d"] - panel["adj_close"]) / panel["adj_close"] * 100.0
    
    # Market return (equal-weighted average across 32 tickers)
    market_ret = panel.groupby("trade_date")["return_20d_pct"].mean().rename("mkt_return_20d_pct")
    panel = panel.merge(market_ret, on="trade_date", how="left")
    panel["excess_return_20d_pct"] = panel["return_20d_pct"] - panel["mkt_return_20d_pct"]

    # -------------------------------------------------------------
    # 1. Target Dynamics Across All Historical Data (2020-2024)
    # -------------------------------------------------------------
    df_eval = panel[(panel["trade_date"] >= "2021-01-01") & (panel["trade_date"] < "2025-01-01")].dropna(subset=["return_20d_pct", "max_adverse_drawdown_pct"]).copy()
    
    threshold = -7.4293
    df_eval["is_endpoint_drop"] = (df_eval["return_20d_pct"] <= threshold).astype(int)
    df_eval["is_interim_drawdown"] = (df_eval["max_adverse_drawdown_pct"] <= threshold).astype(int)
    df_eval["is_rebound"] = ((df_eval["is_interim_drawdown"] == 1) & (df_eval["is_endpoint_drop"] == 0)).astype(int)
    df_eval["is_idiosyncratic_drop"] = ((df_eval["is_endpoint_drop"] == 1) & (df_eval["excess_return_20d_pct"] <= -5.0)).astype(int)
    df_eval["is_systematic_drop"] = ((df_eval["is_endpoint_drop"] == 1) & (df_eval["excess_return_20d_pct"] > -5.0)).astype(int)

    n_tot = len(df_eval)
    n_endpoint = df_eval["is_endpoint_drop"].sum()
    n_interim = df_eval["is_interim_drawdown"].sum()
    n_rebound = df_eval["is_rebound"].sum()
    n_idio = df_eval["is_idiosyncratic_drop"].sum()
    n_syst = df_eval["is_systematic_drop"].sum()

    print("\n=======================================================")
    print("1. TARGET DYNAMICS (2021-2024, N={})".format(n_tot))
    print("=======================================================")
    print(f"Primary Endpoint Drop (R_20d <= {threshold}%): {n_endpoint} ({n_endpoint/n_tot*100:.2f}%)")
    print(f"Interim Drawdown Drop (Min_R <= {threshold}%):   {n_interim} ({n_interim/n_tot*100:.2f}%)")
    print(f"Rebound Events (Interim Drop, but Day 20 Flat/Up): {n_rebound} ({n_rebound/n_tot*100:.2f}%, {n_rebound/max(1,n_interim)*100:.1f}% of interim drops)")
    print(f"Idiosyncratic Drops (Stock fell >5% worse than market): {n_idio} ({n_idio/max(1,n_endpoint)*100:.1f}% of endpoint drops)")
    print(f"Systematic / Beta Drops (Stock fell with general market): {n_syst} ({n_syst/max(1,n_endpoint)*100:.1f}% of endpoint drops)")

    # -------------------------------------------------------------
    # 2. Market Regime Breakdown
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("2. TARGET BEHAVIOR BY MARKET REGIME")
    print("=======================================================")
    regimes = [
        ("2021 (Bull Market / Stimulus Expansion)", "2021-01-01", "2022-01-01"),
        ("2022 (Severe Bear Market / Fed Rate Hikes)", "2022-01-01", "2023-01-01"),
        ("2023 (Tech Rebound / Narrow Mega-Cap Rally)", "2023-01-01", "2024-01-01"),
        ("2024 (Broad Market Expansion / Low Vol)", "2024-01-01", "2025-01-01"),
    ]
    for r_name, start, end in regimes:
        sub = df_eval[(df_eval["trade_date"] >= start) & (df_eval["trade_date"] < end)]
        sub_n = len(sub)
        sub_end = sub["is_endpoint_drop"].sum()
        sub_dd = sub["is_interim_drawdown"].sum()
        sub_reb = sub["is_rebound"].sum()
        sub_mkt_ret = sub["mkt_return_20d_pct"].mean()
        sub_vol = sub["volatility_20d"].mean() if "volatility_20d" in sub else 0
        print(f"\n{r_name}: N={sub_n}")
        print(f"  Mean Market 20d Return: {sub_mkt_ret:+.2f}%, Mean 20d Vol: {sub_vol*100:.2f}%")
        print(f"  Endpoint Drop Prevalence: {sub_end}/{sub_n} ({sub_end/sub_n*100:.2f}%)")
        print(f"  Interim Drawdown Prevalence: {sub_dd}/{sub_n} ({sub_dd/sub_n*100:.2f}%)")
        print(f"  Rebound Rate: {sub_reb}/{sub_dd} ({sub_reb/max(1,sub_dd)*100:.1f}% of drawdowns recovered)")

    # -------------------------------------------------------------
    # 3. Sector Breakdown
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("3. SECTOR BREAKDOWN (2021-2024)")
    print("=======================================================")
    sector_stats = []
    for sec, grp in df_eval.groupby("sector"):
        sec_n = len(grp)
        sec_end = grp["is_endpoint_drop"].sum()
        sec_dd = grp["is_interim_drawdown"].sum()
        sec_reb = grp["is_rebound"].sum()
        sec_mkt_rel = grp["excess_return_20d_pct"].mean()
        sector_stats.append({
            "sector": sec,
            "n_obs": sec_n,
            "tickers": grp["ticker"].nunique(),
            "endpoint_prev": round(sec_end / sec_n * 100, 2),
            "interim_prev": round(sec_dd / sec_n * 100, 2),
            "rebound_pct": round(sec_reb / max(1, sec_dd) * 100, 1),
            "mean_excess_ret": round(sec_mkt_rel, 2)
        })
    df_sec = pd.DataFrame(sector_stats).sort_values("endpoint_prev", ascending=False)
    print(df_sec.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Feature Family Diagnostics
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("4. FEATURE CORRELATIONS WITH TARGET (POINT-IN-TIME)")
    print("=======================================================")
    feat_candidates = [
        "return_5d", "return_20d", "return_60d", "volatility_20d", "volatility_60d",
        "downside_dev_20d", "max_drawdown_20d", "max_drawdown_60d", "volume_ratio_20d",
        "return_20d_rel", "vol_ratio_20d", "altman_z_score", "current_ratio", "debt_to_equity"
    ]
    corrs = {}
    for feat in feat_candidates:
        if feat in df_eval.columns:
            s = df_eval.dropna(subset=[feat, "is_endpoint_drop"])
            r = np.corrcoef(s[feat], s["is_endpoint_drop"])[0, 1]
            missing_pct = df_eval[feat].isna().mean() * 100
            corrs[feat] = {"corr_with_drop": round(r, 4), "missing_pct": round(missing_pct, 1)}
    df_corr = pd.DataFrame(corrs).T.sort_values(by="corr_with_drop", key=abs, ascending=False)
    print(df_corr.to_string())

if __name__ == "__main__":
    run_diagnostics()
