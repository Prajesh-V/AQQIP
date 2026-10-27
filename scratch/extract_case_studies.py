"""
Extracts concrete case studies from the authoritative evaluation and historical panel:
1. High-confidence False Positives
2. Severe False Negatives
3. Clean True Positives
4. Alerts Rejected by Evidence-Gated Policy
"""

import pandas as pd
import numpy as np
from src.quant.authoritative_eval_artifact import PARQUET_PATH
from src.quant.train_pipeline import load_raw_data
from src.quant.exploratory_target_drawdown import compute_maximum_adverse_drawdowns

def extract_case_studies():
    eval_df = pd.read_parquet(PARQUET_PATH)
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    
    # Merge drawdowns and actual forward returns
    market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])
    eval_df["trade_date"] = pd.to_datetime(eval_df["trade_date"])
    
    df_dd = compute_maximum_adverse_drawdowns(market_df, horizon=20)
    df_dd["future_price_20d"] = df_dd.groupby("ticker")["adj_close"].shift(-20)
    df_dd["realized_ret_20d_pct"] = (df_dd["future_price_20d"] - df_dd["adj_close"]) / df_dd["adj_close"] * 100.0
    
    merged = pd.merge(
        eval_df,
        df_dd[["ticker", "trade_date", "adj_close", "realized_ret_20d_pct", "max_adverse_drawdown_pct"]],
        on=["ticker", "trade_date"],
        how="inner"
    )
    
    op_t = float(merged["op_thresh_M_Regularized"].iloc[0]) # 0.12
    merged["is_alert"] = (merged["prob_M_Regularized"] >= op_t).astype(int)
    
    # 1. High-Confidence False Positives (Highest prob, but target_label == 0)
    fps = merged[(merged["is_alert"] == 1) & (merged["target_label"] == 0)].sort_values(by="prob_M_Regularized", ascending=False)
    print("=== TOP 5 HIGH-CONFIDENCE FALSE POSITIVES ===")
    cols = ["ticker", "trade_date", "prob_M_Regularized", "realized_ret_20d_pct", "max_adverse_drawdown_pct"]
    print(fps[cols].head(5).to_string(index=False))
    
    # 2. Severe False Negatives (Lowest prob among severe drops <= -15%)
    fns = merged[(merged["is_alert"] == 0) & (merged["realized_ret_20d_pct"] <= -15.0)].sort_values(by="prob_M_Regularized", ascending=True)
    print("\n=== TOP 5 SEVERE FALSE NEGATIVES (SEVERE CRASH, LOW PREDICTED PROB) ===")
    print(fns[cols].head(5).to_string(index=False))
    
    # 3. Clean True Positives (High prob and severe crash)
    tps = merged[(merged["is_alert"] == 1) & (merged["target_label"] == 1)].sort_values(by="prob_M_Regularized", ascending=False)
    print("\n=== TOP 5 CLEAN TRUE POSITIVES (HIGH PROB AND REALIZED CRASH) ===")
    print(tps[cols].head(5).to_string(index=False))
    
    # Look at qualitative rejections from the 2021-2024 matched dataset
    # We can inspect scratch/alert_budget_evaluation_output.json
    print("\nDone extracting case studies.")

if __name__ == "__main__":
    extract_case_studies()
