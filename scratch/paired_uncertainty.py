"""
Script to compute paired uncertainty estimates between M_Regularized and M_Reg_Plus_SecBert (v1)
across the 4 chronological pre-2025 rolling-origin folds.
"""
import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
import numpy as np
import pandas as pd
from scipy import stats

from src.quant.rolling_origin_transformer_experiment import RollingOriginTransformerExperiment

def compute_paired_uncertainty():
    print("=== PRIORITY 1: REPRODUCE ROLLING-ORIGIN RESULTS & PAIRED UNCERTAINTY ===")
    exp = RollingOriginTransformerExperiment()
    res = exp.run_experiment()
    
    summary = res["summary"]
    matched = res["matched_fold_results"]
    
    m_reg = summary["M_Regularized"]["fold_details"]
    m_sec = summary["M_Reg_Plus_SecBert"]["fold_details"]
    
    folds = [f["fold"] for f in m_reg]
    
    # 1. Broader Population Paired Comparisons
    print("\n--- BROADER POPULATION FOLD-BY-FOLD COMPARISON ---")
    roc_m = np.array([f["roc_auc"] for f in m_reg])
    roc_s = np.array([f["roc_auc"] for f in m_sec])
    pr_m = np.array([f["pr_auc"] for f in m_reg])
    pr_s = np.array([f["pr_auc"] for f in m_sec])
    bss_m = np.array([f["bss"] for f in m_reg])
    bss_s = np.array([f["bss"] for f in m_sec])
    
    diff_roc = roc_s - roc_m
    diff_pr = pr_s - pr_m
    diff_bss = bss_s - bss_m
    
    df_broader = pd.DataFrame({
        "Fold": folds,
        "M_Reg ROC": roc_m,
        "M_SecBert ROC": roc_s,
        "Delta ROC": diff_roc,
        "M_Reg PR": pr_m,
        "M_SecBert PR": pr_s,
        "Delta PR": diff_pr,
        "M_Reg BSS": bss_m,
        "M_SecBert BSS": bss_s,
        "Delta BSS": diff_bss
    })
    print(df_broader.to_string(index=False))
    
    # Paired Statistics
    mean_d_roc = float(np.mean(diff_roc))
    se_d_roc = float(stats.sem(diff_roc))
    ci_d_roc = (mean_d_roc - 1.96 * se_d_roc, mean_d_roc + 1.96 * se_d_roc)
    t_stat_roc, p_val_roc = stats.ttest_1samp(diff_roc, 0.0)
    
    mean_d_pr = float(np.mean(diff_pr))
    se_d_pr = float(stats.sem(diff_pr))
    ci_d_pr = (mean_d_pr - 1.96 * se_d_pr, mean_d_pr + 1.96 * se_d_pr)
    t_stat_pr, p_val_pr = stats.ttest_1samp(diff_pr, 0.0)
    
    print(f"\nPaired ROC-AUC Delta: {mean_d_roc:+.4f} (SE: {se_d_roc:.4f}, 95% CI: [{ci_d_roc[0]:+.4f}, {ci_d_roc[1]:+.4f}], t={t_stat_roc:.2f}, p={p_val_roc:.4f})")
    print(f"Paired PR-AUC Delta:  {mean_d_pr:+.4f} (SE: {se_d_pr:.4f}, 95% CI: [{ci_d_pr[0]:+.4f}, {ci_d_pr[1]:+.4f}], t={t_stat_pr:.2f}, p={p_val_pr:.4f})")
    
    # 2. Matched Sample Paired Comparisons
    print("\n--- MATCHED SAMPLE FOLD-BY-FOLD COMPARISON ---")
    m_matched = matched["M_Regularized"]
    s_matched = matched["M_Reg_Plus_SecBert"]
    
    m_roc_match = np.array([f["roc_auc"] for f in m_matched])
    s_roc_match = np.array([f["roc_auc"] for f in s_matched])
    m_pr_match = np.array([f["pr_auc"] for f in m_matched])
    s_pr_match = np.array([f["pr_auc"] for f in s_matched])
    n_match = [f["n_matched"] for f in m_matched]
    
    diff_roc_m = s_roc_match - m_roc_match
    diff_pr_m = s_pr_match - m_pr_match
    
    df_matched = pd.DataFrame({
        "Fold": [f["fold"] for f in m_matched],
        "N Matched": n_match,
        "M_Reg ROC": m_roc_match,
        "M_SecBert ROC": s_roc_match,
        "Delta ROC": diff_roc_m,
        "M_Reg PR": m_pr_match,
        "M_SecBert PR": s_pr_match,
        "Delta PR": diff_pr_m
    })
    print(df_matched.to_string(index=False))
    
    mean_dm_roc = float(np.mean(diff_roc_m))
    se_dm_roc = float(stats.sem(diff_roc_m))
    ci_dm_roc = (mean_dm_roc - 1.96 * se_dm_roc, mean_dm_roc + 1.96 * se_dm_roc)
    t_stat_m_roc, p_val_m_roc = stats.ttest_1samp(diff_roc_m, 0.0)
    
    mean_dm_pr = float(np.mean(diff_pr_m))
    se_dm_pr = float(stats.sem(diff_pr_m))
    ci_dm_pr = (mean_dm_pr - 1.96 * se_dm_pr, mean_dm_pr + 1.96 * se_dm_pr)
    t_stat_m_pr, p_val_m_pr = stats.ttest_1samp(diff_pr_m, 0.0)
    
    print(f"\nMatched ROC-AUC Delta: {mean_dm_roc:+.4f} (SE: {se_dm_roc:.4f}, 95% CI: [{ci_dm_roc[0]:+.4f}, {ci_dm_roc[1]:+.4f}], t={t_stat_m_roc:.2f}, p={p_val_m_roc:.4f})")
    print(f"Matched PR-AUC Delta:  {mean_dm_pr:+.4f} (SE: {se_dm_pr:.4f}, 95% CI: [{ci_dm_pr[0]:+.4f}, {ci_dm_pr[1]:+.4f}], t={t_stat_m_pr:.2f}, p={p_val_m_pr:.4f})")

if __name__ == "__main__":
    compute_paired_uncertainty()
