import json
import numpy as np

with open("scratch/alert_budget_evaluation_output.json") as f:
    data = json.load(f)

matched = data["matched_results"]
broad = data["broad_results"]
boot = data["bootstrap_summary"]

print("=== 1. COMMON MATCHED UNIVERSE: 4-FOLD AGGREGATE RESULTS ===")
for m in ["M_Regularized", "M_Reg_Plus_SecBert_v1", "M_Reg_Plus_SecBert_v2", "Evidence_Gated_Ranked"]:
    folds = matched[m]
    rocs = [f["roc_auc"] for f in folds]
    prs = [f["pr_auc"] for f in folds]
    bss = [f["brier_score"] for f in folds]
    print(f"{m:<25}: ROC={np.mean(rocs):.4f} +/- {np.std(rocs):.4f}, PR={np.mean(prs):.4f}, Brier={np.mean(bss):.4f}")

print("\n=== 2. MATCHED UNIVERSE: DAILY BUDGET PERFORMANCE (IDENTICAL ALERT COUNTS) ===")
for b in ["budget_5pct", "budget_10pct", "budget_20pct"]:
    print(f"\n--- {b.upper()} ---")
    h = f"{'Model':<25} | {'Precision':<10} | {'Recall':<8} | {'FP Share':<10} | {'Total Alerts':<12} | {'TP':<6} | {'FP':<6}"
    print(h)
    print("-" * len(h))
    for m in ["M_Regularized", "M_Reg_Plus_SecBert_v1", "M_Reg_Plus_SecBert_v2", "Evidence_Gated_Ranked"]:
        folds = matched[m]
        tot_tp = sum(f["budgets"][b]["tp"] for f in folds)
        tot_fp = sum(f["budgets"][b]["fp"] for f in folds)
        tot_alerts = sum(f["budgets"][b]["alerts"] for f in folds)
        tot_pos = sum(f["n_pos"] for f in folds)
        prec = tot_tp / tot_alerts
        rec = tot_tp / tot_pos
        fp_share = tot_fp / tot_alerts
        print(f"{m:<25} | {prec:>10.4f} | {rec:>8.4f} | {fp_share:>10.4f} | {tot_alerts:>12} | {tot_tp:>6} | {tot_fp:>6}")

print("\n--- EVIDENCE-GATED ABSTAINING POLICY (FILTERED) ---")
h2 = f"{'Budget Target':<15} | {'Precision':<10} | {'Recall':<8} | {'FP Share':<10} | {'Realized Alerts':<16} | {'Alert Rate':<11}"
print(h2)
print("-" * len(h2))
for b in ["budget_5pct", "budget_10pct", "budget_20pct"]:
    folds = matched["Evidence_Gated_Abstaining"]
    tot_tp = sum(f["budgets"][b]["tp"] for f in folds)
    tot_fp = sum(f["budgets"][b]["fp"] for f in folds)
    tot_alerts = sum(f["budgets"][b]["realized_alerts"] for f in folds)
    tot_obs = sum(f["n_obs"] for f in folds)
    tot_pos = sum(f["n_pos"] for f in folds)
    prec = tot_tp / max(1, tot_alerts)
    rec = tot_tp / tot_pos
    fp_share = tot_fp / max(1, tot_alerts)
    alert_rate = tot_alerts / tot_obs
    print(f"{b:<15} | {prec:>10.4f} | {rec:>8.4f} | {fp_share:>10.4f} | {tot_alerts:>16} | {alert_rate:>11.4f}")

print("\n=== 3. BROADER AVAILABLE POPULATION (MODEL COVERAGE) ===")
for m in ["M_Regularized", "M_Reg_Plus_SecBert_v1", "M_Reg_Plus_SecBert_v2"]:
    folds = broad[m]
    tot_obs = sum(f["n_obs"] for f in folds)
    tot_pos = sum(f["n_pos"] for f in folds)
    rocs = [f["roc_auc"] for f in folds]
    prs = [f["pr_auc"] for f in folds]
    print(f"{m:<25}: N={tot_obs:>5}, Pos={tot_pos:>4} ({tot_pos/tot_obs*100:.1f}%), Mean ROC={np.mean(rocs):.4f}, Mean PR={np.mean(prs):.4f}")

print("\n=== 4. PAIRED DATE-LEVEL BLOCK BOOTSTRAP (1,000 RESAMPLED ITERATIONS) ===")
print(json.dumps(boot, indent=2))
