import json
import numpy as np

with open(r'C:\Users\prajesh\.gemini\antigravity-ide\brain\6cd62b1a-81e0-4795-a91a-7069a3fa5900\.system_generated\tasks\task-2151.log', 'r') as f:
    text = f.read()

s_idx = text.find('=== ROLLING-ORIGIN EXPERIMENT V2 SUMMARY ===')
e_idx = text.find('=== MATCHED SAMPLE COMPARISON V2 ===')
summary_json = json.loads(text[s_idx + len('=== ROLLING-ORIGIN EXPERIMENT V2 SUMMARY ==='):e_idx].strip())
matched_json = json.loads(text[e_idx + len('=== MATCHED SAMPLE COMPARISON V2 ==='):].strip())

print("=== BROADER POPULATION (4-FOLD MEAN) ===")
header = f"{'Model':<25} | {'Mean ROC':<9} | {'Std ROC':<8} | {'Mean PR':<8} | {'Mean BSS':<9} | {'Mean F1':<8}"
print(header)
print("-" * len(header))
for m, d in summary_json.items():
    print(f"{m:<25} | {d['mean_roc_auc']:>9.4f} | {d['std_roc_auc']:>8.4f} | {d['mean_pr_auc']:>8.4f} | {d['mean_bss']:>9.4f} | {d['mean_f1']:>8.4f}")

print("\n=== MATCHED SAMPLE ROC BY FOLD ===")
folds = ['Fold_1_2021', 'Fold_2_2022', 'Fold_3_2023', 'Fold_4_2024']
h2 = f"{'Fold':<12} | {'M_Regularized':<14} | {'M_Reg+SecBert_v1':<16} | {'M_Reg+SecBert_v2':<16} | {'Evidence_Gated':<14}"
print(h2)
print("-" * len(h2))
for i, fld in enumerate(folds):
    m_reg = matched_json['M_Regularized'][i]['roc_auc']
    m_v1 = matched_json['M_Reg_Plus_SecBert_v1'][i]['roc_auc']
    m_v2 = matched_json['M_Reg_Plus_SecBert_v2'][i]['roc_auc']
    m_gate = matched_json['Evidence_Gated_Policy'][i]['roc_auc']
    print(f"{fld:<12} | {m_reg:>14.4f} | {m_v1:>16.4f} | {m_v2:>16.4f} | {m_gate:>14.4f}")

print("\n=== FIXED ALERT BUDGETS (5%, 10%, 20%) ACROSS FOLDS ===")
for budget in ["budget_5pct", "budget_10pct", "budget_20pct"]:
    print(f"\n--- {budget.upper()} ---")
    h3 = f"{'Model':<25} | {'Mean Prec':<10} | {'Mean Recall':<11} | {'Mean FP Share':<13} | {'Mean Alerts':<11}"
    print(h3)
    print("-" * len(h3))
    for m in summary_json.keys():
        precs = [f['fixed_budgets'][budget]['precision'] for f in summary_json[m]['fold_details']]
        recs = [f['fixed_budgets'][budget]['recall'] for f in summary_json[m]['fold_details']]
        fps = [f['fixed_budgets'][budget]['fp_share'] for f in summary_json[m]['fold_details']]
        alerts = [f['fixed_budgets'][budget]['alerts'] for f in summary_json[m]['fold_details']]
        print(f"{m:<25} | {np.mean(precs):>10.4f} | {np.mean(recs):>11.4f} | {np.mean(fps):>13.4f} | {np.mean(alerts):>11.1f}")
