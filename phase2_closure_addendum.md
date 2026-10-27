# AQQIP — Final Phase 2 Closure Addendum: Statistical Interpretation and Provenance

**Document Type:** Final Research-Governance Addendum  
**Date:** 2026-10-21  
**Audience:** Quantitative Governance Board & Research Leadership  
**Authoritative Specification:** `aqqip_volatility_spec_v4_7.md` (SHA-256: `eabd5e448d038dd26ce5168c4c1bb652d97980e508da53805710d5cfd0d69ec1`)  
**Audited Run Archive:** `C:\dev\AQQIP\experiments\asymmetric_volatility_phase2_20261021_203158`  
**Reference Reports:**
- Phase 2 Execution Report: `phase2_execution_report.md`
- Phase 2B Audit Report: `phase2b_audit_report.md`
- Pre-Training Freeze Manifest: `C:\dev\AQQIP\manifests\pre_training_freeze_manifest_phase1c.json`

---

## 1. Registered Disposition

The preregistered Phase 2 rolling-origin evaluation yielded the following authoritative empirical results:

- **Primary Comparison:** Candidate C versus Baseline A.
- **Pooled False-Positive-Share Change ($\Delta \text{FP\_Share}$):** $+0.7568\text{ percentage points}$ ($+0.007568\text{ proportion}$).
- **Two-Sided 95% Confidence Interval:** $[-0.2162, +1.8378]\text{ percentage points}$ ($[-0.002162, +0.018378]\text{ proportion}$).
- **Recall Non-Inferiority Lower Bound ($q_{0.05}(\Delta \text{Recall})$):** $-0.7955\text{ percentage points}$ ($-0.007955\text{ proportion}$).
- **Mechanistic Upside Attribution Ratio ($\rho_{\text{upside}}$):** $74.18\%$ ($158 / 213$).
- **Registered Decision Outcome:** **Outcome O-8B** (Tier 3: Zero improvement or deterioration; Baseline A unchanged).
- **Operational Disposition:** **Baseline A remains unchanged; Candidate C is ineligible for production promotion.**

---

## 2. Correct Statistical Interpretation

To maintain rigorous scientific and methodological standards, the empirical findings must be interpreted according to the following distinctions:

1. **Point Estimate Worsening:** The pooled point estimate of false-positive share worsened under Candidate C, increasing from $74.8108\%$ (Baseline A) to $75.5676\%$ (Candidate C), representing an observed delta of $+0.7568\text{ percentage points}$ ($+14$ net false positives across the evaluation panel).
2. **Absence of Statistically Significant Deterioration:** The two-sided 95% confidence interval spans $[-0.2162, +1.8378]\text{ percentage points}$ and includes zero ($0.0\text{ pp}$). Under standard Neyman-Pearson hypothesis testing at the preregistered $\alpha = 0.05$ significance level, the null hypothesis of equal false-positive share cannot be rejected. Therefore, **this result does not establish statistically significant deterioration.** The empirical outcome must not be characterized as proof that Candidate C or downside semivolatility is universally or structurally inferior in general.
3. **Decisive Failure of Preregistered Improvement Criteria:** While deterioration is not statistically significant, Candidate C decisively failed both registered primary efficacy hurdles:
   - *Target effect size:* The observed $+0.7568\text{ pp}$ falls far short of the required $\le -4.0\text{ percentage points}$ reduction.
   - *Statistical detectability:* The upper confidence bound of $+1.8378\text{ pp}$ fails the requirement of being strictly negative ($< 0.0\text{ pp}$).
4. **Passing of Recall Non-Inferiority:** Recall non-inferiority passed unambiguously. The one-sided 5th-percentile bootstrap lower bound for recall difference is $-0.7955\text{ percentage points}$, which satisfies the preregistered non-inferiority bound of $\ge -1.0\text{ percentage point}$ (observed point recall delta was $-0.3447\text{ pp}$, moving from $11.4722\%$ to $11.1275\%$). The substitution of downside semivolatility did not degrade hazard-event capture beyond acceptable limits.
5. **Exploratory Role of Mechanistic Attribution:** The upside-attribution diagnostic ($\rho_{\text{upside}} = 74.18\%$, with $158$ of $213$ displaced false positives possessing positive forward returns) met the diagnostic threshold of $> 60.0\%$. This confirms the mechanistic hypothesis that when Candidate C displaces a baseline false positive, it predominantly does so on securities experiencing positive forward price movement. However, under the preregistered decision hierarchy, **this mechanistic diagnostic is strictly explanatory and cannot override or compensate for failing the primary false-positive reduction criteria.** Outcome O-8B cannot be reinterpreted as a favorable or successful research result.

---

## 3. Exact Threshold Semantics

The decision matrix thresholds defined in Specification v4.7 are restated below in their exact mathematical formulation:

- **Primary False-Positive-Share Improvement:** Effect size $\le -4.0\text{ percentage points}$ (proportion $\le -0.0400$).
- **Statistical Detectability:** Upper 95% confidence bound $< 0\text{ percentage points}$ (proportion $< 0.0000$; strictly negative).
- **Recall Non-Inferiority:** One-sided lower 95% bound $\ge -1.0\text{ percentage point}$ (proportion $\ge -0.0100$).
- **Upside Attribution:** Strictly $> 60.0\%$ (proportion $> 0.6000$) with $|D_{\text{FP}}| > 0$.

### Irrelevance of Minor Operator Inconsistencies
Earlier working notes in the audit trail occasionally referenced the detectability criterion as $\le 0.0\text{ pp}$ (non-strict) or the upside diagnostic as $\ge 60.0\%$ (non-strict). As demonstrated in the Phase 2B audit, the observed empirical values resolve unambiguously under both strict and non-strict interpretations:
- Observed upper confidence bound $+1.8378\text{ pp}$ violates both $< 0.0\text{ pp}$ and $\le 0.0\text{ pp}$.
- Observed effect size $+0.7568\text{ pp}$ violates both $\le -4.0\text{ pp}$ and $< -4.0\text{ pp}$.
- Observed recall lower bound $-0.7955\text{ pp}$ satisfies both $\ge -1.0\text{ pp}$ and $> -1.0\text{ pp}$.
- Observed upside attribution $74.18\%$ satisfies both $> 60.0\%$ and $\ge 60.0\%$.

The classification of Outcome O-8B is robust, exact, and completely invariant to these minor operator differences.

---

## 4. Provenance and Reproducibility Limitations

The Phase 2B audit established a clear demarcation between verified empirical facts and documented provenance limitations:

### 4.1 Verified Empirical Facts
1. **Invocation Separation:** The first runner invocation (`task-2766`, 20:30:23 local) crashed after 6.2 seconds with `KeyError: 'Candidate_A'` and exited with code 1 before writing any model, prediction, or metric files. The second invocation (`task-2784`, 20:31:54 local) executed sequentially in its own isolated directory (`experiments/asymmetric_volatility_phase2_20261021_203158`), completing cleanly in 76.13 seconds with zero process concurrency, overlap, or directory collisions.
2. **Prediction and Metric Reconciliation:** The 29,600 saved prediction observations across 4 folds (925 dates $\times$ 32 tickers) exhibit zero duplicate rows, zero missing values, exactly 2 alerts per candidate per date, and zero ranking violations. Independent re-tabulation from raw predictions reproduces all reported pooled metrics, fold-level metrics, and alert-transition counts ($D_{\text{FP}}=213, D_{\text{TP}}=53, I_{\text{TP}}=39, I_{\text{FP}}=227$; budget identity $266 == 266$) to exact machine precision.
3. **Booster Structure:** All 16 booster text files match their preregistered candidate feature counts (11 for A and C; 12 for B and D) and their exact Stage 1 selected iteration counts (11 to 117 trees).
4. **Protected State Invariance:** All 16 files tracked in `pre_training_freeze_manifest_phase1c.json` remain byte-for-byte identical to their pre-training hashes. The production database and MLflow tracking store were not modified.

### 4.2 Documented Limitations
1. **Omission of Runner Hash in Summary:** `execution_summary.json` captured `spec_hash`, start/end timestamps, duration, and output directory, but omitted the runner script's SHA-256 hash. While runner source lineage (`run_asymmetric_volatility_experiment.py`, 23,383 bytes, SHA-256 `7cd8f313330f13c59fda4042107562864588833345540895403600a6913a109c`) was reconstructed from filesystem timestamps, shell logs, and transcript diffs, the runner hash was not embedded directly within the execution summary.
2. **Replicate Draw Persistence:** The formal bootstrap confidence intervals and percentiles were independently reproduced from the saved prediction tables using the deterministic PRNG seed (`20240101`). However, the individual resampled block-index sequences ($1,000\text{ draws} \times 4\text{ folds}$) were not persisted as a dedicated diagnostic table. Replicability relies on deterministic algorithmic pseudo-random number generator reconstruction.
3. **Residual Directory:** The aborted directory from Invocation 1 (`experiments/asymmetric_volatility_phase2_20261021_203027`) remains on disk as empty subdirectories; it was not cleaned up, though it contains zero data.

---

## 5. Final Research and Governance Decision

1. **Formal Closure:** Phase 2 and Phase 2B of the AQQIP Asymmetric Volatility research program are formally **closed**.
2. **No Additional Modelling or Refitting:** No additional model fitting, feature re-engineering, hyperparameter tuning, holdout scoring, or iterative exploration against the 2021–2024 panel is authorized.
3. **Evidence Status:** The 2021–2024 rolling-origin panel represents historical development evidence, not an untouched final holdout test. Further iteration against this panel would constitute p-hacking and violate quantitative research governance.
4. **Production Standard Retained:** **Baseline A remains the authoritative production standard.** Candidate C is rejected for deployment.
5. **Exploratory Candidate Status:** Empirical results for Candidate B ($\text{FP\_Share} = 75.0811\%$) and Candidate D ($\text{FP\_Share} = 74.5405\%$) remain strictly exploratory and descriptive benchmarks, providing no confirmatory basis for model replacement or promotion.
