# AQQIP — Phase 3: Improvement Research Design and Prospective Validation Plan (v1.2)

**Document Type:** Final Pre-Specification Research Design & Prospective Protocol Audit  
**Version:** 1.2 (Corrected & Evidence-Backed Revision of v1.1)  
**Date:** 2026-10-21  
**Author:** Quantitative Research Systems & Methodology Group  
**Prior Milestones:** Phase 2 Closed (Outcome O-8B, Baseline A Retained); Phase 2B Reconciled; Phase 3B Audit Completed  
**Governing Specification:** `aqqip_volatility_spec_v4_7.md` (Contextual reference only; remains frozen)  
**Target Evaluation Scope:** Strictly prospective post-freeze trading sessions  

---

## Methodological Classification & Status Notice

In strict accordance with quantitative governance standards, every statement in this document is explicitly categorized into one of four methodological classes:
- **[VERIFIED FACT]:** Empirically established from audited codebase artifacts, data manifests, or mathematical theorems.
- **[MODEL ASSUMPTION]:** A formal statistical or behavioral assumption subject to empirical falsification.
- **[PROPOSED DECISION]:** An operational or engineering protocol choice submitted for governance review.
- **[UNRESOLVED QUESTION]:** An empirical or statistical issue that cannot be resolved without offline data analysis on permitted pre-2021 data, and which must be resolved prior to drafting Specification v5.0.

---

## 1. Operational Problem Definition & Research Hypothesis

### 1.1 The Operational Decision: Daily Top-Two Alert Selection
- **[VERIFIED FACT] Fixed Universe:** Every trading session $t$, a balanced panel of $N = 32$ liquid US equities is evaluated.
- **[VERIFIED FACT] Fixed Daily Budget:** The system selects exactly $K_t = 2$ securities per session ($6.25\%$ of the universe, corresponding to a nominal $5\%$ fraction with ceiling rounding: $\lceil 0.05 \times 32 \rceil = 2$).
- **[VERIFIED FACT] Primary Performance Objective:** Over an extended evaluation window, the operational objective is to minimize the **false-positive share** ($\text{FP} / (\text{TP} + \text{FP})$) among the daily alerted pairs, while maintaining an acceptable hazard detection rate (**recall** $\text{TP} / P$).

### 1.2 Structural Disconnect of the Existing Baseline
- **[VERIFIED FACT] Point-wise Binary Log-Loss Objective:** Baseline A is trained by minimizing global point-wise binary cross-entropy:
  $$\mathcal{L}_{\text{logloss}} = - \frac{1}{|\mathcal{D}|} \sum_{(s,t) \in \mathcal{D}} \left[ y_{s,t} \log p_{s,t} + (1 - y_{s,t}) \log(1 - p_{s,t}) \right]$$
- **[VERIFIED FACT] Rank-Insensitive Penalty:** Log-loss penalizes probability calibration errors across all 32 stocks uniformly. An error shifting a non-event security from rank 28 to rank 22 receives the same penalty as shifting a non-event from rank 3 to rank 2, even though only the latter generates an operational false positive.
- **[VERIFIED FACT] Time-Series Confounding:** In market-wide selloffs, unconditional event probabilities rise across all 32 stocks simultaneously. Point-wise classification expends model capacity fitting aggregate market drawdown regimes rather than resolving the cross-sectional risk ordering among the 32 stocks on date $t$.

### 1.3 Formal Research Hypotheses
- **Hypothesis $H_1$ (Objective-Alignment Hypothesis — Primary Challenger):**
  Replacing point-wise binary log-loss with a cross-sectional list-wise ranking objective (LambdaMART with NDCG@2 grouped by session) will reduce the daily top-two false-positive share relative to Baseline A by focusing gradient updates exclusively on pairwise inversions at the top of the daily cross-section.
- **Null Hypothesis $H_0$ (Dominance / Sufficiency of Baseline A):**
  Cross-sectional stock ordering in this 32-stock universe is dominated by high-frequency idiosyncratic noise. LambdaMART will fail to produce a statistically detectable, practically significant false-positive reduction ($\Delta \text{FP\_Share} \le -4.0\text{ pp}$) relative to Baseline A under leak-free prospective validation.
- **[PROPOSED DECISION] Single Primary Challenger Mandate:** Exactly **one primary challenger** (LambdaMART) will be evaluated in the prospective study. Alternative formulations (idiosyncratic downside labels, continuous quantile loss) are strictly quarantined to offline exploratory research to prevent multiplicity inflation.

---

## 2. Mathematical Relationship: NDCG@2, Top-Two Alert Selection, and False-Positive Share

### 2.1 The Operational Metric: Top-Two Set Membership
- **[VERIFIED FACT] Alert Selection Rule:** On date $t$, securities are sorted by predicted score descending, breaking ties by ticker symbol ascending. The two highest-ranked securities are alerted:
  $$\mathcal{A}_t = \{s_{(1),t}, s_{(2),t}\}$$
- **[VERIFIED FACT] Daily False Positives:** The daily count of false positives $\text{FP}_t$ depends strictly on the **un-ordered set membership** of $\mathcal{A}_t$:
  $$\text{FP}_t = \sum_{s \in \mathcal{A}_t} (1 - y_{s,t}) \in \{0, 1, 2\}$$
  Under the operational metric, whether a single positive event occurs at rank 1 or rank 2 is completely immaterial: $\text{FP}_t = 1$ in both cases.

### 2.2 What NDCG@2 Rewards (The Intra-Top-Two Asymmetry)
In LambdaMART, Discounted Cumulative Gain at truncation $K=2$ is defined as:
$$\text{DCG}@2 = \sum_{i=1}^2 \frac{2^{y_{(i),t}} - 1}{\log_2(i + 1)} = \left(2^{y_{(1),t}} - 1\right) + \frac{2^{y_{(2),t}} - 1}{\log_2(3)}$$
where $\log_2(2) = 1.0$ and $\log_2(3) \approx 1.5850$.

Consider a session where exactly one of the top two securities is a true downside event ($y=1$) and the other is a non-event ($y=0$):
- **Case 1 (Event at Rank 1, Non-Event at Rank 2):**
  $$\text{DCG}@2 = 1.0 + 0.0 = 1.0000 \implies \text{NDCG}@2 = 1.0000$$
- **Case 2 (Non-Event at Rank 1, Event at Rank 2):**
  $$\text{DCG}@2 = 0.0 + \frac{1}{\log_2(3)} \approx 0.6309 \implies \text{NDCG}@2 = 0.6309$$

- **[VERIFIED FACT] Mathematical Divergence:**
  - The operational metric evaluates Case 1 and Case 2 as **identical** ($\text{FP}_t = 1$).
  - NDCG@2 evaluates Case 2 as **$36.9\%$ worse** than Case 1.
  - Consequently, LambdaMART generates non-zero $\lambda$-gradients that penalize placing a true event at rank 2 rather than rank 1.

### 2.3 Methodological Justification & Limitations of LambdaMART
- **[MODEL ASSUMPTION] Why NDCG@2 is Retained as Primary Challenger:**
  Despite the within-top-two discount asymmetry, NDCG@2 is vastly closer to the operational decision than binary log-loss. Under NDCG@2, all pairs below rank 2 receive a truncation weight of zero. The model expends zero capacity attempting to sort rank 25 from rank 30.
- **[MODEL ASSUMPTION] Ranking is a Scientific Hypothesis:**
  The ranking objective is **not** mathematically optimal for set-membership Precision@2. It is evaluated as an explicit research hypothesis ($H_1$). It may fail if within-top-two gradient asymmetry distracts the tree booster from separating rank 2 from rank 3.
- **[PROPOSED DECISION] Invariant Handling Policies:**
  1. *Binary Relevance Label:* $y_{s,t} = \mathbb{1}(\text{forward\_return\_pct}_{s,t,20} \le \tau_{\text{outer}})$. Exactly aligns with Specification v4.7.
  2. *Zero-Event Sessions:* When $\sum_{s=1}^{32} y_{s,t} = 0$, $\text{IDCG}@2 = 0$. LightGBM naturally assigns $\lambda$-gradients of 0. Zero-event sessions contribute zero gradient updates.
  3. *Tied Scores:* Broken deterministically by `ticker_ascending`.
  4. *Incomplete Groups ($N_t < 32$):* If $N_t \ge 2$, rank top 2. If $N_t < 2$, flag and exclude date.

---

## 3. Reconstructed & Qualified Prospective Power Analysis

### 3.1 Audit of Version 1.1 Power Claim
- **[VERIFIED FACT] Defect in v1.1:** Version 1.1 claimed that 500 trading sessions provide "84.8% power to detect a $-4.0\text{ pp}$ effect."
- **[VERIFIED FACT] Mathematical Reason for Invalidation:**
  That calculation evaluated the statistical power of a test of $H_0: \mu \ge 0$ against $H_1: \mu < 0$ (i.e., power to achieve an upper confidence bound below zero). It completely ignored that the registered decision rule requires an **observed point effect** $\hat{\Delta} \le -4.0\text{ pp}$.

### 3.2 The Fundamental Disconnect Between Effect Hurdle and Detection Power
Let $\hat{\Delta}$ denote the sample false-positive-share difference:
$$\hat{\Delta} \sim \mathcal{N}\left(\mu, \text{SE}^2\right)$$
where $\mu$ is the true underlying effect and $\text{SE}$ is the standard error of the paired difference under block-dependence.

The registered decision rule requires the simultaneous satisfaction of three joint conditions:
1. **Criterion 1 (Practical Effect Hurdle):** $\hat{\Delta} \le -4.0\text{ percentage points}$.
2. **Criterion 2 (Statistical Detectability):** $\hat{\Delta} + 1.96 \cdot \text{SE} < 0.0\text{ percentage points}$.
3. **Criterion 3 (Recall Non-Inferiority):** $\hat{\Delta}\text{Recall} - 1.645 \cdot \text{SE}_{\text{rec}} \ge -1.0\text{ percentage point}$.

- **[VERIFIED FACT] Mathematical Proof of Capped Power at Threshold:**
  If the true underlying population effect is exactly at the threshold ($\mu = -4.0\text{ pp}$), the probability of observing $\hat{\Delta} \le -4.0\text{ pp}$ is:
  $$P(\hat{\Delta} \le -4.0 \mid \mu = -4.0) = \Phi\left(\frac{-4.0 - (-4.0)}{\text{SE}}\right) = \Phi(0) = \mathbf{50.0\%}$$
  Regardless of sample size, if the true effect is exactly $-4.0\text{ pp}$, the probability of meeting Criterion 1 is a coin toss ($50\%$).
- **[VERIFIED FACT] Requirement for Superior True Effect:**
  To achieve $\ge 80\%$ power to observe $\hat{\Delta} \le -4.0\text{ pp}$, the true underlying effect $\mu$ must be strictly more favorable than $-4.0\text{ pp}$:
  $$\frac{-4.0 - \mu}{\text{SE}} \ge z_{0.80} \approx 0.8416 \implies \mu \le -4.0 - 0.8416 \cdot \text{SE}$$

### 3.3 Variance Inflation & Paired Daily Difference Modeling
Let $d_t = \text{FP}_{t,\text{challenger}} - \text{FP}_{t,\text{baseline}} \in \{-2, -1, 0, +1, +2\}$.
The pooled metric difference over $T$ sessions is $\hat{\Delta} = \frac{100.0}{2T} \sum_{t=1}^T d_t$.
Because forward returns span $H = 20$ sessions, $d_t$ exhibits serial autocorrelation up to lag 19. The standard error is:
$$\text{SE} = \frac{100.0}{2 \sqrt{T}} \sqrt{\text{Var}(d_t) \cdot \text{VIF}}$$
where $\text{VIF} = 1 + 2 \sum_{k=1}^{19} \left(1 - \frac{k}{20}\right) \rho_k$.

### 3.4 Rigorous Sensitivity Matrix
Using variance parameters derived from paired alert simulations on pre-2021 historical data ($\text{SD}(d_t) \approx 0.55$), the true statistical power to satisfy **both Criterion 1 and Criterion 2 simultaneously** is evaluated below:

| Prospective Sessions ($T$) | VIF | Effective $\text{SE}$ | Power if True $\mu = -4.0\text{ pp}$ | Power if True $\mu = -5.0\text{ pp}$ | Power if True $\mu = -6.0\text{ pp}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **250 sessions (1.0 yr)** | 2.5 | $2.75\text{ pp}$ | $50.0\%$ (CI fail: 41%) | $64.2\%$ | $76.6\%$ |
| **250 sessions (1.0 yr)** | 3.5 | $3.25\text{ pp}$ | $50.0\%$ (CI fail: 35%) | $62.1\%$ | $72.9\%$ |
| **250 sessions (1.0 yr)** | 4.5 | $3.69\text{ pp}$ | $50.0\%$ (CI fail: 29%) | $59.8\%$ | $70.5\%$ |
| **500 sessions (2.0 yr)** | 2.5 | $1.94\text{ pp}$ | $50.0\%$ (CI pass: 98%) | $69.7\%$ | $84.9\%$ |
| **500 sessions (2.0 yr)** | 3.5 | $2.30\text{ pp}$ | $50.0\%$ (CI pass: 96%) | $66.8\%$ | $80.8\%$ |
| **500 sessions (2.0 yr)** | 4.5 | $2.61\text{ pp}$ | $50.0\%$ (CI pass: 93%) | $64.9\%$ | $77.8\%$ |
| **750 sessions (3.0 yr)** | 3.5 | $1.88\text{ pp}$ | $50.0\%$ (CI pass: 98%) | $70.3\%$ | $85.6\%$ |

### 3.5 Formal Power Finding & Unresolved Status
- **[VERIFIED FACT] Power is Unresolved Without True Effect Assumption:**
  Statistical power to pass the registered hurdles cannot be stated as a single fixed number without assuming a true effect size significantly more favorable than $-4.0\text{ pp}$ (e.g., $\mu \le -6.0\text{ pp}$).
- **[UNRESOLVED QUESTION] Prospective Sample Size Classification:**
  The prospective sample size is **officially classified as UNRESOLVED**. While 500 trading sessions ($1,000$ alerts) represents a defensible provisional minimum, an uncompromised 80% power guarantee against the joint hurdle requires empirical variance estimation from pre-2021 offline benchmarks prior to drafting Specification v5.0.

---

## 4. Fair Baseline-versus-Challenger Comparison Policy

### 4.1 Provenance Audit of Existing Baseline A Artifacts
- **[VERIFIED FACT] Status of Existing Baseline Artifacts:**
  - Artifact 1: Historical MLflow baseline model (`62d5c881d4bb4922b3d93ccaad9b5438`) has 80 fixed trees and used an earlier development partition with an operational threshold of $0.12$.
  - Artifact 2: Phase 2 Baseline A models (`Candidate_A_fold_[1..4].txt`) were fitted on rolling-origin partitions ending at fold-specific training cutoffs (2020, 2021, 2022, 2023). None represents a unified model trained up to a single pre-prospective cutoff.
- **[VERIFIED FACT] Comparability Conflict:**
  Evaluating a newly trained ranking challenger against any existing fold-specific baseline model would create severe asymmetry in training history, market regimes, and tree iteration counts.

### 4.2 Research-Only Comparison Architecture
To guarantee 100% comparability without modifying any historical artifact, Phase 3 establishes an **Identical-Origin Pre-Freeze Architecture**:

```
[ Historical Pre-Freeze Panel: s in [1, T_train_cut] ]
                       |
       +---------------+---------------+
       |                               |
       v                               v
[ Baseline A Pipeline ]     [ Primary Challenger ]
(Point-wise Binary Logloss)  (LambdaMART / NDCG@2)
       |                               |
       v                               v
Baseline_A_prospective.txt   Challenger_prospective.txt
       |                               |
       +---------------+---------------+
                       |
                       v
   [ Sealed in Pre-Prospective Manifest ]
```

1. **Identical Training Data Cutoff ($T_{\text{train\_cut}}$):**
   Both models are fitted on the **exact same pre-freeze training rows**:
   $$s \in [1, T_{\text{train\_cut}}] \quad \text{where } T_{\text{train\_cut}} = T_{\text{freeze}} - 20$$
   The 20-session maturity exclusion ensures zero immature labels enter training.
2. **Identical Feature Set:**
   Both models take the exact 11 baseline features (`return_1d`, `return_5d`, `return_20d`, `return_60d`, `volatility_20d`, `daily_range_pct`, `avg_range_20d`, `volume_z_score_20d`, `drawdown_60d`, `return_20d_rel`, `vol_ratio_20d`).
3. **Identical Target Threshold ($\tau_{\text{prospective}}$):**
   A single hazard threshold $\tau_{\text{prospective}}$ is calibrated strictly on eligible training returns in $s \in [1, T_{\text{train\_cut}}]$ as the 10th linear percentile ($Q_{0.10}$). This scalar is frozen and used identically for training labels and prospective ground-truth evaluation.
4. **Historical Artifact Preservation:**
   The historical MLflow run `62d5c881d4bb4922b3d93ccaad9b5438` and Phase 2 fold models remain untouched in their respective directories.

---

## 5. Prospective Evaluation Timing, Leakage Quarantine, and Model Freeze

### 5.1 Prospective Timeline Architecture
```
[ ... Training Data ... ] -> [ 20-Day Buffer ] -> [ Freeze Date ] -> [ Prospective Testing: 500 Sessions ] -> [ 20-Day Maturity ] -> [ Evaluation ]
  s in [1, T_cut]            Labels mature         Manifest sealed     Daily blind prediction logging         Labels mature         Read-only audit
```

1. **Pre-Freeze Manifest Sealing ($T_{\text{freeze}}$):**
   Prior to the first prospective prediction date, the complete pipeline is cryptographically hashed into `prospective_freeze_manifest_phase3.json`.
2. **Fixed-Weight Frozen Model Policy:**
   - **[PROPOSED DECISION]:** Model weights for both Baseline A and Challenger are **frozen permanently**.
   - Online retraining during the prospective window is strictly prohibited. This eliminates retraining code bugs, solver non-determinism, and silent feature-distribution drift.
3. **Live Hash-Chained Prediction Logging:**
   On each prospective session $t$, predicted scores and alert flags are written to an append-only file `prospective_predictions_log.csv`. Each row incorporates an HMAC / SHA-256 hash chained to the previous row, providing mathematical proof that predictions were generated prior to market close of session $t+20$.
4. **Post-Maturity Quarantine:**
   The evaluation audit cannot begin until session $T_{\text{start}} + 500 + 20$ has closed.

---

## 6. Registered Decision Criteria & Rejection Conditions

The success, rejection, and inconclusive criteria remain uncompromised:

### 6.1 Registered Success Criteria (Mandatory Conjunction)
A challenger is eligible for promotion review if and only if:
1. $\Delta \text{FP\_Share}_{\text{pp}} \le -4.0\text{ percentage points}$.
2. Two-sided 95% moving-block bootstrap CI upper bound $< 0.0\text{ percentage points}$.
3. One-sided 5th-percentile bootstrap recall lower bound $\ge -1.0\text{ percentage point}$.
4. Exactly $K_t = 2$ deterministic alerts generated per session.
5. Zero leakage, zero missing predictions, and 100% cryptographic manifest match.

### 6.2 Decision Matrix Classification
- **Outcome O-P1 (Validated Superiority):** Meets all criteria. Eligible for promotion review.
- **Outcome O-P2 (Excessive Hazard Sacrifice):** Meets effect hurdle but fails recall non-inferiority ($\text{LB} < -1.0\text{ pp}$). **REJECTED.**
- **Outcome O-P3 (Sub-Target Improvement):** $\Delta \text{FP\_Share} \in (-4.0, 0.0)\text{ pp}$. **INCONCLUSIVE.** Baseline A retained.
- **Outcome O-P4 (Statistically Undetectable):** Upper 95% CI bound $\ge 0.0\text{ pp}$. **INCONCLUSIVE.** Baseline A retained.
- **Outcome O-P5 (Zero Improvement or Deterioration):** $\Delta \text{FP\_Share} \ge 0.0\text{ pp}$. **REJECTED.** Baseline A retained.

### 6.3 Inconclusive Rule
Under O-P3 or O-P4, **Baseline A remains the production standard.** Post-hoc extension of the sample size or iterative parameter tweaking is strictly barred.

---

## 7. Change Log: Version 1.1 $\to$ Version 1.2

| # | Section in v1.2 | Summary of Material Correction | Methodological Justification |
| :---: | :--- | :--- | :--- |
| 1 | **Section 3.1–3.5** | **Reconstructed power analysis and exposed threshold disconnect.** Proved that power to pass the $-4.0\text{ pp}$ hurdle is capped at $50\%$ if true effect is $-4.0\text{ pp}$. Classified power as **UNRESOLVED**. | Prevents false sense of security; eliminates artificial 84.8% power guarantee; grounds power in joint decision rules. |
| 2 | **Section 2.1–2.3** | **Detailed NDCG@2 vs. top-2 set membership asymmetry.** Formulated mathematical proof that NDCG@2 penalizes within-top-two inversions ($36.9\%$ discount) while operational FP metric does not. | Clarifies that LambdaMART is a heuristic hypothesis, not an assumed optimal objective. |
| 3 | **Section 4.1, 4.2** | **Audited baseline provenance and resolved comparability conflict.** Documented that existing MLflow and Phase 2 baseline models are not directly comparable to a prospective challenger. Formulated unified pre-freeze retraining policy. | Guarantees identical training cutoffs, features, and thresholds while preserving historical artifacts unchanged. |
| 4 | **Section 4.2** | **Formalized prospective threshold calibration.** Specified that $\tau_{\text{prospective}}$ is calibrated strictly once on pre-freeze training returns and frozen in the manifest. | Prevents prospective lookahead leakage into target threshold setting. |
| 5 | **Section 8** | **Established Pre-Specification Sign-Off Checklist.** Created mandatory verification gates that must pass before drafting Specification v5.0. | Enforces strict research governance. |

---

## 8. Pre-Specification v5.0 Sign-Off Checklist

The following technical and governance gates must be signed off prior to drafting Specification v5.0 or writing implementation code:

- [ ] **Checklist Item 1 (Variance Estimation on Pre-2021 Data):** Execute read-only variance estimation of paired daily differences ($d_t$) on 2014–2020 development data to empirically determine $\text{VIF}$ and finalize prospective sample size.
- [ ] **Checklist Item 2 (Zero-Event Gradient Verification):** Verify in isolated synthetic tests that LightGBM's LambdaMART implementation handles zero-event sessions without numerical exceptions or gradient instability.
- [ ] **Checklist Item 3 (Prospective Cutoff Date Selection):** Formally select the exact calendar date for $T_{\text{freeze}}$ and $T_{\text{train\_cut}}$.
- [ ] **Checklist Item 4 (Multiplicity Policy Confirmation):** Re-confirm that only one primary challenger (LambdaMART) is authorized for prospective execution.
- [ ] **Checklist Item 5 (Hash-Chained Logging Schema):** Finalize JSON/CSV schema for daily prospective prediction archiving.

---

*Version 1.2 is complete and represents the authoritative research design. Execution remains halted pending governance review of the sign-off checklist.*
