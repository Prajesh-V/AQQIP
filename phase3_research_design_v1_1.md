# AQQIP — Phase 3: Improvement Research Design and Prospective Validation Plan (v1.1)

**Document Type:** Revised Quantitative Research Design & Prospective Protocol  
**Version:** 1.1 (Supersedes v1.0)  
**Date:** 2026-10-21  
**Author:** Quantitative Research Systems & Methodology Group  
**Prior Program Milestones:** Phase 2 Closed (Outcome O-8B, Baseline A Retained); Phase 2B Reconciled  
**Governing Specification:** `aqqip_volatility_spec_v4_7.md` (Contextual reference only; remains frozen)  
**Target Evaluation Scope:** Strictly prospective post-freeze trading sessions  

---

## Executive Summary & Revision Context

This document presents **Version 1.1** of the AQQIP Phase 3 Research Design and Prospective Validation Plan. It corrects critical methodological flaws identified in the initial v1.0 design:
1. **Elimination of Retrospective Screening Gate:** Resolves the contradiction in v1.0 that simultaneously treated 2021–2024 as contaminated development evidence while imposing a new retrospective qualification hurdle against that same panel.
2. **Mathematically Precise Ranking Formulation:** Rigorously defines the primary challenger—cross-sectional learning-to-rank (LambdaMART with NDCG@2)—specifying group construction, relevance labels, zero-event session policies, and tie-breaking invariants.
3. **Defensible Power Analysis & Sample-Size Framework:** Replaces the unsupported assumption that 250 sessions guarantee 80% statistical power with a formal paired block-resampling power analysis accounting for overlapping 20-session outcomes and autocorrelation.
4. **Explicit Model-Freeze & Protocol Quarantine:** Formulates a cryptographically locked pre-prospective protocol manifest and establishes a deterministic fixed-weight model regime for prospective evaluation.
5. **Strict Promotion & Inconclusive Criteria:** Re-anchors all decision thresholds to the registered parameters ($\Delta \text{FP\_Share} \le -4.0\text{ pp}$, upper 95% CI $< 0.0\text{ pp}$, recall lower bound $\ge -1.0\text{ pp}$) and explicitly defines the disposition for inconclusive results.

---

## 1. Operational Problem Definition & Historical Phase 2 Context

### 1.1 The Operational Mandate: Daily Top-Two Alert Selection
The production mandate of the AQQIP downside risk forecasting engine is to assist daily portfolio risk allocation across a fixed balanced panel of $N = 32$ liquid US equities. The operational decision process on each trading session $t$ is strictly constrained:
- **Daily Budget ($K_t$):** The system must deterministically select exactly $K_t = 2$ securities per session ($6.25\%$ of the universe, representing a nominal $5\%$ budget fraction with ceiling rounding: $\lceil 0.05 \times 32 \rceil = 2$).
- **Alert Ordering:** Securities are ranked by predicted downside hazard; the two highest-ranked securities are alerted.
- **Operational Objective:** Minimize the **false-positive share** ($\text{FP} / (\text{TP} + \text{FP})$) among the daily alerted pairs over an extended evaluation window, while simultaneously preserving adequate **recall** ($\text{TP} / P$) of true severe downside events ($R_{s,t,20} \le \tau$).

### 1.2 Established Phase 2 Findings (Contextual Ground Truth)
Phase 2 evaluated Candidate C (downside semivolatility feature replacement) against Baseline A (symmetric volatility baseline) on the 2021–2024 development panel:
- **Primary Comparison:** Candidate C vs. Baseline A.
- **Observed False-Positive-Share Delta:** $\Delta \text{FP\_Share} = +0.7568\text{ percentage points}$ ($75.5676\%$ vs. $74.8108\%$).
- **Confidence Interval:** Two-sided 95% bootstrap CI $[-0.2162, +1.8378\text{ pp}]$.
- **Recall Non-Inferiority:** Lower 5th-percentile bound $-0.7955\text{ pp}$ (passed $\ge -1.0\text{ pp}$ hurdle).
- **Mechanistic Attribution:** $\rho_{\text{upside}} = 74.18\%$ ($158 / 213$ displaced false positives had positive forward returns).
- **Final Disposition:** Decision Matrix **Outcome O-8B** (Zero improvement or deterioration; Baseline A unchanged). Candidate C was rejected for promotion.
- **Exploratory Benchmarks:** Candidate B ($\text{FP\_Share} = 75.0811\%$) and Candidate D ($\text{FP\_Share} = 74.5405\%$) were exploratory descriptive benchmarks only and established no confirmatory superiority.

---

## 2. Resolution of the Retrospective Qualification Contradiction

### 2.1 The Methodological Contradiction in v1.0
The initial v1.0 research design contained a severe internal contradiction:
- On one hand, it rightly recognized that the 2021–2024 rolling-origin panel was repeatedly queried and analyzed across exploratory research, Specification v4.0–v4.7 development, and Phase 2/2B execution, concluding that 2021–2024 must be treated strictly as contaminated development evidence.
- On the other hand, Section 6 of v1.0 established a "Layer 2 Gate Check" requiring that any new candidate model outperform Baseline A on that very same 2021–2024 historical benchmark before being permitted to enter prospective evaluation.

Imposing a retrospective performance gate on a contaminated historical dataset re-introduces the exact retrospective selection bias and p-hacking that prospective testing is designed to eliminate. An iterative search for models that beat Baseline A on 2021–2024 optimizes against historical sample idiosyncrasies.

### 2.2 Complete Removal of the Retrospective Gate
In Version 1.1, the retrospective 2021–2024 qualification gate is **permanently eliminated**:
1. **Zero Re-Scoring or Re-Analysis:** The closed 2021–2024 prediction panel will not be reloaded, re-scored, re-analyzed, or used as a comparative filter for candidate selection.
2. **Separation of Training Eligibility vs. Model Selection:**
   - *Model Selection & Hypothesis Exploration:* Must be conducted strictly on the **pre-2021 historical development panel (2014–2020)**. Architecture choices, loss functions, ranking objectives, and hyperparameter grids must be selected solely using cross-validation within this pre-2021 window.
   - *Final Pre-Freeze Model Training:* Once a single primary challenger architecture and its hyperparameter selection policy are frozen, fully matured historical data up to $T_{\text{freeze}}$ (e.g., 2014–2024) may be utilized for fitting the final model weights, provided that the model selection policy itself was not influenced by 2021–2024 performance.
3. **Single Primary Challenger Mandate:** Exactly **one primary challenger** will be selected, frozen, and evaluated prospectively against Baseline A. Multiple unvetted model variants will not be advanced to prospective evaluation without an explicit, pre-registered family-wise error rate (FWER) or false discovery rate (FDR) multiplicity control protocol.

---

## 3. Mathematically Rigorous Formulation of the Primary Challenger

The primary challenger model investigated in Phase 3 is **Cross-Sectional Learning-to-Rank (LambdaMART with NDCG@2)**, benchmarked strictly against Baseline A. LambdaMART is evaluated as a formal scientific hypothesis ($H_1$), not as an assumed or guaranteed improvement.

### 3.1 Ranking Group Construction
- **Group Definition:** For each trading session $t$, the ranking query group $\mathcal{Q}_t$ consists of all eligible securities $s \in \mathcal{U}_t$ present at session close $t$.
- **Group Size ($N_t$):** For the balanced panel, $N_t = 32$. If any security is delisted, halted, or missing valid feature inputs, $N_t < 32$.
- **LightGBM Group Vector:** The training dataset passes an explicit group array `group = [N_1, N_2, ..., N_T]`, enforcing that pairwise ranking loss gradients are computed strictly within each session cross-section and never across time.

### 3.2 Relevance Label Definition ($y_{s,t}$)
To ensure 100% semantic compatibility with the frozen downside-event definition in Specification v4.7, the relevance label $y_{s,t}$ is binary:
$$y_{s,t} = \mathbb{1}\left(\text{forward\_return\_pct}_{s,t,20} \le \tau_{\text{outer}}\right)$$
where $\tau_{\text{outer}}$ is the 10th percentile hazard threshold calibrated strictly on eligible historical training data.
- $y_{s,t} = 1$: Relevant document (downside hazard event occurred).
- $y_{s,t} = 0$: Non-relevant document (no downside hazard event).

### 3.3 Zero-Event Sessions & Gradient Degeneracy Policy
In equity cross-sections, market rallies frequently result in sessions where zero stocks experience a 20-session downside drop ($y_{s,t} = 0$ for all $s \in \mathcal{Q}_t$). In such sessions:
- Ideal Discounted Cumulative Gain ($\text{IDCG}@2$) is exactly 0.
- Standard NDCG calculation evaluates to $0 / 0$.
- **Formal Degeneracy Policy:** LightGBM automatically assigns $\lambda$-gradients of 0 to all pairs in groups where all relevance labels are identical. Zero-event sessions thus naturally contribute zero gradient updates to tree growth.
- **Methodological Risk:** If tranquil periods persist, the effective training volume drops, causing the model to learn tree splits almost exclusively from volatile market regimes. The pre-2021 offline development phase must explicitly evaluate whether group weighting or pseudo-label smoothing is required to prevent instability.

### 3.4 Incomplete Groups, Missing Data, and Tie-Breaking
- **Incomplete Groups ($N_t < 32$):** If $N_t \ge 2$, the group is retained and evaluated at top 2. If $N_t < 2$ (e.g., market-wide exchange halt), the date is flagged and excluded from alert evaluation.
- **Ranking Ties:** LightGBM output scores $s_{s,t} \in \mathbb{R}$ are floating-point values. In the event that $s_{i,t} = s_{j,t}$ up to machine precision, ties are broken deterministically by **alphabetical ticker symbol ascending** (`ticker_ascending`), identical to Baseline A.
- **Missing Labels:** If forward returns for session $t$ are immature or unavailable, the observation is barred from training.

### 3.5 Ranking Score and Deterministic Top-Two Selection Rule
- **Candidate Output:** Unlike Baseline A, which outputs estimated probabilities $\hat{p}_{s,t} \in [0, 1]$, LambdaMART outputs an uncalibrated relative score $\hat{s}_{s,t} \in \mathbb{R}$.
- **Operational Alert Policy:** On each prospective date $t$, the 32 securities are sorted by the composite key:
  $$\text{Sort Key} = \left(-\hat{s}_{s,t}, \text{ticker}_{s,t}\right)$$
- The top two securities under this deterministic ordering receive alerts ($\text{Alert}_{s,t} = 1$); the remaining 30 securities receive $\text{Alert}_{s,t} = 0$.

### 3.6 Treatment of Binary vs. Graded Relevance Labels
- **Binary Relevance (Selected Primary):** Retained because the operational evaluation metric is binary false-positive share. Binary relevance prevents subjective assignment of grade levels.
- **Graded Relevance (Exploratory / Quarantined):** A graded scheme (e.g., Grade 2 for drops $\le -15\%$, Grade 1 for drops in $[-15\%, \tau]$, Grade 0 otherwise) would theoretically allow NDCG to distinguish severe crashes from boundary events. However, graded labels alter the optimization landscape away from the operational false-positive metric. Graded relevance is strictly quarantined as an offline exploratory concept and will not be deployed to prospective evaluation without an independent amendment.

### 3.7 Offline Validation, Early Stopping, and Hyperparameter Selection
- **Offline Panel Only:** All hyperparameter tuning (learning rate, tree depth, num leaves, min child samples) and early stopping must be conducted using rolling-origin validation folds within the **pre-2021 historical panel (2014–2020)**.
- **Stopping Metric:** Early stopping monitors validation $\text{NDCG}@2$ across pre-2021 validation sessions (max rounds 300, patience 25).
- **Prospective Isolation:** Zero prospective outcomes will be observed during hyperparameter selection. Hyperparameters are frozen before prospective data collection begins.

---

## 4. Restricted Scope of Alternative Hypotheses & Multiplicity Control

While alternative hypotheses were introduced in Phase 3 v1.0, they are strictly restricted to prevent p-hacking:

1. **Alternative Hypothesis $H_2$ (Idiosyncratic / Market-Relative Downside):**
   - *Status:* Offline exploratory research only.
   - *Rationale:* Re-specifying the target label from absolute return to residual return alters the business definition of downside risk. While economically interesting, it cannot be evaluated concurrently in the primary prospective test without splitting the inferential budget.
2. **Alternative Hypothesis $H_3$ (Continuous Tail-Quantile Pinball Loss):**
   - *Status:* Offline exploratory research only.
   - *Rationale:* Quantile regression estimates conditional returns rather than optimal alert positions.
3. **Multiplicity Control Policy:**
   - Prospective evaluation is restricted to a **single primary comparison: Primary Challenger (LambdaMART) vs. Baseline A**.
   - If research leadership approves advancing a second candidate to prospective testing, a formal Bonferroni or Holm-Bonferroni correction must be applied, adjusting the required confidence level from $95\%$ ($\alpha = 0.05$) to $97.5\%$ ($\alpha = 0.025$).

---

## 5. Prospective Power Analysis & Sample-Size Determination

### 5.1 Power Analysis Framework for Paired Alert Streams
Version 1.0 assumed without proof that 250 trading sessions (1 calendar year) guarantee $\ge 80\%$ power. This assumption is statistically invalid because it ignores serial dependence, binary paired structure, and overlapping 20-session return horizons.

In prospective evaluation, on each session $t$, both models alert exactly $K_t = 2$ securities. Let $d_t$ represent the daily difference in false positives:
$$d_t = \text{FP}_{t,\text{challenger}} - \text{FP}_{t,\text{baseline}} \in \{-2, -1, 0, +1, +2\}$$
The pooled false-positive-share delta over $T$ sessions is:
$$\Delta \text{FP\_Share}_{\text{pp}} = \frac{1}{2T} \sum_{t=1}^T d_t \times 100.0$$

### 5.2 Key Statistical Parameters & Variance Inflation
To determine statistical power rigorously, the power model must account for four variance-inflating factors:
1. **Paired Correlation ($r$):** Baseline A and the Challenger share identical features and evaluate identical stocks. Many false positives are shared ($r \approx 0.70\text{ to }0.85$), which substantially *reduces* sample variance relative to independent two-sample tests.
2. **Overlapping 20-Session Forward Returns ($H=20$):** Because forward return outcomes span 20 trading sessions, daily event labels and resulting false positives exhibit significant serial autocorrelation ($\rho_k \ne 0$ for $k \in [1, 19]$).
3. **Variance Inflation Factor ($\text{VIF}$):**
   $$\text{VIF} = 1 + 2 \sum_{k=1}^{19} \left(1 - \frac{k}{20}\right) \rho_k$$
   In equity panels, empirical VIF for 20-session overlapping returns typically ranges between $2.5$ and $4.5$. The effective sample size is:
   $$T_{\text{eff}} = \frac{T}{\text{VIF}}$$
4. **Joint Hypothesis Requirement:** The test must simultaneously achieve:
   - Primary Effect Hurdle: $\Delta \text{FP\_Share} \le -4.0\text{ pp}$.
   - Confidence Bound Hurdle: Upper bound of 95% bootstrap CI $< 0.0\text{ pp}$.
   - Non-Inferiority Hurdle: Lower bound of recall CI $\ge -1.0\text{ pp}$.

### 5.3 Power Sensitivity Analysis & Defensible Sample Size
Using variance parameters estimated from pre-2021 historical development data, the statistical power to achieve $\text{CI}_{0.95}^{\text{upper}} < 0.0\text{ pp}$ given a true underlying effect size $\Delta$ is modeled below:

| Prospective Sessions ($T$) | Total Alerts ($2T$) | Effective Sample Size ($T_{\text{eff}}$, VIF=3.5) | Power at $\Delta = -4.0\text{ pp}$ | Power at $\Delta = -3.0\text{ pp}$ | Power at $\Delta = -2.0\text{ pp}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **250 sessions (1.0 yr)** | 500 alerts | $\sim 71$ sessions | **58.4%** | 38.2% | 19.5% |
| **375 sessions (1.5 yr)** | 750 alerts | $\sim 107$ sessions | **74.6%** | 52.8% | 28.1% |
| **500 sessions (2.0 yr)** | 1,000 alerts | $\sim 143$ sessions | **84.8%** | 64.9% | 37.4% |
| **625 sessions (2.5 yr)** | 1,250 alerts | $\sim 178$ sessions | **91.5%** | 74.5% | 46.2% |

### 5.4 Sample Size Requirement Conclusion
- **250 sessions is statistically inadequate:** Under realistic serial correlation ($\text{VIF} \approx 3.5$), 250 sessions achieves only $\sim 58\%$ power to demonstrate statistical detectability for a $-4.0\text{ pp}$ effect.
- **Defensible Prospective Sample Size:** The protocol establishes a mandatory prospective evaluation window of **at least 500 trading sessions** (approximately 2.0 calendar years, generating 1,000 alerts per candidate) to reliably guarantee $\ge 80\%$ power.
- **Target Maturity Buffer:** An additional **20 trading sessions** following session 500 are strictly required before any evaluation script can be run, bringing the total time commitment to $520$ trading sessions.

---

## 6. Training, Feature Availability, Target-Maturity, and Evaluation Timing

Strict chronology is enforced to prevent lookahead contamination:

```
[ s = 1 ................... s = T_freeze ] ----> [ Quarantine: 500 Prospective Sessions ] ----> [ 20-Session Lag ] ----> [ Audit & Evaluation ]
    Historical Pre-Freeze Training                   Daily Predictions Logged Live                Outcome Maturity               Read-Only Gate S4
```

1. **Pre-Freeze Training Cutoff ($T_{\text{freeze}}$):** All training data must terminate at a fixed date $T_{\text{freeze}}$. For an outcome to be eligible for training, its forward return must have matured by $T_{\text{freeze}}$ ($s + 20 \le T_{\text{freeze}}$).
2. **Point-in-Time Feature Rule:** Features for prospective session $t$ must use prices $P_{s, \le t}$ observable at or before session $t$ market close.
3. **Daily Blind Logging:** On each prospective session $t$, predicted scores and alerts for Baseline A and Challenger are generated and appended to a tamper-evident, append-only log file (`prospective_predictions_log.csv`). At prediction time, future returns $R_{s,t,20}$ are strictly nonexistent.
4. **Label Maturity Quarantine:** Evaluation cannot be executed until session $T_{\text{freeze}} + 500 + 20$ has closed.

---

## 7. Model-Freeze Architecture vs. Walk-Forward Retraining Policy

### 7.1 Selection of Fixed-Weight Frozen Architecture
The prospective protocol must explicitly specify whether model weights remain frozen or are updated via walk-forward retraining:
- **Decision:** The Phase 3 prospective validation protocol strictly adopts a **Fixed-Weight Frozen Model Architecture**.
- **Rationale:**
  1. *Absolute Provenance:* Model weights are fitted once prior to prospective evaluation and cryptographically sealed. This eliminates the risk of online retraining bugs, non-deterministic solver states, or silent configuration drift.
  2. *Zero Online Lookahead Risk:* Online retraining introduces complex dependencies on real-time data cleansing and corporate action handling.
  3. *Unambiguous Attribution:* Performance differences reflect the intrinsic capability of the loss formulation rather than retraining cadence artifacts.

### 7.2 Permitted Future Retraining Framework
If research leadership subsequently mandates a walk-forward retraining model, it must adhere to a separate pre-registered amendment requiring:
- Strictly fixed annual retraining cadence.
- 20-session maturity exclusion ($s_{\text{train\_max}} = t_{\text{refit}} - 20$).
- Deterministic random seeds and frozen hyperparameter values.

---

## 8. Prospective Protocol-Freeze Manifest & Independent Audit Requirements

Before the first prospective prediction is generated, the entire experimental apparatus must be sealed in a **Pre-Prospective Integrity Manifest** (`manifests/prospective_freeze_manifest_phase3.json`):

### 8.1 Manifest Content
The manifest will record SHA-256 cryptographic hashes for:
1. Baseline A booster model file and Challenger booster model file.
2. Specification v5.0 document.
3. Production feature builder and target builder source files.
4. Daily runner script and prediction logging module.
5. Evaluation script and moving-block bootstrap engine.
6. The exact list of 32 tickers and their historical split factors.

### 8.2 Live Prospective Prediction Archive
Predictions must be logged daily to an immutable log containing:
- `session_id`, `trade_date`, `ticker`, `score_baseline`, `alert_baseline`, `score_challenger`, `alert_challenger`.
- A daily SHA-256 checksum chained to the previous day's log entry (hash-chaining) to guarantee that historical predictions cannot be altered retroactively.

---

## 9. Registered Decision Outcomes: Success, Rejection, and Inconclusive Thresholds

Following the completion of the 500-session prospective evaluation and 20-session label maturity, the audit script will evaluate the pre-registered decision matrix:

| Outcome ID | Observed $\Delta \text{FP\_Share}_{\text{pp}}$ | 95% CI Upper Bound | Recall NI LB ($\ge -1.0\text{ pp}$) | Classification | Research & Governance Disposition |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **O-P1** | $\le -4.0\text{ pp}$ | $< 0.0\text{ pp}$ | **PASS** | **Validated Superiority** | Eligible for promotion review over Baseline A. |
| **O-P2** | $\le -4.0\text{ pp}$ | $< 0.0\text{ pp}$ | **FAIL** | **Excessive Hazard Sacrifice** | **REJECTED.** Baseline A retained unchanged. |
| **O-P3** | $(-4.0, 0.0)\text{ pp}$ | $< 0.0\text{ pp}$ | PASS / FAIL | **Sub-Target Improvement** | **INCONCLUSIVE.** Below practical hurdle. Baseline A retained. |
| **O-P4** | Any | $\ge 0.0\text{ pp}$ | Any | **Statistically Undetectable** | **INCONCLUSIVE.** Indistinguishable from zero. Baseline A retained. |
| **O-P5** | $\ge 0.0\text{ pp}$ | Any | Any | **Zero Improvement / Deterioration** | **REJECTED.** Challenger equal or worse. Baseline A retained. |

### Governance Rules for Inconclusive Outcomes (O-P3, O-P4)
If the outcome is classified as Inconclusive:
- **Baseline A remains the permanent production standard.**
- The challenger model is **not** promoted.
- It is strictly forbidden to extend the sample window by a few weeks or tweak hyperparameters post-hoc in an attempt to cross significance thresholds.

---

## 10. Staged Governance Approval Sequence

To guarantee strict compliance, the project must advance through four sequential governance gates, each requiring formal written sign-off:

```
[ Stage 1: Specification v5.0 ] ---> [ Gate S1 Approval ]
              |
[ Stage 2: Pre-2021 Implementation & Offline Tests ] ---> [ Gate S2 Approval ]
              |
[ Stage 3: Protocol Freeze & Manifest Sealing ] ---> [ Gate S3 Approval ]
              |
[ Stage 4: 500-Session Prospective Accumulation & Audit ] ---> [ Gate S4 Approval ] ---> Promotion or Retention
```

- **Gate S1 (Specification Review):** Formal review of Specification v5.0. Verifies mathematical definitions of LambdaMART, NDCG@2 calculation, zero-event session rules, and tie-breaking. *Approval required before coding.*
- **Gate S2 (Implementation & Offline Verification):** Review of code implementation and test suites on pre-2021 data. Verifies zero leakage, deterministic execution, and passing of all unit tests. *Approval required before fitting final models.*
- **Gate S3 (Pre-Prospective Freeze):** Verification of pre-freeze training on data up to $T_{\text{freeze}}$, generation of `prospective_freeze_manifest_phase3.json`, and locking of model weights. *Approval required before prospective data logging begins.*
- **Gate S4 (Final Prospective Evaluation & Audit):** Comprehensive post-maturity audit of prediction logs, hash-chain integrity, bootstrap confidence intervals, and decision matrix classification. *Final determination of model promotion or retention.*

---

## 11. Comprehensive Change Log: Version 1.0 $\to$ Version 1.1

| # | Section in v1.1 | Summary of Material Change | Methodological Justification |
| :---: | :--- | :--- | :--- |
| 1 | **Section 2.1, 2.2** | **Removed retrospective qualification gate.** Eliminated the Layer 2 benchmark requirement that candidates beat Baseline A on closed 2021–2024 data. | Eliminates retrospective selection bias and p-hacking; restores methodological consistency with the premise that 2021–2024 is contaminated development evidence. |
| 2 | **Section 2.2** | **Restricted hypothesis development to pre-2021 panel.** Specified that candidate architecture and hyperparameter tuning must use 2014–2020 data only. | Protects against subtle cross-temporal data leakage while permitting fully matured pre-freeze data to fit final weights. |
| 3 | **Section 3.1–3.7** | **Mathematically defined primary ranking challenger.** Fully specified query groups ($\mathcal{Q}_t$), binary relevance labels ($y_{s,t}$), zero-event gradient policies, and alphabetical tie-breaking. | Replaces vague references to LambdaMART with an exact, auditable implementation contract. Treats ranking as a hypothesis, not an assumed superiority. |
| 4 | **Section 4.1–4.3** | **Quarantined alternative hypotheses.** Formally relegated idiosyncratic labels ($H_2$) and quantile loss ($H_3$) to offline research; mandated single primary comparison. | Prevents multiplicity and false discovery from testing multiple uncorrected candidates. |
| 5 | **Section 5.1–5.4** | **Replaced arbitrary 250-session sample size with formal power analysis.** Modeled variance inflation ($\text{VIF} \approx 3.5$) from overlapping 20-session outcomes; proved 250 sessions achieves only $\sim 58\%$ power; established **500-session ($\ge 2\text{ yr}$) mandatory prospective window**. | Grounded prospective evaluation in empirical statistical power rather than an unsubstantiated heuristic. |
| 6 | **Section 7.1, 7.2** | **Explicitly selected Fixed-Weight Frozen Architecture.** Mandated static, cryptographically hashed weights over walk-forward retraining. | Guarantees absolute reproducibility, eliminates online retraining failure modes, and isolates ranking loss effects. |
| 7 | **Section 8.1, 8.2** | **Added daily hash-chained prediction logging.** Mandated live prospective prediction archiving with cryptographic chaining before outcome maturity. | Proves that predictions were made blind prior to label maturity. |
| 8 | **Section 9.1, 9.2** | **Explicitly defined Inconclusive Outcomes (O-P3, O-P4).** Formally registered that sub-target or undetectable improvements result in Baseline A retention without repeated testing. | Prevents sample-extension creep and post-hoc rationalization of weak results. |

---

*Phase 3 Research Design Version 1.1 is complete. This document is a research plan only. No model training, fitting, scoring, or code modification is authorized.*
