# AQQIP — Phase 3: Improvement Research Design and Prospective Validation Plan

**Document Type:** Quantitative Research Design & Prospective Validation Protocol  
**Date:** 2026-10-21  
**Author:** Quantitative Research Systems & Methodology Group  
**Prior Program Milestone:** Phase 2 Closed (Outcome O-8B, Baseline A Retained)  
**Governing Context:** Phase 2 Closure Addendum (`phase2_closure_addendum.md`)  
**Target Evaluation Scope:** Strictly prospective post-freeze trading sessions  

---

## 1. Precise Operational Problem and Proposed Research Hypotheses

### 1.1 The Operational Problem: Daily Top-Two Alert Selection
The operational mandate of the AQQIP downside risk forecasting engine is not generic cross-sectional classification or global probability calibration. Rather, the system executes a highly specific daily portfolio-selection decision:

1. **Fixed Universe:** Every trading session $t$, a fixed balanced panel of $N = 32$ liquid US equities is evaluated.
2. **Fixed Daily Budget:** The system selects an alert budget of exactly $K_t = 2$ securities per session ($6.25\%$ of the universe, corresponding to a nominal $5\%$ fraction with ceiling rounding: $\lceil 0.05 \times 32 \rceil = 2$).
3. **Operational Objective:** Minimize the **false-positive share** ($\text{FP} / (\text{TP} + \text{FP})$) among those two selected alerts over an extended horizon, while simultaneously preserving an acceptable rate of hazard detection (**recall** $\text{TP} / P$). Over a typical 250-session trading year, exactly 500 alerts are issued.

### 1.2 The Structural Disconnect in Baseline A
Baseline A minimizes global point-wise binary log-loss:
$$\mathcal{L}_{\text{logloss}} = - \frac{1}{|\mathcal{D}|} \sum_{(s,t) \in \mathcal{D}} \left[ y_{s,t} \log p_{s,t} + (1 - y_{s,t}) \log(1 - p_{s,t}) \right]$$

This objective exhibits three severe structural misalignments with the operational decision:
- **Rank-Insensitive Loss Distribution:** Binary log-loss penalizes probability estimation errors uniformly across all 32 stocks. An error that promotes a non-event security from rank 28 to rank 22 incurs an identical loss penalty to an error that promotes a non-event security from rank 3 to rank 2. Yet in production, the 28-to-22 shift has zero impact, whereas the 3-to-2 shift displaces a potential true alert and generates an operational false positive.
- **Confounding by Macro Time-Series Shifts:** Under market-wide drawdown regimes (e.g., March 2020 or 2022 rate-hike selloffs), the unconditional hazard probability across all 32 stocks surges simultaneously. Under tranquil regimes, all probabilities drop. Binary cross-entropy expends model capacity fitting aggregate market volatility rather than resolving the cross-sectional risk ordering among the 32 stocks on day $t$.
- **Artificial Target Discretization:** Transforming forward returns $R_{s,t,20}$ into a binary indicator $y_{s,t} = \mathbb{1}(R_{s,t,20} \le \tau)$ completely discards information regarding the severity of price drops (a drop of $-25\%$ is treated identically to a drop of $-7.1\%$ when $\tau = -7.0\%$).

### 1.3 Competing Research Hypotheses
Rather than assuming that a specific technique will succeed, Phase 3 establishes four competing, mutually exclusive research hypotheses:

- **Hypothesis $H_1$ (Objective-Alignment Hypothesis — Learning to Rank):**
  Directly optimizing a list-wise ranking objective (e.g., LambdaMART with NDCG@2 or pairwise ranking margin grouped by trading session) will reduce daily top-two false-positive share relative to Baseline A by focusing tree-split gain exclusively on pairwise inversions at the top of the daily cross-section, while ignoring irrelevant ordering in the bottom 30 stocks.
- **Hypothesis $H_2$ (Target-Decomposition Hypothesis — Idiosyncratic Downside):**
  The current absolute return target conflates systematic market downturns with stock-specific collapse. Re-specifying the downside hazard label as relative/idiosyncratic downside return ($R_{s,t,20} - R_{\text{mkt},t,20} \le \tau_{\text{idio}}$) will eliminate macro confounding and improve stock-picking selectivity.
- **Hypothesis $H_3$ (Continuous Tail-Quantile Hypothesis — Pinball Loss):**
  Replacing binary classification with continuous extreme-quantile regression at the 10th percentile ($\alpha = 0.10$) preserves continuous tail-magnitude information during tree construction, producing more granular risk estimates that translate to superior top-two ordering.
- **Null Hypothesis $H_0$ (Dominance / Sufficiency of Baseline A):**
  Cross-sectional stock ordering in this liquid 32-stock universe is dominated by idiosyncratic high-frequency noise. Neither ranking loss, label restructuring, nor quantile regression will generate genuine out-of-sample false-positive reduction relative to Baseline A under strictly prospective, leak-free evaluation.

---

## 2. Comparison of Plausible Training Objectives

A rigorous model-improvement program must evaluate alternative objective functions against mathematical properties, operational suitability, and implementation risks before committing to code:

| Evaluation Dimension | 1. Point-wise Binary Log-Loss (Baseline A) | 2. List-wise Ranking (LambdaMART / NDCG@2) | 3. Continuous Quantile Loss (Pinball Loss $\alpha=0.10$) | 4. Cost-Sensitive Focal Cross-Entropy |
| :--- | :--- | :--- | :--- | :--- |
| **Mathematical Formulation** | $-\sum [y \log p + (1-y)\log(1-p)]$ | Lambda-gradient scaled pairwise loss: $\Delta \text{NDCG} \cdot \frac{\partial \mathcal{L}_{\text{pair}}}{\partial s}$ | $\max(\alpha(y - \hat{y}), (\alpha - 1)(y - \hat{y}))$ on continuous returns $y = R_{s,t,20}$ | $-\alpha_t (1 - p_t)^\gamma \log(p_t)$ with focusing parameter $\gamma$ |
| **Operational Alignment** | **Poor.** Optimizes global calibration across all ranks. | **Exact.** Directly optimizes top-two positions within each daily session. | **Moderate.** Preserves tail ordering; does not explicitly optimize top-two rank. | **Poor.** Emphasizes hard individual examples, but remains point-wise. |
| **Time-Series Confounding** | **High.** Affected by macro regime shifts in base rates. | **Zero.** Ranking is strictly intra-day (cross-sectional within session $t$). | **Moderate.** Subject to market-wide volatility shifts in return levels. | **High.** Still suffers from time-series probability shifts. |
| **Output Type** | Calibrated posterior probability $p_{s,t} \in [0, 1]$. | Unbounded relative ranking score $s_{s,t} \in \mathbb{R}$. | Estimated conditional return quantile $\hat{q}_{0.10} \in \mathbb{R}$. | Uncalibrated pseudo-probability $p_{s,t}^* \in [0, 1]$. |
| **Sparsity & Gradient Risks** | Smooth, convex gradients for every observation. | **High risk on sparse days:** On sessions where 0 of 32 stocks experience an event, $\lambda$-gradients vanish or become ill-conditioned. | Piecewise linear loss; subgradient is a step function ($\alpha$ or $\alpha-1$), requiring careful learning rate control. | Non-convex parameter tuning for $\gamma$ and $\alpha_t$. |
| **Implementation Complexity** | Minimal. Existing native LightGBM setup. | **High.** Requires session grouping (`Dataset(group=...)`), custom evaluation hooks, and empty-group handling. | Moderate. Native `objective="quantile"` in LightGBM, but changes label pipeline to continuous returns. | Low to moderate. Custom objective or standard LightGBM sample weights. |

### Methodological Recommendation
**List-wise Ranking (LambdaMART with NDCG@2)** represents the most theoretically sound alternative to Baseline A, because it directly matches the operational metric. However, its primary vulnerability is **label sparsity**: when fewer than two stocks experience downside events on day $t$, NDCG@2 is either undefined or zero-gradient. Therefore, any ranking candidate must incorporate a mathematically sound daily weighting or smoothing rule to prevent degeneracy on tranquil days.

---

## 3. Primary Metric and Registered Success Criteria

To prevent post-hoc reinterpretation, Phase 3 establishes strict, immutable success hurdles:

### 3.1 Primary Metric: Daily False-Positive-Share Difference
The primary evaluative metric is the difference in pooled false-positive share between Candidate and Baseline A across all selected top-two alerts over the prospective evaluation window:
$$\Delta \text{FP\_Share}_{\text{pp}} = \text{FP\_Share}(\text{Candidate}) - \text{FP\_Share}(\text{Baseline A})$$
where $\text{FP\_Share} = \frac{\text{FP}}{\text{TP} + \text{FP}} \times 100.0$.

### 3.2 Registered Success Criteria (Mandatory Hurdles)
A candidate model will be considered successful if and only if it simultaneously satisfies all four of the following criteria:

1. **Practical Effect Size Hurdle:**
   $$\Delta \text{FP\_Share}_{\text{pp}} \le -4.0\text{ percentage points}$$
   The candidate must reduce the pooled false-positive share by at least 4.0 percentage points relative to Baseline A (e.g., if Baseline A achieves $74.8\%$, Candidate must achieve $\le 70.8\%$).
2. **Statistical Detectability Hurdle:**
   $$\text{CI}_{0.95}^{\text{upper}}(\Delta \text{FP\_Share}_{\text{pp}}) < 0.0\text{ percentage points}$$
   The upper bound of the two-sided 95% moving-block bootstrap confidence interval (block length $L=20$ sessions, $B=1,000$ replicates) must be strictly negative, establishing that the observed reduction is statistically distinguishable from zero at $\alpha = 0.05$.
3. **Recall Non-Inferiority Hurdle (Bounded Degradation):**
   $$\text{LB}_{0.05}(\Delta \text{Recall}_{\text{pp}}) \ge -1.0\text{ percentage point}$$
   The one-sided 5th-percentile bootstrap lower bound for recall change must be $\ge -1.0\text{ percentage point}$. A reduction in false positives achieved solely by abandoning hazard detection is unacceptable.
4. **Budget Invariant:**
   The candidate must deterministically generate exactly $K_t = 2$ alerts per session across all 32-stock dates, breaking ties deterministically by ascending ticker symbol.

---

## 4. Design for Strictly Prospective Evaluation

### 4.1 Justification for Prospective Protocol
The 2021–2024 development panel was repeatedly queried across exploratory phases, Specification v4.0–v4.7 development, and Phase 2 execution. The 2025–2026 data was previously inspected for diagnostic rebound analyses. Consequently, **both periods represent historical development evidence, not uncorrupted holdouts.** Any further model selection on 2014–2026 data would constitute retrospective over-fitting and p-hacking.

Phase 3 requires a **strictly prospective evaluation design**:

```
[ Historical Pre-Freeze Panel (2014-2024) ] ---> [ Protocol & Code Freeze ] ---> [ Quarantine / Clock Running ] ---> [ Label Maturity ] ---> [ Prospective Evaluation ]
         (Model Training & Tuning)                      (SHA-256 Manifest)             (No Peeking / Scoring)           (Wait H=20 Days)          (Single Read-Only Audit)
```

### 4.2 Timeline and Quarantine Architecture
1. **Pre-Freeze Training Panel (2014–2024):** All model fitting, hyperparameter selection, and algorithm development are conducted exclusively on data up to a specified historical cutoff $T_{\text{freeze}}$ (e.g., 2024-12-31).
2. **Protocol and Artifact Freeze:** Once the candidate architecture and hyperparameters are selected, the codebase, model weights, feature builder, and evaluation code are cryptographically hashed and sealed in an integrity manifest.
3. **Quarantine Accumulation Window:** The prospective evaluation window begins at $T_{\text{start}} > T_{\text{freeze}}$. During this window, daily market data accumulates in real time. **Zero model retraining, hyperparameter tuning, or preliminary scoring against the accumulating panel is permitted.**
4. **Minimum Evaluation Sample Size:** To achieve adequate statistical power ($\ge 80\%$) to detect a 4.0 percentage point reduction in false-positive share, the prospective test window must encompass:
   - Minimum Duration: $\ge 250$ consecutive trading sessions (approx. 1 calendar year).
   - Universe: 32 stocks balanced panel ($250 \times 32 = 8,000$ observations).
   - Total Alerts: Exactly 500 alerts per candidate ($250 \times 2 = 500$).

---

## 5. Point-in-Time Data Requirements and Leakage Prevention

### 5.1 The 20-Session Target Maturity Invariant
The prediction target is the forward 20-session return outcome:
$$R_{s,t,20} = \frac{P_{s,t+20} - P_{s,t}}{P_{s,t}}$$

- **Unobservability Rule:** For decision date $t$, the label is strictly unobservable until the official market close of session $t+20$.
- **Evaluation Timing Restriction:** The prospective evaluation audit cannot be initiated until **at least 20 trading sessions have elapsed** following the final date of the prospective window ($T_{\text{end}} + 20\text{ sessions}$). Premature calculation of labels using interpolated or truncated returns is strictly prohibited.

### 5.2 Point-in-Time Feature Alignment
- All feature vectors for session $t$ must be computed strictly using price and volume history available at or before session $t$ close ($P_{s, \le t}$).
- Historical corporate actions (splits, cash dividends) must be adjusted using split factors known as of date $t$. Future corporate actions must not be retroactively back-adjusted into historical features prior to their effective date.

### 5.3 Target Threshold Quarantine ($\tau_{\text{outer}}$)
The outer evaluation threshold $\tau_{\text{outer}}$ must be calibrated strictly on historical training observations preceding $T_{\text{freeze}}$. Under no circumstances may prospective evaluation returns be included in threshold quantile calculations.

---

## 6. Training / Validation / Evaluation Separation Strategy

To ensure zero repeated model selection on the prospective panel, Phase 3 establishes a strict multi-layer separation strategy:

1. **Layer 1: Inner Development & Hyperparameter Tuning (Historical Pre-2021)**
   - Rolling origin folds spanning 2014–2020 are used for exploratory architecture comparisons, loss formulation testing, and hyperparameter tuning.
2. **Layer 2: Sanity / Benchmark Replication Panel (Historical 2021–2024)**
   - Acknowledged as previously examined development evidence.
   - Used strictly as a **Gate Check**: If a new candidate cannot outperform Baseline A on this historical benchmark, it is eliminated immediately and will not proceed to prospective testing.
3. **Layer 3: Sealed Prospective Evaluation Panel (Post-Freeze)**
   - Strictly single-use.
   - Evaluated exactly once.
   - If the candidate fails the prospective evaluation, it is permanently rejected. No iterative re-tuning or re-testing against this panel is allowed.

---

## 7. Explicit Rejection Conditions

A proposed Candidate model will be **instantly and permanently rejected**, and Baseline A maintained as the production model, if any of the following conditions occur:

1. **Practical Failure:** $\Delta \text{FP\_Share}_{\text{pp}} > -4.0\text{ percentage points}$ on the prospective panel.
2. **Detectability Failure:** The upper 95% bootstrap confidence bound $\text{CI}_{0.95}^{\text{upper}}(\Delta \text{FP\_Share}_{\text{pp}}) \ge 0.0\text{ percentage points}$.
3. **Excessive Recall Degradation:** The lower 5th percentile bootstrap bound for recall $\text{LB}_{0.05}(\Delta \text{Recall}_{\text{pp}}) < -1.0\text{ percentage point}$.
4. **Structural / Budget Failure:** The candidate outputs non-finite scores, produces ties that prevent selecting exactly 2 alerts, or fails to alert on exactly 2 securities per date.
5. **Leakage or Provenance Defect:** Any audit finding demonstrating lookahead leakage, premature label maturity scoring, or failure of code hashes to match the pre-freeze manifest.

---

## 8. Staged Implementation Plan and Governance Gates

Phase 3 requires formal, independent approval at four successive gates before any candidate can be promoted:

```
[ Stage 1: Specification ] ---> [ Gate S1 Approval ]
            |
[ Stage 2: Implementation ] ---> [ Gate S2 Approval ]
            |
[ Stage 3: Code Freeze ]    ---> [ Gate S3 Approval ]
            |
[ Stage 4: Prospective Run ] --> [ Gate S4 Final Audit ] ---> Promotion / Retention
```

### Stage 1: Mathematical Specification & Protocol Design
- Produce **Specification v5.0** detailing exact mathematical formulations for candidate objectives (e.g., LambdaMART group construction, tie-breaking rules, quantile loss definition).
- *Gate S1 Review:* Independent audit verifying MECE properties, lack of leakage, and strict threshold inequalities. **Approval required before writing code.**

### Stage 2: Code Implementation & Test Suite Validation
- Implement candidate modules in isolated, modular packages without touching Baseline A code.
- Write automated test suite verifying:
  - Deterministic group-wise sorting.
  - Zero gradient NaN exceptions on zero-event sessions.
  - 100% finite outputs across edge cases.
- *Gate S2 Review:* Verification that all unit tests pass, pre-execution gates pass, and code integrity hashes are generated. **Approval required before model fitting.**

### Stage 3: Benchmark Replication and Protocol Freeze
- Fit Candidate and Baseline A on historical training partitions.
- Verify candidate achieves expected operational behavior on historical data.
- Generate and seal the **Pre-Prospective Integrity Manifest**, locking all source code, model weights, and protocol parameters.
- *Gate S3 Review:* Formal authorization to lock the models and begin prospective data accumulation. **Zero changes permitted post-freeze.**

### Stage 4: Prospective Evaluation Execution and Read-Only Audit
- Wait for prospective window accumulation plus 20 sessions of label maturity.
- Execute strictly read-only scoring and moving-block bootstrap evaluation.
- Conduct independent provenance audit (analogous to Phase 2B).
- *Gate S4 Review:* Quantitative Governance Board review of audit results to determine final disposition.

---

## 9. Minimum Evidence Required Before Promotion Over Baseline A

Baseline A represents the battle-tested, fully audited production standard. To justify the operational disruption and model risk of replacing Baseline A, the following minimum body of empirical evidence is mandatory:

1. **Prospective Duration:** At least 250 prospective trading days ($\ge 500$ alerts) gathered post-freeze.
2. **Statistically Significant Improvement:** Confirmed $\ge 4.0\text{ percentage point}$ reduction in false-positive share with a two-sided 95% confidence interval strictly below zero.
3. **Preserved Recall:** Confirmed recall lower bound within 1.0 percentage point of Baseline A.
4. **Complete Audit Trail:** 100% cryptographic match on all sealed runner scripts, models, and prediction tables.
5. **Multi-Stakeholder Governance Sign-Off:** Unanimous written authorization from Quantitative Research, Portfolio Risk Management, and MLOps Engineering.

In the absence of this complete evidentiary package, **Baseline A remains the permanent, uncompromised production standard.**
