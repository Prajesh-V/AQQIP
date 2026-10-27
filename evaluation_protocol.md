# AQQIP Evaluation Protocol

## 1. Purpose

This document defines how AQQIP will be evaluated experimentally. It is the source of truth for benchmarks, baselines, validation splits, metrics, acceptance criteria, and reporting rules.

The principle is simple:

> Every claimed capability must have a measurable test, a baseline where appropriate, and a reproducible experiment record.

## 2. Evaluation hierarchy

```text
Data integrity
   ↓
Deterministic formula correctness
   ↓
Model predictive evaluation
   ↓
Qualitative evidence evaluation
   ↓
MARL environment evaluation
   ↓
MLOps feedback evaluation
   ↓
End-to-end system evaluation
```

A later stage cannot be considered successful merely because it produces a visually convincing output.

## 3. Data leakage policy

The primary financial prediction evaluation must be chronological.

Do not use random row-level train/test splits when they permit information from the future to influence training examples from the past.

The preferred structure is:

```text
Historical period A ──► Train
Historical period B ──► Validation / tuning
Historical period C ──► Final holdout test
```

The exact date ranges are to be determined from the actual dataset after ingestion and recorded in the experiment configuration.

Any preprocessing that learns from data, including normalization, calibration and feature selection, must be fit only on training data.

## 4. Prediction target governance

Before the first official LightGBM benchmark, freeze the target definition in configuration.

Required fields:

```text
prediction_horizon_trading_days
sharp_drop_threshold_pct
price_field_used_for_outcome
```

A proposed starting design is a binary event defined by a selected percentage decline in adjusted close over a selected trading-day horizon. This is a design choice, not an established project result, and must be confirmed before the benchmark run.

## 5. Quantitative engine evaluation

### 5.1 Deterministic formula tests

Beneish and Altman implementations must be compared against known hand-calculated or reference test cases.

Tests:

- normal values,
- edge values,
- missing values,
- denominator zero protection,
- sign-sensitive fields,
- repeated calculation determinism.

### 5.2 Model baselines

Benchmark at least:

1. majority-class baseline,
2. simple heuristic baseline,
3. price-only model,
4. ratio-only model,
5. ratio + price model,
6. integrated model with qualitative signal where the experiment is explicitly testing that extension.

### 5.3 Primary metrics

Report:

- ROC-AUC,
- PR-AUC,
- precision,
- recall,
- F1,
- confusion matrix,
- Brier score,
- calibration curve / calibration error where sample size permits.

The PRD currently proposes AUC-ROC >= 0.70 and top-decile anomaly precision >= 60% as aspirational success targets; these must be treated as project acceptance targets, not guaranteed outcomes. fileciteturn17file4L78-L89

### 5.4 Stability tests

Repeat the evaluation across multiple time windows or rolling temporal folds where dataset size permits.

Report mean and dispersion across folds rather than a single lucky holdout number.

### 5.5 Optuna evaluation

Record:

- search-space definition,
- number of trials,
- best validation metric,
- test metric for the selected model,
- random seed,
- runtime.

The test set must never be used to choose the Optuna configuration.

## 6. Qualitative engine evaluation

### 6.1 Extraction correctness

Create a manually checked sample containing known Risk Factors sections.

Measure:

- successful section extraction rate,
- extraction failure rate,
- false/truncated extraction rate.

A failed extraction must be recorded as failed, not silently converted into an empty or tiny document.

### 6.2 Semantic-diff benchmark

Prepare a labeled sample of genuine before/after filing changes.

For each pair, manually record:

- expected additions,
- expected deletions,
- expected meaningful shifts.

Measure:

- agreement rate,
- false positive change rate,
- false negative change rate,
- threshold sensitivity.

### 6.3 Threshold sensitivity

Run the semantic-diff detector at multiple cosine thresholds around the configured default and observe how the change set size and agreement rate vary.

Do not select a threshold on the final evaluation sample.

### 6.4 LLM interpretation evaluation

Gemini and Ollama are interpretation engines, not the authoritative detector.

For a sampled set, reviewers should score:

- groundedness in supplied evidence,
- factual consistency with the diff,
- unsupported claims,
- clarity.

A summary must not introduce a claim not supported by the retrieved filing text.

## 7. MARL evaluation

The MARL environment is evaluated at three levels.

### Level 1 — Environment sanity

- reset/step API correctness,
- valid observation/action spaces,
- finite rewards,
- positive prices,
- bounded positions,
- deterministic output under fixed seed.

### Level 2 — Behavior sanity

Use deterministic rule agents and verify expected directional responses to controlled signal changes.

Example:

```text
Increase quantitative pressure
        ↓
Short-seller conviction should not decrease
```

This is a monotonicity sanity test, not a claim about actual markets.

### Level 3 — Scenario experiments

Use the experiment matrix in `marl_experiment_matrix.md`.

Primary comparisons:

- baseline,
- quant-only,
- qualitative-only,
- combined evidence,
- stress condition.

## 8. MARL stability benchmark

Each reported MARL result should be averaged across repeated random seeds where computationally reasonable.

Report:

```text
mean
standard deviation
minimum
maximum
```

This is more informative than one animated run.

## 9. MLOps evaluation

Verify the complete lifecycle:

```text
Prediction created
    ↓
Target date passes
    ↓
Resolver retrieves actual outcome
    ↓
Prediction marked resolved
    ↓
Training dataset updates
    ↓
New MLflow run created
    ↓
New model artifact recorded
```

Measure:

- resolver correctness,
- idempotency of repeated resolver runs,
- percentage of eligible predictions resolved,
- retrain execution success,
- run traceability.

## 10. End-to-end acceptance

A complete ticker-analysis test case must demonstrate:

```text
Ticker selected
   ↓
CIK resolved
   ↓
Price data available
   ↓
10-K / 10-Q status known
   ↓
Quantitative signals computed
   ↓
Qualitative comparison computed
   ↓
SignalBundle created
   ↓
MARL scenario executed
   ↓
Results persisted
   ↓
Dashboard renders the same persisted result
```

The test passes only if the identifiers and timestamps remain traceable across every stage.

## 11. Performance benchmarks

Measure separately:

- price ingestion latency,
- filing detection latency,
- filing download latency,
- Risk Factors extraction latency,
- embedding latency,
- semantic diff latency,
- LLM interpretation latency,
- LightGBM inference latency,
- MARL runtime,
- total analysis time.

The current PRD sets an aspirational end-to-end processing target of under five minutes per new ticker and a single MARL scenario target under 30 seconds on the development laptop. fileciteturn17file4L82-L88

## 12. Real-world reliability tests

Simulate:

- SEC API timeout,
- SEC rate limiting,
- malformed filing,
- unavailable primary document,
- missing previous filing,
- yfinance failure,
- incomplete financial facts,
- Gemini unavailable,
- Ollama unavailable,
- database restart,
- duplicate ingestion,
- stale cached data.

The expected behavior is graceful degradation, explicit status, retry where appropriate, and no silent corruption.

## 13. Result-reporting rules

Every reported result must state:

- dataset definition,
- time window,
- target definition,
- model/config version,
- random seed(s),
- metrics,
- baseline,
- limitations.

Never present an achieved result that was not produced by an actual experiment run.

