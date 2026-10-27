# AQQIP MARL Experiment Matrix

## Purpose

This document defines the controlled MARL experiments that will demonstrate whether the quantitative and qualitative evidence streams change simulated market dynamics in a measurable and reproducible way.

## 1. Required experiment matrix

| ID | Scenario | Quant signal | Qualitative signal | Purpose |
|---|---|---:|---:|---|
| M0 | Random baseline | 0 | 0 | Establish chance behavior |
| M1 | Rule-based baseline | 0 | 0 | Verify environment mechanics |
| M2 | Quantitative only | ON | 0 | Test accounting/market anomaly influence |
| M3 | Qualitative only | 0 | ON | Test disclosure-shift influence |
| M4 | Combined evidence | ON | ON | Test integrated evidence |
| M5 | Stress market | ON | ON | Test robustness under higher volatility/liquidity stress |

M4 is the primary demonstration scenario.

## 2. M0 — Random baseline

Purpose:

- establish a lower-bound behavioral reference;
- detect whether the environment produces extreme outcomes even when no signal is supplied;
- verify that downstream results are not being caused solely by implementation artifacts.

Run across multiple seeds.

Record:

- drawdown,
- agent P&L,
- inventory,
- volatility,
- runtime.

## 3. M1 — Rule-based baseline

Implement transparent policies before learned agents.

Example policy logic:

```text
if Q > threshold_high:
    Short Seller → Increase Short
elif Q < threshold_low:
    Short Seller → Hold/Reduce
```

Retail reacts to recent momentum, while the Market Maker reacts to inventory/adverse-selection pressure.

This experiment is primarily a **software correctness benchmark**.

## 4. M2 — Quantitative-only experiment

Input:

```text
Q = quant_pressure_score
L = 0
```

Question:

> What simulated dynamics emerge when market participants receive only the quantitative forensic signal?

## 5. M3 — Qualitative-only experiment

Input:

```text
Q = 0
L = semantic_shift_score
```

Question:

> What simulated dynamics emerge when the numerical forensic engine is removed and only disclosure-language evidence remains?

## 6. M4 — Combined evidence

Input:

```text
Q = quant_pressure_score
L = semantic_shift_score
```

This is the core AQQIP experiment.

The important comparison is not "did the simulator crash?" but:

```text
M4 versus M2
M4 versus M3
M4 versus M1
```

Metrics of interest:

- maximum drawdown,
- simulated price change,
- Short Seller P&L,
- Retail Herd P&L,
- Market Maker inventory risk,
- volatility,
- response timing.

## 7. M5 — Stress experiment

Hold forensic evidence fixed while increasing market stress parameters.

Example dimensions:

```text
volatility: low → medium → high
liquidity: high → medium → low
```

The purpose is robustness, not proving that real markets follow the model.

## 8. Signal-strength sweep

Run controlled experiments over:

```text
Q ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
L ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

This creates a grid of signal combinations.

The resulting surface can show whether the simulator responds smoothly or exhibits unstable threshold behavior.

## 9. Evidence-concordance experiment

Compare cases where the two channels agree versus disagree.

### Concordant high

```text
Q = 0.8
L = 0.8
```

### Quant-dominant

```text
Q = 0.8
L = 0.2
```

### Qual-dominant

```text
Q = 0.2
L = 0.8
```

### Weak evidence

```text
Q = 0.2
L = 0.2
```

This is important because the whole AQQIP value proposition is not merely "more signals" but **complementary evidence streams**.

## 10. Seed stability

Each core scenario should be repeated over multiple seeds, for example:

```text
101
202
303
404
505
```

Store each run independently.

Report:

```text
mean ± standard deviation
min
max
```

## 11. Agent-behavior sanity checks

### Short Seller

Higher quantitative/qualitative downside evidence should not systematically cause lower short conviction under an otherwise identical controlled setup.

### Retail Herd

A stronger negative simulated price signal should not systematically make the retail agent more bullish if the policy explicitly encodes momentum/herd response in the same direction.

### Market Maker

Higher adverse-selection/inventory risk should not produce increased market-making exposure when the policy is designed to reduce that exposure.

These are **internal consistency tests**, not claims about universal real-world behavior.

## 12. Policy-learning experiments

Only after the environment passes rule-based tests:

```text
random policy
       ↓
independent learned policies
       ↓
MARL policy variant
```

The exact learning algorithm is to be selected after measuring environment stability and compute cost. The v1 success criterion is bounded, reproducible simulation behavior; learned-equilibrium quality is not a release blocker.

## 13. Ablation experiments

Remove one component at a time:

```text
full system
  ├── remove quantitative evidence
  ├── remove qualitative evidence
  ├── remove market-maker interaction
  ├── remove retail feedback
  └── remove stochastic noise
```

This helps determine whether the final behavior is genuinely produced by the intended architecture.

## 14. Reproducibility artifact

Every MARL experiment must save:

```yaml
ExperimentRecord:
  experiment_id: string
  scenario_id: string
  ticker: string
  seed: int
  environment_version: string
  signal_bundle_hash: string
  agent_policy_version: string
  reward_version: string
  environment_parameters: object
  outputs: object
  runtime_ms: float
```

## 15. Presentation-ready final comparison

The final demo should show a compact table:

| Scenario | Q | L | Max Drawdown | Short P&L | Retail P&L | MM Risk |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 0 | measured | measured | measured | measured |
| Quant only | ✓ | — | measured | measured | measured | measured |
| Qual only | — | ✓ | measured | measured | measured | measured |
| Combined | ✓ | ✓ | measured | measured | measured | measured |

Values must be populated from actual experiment runs only.

