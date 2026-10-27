# AQQIP MARL Design Specification

## 1. Objective

The AQQIP MARL subsystem is a **controlled adversarial market simulation**, not a live trading engine and not a stock-price forecasting model.

Its purpose is to answer:

> Given a defined set of quantitative and qualitative financial-forensics evidence, what market dynamics can emerge when different participant types respond according to different incentives?

The simulation must remain reproducible, bounded, explainable, and experimentally testable.

## 2. User-facing meaning

The user never directly programs an agent. The user:

1. selects a supported security/ticker;
2. requests an analysis using the latest available data;
3. reviews quantitative and qualitative evidence;
4. chooses a simulation scenario;
5. runs the MARL environment;
6. sees simulated price path, drawdown, agent positions/rewards, and scenario assumptions.

The output must be labelled as a simulation scenario.

## 3. Real-world boundary

AQQIP models three stylized market participants:

- **Short Seller:** reacts to perceived downside evidence and seeks profit from price decline.
- **Retail Herd:** reacts to price/sentiment dynamics and may amplify momentum or panic.
- **Market Maker:** provides/reduces liquidity and manages inventory/adverse-selection risk.

These are abstractions. The environment is not claiming that all real market participants behave this way or that its transition equations are a complete market model.

## 4. Simulation clock

- One environment step = one trading day.
- Default episode length should be configurable and recorded; a 20-trading-day scenario is a reasonable initial benchmark because it is interpretable and bounded.
- Every episode begins from a real selected security's recent market state.
- Randomness is controlled with an explicit seed.

## 5. Environment state

Recommended state vector:

```text
market:
  normalized price return
  rolling volatility
  volume/liquidity proxy
  recent drawdown

forensic:
  quant_pressure_score
  semantic_shift_score
  evidence_concordance

agent state:
  short position
  retail position
  market-maker inventory
  cash / capital utilization

scenario state:
  day_index
  remaining_horizon
  volatility regime
```

All features are normalized before being passed to learned policies.

## 6. Action spaces

Use discrete actions for v1.

### Short Seller

```text
0 = Hold / maintain
1 = Increase short
2 = Reduce short / cover
```

### Retail Herd

```text
0 = Hold
1 = Buy
2 = Sell
```

### Market Maker

```text
0 = Normal liquidity
1 = Increase liquidity
2 = Reduce liquidity
```

Discrete actions make the first version more stable and align with the project rulebook's explicit v1 requirement.

## 7. Agent objectives

### 7.1 Short Seller

The Short Seller seeks positive P&L from downward price movement while being penalized for:

- excessive risk,
- transaction cost,
- excessive leverage/position size,
- invalid short inventory.

Conceptually:

```text
R_short = PnL - risk_penalty - transaction_cost - constraint_penalty
```

### 7.2 Retail Herd

The Retail Herd seeks portfolio P&L while incorporating price momentum and a herd-response term.

```text
R_retail = portfolio_return - transaction_cost - risk_penalty
```

The herd response should be configurable rather than hard-coded as a universal psychological law.

### 7.3 Market Maker

The Market Maker seeks spread/liquidity benefit while penalizing adverse selection and inventory risk.

```text
R_mm = liquidity/spread_benefit - inventory_risk - adverse_selection_penalty
```

The simulation does not need to implement a full limit-order book in v1.

## 8. Market transition model

The market transition should be intentionally simple enough to test.

A recommended conceptual structure is:

```text
net_pressure_t =
    short_pressure(A_short, Q, L)
  + retail_pressure(A_retail, price_momentum, sentiment_proxy)
  + liquidity_effect(A_mm)

return_t+1 =
    baseline_return
  + market_impact(net_pressure_t)
  + forensic_effect(Q, L)
  + random_noise_t
```

The exact coefficients must be configuration-driven and logged with each experiment.

The environment must prevent impossible outputs using hard bounds. Examples:

- price > 0,
- finite rewards,
- finite positions,
- bounded daily return shock,
- bounded inventory,
- no NaN/Infinity propagation.

## 9. How real model inference enters MARL

The quantitative and qualitative systems run **before** the simulation.

The integration sequence is:

```text
Selected ticker
    ↓
Data freshness check
    ↓
Quantitative inference
    ↓
Qualitative semantic analysis
    ↓
SignalBundle creation
    ↓
validation + normalization
    ↓
MARL scenario construction
    ↓
environment reset()
    ↓
agent observations
    ↓
agent actions
    ↓
market transition
    ↓
rewards + next state
    ↓
episode metrics
```

The MARL environment receives the complete `SignalBundle`, not a text prompt and not a raw LLM response.

## 10. Required MARL scenario modes

### Mode A — Baseline

No forensic signal influence.

### Mode B — Quantitative only

Use quantitative signal; clamp qualitative signal influence to zero.

### Mode C — Qualitative only

Use qualitative signal; clamp quantitative signal influence to zero.

### Mode D — Combined evidence

Use both channels.

### Mode E — Stress scenario

Use stronger volatility/liquidity stress without altering the evidence itself.

The dashboard must identify the scenario mode clearly.

## 11. Rule-based to learned progression

Do not start by requiring successful policy learning.

### Stage 1 — deterministic sanity agents

Implement transparent rules to prove the environment is correct.

Example:

```text
if quant_pressure_score > threshold:
    Short Seller increases short exposure
```

### Stage 2 — random policy baseline

Verify the learned/controlled system can be compared against chance behavior.

### Stage 3 — learned policies

Introduce the selected RL/MARL algorithm only after the environment passes deterministic tests.

This separation prevents policy-learning instability from hiding an environment bug.

## 12. Real-world market state initialization

A simulation begins using observed market context from the selected security:

- latest daily adjusted close,
- recent return window,
- recent volatility,
- volume/liquidity proxy.

The forensic signals modify scenario conditions; they do not rewrite the historical market record.

## 13. Reproducibility

Every run must store:

- random seed,
- signal bundle hash,
- model version(s),
- environment version,
- reward weights,
- market parameters,
- episode length,
- scenario mode.

Two runs with the same inputs, parameters and seed should reproduce the same result within numerical tolerance.

## 14. Primary metrics

At minimum:

- maximum drawdown,
- simulated price change,
- Short Seller P&L,
- Retail Herd P&L,
- Market Maker P&L,
- peak market-maker inventory,
- volatility,
- episode length,
- runtime,
- invalid/terminated steps.

## 15. What AQQIP must never claim

The MARL engine must never say:

- "the stock will fall X%,"
- "the real market will crash,"
- "the system guarantees a short profit,"
- "the simulation proves fraud."

The correct wording is:

> "Under the defined scenario assumptions, the simulator generated the following market-reaction scenario."

## 16. Failure containment

If the signal bundle is invalid, the MARL engine must refuse the run with a diagnostic error.

If one evidence channel is unavailable, the simulation can run in an explicitly labelled single-channel mode only when the user selects that mode.

If price, volatility, or liquidity inputs are missing, no simulation is generated.

