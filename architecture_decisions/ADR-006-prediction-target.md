# ADR 006: Prediction Target Configuration

## Status
Accepted

## Context
Phase 3 requires a precise, configuration-driven target for the quantitative benchmark.
The model needs to predict a negative-tail event (e.g., a "sharp drop"). 
We need to define exactly what constitutes this event and avoid arbitrary static thresholds that might not fit the specific asset universe or market regime.

## Decision
We will define the target as a binary label:
`y_t = 1[R(t -> t+H) <= q_alpha(train)]`

Where:
- `H` (Target Horizon): 20 trading days. This captures approximately 1 calendar month of market reaction to SEC filings.
- `alpha` (Quantile): 0.10 (10th percentile). We define a "sharp drop" as performing in the bottom 10% of forward returns across the training set distribution.
- The threshold `q_alpha(train)` is calculated **exclusively on the training partition**.

## Consequences
- The event rate in the training set is strictly bounded near 10%.
- We prevent data leakage by isolating the threshold calculation from validation and test distributions.
- The binary classification problem has an explicit class imbalance (approx 9:1), which will be handled during model training using class weighting or calibrated probabilities.
