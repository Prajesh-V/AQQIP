# ADR-004: Time-Aware Validation

## Status
Accepted

## Context
Financial prediction models are especially vulnerable to data leakage because time-series data has natural temporal ordering. A model trained on future data and evaluated on past data produces artificially inflated metrics.

## Decision
All model training, tuning, and evaluation must use chronological splits:
```
Historical period A → Train
Historical period B → Validation / tuning
Historical period C → Final holdout test
```

Random row-level train/test splits are prohibited for the primary evaluation. Any preprocessing that learns from data (normalization, calibration, feature selection) must be fit only on training data.

## Alternatives Considered
- Random K-fold cross-validation → rejected (permits future information leakage)
- Rolling-window validation → considered and may be used additionally where dataset size permits, but base requirement is chronological split

## Rationale
Financial time-series has autocorrelation and regime changes. Random splits create optimistic bias. The evaluation protocol requires temporal integrity for any claimed metric to be meaningful.

## Consequences
- Dataset must be sorted by date before splitting
- Feature engineering must only use information available as of prediction timestamp
- The target definition's `target_date` must always be in the future relative to `predicted_at`
- Optuna must optimize on validation set only; test set is never used for hyperparameter selection
- Smaller effective training sets in early time periods
