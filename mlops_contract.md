# AQQIP MLOps Contract & Outcome Feedback Specification

## 1. Purpose

The MLOps layer closes the loop between a prediction made today and what actually happens afterward.

The system therefore treats every prediction as a hypothesis with a future evaluation date.

## 2. Prediction lifecycle

```text
GENERATED
    ↓
OPEN
    ↓
TARGET DATE PENDING
    ↓
ELIGIBLE FOR RESOLUTION
    ↓
RESOLVING
    ↓
RESOLVED
    ↓
TRAINING CANDIDATE
```

Failure states:

```text
RESOLUTION_FAILED
DATA_UNAVAILABLE
INVALID_TARGET
```

## 3. Prediction record

The current MySQL design stores:

- ticker,
- model_run_id,
- predicted probability,
- predicted_at,
- target_date,
- actual outcome fields,
- resolution timestamp,
- open/resolved status.

This is the canonical record for model accountability.

## 4. Daily resolver

The resolver performs:

```text
1. Query predictions where status = open.
2. Select predictions whose target_date has passed.
3. Retrieve the required market observation.
4. Compute actual outcome according to frozen target definition.
5. Write actual outcome fields.
6. Set status = resolved.
7. Record resolved_at.
```

## 5. Idempotent resolution

Running the resolver twice must not create two outcomes.

A resolved prediction remains resolved unless a controlled correction/reprocessing workflow is explicitly invoked.

## 6. Hard examples

After resolution, construct training metadata:

```text
true positive
false positive
true negative
false negative
```

The exact definition depends on the frozen prediction threshold and target label.

The retraining pipeline can prioritize hard examples for analysis, but must not automatically discard ordinary examples.

## 7. Retraining trigger

For v1, use a simple deterministic policy such as:

```text
scheduled retraining window
AND
sufficient new resolved observations
```

Do not retrain after every single prediction resolution unless an experiment explicitly tests that policy.

The threshold for "sufficient new observations" is configuration, not magic.

## 8. Model versioning

Every retrain creates a new MLflow run.

Required metadata:

```text
training dataset cutoff
feature schema version
prediction target version
training code version
Optuna configuration
random seed
metrics
artifact location
```

No model replaces the previous model without preserving the old artifact/run.

## 9. Pre/post feedback evaluation

A retraining cycle should be evaluated as an experiment:

```text
Model N
   ↓
resolve new observations
   ↓
Model N+1
```

Compare Model N and Model N+1 on an untouched temporal evaluation slice.

Do not declare improvement merely because training-set performance increased.

## 10. MARL reward adaptation

The existing architecture proposes adjusting MARL reward penalty coefficients based on rolling model accuracy.

For v1, make this conservative.

The reward adjustment module should output a versioned coefficient set:

```yaml
RewardConfig:
  risk_penalty: float
  inventory_penalty: float
  transaction_cost_weight: float
  adverse_selection_weight: float
  version: string
  source_metrics_window: object
```

Every simulation stores the reward configuration version it used.

## 11. Safe reward adaptation rule

Reward parameters should not be changed directly from one noisy observation.

Use a rolling evaluation window and bounded parameter updates.

Example concept:

```text
new_weight = clamp(
    old_weight + learning_rate * bounded_feedback_signal,
    min_weight,
    max_weight
)
```

The exact policy must be benchmarked before being adopted as the final configuration.

## 12. Data lineage

A user should be able to trace:

```text
Prediction P123
   ↓
MLflow run ABC
   ↓
Feature snapshot HASH1
   ↓
Target date D
   ↓
Actual outcome
   ↓
Training dataset version V2
   ↓
New MLflow run DEF
   ↓
New model artifact
   ↓
MARL reward version R3
```

## 13. MLOps monitoring metrics

Track:

- open predictions,
- resolved predictions,
- resolution success rate,
- mean resolution latency,
- false-positive rate,
- false-negative rate,
- rolling hit rate,
- calibration drift where measurable,
- model version currently used,
- number of retrains,
- reward configuration changes.

## 14. Backpressure and failure handling

If yfinance cannot retrieve the actual target-date observation:

```text
prediction remains OPEN
resolver records failure
retry later
```

Do not mark unresolved data as a negative example.

## 15. Governance

The MLOps loop is an **evaluation and adaptation mechanism**, not evidence that the system is autonomously improving in a scientifically proven sense.

Any claim such as "Model N+1 improved performance" must be supported by a controlled before/after evaluation.

