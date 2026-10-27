# ADR-008: Evidence-Based Model Promotion

## Status
Accepted

## Context
The MLOps loop retrains models as new resolved predictions provide ground truth. A naive approach would automatically replace the active model with any newly trained model. This risks deploying a worse model due to noisy training data or overfitting.

## Decision
Maintain the concept of:
- **Current production-eligible model** — the model actively used for predictions
- **Candidate model** — a newly trained model awaiting evaluation

A candidate model must pass the following before promotion:
1. Schema validation (features match expected schema)
2. Metric comparison against incumbent on temporal holdout
3. Leakage check (no future data in training set)
4. Calibration check where sample size permits
5. Reproducibility check (same seed produces same output)

Only if the candidate demonstrates improvement (or non-degradation within tolerance) on the holdout set is it promoted.

## Alternatives Considered
- Always-latest model → rejected (risks silent degradation)
- Manual human review of every model → impractical for automated loop
- Champion/challenger A/B testing → deferred for v1 (requires concurrent prediction infrastructure)

## Rationale
Evidence-based promotion prevents a single noisy retraining from corrupting the active model. All model artifacts remain in MLflow for comparison and rollback.

## Consequences
- MLflow must store all candidate runs, not just promoted ones
- The promotion check adds latency to the retraining cycle
- Old model artifacts are never deleted
- The resolver always records which model version made each prediction
