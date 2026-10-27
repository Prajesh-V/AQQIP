import pytest
from datetime import date, datetime, timedelta
from src.mlops.models import PredictionRecord
from src.mlops.resolver import MockMarketDataStore, MLOpsResolver, ModelPromoter, RewardAdapter

class MissingDataMarketStore(MockMarketDataStore):
    def get_price(self, ticker: str, trade_date: date) -> float | None:
        if ticker == "MISSING":
            return None
        # Entry price on 2025-01-01 is 100.0, exit on 2025-01-15 is 90.0
        if trade_date == date(2025, 1, 1):
            return 100.0
        return 90.0

def test_prediction_resolution():
    store = MissingDataMarketStore()
    resolver = MLOpsResolver(store)
    
    current_date = date(2025, 2, 1)
    
    preds = [
        # Target passed, price available -> should resolve (-10.0% drop)
        PredictionRecord(prediction_id=1, ticker="AAPL", model_run_id="run1", predicted_drop_probability=0.8, predicted_at=datetime(2025,1,1), target_date=date(2025,1,15)),
        # Target not passed -> should stay open
        PredictionRecord(prediction_id=2, ticker="MSFT", model_run_id="run1", predicted_drop_probability=0.2, predicted_at=datetime(2025,1,1), target_date=date(2025,2,15)),
        # Target passed, but market data missing -> should stay open
        PredictionRecord(prediction_id=3, ticker="MISSING", model_run_id="run1", predicted_drop_probability=0.5, predicted_at=datetime(2025,1,1), target_date=date(2025,1,15)),
        # Already resolved -> should be ignored (idempotent)
        PredictionRecord(prediction_id=4, ticker="TSLA", model_run_id="run1", predicted_drop_probability=0.9, predicted_at=datetime(2025,1,1), target_date=date(2025,1,15), status="resolved")
    ]
    
    resolved = resolver.resolve_predictions(preds, current_date)
    
    assert len(resolved) == 1
    assert resolved[0].prediction_id == 1
    assert resolved[0].status == "resolved"
    assert resolved[0].actual_price_change_pct == -10.0
    assert resolved[0].actual_drop_observed is True
    assert resolved[0].resolved_at is not None
    
    assert preds[1].status == "open"
    assert preds[2].status == "open"


def test_model_promotion():
    promoter = ModelPromoter()
    
    current = {"pr_auc": 0.80, "brier_score": 0.15}
    candidate_better = {"pr_auc": 0.85, "brier_score": 0.10}
    candidate_worse = {"pr_auc": 0.79, "brier_score": 0.18}
    candidate_mixed = {"pr_auc": 0.82, "brier_score": 0.16}
    
    assert promoter.evaluate_candidate(current, candidate_better) == True
    assert promoter.evaluate_candidate(current, candidate_worse) == False
    assert promoter.evaluate_candidate(current, candidate_mixed) == False

def test_reward_adaptation():
    adapter = RewardAdapter()
    
    initial = adapter.current_config
    
    # Adapt with poor performance
    adapted1 = adapter.adapt({"error_rate": 0.2})
    assert adapted1.risk_penalty > initial.risk_penalty
    assert adapted1.version == "v1.0.1"
    
    # Cap test
    adapted2 = adapter.adapt({"error_rate": 100.0})
    assert adapted2.risk_penalty <= 0.05 # Bounded
