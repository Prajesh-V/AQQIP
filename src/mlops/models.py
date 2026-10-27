from typing import Optional, List
from datetime import date, datetime
from pydantic import BaseModel

class PredictionRecord(BaseModel):
    prediction_id: Optional[int] = None
    ticker: str
    cik: Optional[str] = None
    model_run_id: str
    model_version: Optional[str] = None
    predicted_drop_probability: float
    calibrated_probability: Optional[float] = None
    input_feature_hash: Optional[str] = None
    predicted_at: datetime
    target_date: date
    prediction_horizon_days: int = 20
    drop_threshold_pct: float = 10.00
    actual_drop_observed: Optional[bool] = None
    actual_price_change_pct: Optional[float] = None
    resolved_at: Optional[datetime] = None
    status: str = "open"


class RewardConfig(BaseModel):
    version: str
    risk_penalty: float
    inventory_penalty: float
    transaction_cost_weight: float
    adverse_selection_weight: float
    source_metrics_window: int
    updated_at: datetime
