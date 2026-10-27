"""
AQQIP Predictor

Runs inference for new/unlabeled data and writes predictions to MySQL via MLOpsRepository.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any, List, Optional
import pandas as pd

from src.mlops.models import PredictionRecord
from src.mlops.repository import MLOpsRepository
from src.utils.logging import get_logger

logger = get_logger("predictor")


def compute_feature_hash(feature_series: pd.Series) -> str:
    """Computes a deterministic SHA-256 fingerprint for a single feature vector."""
    clean_dict = {k: round(float(v), 6) if pd.notnull(v) else None for k, v in feature_series.to_dict().items()}
    serialized = json.dumps(clean_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class Predictor:
    """Handles inference and persistence using validated feature sets and models."""

    def __init__(
        self,
        model: Any,
        features: list[str],
        model_run_id: str,
        model_version: Optional[str] = "quant_integrated_v1",
        drop_threshold_pct: float = 10.0,
        prediction_horizon_days: int = 20
    ):
        self.model = model
        self.features = features
        self.model_run_id = model_run_id
        self.model_version = model_version
        self.drop_threshold_pct = drop_threshold_pct
        self.prediction_horizon_days = prediction_horizon_days
        self.repository = MLOpsRepository()

    def predict_panel(
        self,
        panel: pd.DataFrame,
        target_dates: Optional[dict[str, date]] = None,
        default_target_date: Optional[date] = None
    ) -> List[PredictionRecord]:
        """
        Generates prediction records for rows in panel.
        Ensures exact feature ordering matching the trained model.
        """
        if panel.empty:
            logger.info("No data to predict")
            return []

        # Validate feature presence
        missing = [f for f in self.features if f not in panel.columns]
        if missing:
            logger.error("Missing features for prediction", missing=missing)
            raise ValueError(f"Missing required features: {missing}")

        # Ensure exact ordering
        X = panel[self.features]
        probs = self.model.predict(X)

        records = []
        for idx, (_, row) in enumerate(panel.iterrows()):
            ticker = str(row["ticker"])
            
            # Determine target date
            if target_dates and ticker in target_dates:
                t_date = target_dates[ticker]
            elif "target_date" in row and pd.notnull(row["target_date"]):
                t_val = row["target_date"]
                t_date = t_val.date() if isinstance(t_val, (datetime, pd.Timestamp)) else t_val
            elif default_target_date:
                t_date = default_target_date
            else:
                raise ValueError(f"Target date must be provided for ticker {ticker}")

            # Trade date / predicted_at
            trade_dt = row.get("trade_date")
            if isinstance(trade_dt, (pd.Timestamp, datetime)):
                predicted_at = trade_dt.to_pydatetime() if isinstance(trade_dt, pd.Timestamp) else trade_dt
            elif isinstance(trade_dt, date):
                predicted_at = datetime.combine(trade_dt, datetime.min.time())
            else:
                predicted_at = datetime.now(timezone.utc).replace(tzinfo=None)

            feat_hash = compute_feature_hash(row[self.features])
            pred_prob = float(probs[idx])

            record = PredictionRecord(
                ticker=ticker,
                cik=str(row.get("cik")) if pd.notnull(row.get("cik")) else None,
                model_run_id=self.model_run_id,
                model_version=self.model_version,
                predicted_drop_probability=pred_prob,
                input_feature_hash=feat_hash,
                predicted_at=predicted_at,
                target_date=t_date,
                prediction_horizon_days=self.prediction_horizon_days,
                drop_threshold_pct=self.drop_threshold_pct,
                status="open"
            )
            records.append(record)

        return records

    def predict_and_store(
        self,
        panel: pd.DataFrame,
        target_dates: Optional[dict[str, date]] = None,
        default_target_date: Optional[date] = None
    ) -> List[int]:
        """Generates predictions and persists them through MLOpsRepository."""
        records = self.predict_panel(panel, target_dates, default_target_date)
        stored_ids = []
        for rec in records:
            pred_id = self.repository.create_prediction(rec)
            stored_ids.append(pred_id)
        logger.info("Persisted predictions through MLOpsRepository", count=len(stored_ids))
        return stored_ids
