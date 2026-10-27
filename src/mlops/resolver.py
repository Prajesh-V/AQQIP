"""
AQQIP MLOps Prediction Resolver

Resolves open predictions against real market data using the frozen target contract.
Queries MySQL ohlcv_prices for actual market observations.
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import date, datetime, timezone
from src.mlops.models import PredictionRecord, RewardConfig
from src.database.mysql_client import get_mysql_connection
from src.utils.logging import get_logger

logger = get_logger("mlops_resolver")


class MySQLMarketDataStore:
    """Queries real OHLCV market observations from MySQL ohlcv_prices table."""

    def __init__(self, price_field: str = "adj_close"):
        self.price_field = price_field

    def get_price(self, ticker: str, trade_date: date) -> Optional[float]:
        """
        Fetches the observed price for ticker on trade_date.
        Never substitutes hardcoded or fabricated prices.
        Returns None if no price row exists in MySQL.
        """
        query = f"""
            SELECT {self.price_field}
            FROM ohlcv_prices
            WHERE ticker = %s AND trade_date = %s
            LIMIT 1
        """
        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (ticker, trade_date))
            row = cursor.fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return None


class MockMarketDataStore:
    """In-memory market data store for isolated unit tests."""

    def __init__(
        self,
        prices: Optional[Dict[Tuple[str, date], float]] = None,
        default_price: Optional[float] = None
    ):
        self.prices: Dict[Tuple[str, date], float] = prices or {}
        self.default_price = default_price

    def set_price(self, ticker: str, trade_date: date, price: float) -> None:
        self.prices[(ticker, trade_date)] = price

    def get_price(self, ticker: str, trade_date: date) -> Optional[float]:
        if (ticker, trade_date) in self.prices:
            return self.prices[(ticker, trade_date)]
        return self.default_price


class MLOpsResolver:
    """
    Resolves predictions using the security, prediction date, and target date.
    Calculates actual price change using the frozen target contract convention:
    actual_price_change_pct = (exit_price - entry_price) / entry_price * 100.0.
    Remains idempotent across repeated runs.
    Leaves predictions open when target date hasn't occurred or market data is missing.
    """

    def __init__(self, market_store: Optional[Any] = None):
        self.market_store = market_store if market_store is not None else MySQLMarketDataStore()

    def resolve_predictions(
        self,
        open_predictions: List[PredictionRecord],
        current_date: Optional[date] = None
    ) -> List[PredictionRecord]:
        resolved = []
        for pred in open_predictions:
            # 1. Idempotency: Skip already resolved predictions
            if pred.status != "open":
                continue

            # 2. Target date check: Leave open if target date hasn't passed relative to current_date
            if current_date is not None and pred.target_date > current_date:
                continue

            # 3. Determine entry date and fetch entry price
            entry_date = (
                pred.predicted_at.date()
                if isinstance(pred.predicted_at, datetime)
                else pred.predicted_at
            )
            entry_price = self.market_store.get_price(pred.ticker, entry_date)
            if entry_price is None or entry_price <= 0:
                logger.debug(
                    "Missing entry price observation; leaving prediction open",
                    ticker=pred.ticker,
                    entry_date=str(entry_date)
                )
                continue

            # 4. Fetch target date exit price
            exit_price = self.market_store.get_price(pred.ticker, pred.target_date)
            if exit_price is None or exit_price <= 0:
                logger.debug(
                    "Missing future market observation; leaving prediction open",
                    ticker=pred.ticker,
                    target_date=str(pred.target_date)
                )
                continue

            # 5. Calculate actual price change per frozen target contract (percentage return)
            price_change_pct = (exit_price - entry_price) / entry_price * 100.0

            # 6. Evaluate sharp drop condition against drop_threshold_pct
            threshold = pred.drop_threshold_pct if pred.drop_threshold_pct is not None else 10.0
            drop_cutoff = -abs(threshold)
            drop_observed = bool(price_change_pct <= drop_cutoff)

            # 7. Record resolved state
            pred.actual_price_change_pct = round(price_change_pct, 4)
            pred.actual_drop_observed = drop_observed
            pred.status = "resolved"
            pred.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
            resolved.append(pred)

        return resolved

    def resolve_open_predictions_in_db(
        self,
        repo: Optional[Any] = None,
        current_date: Optional[date] = None
    ) -> List[PredictionRecord]:
        """Resolves open predictions in the database and persists updates."""
        from src.mlops.repository import MLOpsRepository
        if repo is None:
            repo = MLOpsRepository()
        open_preds = repo.get_open_predictions()
        resolved_preds = self.resolve_predictions(open_preds, current_date=current_date)
        for pred in resolved_preds:
            repo.update_resolved(pred)
        return resolved_preds


class ModelPromoter:
    def evaluate_candidate(self, current_metrics: dict, candidate_metrics: dict) -> bool:
        """
        Compare candidate vs current on untouched evaluation slice.
        Returns True if candidate should be promoted.
        Uses PR-AUC and Brier score as primary criteria per project standards.
        """
        curr_brier = current_metrics.get("brier_score", current_metrics.get("brier", float("inf")))
        cand_brier = candidate_metrics.get("brier_score", candidate_metrics.get("brier", float("inf")))
        
        if candidate_metrics.get("pr_auc", 0) > current_metrics.get("pr_auc", 0):
            if cand_brier < curr_brier:
                return True
        return False


class RewardAdapter:
    def __init__(self):
        self.current_config = RewardConfig(
            version="v1.0.0",
            risk_penalty=0.01,
            inventory_penalty=0.01,
            transaction_cost_weight=0.001,
            adverse_selection_weight=0.005,
            source_metrics_window=30,
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )

    def adapt(self, performance_metrics: dict) -> RewardConfig:
        """Conservative bounded reward adaptation."""
        error_rate = performance_metrics.get("error_rate", 0.0)
        new_risk_penalty = min(0.05, max(0.001, self.current_config.risk_penalty * (1.0 + error_rate * 0.1)))

        new_config = RewardConfig(
            version=f"v1.0.{int(self.current_config.version.split('.')[-1]) + 1}",
            risk_penalty=new_risk_penalty,
            inventory_penalty=self.current_config.inventory_penalty,
            transaction_cost_weight=self.current_config.transaction_cost_weight,
            adverse_selection_weight=self.current_config.adverse_selection_weight,
            source_metrics_window=self.current_config.source_metrics_window,
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
        self.current_config = new_config
        return new_config
