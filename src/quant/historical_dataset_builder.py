"""
AQQIP Historical Dataset Builder

Orchestrates the PointInTimeJoiner, FeatureBuilder, and TargetBuilder 
to generate the training/validation/test panels.
"""
from __future__ import annotations

import pandas as pd
from typing import Tuple

from src.quant.point_in_time_join import PointInTimeJoiner
from src.quant.feature_builder import FeatureBuilder
from src.quant.target_builder import TargetBuilder
from src.quant.leakage_validator import LeakageValidator
from src.utils.logging import get_logger

logger = get_logger("dataset_builder")


class HistoricalDatasetBuilder:
    """Builds leak-free panels for modeling."""

    def __init__(self):
        self.target_builder = TargetBuilder()

    def build_dataset(
        self, 
        market_df: pd.DataFrame, 
        fundamentals_df: pd.DataFrame,
        qualitative_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Creates a raw panel with market, fundamental, and qualitative features, and forward returns.
        Does NOT apply labels yet (labels require fitting a threshold on train split).
        """
        logger.info("Building historical dataset panel")
        
        # 1. Point in time join
        panel = PointInTimeJoiner.construct_panel(market_df, fundamentals_df, qualitative_df)
        
        # 2. Build market features
        panel = FeatureBuilder.build_market_features(panel)
        panel = FeatureBuilder.build_cross_sectional_features(panel)
        
        # 3. Build fundamental features
        panel = FeatureBuilder.extract_fundamental_features(panel)

        # 4. Build qualitative features
        panel = FeatureBuilder.extract_qualitative_features(panel)
        
        # 5. Compute forward returns
        panel = self.target_builder.compute_forward_returns(panel)
        
        # 6. Leakage check
        errors = LeakageValidator.validate_panel(panel)
        if errors:
            for e in errors:
                logger.error("LEAKAGE DETECTED: " + e)
            raise ValueError(f"Leakage detected in historical dataset: {errors}")
            
        return panel

    def split_and_label(
        self, 
        panel: pd.DataFrame, 
        val_start: str, 
        test_start: str,
        alpha: float,
        enforce_temporal_purging: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits chronologically, purges boundary-crossing forward outcome windows,
        and applies the target threshold fitted ONLY on train.
        """
        logger.info(f"Splitting dataset: val_start={val_start}, test_start={test_start}, purging={enforce_temporal_purging}")
        
        panel = panel.copy()
        val_start_dt = pd.to_datetime(val_start)
        test_start_dt = pd.to_datetime(test_start)

        # Ensure trade_date is datetime for comparison
        if not pd.api.types.is_datetime64_any_dtype(panel['trade_date']):
            panel['trade_date_dt'] = pd.to_datetime(panel['trade_date'])
        else:
            panel['trade_date_dt'] = panel['trade_date']

        # Ensure target_date is available and datetime
        if 'target_date' not in panel.columns:
            panel['target_date'] = panel.groupby('ticker')['trade_date'].shift(-self.target_builder.horizon)
            
        if not pd.api.types.is_datetime64_any_dtype(panel['target_date']):
            panel['target_date_dt'] = pd.to_datetime(panel['target_date'])
        else:
            panel['target_date_dt'] = panel['target_date']
            
        train = panel[panel['trade_date_dt'] < val_start_dt].copy()
        val = panel[(panel['trade_date_dt'] >= val_start_dt) & 
                    (panel['trade_date_dt'] < test_start_dt)].copy()
        test = panel[panel['trade_date_dt'] >= test_start_dt].copy()
        
        # Enforce temporal purging:
        # Exclude training observations whose forward outcome window crosses into validation
        # Exclude validation observations whose forward outcome window crosses into test holdout
        if enforce_temporal_purging:
            orig_train_len, orig_val_len = len(train), len(val)
            train = train[train['target_date_dt'] < val_start_dt].copy()
            val = val[val['target_date_dt'] < test_start_dt].copy()
            logger.info(
                "Enforced temporal purging on partition boundaries",
                train_purged_rows=orig_train_len - len(train),
                val_purged_rows=orig_val_len - len(val)
            )

        # Clean up temporary column
        for df in [train, val, test]:
            df.drop(columns=['trade_date_dt', 'target_date_dt'], inplace=True, errors='ignore')

        # Validate that chronological partitions have no temporal inversion or overlap
        split_errs = LeakageValidator.validate_split_boundaries(train, val, test)
        if split_errs:
            raise ValueError(f"Chronological split boundary error: {split_errs}")

        # Fit threshold on train ONLY
        threshold = self.target_builder.fit_threshold(train['forward_return_pct'], alpha)
        logger.info(f"Fitted target threshold (alpha={alpha}) on train set: {threshold:.4f}%")
        
        # Apply labels to all splits using the train threshold
        train = self.target_builder.apply_labels(train, threshold)
        val = self.target_builder.apply_labels(val, threshold)
        test = self.target_builder.apply_labels(test, threshold)
        
        return train, val, test
