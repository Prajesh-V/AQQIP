import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
import mysql.connector
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score, average_precision_score

from config.settings import get_settings
from src.quant.train_pipeline import load_raw_data, DEFAULT_MARKET_FEATURES
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.point_in_time_join import PointInTimeJoiner
from src.quant.feature_builder import FeatureBuilder

def investigate_discrepancy():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    
    market_df, fund_df, qual_df = load_raw_data(include_qualitative=True)
    
    # Run 1: With PointInTimeJoiner WITHOUT conservative lag vs WITH conservative lag
    # Run 2: With FeatureBuilder WITHOUT cross-sectional features vs WITH cross-sectional features
    # Run 3: Different val_start (e.g. 2023-01-01 vs 2024-01-01)
    
    model_m = mlflow.lightgbm.load_model("runs:/59a7f799181243c796b73ee3b03f5686/model")
    
    for val_start in ['2023-01-01', '2024-01-01']:
        for lag in [0, 1]:
            builder = HistoricalDatasetBuilder()
            panel = PointInTimeJoiner.construct_panel(market_df, fund_df, qual_df, conservative_availability_lag_days=lag)
            panel = FeatureBuilder.build_market_features(panel)
            panel = FeatureBuilder.build_cross_sectional_features(panel)
            panel = FeatureBuilder.extract_fundamental_features(panel)
            panel = FeatureBuilder.extract_qualitative_features(panel)
            panel = builder.target_builder.compute_forward_returns(panel)
            
            train, val, test = builder.split_and_label(
                panel, val_start=val_start, test_start='2025-01-01', alpha=0.10, enforce_temporal_purging=True
            )
            
            X_test = test[DEFAULT_MARKET_FEATURES]
            y_test = test['target_label'].values
            preds = model_m.predict(X_test)
            
            roc = roc_auc_score(y_test, preds)
            pr = average_precision_score(y_test, preds)
            thresh = builder.target_builder.fit_threshold(train['forward_return_pct'], 0.10)
            
            print(f"val_start={val_start}, lag={lag}: N_test={len(test)}, Test_Prev={y_test.mean():.4f}, Train_thresh={thresh:.4f}%, ROC={roc:.4f}, PR={pr:.4f}")

if __name__ == "__main__":
    investigate_discrepancy()
