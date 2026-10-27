import os
import sys
sys.path.insert(0, r"c:\dev\AQQIP")
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from config.settings import get_settings
from src.quant.train_pipeline import load_raw_data, DEFAULT_MARKET_FEATURES
from src.quant.historical_dataset_builder import HistoricalDatasetBuilder
from src.quant.feature_builder import FeatureBuilder

def check_5878():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    
    # Check all runs in MLflow
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(client.get_experiment_by_name("aqqip_quant_models").experiment_id)
    for r in runs:
        print(f"Run {r.info.run_id} ({r.data.tags.get('mlflow.runName', 'unnamed')}):")
        print(f"  metrics: {r.data.metrics}")
        
if __name__ == "__main__":
    check_5878()
