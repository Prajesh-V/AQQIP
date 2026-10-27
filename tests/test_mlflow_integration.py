import pytest
import mlflow
import os
from config.settings import get_settings

def test_mlflow_integration():
    settings = get_settings().mlflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # We will log a dummy metric and parameter to MLFlow
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment("phase5_mlops_test")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log parameters
        mlflow.log_param("scenario", "baseline")
        mlflow.log_param("seed", 42)
        
        # Log metrics
        mlflow.log_metric("auc", 0.85)
        mlflow.log_metric("log_loss", 0.35)
        
        # Log artifact
        with open("dummy_artifact.txt", "w") as f:
            f.write("Model weights dummy")
        mlflow.log_artifact("dummy_artifact.txt")
        os.remove("dummy_artifact.txt")
        
    assert run_id is not None
    
    # Verify it can be read back
    client = mlflow.tracking.MlflowClient()
    run_info = client.get_run(run_id)
    assert run_info.data.params["scenario"] == "baseline"
    assert run_info.data.metrics["auc"] == 0.85
    
    artifacts = client.list_artifacts(run_id)
    assert any(a.path == "dummy_artifact.txt" for a in artifacts)
