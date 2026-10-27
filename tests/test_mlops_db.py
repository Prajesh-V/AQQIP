import pytest
from datetime import datetime, date
from src.mlops.models import PredictionRecord
from src.mlops.repository import MLOpsRepository

def test_mlops_db_integration():
    repo = MLOpsRepository()
    
    # We will insert a dummy prediction using the raw connection to test reading/updating
    cursor = repo.conn.cursor()
    query = """
    INSERT INTO predictions 
    (ticker, model_run_id, predicted_drop_probability, predicted_at, target_date, status)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    params = ("TESTML", "run_x", 0.8, datetime.utcnow(), date(2025,1,1), "open")
    cursor.execute(query, params)
    pred_id = cursor.lastrowid
    repo.conn.commit()
    cursor.close()
    
    # Read back open predictions
    open_preds = repo.get_open_predictions()
    assert any(p.prediction_id == pred_id for p in open_preds)
    
    pred = repo.get_prediction(pred_id)
    assert pred is not None
    assert pred.status == "open"
    assert pred.ticker == "TESTML"
    
    # Resolve it
    pred.status = "resolved"
    pred.actual_drop_observed = True
    pred.actual_price_change_pct = -0.15
    pred.resolved_at = datetime.utcnow()
    repo.update_resolved(pred)
    
    # Read back to ensure it updated
    updated_pred = repo.get_prediction(pred_id)
    assert updated_pred.status == "resolved"
    assert updated_pred.actual_drop_observed is True
    assert updated_pred.actual_price_change_pct == -0.15
    
    # Ensure it's not in open predictions anymore
    open_preds_new = repo.get_open_predictions()
    assert not any(p.prediction_id == pred_id for p in open_preds_new)
