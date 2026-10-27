"""
Tests for Predictor.
Verifies that Predictor generates valid prediction records and writes them
through MLOpsRepository.
"""
import pandas as pd
import datetime
from unittest.mock import patch, MagicMock
from src.quant.predictor import Predictor
from src.mlops.models import PredictionRecord


class MockModel:
    def predict(self, X):
        return [0.8, 0.2]


@patch("src.quant.predictor.MLOpsRepository")
def test_predictor_writes_open_prediction(mock_repo_cls):
    mock_repo = MagicMock()
    mock_repo_cls.return_value = mock_repo
    mock_repo.create_prediction.side_effect = [101, 102]

    model = MockModel()
    predictor = Predictor(
        model=model,
        features=['feature_a'],
        model_run_id='run_test',
        model_version='v1.0'
    )
    # Ensure predictor uses our mock repo instance
    predictor.repository = mock_repo

    panel = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'feature_a': [1.0, 2.0],
        'trade_date': [datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)]
    })
    
    target_date = datetime.date(2023, 1, 25)
    
    # Predict and store
    records = predictor.predict_panel(panel, default_target_date=target_date)
    assert len(records) == 2
    
    # Verify AAPL record
    assert records[0].ticker == 'AAPL'
    assert records[0].predicted_drop_probability == 0.8
    assert records[0].target_date == target_date
    assert records[0].status == 'open'
    assert records[0].model_run_id == 'run_test'
    assert records[0].input_feature_hash is not None

    # Verify MSFT record
    assert records[1].ticker == 'MSFT'
    assert records[1].predicted_drop_probability == 0.2
    assert records[1].target_date == target_date
    assert records[1].status == 'open'

    # Storing through repository
    ids = predictor.predict_and_store(panel, default_target_date=target_date)
    assert ids == [101, 102]
    assert mock_repo.create_prediction.call_count == 2
