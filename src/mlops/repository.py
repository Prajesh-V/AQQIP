import mysql.connector
from typing import List, Optional
from datetime import date, datetime
from src.mlops.models import PredictionRecord
from src.database.mysql_client import get_mysql_connection
from config.settings import get_settings


class MLOpsRepository:
    def __init__(self):
        settings = get_settings().mysql
        self.settings = settings
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or not self._conn.is_connected():
            self._conn = mysql.connector.connect(
                host=self.settings.host,
                port=self.settings.port,
                user=self.settings.user,
                password=self.settings.password,
                database=self.settings.database
            )
        return self._conn

    def create_prediction(self, record: PredictionRecord) -> int:
        """Persists a new prediction record into MySQL predictions table."""
        cursor = self.conn.cursor()
        query = """
        INSERT INTO predictions (
            ticker, cik, model_run_id, model_version,
            predicted_drop_probability, calibrated_probability, input_feature_hash,
            predicted_at, target_date, prediction_horizon_days, drop_threshold_pct,
            status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        params = (
            record.ticker,
            record.cik,
            record.model_run_id,
            record.model_version,
            float(record.predicted_drop_probability),
            float(record.calibrated_probability) if record.calibrated_probability is not None else None,
            record.input_feature_hash,
            record.predicted_at,
            record.target_date,
            record.prediction_horizon_days,
            float(record.drop_threshold_pct),
            record.status
        )
        cursor.execute(query, params)
        pred_id = cursor.lastrowid
        self.conn.commit()
        cursor.close()
        record.prediction_id = pred_id
        return pred_id

    def _row_to_record(self, row: dict) -> PredictionRecord:
        """Maps a MySQL dictionary row to PredictionRecord."""
        pred_id = row.pop('id', None)
        return PredictionRecord(
            prediction_id=pred_id,
            ticker=row['ticker'],
            cik=row.get('cik'),
            model_run_id=row['model_run_id'],
            model_version=row.get('model_version'),
            predicted_drop_probability=float(row['predicted_drop_probability']),
            calibrated_probability=float(row['calibrated_probability']) if row.get('calibrated_probability') is not None else None,
            input_feature_hash=row.get('input_feature_hash'),
            predicted_at=row['predicted_at'],
            target_date=row['target_date'],
            prediction_horizon_days=int(row.get('prediction_horizon_days', 20)),
            drop_threshold_pct=float(row.get('drop_threshold_pct', 10.00)),
            actual_drop_observed=bool(row['actual_drop_observed']) if row.get('actual_drop_observed') is not None else None,
            actual_price_change_pct=float(row['actual_price_change_pct']) if row.get('actual_price_change_pct') is not None else None,
            resolved_at=row.get('resolved_at'),
            status=row.get('status', 'open')
        )

    def get_open_predictions(self) -> List[PredictionRecord]:
        cursor = self.conn.cursor(dictionary=True)
        query = "SELECT * FROM predictions WHERE status = 'open' ORDER BY id ASC"
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return [self._row_to_record(row) for row in rows]

    def get_prediction(self, prediction_id: int) -> Optional[PredictionRecord]:
        cursor = self.conn.cursor(dictionary=True)
        query = "SELECT * FROM predictions WHERE id = %s"
        cursor.execute(query, (prediction_id,))
        row = cursor.fetchone()
        cursor.close()
        if not row:
            return None
        return self._row_to_record(row)

    def update_resolved(self, prediction: PredictionRecord) -> None:
        cursor = self.conn.cursor()
        query = """
        UPDATE predictions
        SET status = %s, actual_drop_observed = %s, actual_price_change_pct = %s, resolved_at = %s
        WHERE id = %s
        """
        params = (
            prediction.status,
            prediction.actual_drop_observed,
            float(prediction.actual_price_change_pct) if prediction.actual_price_change_pct is not None else None,
            prediction.resolved_at,
            prediction.prediction_id
        )
        cursor.execute(query, params)
        self.conn.commit()
        cursor.close()

