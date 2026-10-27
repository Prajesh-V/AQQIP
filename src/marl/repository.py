import mysql.connector
from typing import Dict, Any, Optional
import json
from src.marl.models import MARLResult
from config.settings import get_settings

class MARLRepository:
    def __init__(self):
        settings = get_settings().mysql
        self.conn = mysql.connector.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database
        )

    def save_run(self, result: MARLResult) -> None:
        cursor = self.conn.cursor()
        query = """
        INSERT INTO marl_simulation_runs 
        (simulation_id, ticker, scenario_mode, scenario_seed, episode_days, initial_price, final_price,
         max_drawdown_pct, price_volatility, short_seller_pnl_pct, retail_herd_pnl_pct,
         market_maker_pnl_pct, market_maker_inventory_peak, terminated_reason,
         scenario_params_json, signal_bundle_hash,
         reward_config_version, model_version, prediction_id, runtime_ms, run_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            result.simulation_id,
            result.ticker,
            result.scenario_params.get("mode", "S0"),
            result.scenario_seed,
            result.episode_days,
            result.initial_price,
            result.final_price,
            result.max_drawdown_pct,
            result.price_volatility,
            result.short_seller_pnl_pct,
            result.retail_herd_pnl_pct,
            result.market_maker_pnl_pct,
            result.market_maker_inventory_peak,
            result.terminated_reason,
            json.dumps(result.scenario_params),
            result.signal_bundle_hash,
            result.reward_config_version,
            result.model_version,
            result.prediction_id,
            result.runtime_ms,
            result.run_at
        )
        cursor.execute(query, params)
        self.conn.commit()
        cursor.close()
        
    def get_run(self, simulation_id: str) -> Optional[MARLResult]:
        cursor = self.conn.cursor(dictionary=True)
        query = "SELECT * FROM marl_simulation_runs WHERE simulation_id = %s"
        cursor.execute(query, (simulation_id,))
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return None
            
        row['scenario_params'] = json.loads(row.pop('scenario_params_json', '{}'))
        row.pop('id', None)
        row.pop('scenario_mode', None)
        return MARLResult(**row)
