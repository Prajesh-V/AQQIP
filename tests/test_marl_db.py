import pytest
from datetime import datetime, date
from src.marl.models import MARLResult
from src.marl.repository import MARLRepository

def test_marl_db_integration():
    repo = MARLRepository()
    
    import uuid
    sim_id = str(uuid.uuid4())
    
    result = MARLResult(
        simulation_id=sim_id,
        ticker="TESTDB",
        scenario_seed=123,
        episode_days=10,
        initial_price=100.0,
        final_price=110.0,
        max_drawdown_pct=5.0,
        price_volatility=2.0,
        short_seller_pnl_pct=-10.0,
        retail_herd_pnl_pct=10.0,
        market_maker_pnl_pct=1.0,
        market_maker_inventory_peak=5.0,
        terminated_reason="max_steps",
        scenario_params={"mode": "S1"},
        signal_bundle_hash="abc",
        model_version="v1"
    )
    
    # Insert
    repo.save_run(result)
    
    # Read back
    read_result = repo.get_run(sim_id)
    
    assert read_result is not None
    assert read_result.simulation_id == sim_id
    assert read_result.ticker == "TESTDB"
    assert read_result.scenario_seed == 123
    assert read_result.scenario_params["mode"] == "S1"
    assert read_result.signal_bundle_hash == "abc"
