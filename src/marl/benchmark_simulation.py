"""
AQQIP MARL Historical Simulation Benchmark Runner

Executes reproducible simulation episodes across:
- Multiple tickers across distinct sectors (AAPL, NVDA, MSFT, BA, JPM)
- Historical regime contexts (2024-01-02, 2025-01-02)
- Scenarios S0–S4 (Baseline, Quant-only, Qual-only, Combined, Stress)
- Policy suites: Heuristic, Random, FixedRule
- Seeds: 42, 123, 999
- Strict point-in-time SignalBundle inputs with zero future data leakage
- Endogenous price transitions
- External reference actual future return (computed post-simulation)
- Complete persistence to MySQL marl_simulation_runs
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.marl.engine import MARLEngine, ScenarioMode
from src.marl.models import SignalBundle, MARLResult
from src.marl.policies import BasePolicySuite, FixedRulePolicy, RandomPolicy, HeuristicPolicySuite
from src.marl.signal_bridge import SignalBundleBridge
from src.marl.repository import MARLRepository
from src.database.mysql_client import get_mysql_connection
from src.utils.logging import get_logger

logger = get_logger("benchmark_simulation")


def fetch_actual_subsequent_return(ticker: str, start_date: date, horizon: int = 20) -> Optional[float]:
    """
    Computes external actual market outcome over subsequent 20 trading sessions.
    Strictly post-hoc reference: NEVER exposed to agents during simulation.
    """
    with get_mysql_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT trade_date, adj_close
            FROM ohlcv_prices
            WHERE ticker = %s AND trade_date >= %s
            ORDER BY trade_date ASC
            LIMIT %s
            """,
            (ticker, start_date, horizon + 1)
        )
        rows = cursor.fetchall()
        if len(rows) >= horizon + 1:
            p_entry = float(rows[0][1])
            p_exit = float(rows[-1][1])
            return round((p_exit - p_entry) / p_entry * 100.0, 4)
        return None


def run_benchmark_matrix(
    tickers: List[str] = ["AAPL", "NVDA", "BA", "JPM"],
    dates: List[date] = [date(2024, 1, 2), date(2025, 1, 2)],
    scenarios: List[ScenarioMode] = [
        ScenarioMode.BASELINE,
        ScenarioMode.QUANT_ONLY,
        ScenarioMode.QUAL_ONLY,
        ScenarioMode.COMBINED,
        ScenarioMode.STRESS
    ],
    seeds: List[int] = [42, 123, 999],
    persist_to_db: bool = True
) -> pd.DataFrame:
    """Executes the benchmark matrix and returns detailed metrics dataframe."""
    bridge = SignalBundleBridge()
    repo = MARLRepository() if persist_to_db else None

    records: List[Dict[str, Any]] = []
    total_runs = 0

    for ticker in tickers:
        for as_of in dates:
            # Build real SignalBundle point-in-time
            try:
                base_bundle = bridge.build_bundle(ticker, as_of_date=as_of, model_variant="M")
            except Exception as e:
                logger.warning("Could not build bundle for ticker/date", ticker=ticker, date=str(as_of), error=str(e))
                continue

            # Query actual future reference return
            actual_ret = fetch_actual_subsequent_return(ticker, as_of, horizon=20)

            for seed in seeds:
                policies: List[BasePolicySuite] = [
                    FixedRulePolicy(),
                    HeuristicPolicySuite(),
                    RandomPolicy(seed=seed)
                ]

                for mode in scenarios:
                    for policy in policies:
                        bundle_copy = base_bundle.model_copy(deep=True)
                        bundle_copy.provenance.random_seed = seed
                        bundle_copy.provenance.scenario_id = f"{mode.value}_{policy.policy_name}_{seed}"

                        try:
                            start_time = time.perf_counter()
                            engine = MARLEngine(bundle=bundle_copy, mode=mode, episode_days=20)
                            obs = engine.reset()

                            while True:
                                acts = policy.get_actions(obs, bundle_copy, engine.prices)
                                obs, _, done, _ = engine.step(acts)
                                if done:
                                    break

                            runtime_ms = (time.perf_counter() - start_time) * 1000.0
                            res = engine.generate_result(policy_type=policy.policy_name, runtime_ms=runtime_ms)

                            if persist_to_db and repo:
                                repo.save_run(res)

                            sim_return = (res.final_price - res.initial_price) / res.initial_price * 100.0

                            records.append({
                                "simulation_id": res.simulation_id,
                                "ticker": ticker,
                                "as_of_date": str(as_of),
                                "scenario": mode.value,
                                "policy": policy.policy_name,
                                "seed": seed,
                                "initial_price": res.initial_price,
                                "final_price": res.final_price,
                                "simulated_return_pct": round(sim_return, 4),
                                "actual_return_20d_pct": actual_ret,
                                "max_drawdown_pct": round(res.max_drawdown_pct, 4),
                                "volatility_pct": round(res.price_volatility, 4),
                                "short_pnl_pct": round(res.short_seller_pnl_pct, 4),
                                "retail_pnl_pct": round(res.retail_herd_pnl_pct, 4),
                                "mm_pnl_pct": round(res.market_maker_pnl_pct, 4),
                                "mm_peak_inventory": res.market_maker_inventory_peak,
                                "inventory_variance": res.scenario_params.get("inventory_variance", 0.0),
                                "runtime_ms": round(runtime_ms, 2),
                                "signal_bundle_hash": res.signal_bundle_hash[:12]
                            })
                            total_runs += 1
                        except ValueError as ve:
                            # E.g. Scenario missing required evidence
                            pass

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("Running reproducible MARL benchmark matrix...")
    df_results = run_benchmark_matrix()
    print(f"\nCompleted {len(df_results)} simulation runs successfully persisted to MySQL.")
    
    # Aggregation by Scenario and Policy
    summary = df_results.groupby(["scenario", "policy"]).agg(
        sim_return_mean=("simulated_return_pct", "mean"),
        sim_return_median=("simulated_return_pct", "median"),
        drawdown_mean=("max_drawdown_pct", "mean"),
        drawdown_p95=("max_drawdown_pct", lambda x: np.percentile(x, 95)),
        short_pnl_mean=("short_pnl_pct", "mean"),
        retail_pnl_mean=("retail_pnl_pct", "mean"),
        mm_pnl_mean=("mm_pnl_pct", "mean"),
        mm_peak_inv_mean=("mm_peak_inventory", "mean"),
        runtime_ms_mean=("runtime_ms", "mean"),
        n_samples=("simulation_id", "count")
    ).reset_index()

    print("\n=== MARL Simulation Benchmark Summary Table ===")
    print(summary.to_string(index=False))
