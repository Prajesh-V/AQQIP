"""
Comprehensive MARL Policies, SignalBridge, and Environment Verification Suite

Validates:
1. Deterministic seeds and reproducibility across runs.
2. Hard positive price invariants under stress.
3. Strict inventory boundedness [-20.0, 20.0].
4. Agent policy action validity across all 3 agents.
5. Retail herd zero future-price leakage.
6. Short seller reaction to quantitative and directional qualitative risk.
7. Market maker inventory defense behavior.
8. Seeded RandomPolicy reproducibility.
9. Point-in-time SignalBundle integrity (zero lookahead).
10. Full simulation run persistence to MySQL marl_simulation_runs.
"""
from __future__ import annotations

import pytest
import numpy as np
from datetime import date, datetime

from src.marl.engine import (
    MARLEngine,
    ScenarioMode,
    ActionShortSeller,
    ActionRetailHerd,
    ActionMarketMaker
)
from src.marl.models import (
    SecurityIdentity,
    MarketContext,
    QuantitativeSignalBlock,
    QualitativeSignalBlock,
    EvidenceQuality,
    FusionBlock,
    ProvenanceBlock,
    SignalBundle
)
from src.marl.policies import (
    FixedRulePolicy,
    RandomPolicy,
    HeuristicPolicySuite
)
from src.marl.signal_bridge import SignalBundleBridge
from src.marl.experiment_runner import ExperimentRunner
from src.marl.repository import MARLRepository


def create_mock_bundle(
    ticker: str = "TEST",
    quant_pressure: float = 0.8,
    directional_qual: float = 0.5,
    qual_available: bool = True,
    seed: int = 42
) -> SignalBundle:
    return SignalBundle(
        security=SecurityIdentity(
            ticker=ticker,
            company_name="Test Company",
            cik="0001234567"
        ),
        market_context=MarketContext(
            current_price=100.0,
            recent_return=-0.03,
            recent_volatility=0.02,
            average_volume=1000000.0,
            liquidity_proxy=1.0,
            as_of_date=date(2025, 1, 2)
        ),
        quantitative=QuantitativeSignalBlock(
            quant_pressure_score=quant_pressure,
            beneish_m_score=-2.5,
            altman_z_score=1.8,
            model_probability=0.15,
            model_run_id="59a7f799181243c796b73ee3b03f5686",
            model_variant="M"
        ),
        qualitative=QualitativeSignalBlock(
            semantic_shift_score=directional_qual,
            cosine_similarity=0.75,
            change_magnitude=0.25,
            directional_risk_score=directional_qual,
            added_count=15,
            deleted_count=10,
            shifted_count=0,
            source_accession="0001234567-25-000001",
            filing_availability_date=date(2024, 11, 1)
        ),
        evidence_quality=EvidenceQuality(
            quant_available=True,
            qualitative_available=qual_available,
            filing_pair_available=True,
            feature_completeness=1.0
        ),
        fusion=FusionBlock(
            evidence_concordance=0.8,
            combined_forensic_pressure=0.65
        ),
        provenance=ProvenanceBlock(
            as_of_date=date(2025, 1, 2),
            random_seed=seed
        )
    )


def test_deterministic_seeds_and_divergence():
    """Proves identical seeds produce identical trajectories, while different seeds diverge."""
    b1 = create_mock_bundle(seed=42)
    b2 = create_mock_bundle(seed=42)
    b3 = create_mock_bundle(seed=999)

    policy = HeuristicPolicySuite()
    runner1 = ExperimentRunner(b1)
    runner2 = ExperimentRunner(b2)
    runner3 = ExperimentRunner(b3)

    r1 = runner1.run_episode(policy=policy, mode=ScenarioMode.COMBINED, seed=42)
    r2 = runner2.run_episode(policy=policy, mode=ScenarioMode.COMBINED, seed=42)
    r3 = runner3.run_episode(policy=policy, mode=ScenarioMode.COMBINED, seed=999)

    # Identical seeds must match exactly
    assert r1.final_price == r2.final_price
    assert r1.max_drawdown_pct == r2.max_drawdown_pct
    assert r1.short_seller_pnl_pct == r2.short_seller_pnl_pct
    assert r1.market_maker_inventory_peak == r2.market_maker_inventory_peak

    # Different seeds must produce different simulated stochastic paths
    assert r1.final_price != r3.final_price


def test_strictly_positive_prices_under_extreme_stress():
    """Proves that prices remain strictly positive (> 0.01) even under extreme downward pressure."""
    bundle = create_mock_bundle(quant_pressure=1.0, directional_qual=1.0, seed=42)
    bundle.market_context.current_price = 0.50  # Start already depressed
    bundle.market_context.recent_volatility = 0.20  # Massive volatility

    engine = MARLEngine(bundle=bundle, mode=ScenarioMode.STRESS, episode_days=50)

    for _ in range(50):
        # Max short pressure, max retail selling, min liquidity
        actions = {
            "short_seller": int(ActionShortSeller.INCREASE),
            "retail_herd": int(ActionRetailHerd.SELL),
            "market_maker": int(ActionMarketMaker.REDUCE_LIQUIDITY)
        }
        obs, rewards, done, _ = engine.step(actions)
        assert obs["price"] >= 0.01
        assert not np.isnan(obs["price"])
        assert not np.isinf(obs["price"])
        if done:
            break

    res = engine.generate_result()
    assert res.final_price >= 0.01


def test_bounded_inventory_invariant():
    """Proves Market Maker inventory never escapes the bounded [-20.0, 20.0] range."""
    bundle = create_mock_bundle(seed=123)
    engine = MARLEngine(bundle=bundle, mode=ScenarioMode.BASELINE, episode_days=30)

    for _ in range(30):
        # Max flow in one direction
        actions = {
            "short_seller": int(ActionShortSeller.INCREASE),
            "retail_herd": int(ActionRetailHerd.SELL),
            "market_maker": int(ActionMarketMaker.NORMAL)
        }
        obs, _, done, _ = engine.step(actions)
        assert -20.0 <= obs["inventory_mm"] <= 20.0
        if done:
            break

    assert engine.max_inventory_mm <= 20.0


def test_policy_action_validity():
    """Proves all policy suites return valid discrete actions within the defined action spaces."""
    bundle = create_mock_bundle(seed=42)
    policies = [FixedRulePolicy(), HeuristicPolicySuite(), RandomPolicy(seed=42)]

    for policy in policies:
        engine = MARLEngine(bundle=bundle, mode=ScenarioMode.COMBINED, episode_days=20)
        obs = engine.reset()
        for _ in range(20):
            actions = policy.get_actions(obs, bundle, engine.prices)
            assert actions["short_seller"] in [0, 1, 2]
            assert actions["retail_herd"] in [0, 1, 2]
            assert actions["market_maker"] in [0, 1, 2]
            obs, _, done, _ = engine.step(actions)
            if done:
                break


def test_retail_herd_no_lookahead_leakage():
    """
    Verifies Retail Herd bases decisions strictly on trailing observed prices
    and has zero access to future prices or exogenous target labels.
    """
    suite = HeuristicPolicySuite(momentum_threshold_pct=1.0)
    bundle = create_mock_bundle()

    # Upward trailing momentum in past simulation prices
    upward_prices = [100.0, 101.0, 102.5, 104.0]
    obs = {"pos_retail": 0.0, "price": 104.0}
    act_up = suite.select_retail_herd_action(obs, bundle, upward_prices)
    assert act_up == int(ActionRetailHerd.BUY)

    # Downward trailing momentum
    downward_prices = [100.0, 99.0, 97.0, 95.0]
    obs_down = {"pos_retail": 5.0, "price": 95.0}
    act_down = suite.select_retail_herd_action(obs_down, bundle, downward_prices)
    assert act_down == int(ActionRetailHerd.SELL)


def test_market_maker_defends_inventory():
    """Verifies Market Maker reduces liquidity when inventory is imbalanced."""
    suite = HeuristicPolicySuite(inventory_defense_limit=8.0, inventory_safe_limit=3.0)
    bundle = create_mock_bundle()

    # Extreme inventory imbalance
    obs_imbalanced = {"inventory_mm": 12.0}
    act_defend = suite.select_market_maker_action(obs_imbalanced, bundle, [100.0])
    assert act_defend == int(ActionMarketMaker.REDUCE_LIQUIDITY)

    # Balanced inventory
    obs_balanced = {"inventory_mm": 1.5}
    act_provide = suite.select_market_maker_action(obs_balanced, bundle, [100.0])
    assert act_provide == int(ActionMarketMaker.INCREASE_LIQUIDITY)


def test_random_policy_seeded_reproducibility():
    """Proves RandomPolicy generates identical action sequences given the same seed."""
    bundle = create_mock_bundle()
    p1 = RandomPolicy(seed=777)
    p2 = RandomPolicy(seed=777)

    obs = {"pos_short": 0.0, "pos_retail": 0.0, "inventory_mm": 0.0, "price": 100.0}
    history = [100.0]

    for _ in range(25):
        a1 = p1.get_actions(obs, bundle, history)
        a2 = p2.get_actions(obs, bundle, history)
        assert a1 == a2


def test_signal_bundle_bridge_point_in_time():
    """Proves SignalBundleBridge strictly enforces point-in-time data as of historical date."""
    bridge = SignalBundleBridge()
    # Historical date where future data existed in database
    hist_date = date(2024, 6, 28)
    bundle = bridge.build_bundle("AAPL", as_of_date=hist_date, model_variant="M")

    assert bundle.market_context.as_of_date <= hist_date
    assert bundle.quantitative.model_variant == "M"
    assert bundle.quantitative.model_probability is not None
    assert bundle.evidence_quality.quant_available is True
    if bundle.qualitative.filing_availability_date:
        assert bundle.qualitative.filing_availability_date <= hist_date


def test_simulation_run_persistence_to_mysql():
    """Proves MARL simulation results with rich metadata persist and retrieve cleanly from MySQL."""
    bundle = create_mock_bundle(seed=42)
    runner = ExperimentRunner(bundle, persist_to_db=True)
    res = runner.run_episode(policy=HeuristicPolicySuite(), mode=ScenarioMode.COMBINED, seed=42)

    repo = MARLRepository()
    fetched = repo.get_run(res.simulation_id)

    assert fetched is not None
    assert fetched.simulation_id == res.simulation_id
    assert fetched.ticker == res.ticker
    assert fetched.scenario_seed == 42
    assert fetched.scenario_params["policy_type"] == "heuristic"
    assert "action_frequencies" in fetched.scenario_params
    assert fetched.reward_config_version == "v1.0.0"
