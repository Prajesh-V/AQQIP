import pytest
import numpy as np
from datetime import date
from src.marl.models import (
    SignalBundle, SecurityIdentity, MarketContext,
    QuantitativeSignalBlock, QualitativeSignalBlock,
    EvidenceQuality, FusionBlock, ProvenanceBlock
)
from src.marl.engine import MARLEngine, ScenarioMode, ActionShortSeller, ActionRetailHerd, ActionMarketMaker

def create_mock_bundle(quant_avail=True, qual_avail=True, seed=42):
    return SignalBundle(
        security=SecurityIdentity(ticker="TEST", company_name="Test Inc", cik="0000000000"),
        market_context=MarketContext(
            current_price=100.0, recent_return=0.01, recent_volatility=0.02,
            average_volume=1000000, liquidity_proxy=1.0, as_of_date=date(2025, 1, 1)
        ),
        quantitative=QuantitativeSignalBlock(
            quant_pressure_score=0.8, model_run_id="run1"
        ),
        qualitative=QualitativeSignalBlock(
            semantic_shift_score=0.7, cosine_similarity=0.9, added_count=10,
            deleted_count=5, shifted_count=20
        ),
        evidence_quality=EvidenceQuality(
            quant_available=quant_avail, qualitative_available=qual_avail,
            filing_pair_available=True, feature_completeness=1.0
        ),
        fusion=FusionBlock(
            evidence_concordance=0.9, combined_forensic_pressure=0.75, fusion_version="v1"
        ),
        provenance=ProvenanceBlock(random_seed=seed)
    )

def test_marl_deterministic():
    bundle = create_mock_bundle()
    
    # Run 1
    engine1 = MARLEngine(bundle, ScenarioMode.COMBINED, episode_days=10)
    obs1 = engine1.reset()
    for _ in range(10):
        engine1.step({"short_seller": ActionShortSeller.INCREASE, "retail_herd": ActionRetailHerd.SELL, "market_maker": ActionMarketMaker.NORMAL})
    res1 = engine1.generate_result()
    
    # Run 2
    engine2 = MARLEngine(bundle, ScenarioMode.COMBINED, episode_days=10)
    obs2 = engine2.reset()
    for _ in range(10):
        engine2.step({"short_seller": ActionShortSeller.INCREASE, "retail_herd": ActionRetailHerd.SELL, "market_maker": ActionMarketMaker.NORMAL})
    res2 = engine2.generate_result()
    
    assert res1.final_price == res2.final_price
    assert res1.short_seller_pnl_pct == res2.short_seller_pnl_pct
    assert res1.scenario_seed == res2.scenario_seed

def test_missing_evidence_rejection():
    bundle = create_mock_bundle(quant_avail=False)
    with pytest.raises(ValueError, match="requires quantitative evidence"):
        MARLEngine(bundle, ScenarioMode.QUANT_ONLY)
        
    bundle2 = create_mock_bundle(qual_avail=False)
    with pytest.raises(ValueError, match="requires qualitative evidence"):
        MARLEngine(bundle2, ScenarioMode.QUAL_ONLY)
        
def test_finiteness_and_invariants():
    bundle = create_mock_bundle()
    engine = MARLEngine(bundle, ScenarioMode.STRESS, episode_days=100)
    engine.reset()
    
    for _ in range(100):
        obs, rewards, done, _ = engine.step({
            "short_seller": ActionShortSeller.INCREASE, 
            "retail_herd": ActionRetailHerd.SELL, 
            "market_maker": ActionMarketMaker.REDUCE_LIQUIDITY
        })
        assert obs["price"] > 0.0
        assert not np.isnan(obs["price"])
        assert -10.0 <= obs["pos_short"] <= 0.0
        assert 0.0 <= obs["pos_retail"] <= 10.0
        assert -20.0 <= obs["inventory_mm"] <= 20.0
        for v in rewards.values():
            assert not np.isnan(v)
            assert not np.isinf(v)
        if done:
            break
            
    res = engine.generate_result()
    assert res.final_price > 0.0
    assert not np.isnan(res.short_seller_pnl_pct)
