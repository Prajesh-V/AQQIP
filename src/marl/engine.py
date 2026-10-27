import numpy as np
import uuid
import hashlib
import json
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List
from src.marl.models import SignalBundle, MARLResult

class ScenarioMode(str, Enum):
    BASELINE = "S0"
    QUANT_ONLY = "S1"
    QUAL_ONLY = "S2"
    COMBINED = "S3"
    STRESS = "S4"

class ActionShortSeller(int, Enum):
    HOLD = 0
    INCREASE = 1
    REDUCE = 2

class ActionRetailHerd(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class ActionMarketMaker(int, Enum):
    NORMAL = 0
    INCREASE_LIQUIDITY = 1
    REDUCE_LIQUIDITY = 2

class MARLEngine:
    def __init__(self, bundle: SignalBundle, mode: ScenarioMode = ScenarioMode.BASELINE, episode_days: int = 20):
        self.bundle = bundle
        self.mode = mode
        self.episode_days = episode_days
        
        # Scenario weights
        self.w_q = 0.0
        self.w_l = 0.0
        self.stress_multiplier = 1.0
        
        if self.mode == ScenarioMode.QUANT_ONLY:
            self.w_q = 1.0
        elif self.mode == ScenarioMode.QUAL_ONLY:
            self.w_l = 1.0
        elif self.mode == ScenarioMode.COMBINED:
            self.w_q = 0.5
            self.w_l = 0.5
        elif self.mode == ScenarioMode.STRESS:
            self.w_q = 0.5
            self.w_l = 0.5
            self.stress_multiplier = 2.0
            
        # Extract required evidence
        if self.w_q > 0 and not bundle.evidence_quality.quant_available:
            raise ValueError(f"Scenario {mode} requires quantitative evidence, but it is missing.")
        if self.w_l > 0 and not bundle.evidence_quality.qualitative_available:
            raise ValueError(f"Scenario {mode} requires qualitative evidence, but it is missing.")
            
        self.rng = np.random.default_rng(bundle.provenance.random_seed)
        self.reset()
        
    def reset(self):
        self.day = 0
        self.prices = [self.bundle.market_context.current_price]
        self.inventory_mm = 0.0
        self.inventory_history = [0.0]
        self.pos_short = 0.0
        self.pos_retail = 0.0
        
        # PnL trackers
        self.pnl_short = 0.0
        self.pnl_retail = 0.0
        self.pnl_mm = 0.0
        
        self.max_inventory_mm = 0.0
        self.terminated_reason = "max_steps"
        
        # Action frequency counters
        self.action_counts = {
            "short_seller": {0: 0, 1: 0, 2: 0},
            "retail_herd": {0: 0, 1: 0, 2: 0},
            "market_maker": {0: 0, 1: 0, 2: 0}
        }
        
        # Base environment dynamics configuration
        self.base_volatility = max(0.001, self.bundle.market_context.recent_volatility)
        
        return self._get_obs()
        
    def _get_obs(self):
        return {
            "price": self.prices[-1],
            "day": self.day,
            "inventory_mm": self.inventory_mm,
            "pos_short": self.pos_short,
            "pos_retail": self.pos_retail
        }
        
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, Dict]:
        if self.day >= self.episode_days:
            return self._get_obs(), {}, True, {}
            
        act_short_int = int(actions.get("short_seller", 0))
        act_retail_int = int(actions.get("retail_herd", 0))
        act_mm_int = int(actions.get("market_maker", 0))
        
        # Record action frequencies
        self.action_counts["short_seller"][act_short_int] = self.action_counts["short_seller"].get(act_short_int, 0) + 1
        self.action_counts["retail_herd"][act_retail_int] = self.action_counts["retail_herd"].get(act_retail_int, 0) + 1
        self.action_counts["market_maker"][act_mm_int] = self.action_counts["market_maker"].get(act_mm_int, 0) + 1
        
        a_short = ActionShortSeller(act_short_int)
        a_retail = ActionRetailHerd(act_retail_int)
        a_mm = ActionMarketMaker(act_mm_int)
        
        # Translate actions to pressures
        short_pressure = 0.0
        if a_short == ActionShortSeller.INCREASE:
            short_pressure = -0.02
        elif a_short == ActionShortSeller.REDUCE:
            short_pressure = 0.02
            
        retail_pressure = 0.0
        if a_retail == ActionRetailHerd.BUY:
            retail_pressure = 0.01
        elif a_retail == ActionRetailHerd.SELL:
            retail_pressure = -0.01
            
        liquidity = 1.0
        if a_mm == ActionMarketMaker.INCREASE_LIQUIDITY:
            liquidity = 1.5
        elif a_mm == ActionMarketMaker.REDUCE_LIQUIDITY:
            liquidity = 0.5
            
        # Forensic pressure
        qual_score = (
            self.bundle.qualitative.directional_risk_score
            if hasattr(self.bundle.qualitative, "directional_risk_score") and self.bundle.qualitative.directional_risk_score > 0
            else self.bundle.qualitative.semantic_shift_score
        )
        forensic_pressure = (
            self.w_q * self.bundle.quantitative.quant_pressure_score + 
            self.w_l * qual_score
        )
        
        # The forensic pressure pushes price DOWN (higher pressure = worse for company)
        # We scale it by stress multiplier
        forensic_impact = -0.01 * forensic_pressure * self.stress_multiplier
        
        # Bounded stochastic shock
        shock = self.rng.normal(0, self.base_volatility * self.stress_multiplier)
        shock = float(np.clip(shock, -0.1, 0.1)) # Hard bound
        
        # Market impact
        net_impact = (short_pressure + retail_pressure + forensic_impact + shock) / max(0.1, liquidity)
        
        # Next price (hard positive protection, NaN protection)
        current_price = self.prices[-1]
        next_price = current_price * (1.0 + net_impact)
        
        if np.isnan(next_price) or np.isinf(next_price):
            next_price = current_price
            self.terminated_reason = "numerical_instability"
            
        next_price = max(0.01, float(next_price)) # Positive price invariant
        self.prices.append(next_price)
        
        # Positions
        trade_short = -1.0 if a_short == ActionShortSeller.INCREASE else (1.0 if a_short == ActionShortSeller.REDUCE else 0.0)
        trade_retail = 1.0 if a_retail == ActionRetailHerd.BUY else (-1.0 if a_retail == ActionRetailHerd.SELL else 0.0)
        
        self.pos_short = float(np.clip(self.pos_short + trade_short, -10.0, 0.0))
        self.pos_retail = float(np.clip(self.pos_retail + trade_retail, 0.0, 10.0))
        
        # MM absorbs net flow (inventory hard bounded within [-20, 20])
        net_flow = trade_short + trade_retail
        self.inventory_mm = float(np.clip(self.inventory_mm - net_flow, -20.0, 20.0))
        self.inventory_history.append(self.inventory_mm)
        self.max_inventory_mm = max(self.max_inventory_mm, abs(self.inventory_mm))
        
        # Rewards (including transaction costs and inventory holding penalty)
        r_short = self.pos_short * net_impact - 0.001 * abs(trade_short)
        r_retail = self.pos_retail * net_impact - 0.001 * abs(trade_retail)
        r_mm = 0.005 * liquidity - 0.01 * abs(self.inventory_mm) - 0.0005 * abs(net_flow)
        
        # NaN protection for rewards
        rewards = {
            "short_seller": float(np.nan_to_num(r_short, nan=0.0, posinf=10.0, neginf=-10.0)),
            "retail_herd": float(np.nan_to_num(r_retail, nan=0.0, posinf=10.0, neginf=-10.0)),
            "market_maker": float(np.nan_to_num(r_mm, nan=0.0, posinf=10.0, neginf=-10.0))
        }
        
        self.pnl_short += rewards["short_seller"]
        self.pnl_retail += rewards["retail_herd"]
        self.pnl_mm += rewards["market_maker"]
        
        self.day += 1
        done = self.day >= self.episode_days or self.terminated_reason != "max_steps"
        
        return self._get_obs(), rewards, done, {}
        
    def generate_result(self, policy_type: str = "fixed_rule", runtime_ms: Optional[float] = None) -> MARLResult:
        prices = np.array(self.prices)
        initial = prices[0]
        final = prices[-1]
        
        # Max drawdown
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        max_drawdown = float(np.max(drawdown))
        
        # Volatility
        returns = np.diff(prices) / prices[:-1]
        vol = float(np.std(returns)) if len(returns) > 1 else 0.0
        
        bundle_hash = hashlib.sha256(self.bundle.model_dump_json().encode()).hexdigest()
        inv_variance = float(np.var(self.inventory_history))
        
        scenario_params = {
            "mode": self.mode.value,
            "policy_type": policy_type,
            "stress_multiplier": self.stress_multiplier,
            "action_frequencies": self.action_counts,
            "inventory_variance": inv_variance,
            "quant_model_run_id": self.bundle.quantitative.model_run_id,
            "directional_qual_score": getattr(self.bundle.qualitative, "directional_risk_score", 0.0)
        }
        
        return MARLResult(
            simulation_id=str(uuid.uuid4()),
            ticker=self.bundle.security.ticker,
            scenario_seed=self.bundle.provenance.random_seed,
            episode_days=self.day,
            initial_price=float(initial),
            final_price=float(final),
            max_drawdown_pct=max_drawdown * 100.0,
            price_volatility=vol * 100.0,
            short_seller_pnl_pct=float(self.pnl_short),
            retail_herd_pnl_pct=float(self.pnl_retail),
            market_maker_pnl_pct=float(self.pnl_mm),
            market_maker_inventory_peak=float(self.max_inventory_mm),
            terminated_reason=self.terminated_reason,
            scenario_params=scenario_params,
            signal_bundle_hash=bundle_hash,
            model_version=self.bundle.quantitative.model_variant,
            reward_config_version="v1.0.0",
            prediction_id=self.bundle.provenance.prediction_id,
            runtime_ms=runtime_ms
        )
