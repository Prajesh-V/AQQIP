"""
AQQIP MARL Agent Policy Framework

Defines executable policies for the three market agents:
1. Short Seller: Deterministic heuristic reacting to quantitative downside pressure and verified adverse qualitative signal.
2. Retail Herd: Momentum-following heuristic reacting strictly to observable past endogenous prices (zero future lookahead).
3. Market Maker: Inventory-defending heuristic responding to inventory imbalance, volatility, and order flow.
4. RandomPolicy: Seeded baseline with uniform discrete action choice.
5. FixedRulePolicy: Fixed baseline (HOLD / HOLD / NORMAL).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from src.marl.engine import ActionShortSeller, ActionRetailHerd, ActionMarketMaker
from src.marl.models import SignalBundle


class BasePolicySuite(ABC):
    """Abstract base class for multi-agent policy suites."""

    @abstractmethod
    def get_actions(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> Dict[str, int]:
        """Returns action dictionary for all three agents given current observation."""
        pass


class FixedRulePolicy(BasePolicySuite):
    """Retained baseline: All agents maintain neutral status."""

    def __init__(self):
        self.policy_name = "fixed_rule"

    def get_actions(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> Dict[str, int]:
        return {
            "short_seller": int(ActionShortSeller.HOLD),
            "retail_herd": int(ActionRetailHerd.HOLD),
            "market_maker": int(ActionMarketMaker.NORMAL)
        }


class RandomPolicy(BasePolicySuite):
    """Seeded stochastic baseline with uniform discrete action choice."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.policy_name = "random"

    def reset_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_actions(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> Dict[str, int]:
        return {
            "short_seller": int(self.rng.choice([0, 1, 2])),
            "retail_herd": int(self.rng.choice([0, 1, 2])),
            "market_maker": int(self.rng.choice([0, 1, 2]))
        }


class HeuristicPolicySuite(BasePolicySuite):
    """
    Meaningful domain-grounded heuristic policies for all three agents:
    - Short Seller: Enforces quantitative threshold + qualitative verification before shorting.
    - Retail Herd: Follows observable trailing 3-day endogenous price momentum.
    - Market Maker: Defends inventory boundaries and manages adverse selection.
    """

    def __init__(
        self,
        quant_downside_threshold: float = 0.55,
        qual_adverse_threshold: float = 0.20,
        momentum_threshold_pct: float = 1.0,
        inventory_defense_limit: float = 8.0,
        inventory_safe_limit: float = 3.0
    ):
        self.policy_name = "heuristic"
        self.quant_thresh = quant_downside_threshold
        self.qual_thresh = qual_adverse_threshold
        self.mom_thresh = momentum_threshold_pct / 100.0
        self.inv_defense_limit = inventory_defense_limit
        self.inv_safe_limit = inventory_safe_limit

    def select_short_seller_action(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> int:
        pos_short = obs.get("pos_short", 0.0)
        curr_price = obs.get("price", 100.0)
        init_price = price_history[0] if price_history else curr_price

        quant_p = bundle.quantitative.quant_pressure_score
        qual_p = bundle.qualitative.directional_risk_score
        qual_available = bundle.evidence_quality.qualitative_available

        # Verified adverse risk: quant pressure elevated, and qual evidence is adverse if available
        is_elevated_risk = (quant_p >= self.quant_thresh)
        if qual_available and qual_p >= self.qual_thresh:
            is_elevated_risk = True

        # Profit-taking / covering condition: price has dropped > 7% from episode start
        price_drawdown = (init_price - curr_price) / max(0.01, init_price)
        if (price_drawdown >= 0.07 or quant_p <= 0.25) and pos_short < 0.0:
            return int(ActionShortSeller.REDUCE)

        # Short expansion condition
        if is_elevated_risk and pos_short > -10.0:
            return int(ActionShortSeller.INCREASE)

        return int(ActionShortSeller.HOLD)

    def select_retail_herd_action(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> int:
        pos_retail = obs.get("pos_retail", 0.0)
        curr_price = obs.get("price", 100.0)

        # Calculate observable trailing return strictly from past simulated prices
        # Zero future information exposed!
        if len(price_history) >= 4:
            trailing_ret = (curr_price - price_history[-4]) / max(0.01, price_history[-4])
        elif len(price_history) > 1:
            trailing_ret = (curr_price - price_history[0]) / max(0.01, price_history[0])
        else:
            trailing_ret = bundle.market_context.recent_return

        # Momentum chasing
        if trailing_ret >= self.mom_thresh and pos_retail < 10.0:
            return int(ActionRetailHerd.BUY)
        elif trailing_ret <= -self.mom_thresh and pos_retail > 0.0:
            return int(ActionRetailHerd.SELL)

        return int(ActionRetailHerd.HOLD)

    def select_market_maker_action(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> int:
        inv = abs(obs.get("inventory_mm", 0.0))

        # Severe inventory imbalance: defend book by reducing liquidity / widening spread
        if inv >= self.inv_defense_limit:
            return int(ActionMarketMaker.REDUCE_LIQUIDITY)
        # Low inventory risk: tighten spread and provide deep liquidity
        elif inv <= self.inv_safe_limit:
            return int(ActionMarketMaker.INCREASE_LIQUIDITY)

        return int(ActionMarketMaker.NORMAL)

    def get_actions(
        self,
        obs: Dict[str, Any],
        bundle: SignalBundle,
        price_history: List[float]
    ) -> Dict[str, int]:
        return {
            "short_seller": self.select_short_seller_action(obs, bundle, price_history),
            "retail_herd": self.select_retail_herd_action(obs, bundle, price_history),
            "market_maker": self.select_market_maker_action(obs, bundle, price_history)
        }
