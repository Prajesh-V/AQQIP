"""
AQQIP MARL Experiment Runner

Executes reproducible multi-agent simulation experiments across:
- Scenarios S0–S4
- Policy suites: Heuristic, Random, FixedRule
- Configurable random seeds
- Real SignalBundles
- Automatic MySQL persistence
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
from src.marl.engine import MARLEngine, ScenarioMode
from src.marl.models import SignalBundle, MARLResult
from src.marl.policies import BasePolicySuite, FixedRulePolicy, RandomPolicy, HeuristicPolicySuite
from src.marl.repository import MARLRepository
from src.utils.logging import get_logger

logger = get_logger("experiment_runner")


class ExperimentRunner:
    """
    Executes MARL simulation experiments across the scenario matrix and policy suites.
    Guarantees reproducible seeds, non-leaking observations, and full provenance.
    """

    def __init__(self, base_bundle: SignalBundle, persist_to_db: bool = False):
        self.base_bundle = base_bundle
        self.persist_to_db = persist_to_db
        self.repo = MARLRepository() if persist_to_db else None

    def run_episode(
        self,
        policy: BasePolicySuite,
        mode: ScenarioMode = ScenarioMode.BASELINE,
        seed: int = 42,
        episode_days: int = 20
    ) -> MARLResult:
        """Executes a single simulation episode under a specified policy and scenario."""
        bundle_copy = self.base_bundle.model_copy(deep=True)
        bundle_copy.provenance.random_seed = seed
        bundle_copy.provenance.scenario_id = f"{mode.value}_{policy.policy_name}_{seed}"

        if isinstance(policy, RandomPolicy):
            policy.reset_seed(seed)

        start_time = time.perf_counter()
        engine = MARLEngine(bundle=bundle_copy, mode=mode, episode_days=episode_days)
        obs = engine.reset()

        while True:
            actions = policy.get_actions(obs, bundle_copy, engine.prices)
            obs, rewards, done, info = engine.step(actions)
            if done:
                break

        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        result = engine.generate_result(policy_type=policy.policy_name, runtime_ms=runtime_ms)

        if self.persist_to_db and self.repo:
            self.repo.save_run(result)

        return result

    def run_matrix(
        self,
        seeds: List[int],
        policies: Optional[List[BasePolicySuite]] = None,
        scenarios: Optional[List[ScenarioMode]] = None
    ) -> List[MARLResult]:
        """
        Executes complete factorial matrix: scenarios x policies x seeds.
        """
        if scenarios is None:
            scenarios = [
                ScenarioMode.BASELINE,
                ScenarioMode.QUANT_ONLY,
                ScenarioMode.QUAL_ONLY,
                ScenarioMode.COMBINED,
                ScenarioMode.STRESS
            ]

        if policies is None:
            policies = [FixedRulePolicy()]

        results: List[MARLResult] = []

        for seed in seeds:
            for mode in scenarios:
                for policy in policies:
                    try:
                        res = self.run_episode(policy=policy, mode=mode, seed=seed)
                        results.append(res)
                    except ValueError as e:
                        # E.g. Scenario requires missing qualitative or quantitative signal
                        logger.warning(
                            "Skipping scenario run due to missing required signal",
                            scenario=mode.value,
                            seed=seed,
                            error=str(e)
                        )

        return results
