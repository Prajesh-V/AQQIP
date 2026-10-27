import pytest
from src.marl.experiment_runner import ExperimentRunner
from tests.test_marl import create_mock_bundle

def test_experiment_matrix():
    bundle = create_mock_bundle()
    runner = ExperimentRunner(bundle)
    results = runner.run_matrix(seeds=[42, 100])
    
    # 5 scenarios * 2 seeds = 10 results
    assert len(results) == 10
    
    modes_seen = set(r.scenario_params["mode"] for r in results)
    assert len(modes_seen) == 5
