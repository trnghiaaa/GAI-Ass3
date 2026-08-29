import json
from pathlib import Path

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.benchmark import MODEL_MATRIX
from gridworld.build_evidence import expected_artifacts
from gridworld.environment import GridWorldEnv, STATE_SCHEMA
from gridworld.train import LOGS_DIR, MODELS_DIR


def test_all_submission_evidence_artifacts_exist_and_are_nonempty():
    missing = [path for path in expected_artifacts() if not path.exists()]
    empty = [path for path in expected_artifacts() if path.exists() and path.stat().st_size == 0]
    assert missing == []
    assert empty == []


def test_saved_models_match_current_state_schema_and_initial_state():
    for level, kind, intrinsic in MODEL_MATRIX:
        suffix = "_intrinsic" if intrinsic else ""
        path = Path(MODELS_DIR) / f"level{level}_{kind}{suffix}.pkl"
        agent = QLearningAgent() if kind == "qlearning" else SARSAAgent()
        metadata = agent.load(path)
        initial_state = GridWorldEnv(level, seed=1).reset()
        assert tuple(metadata["state_schema"]) == STATE_SCHEMA
        assert metadata["level_id"] == level
        assert metadata["algorithm"] == kind
        assert bool(metadata["intrinsic_enabled"]) is intrinsic
        assert initial_state in agent.q_table


def test_final_policy_benchmark_meets_acceptance_thresholds():
    with (Path(LOGS_DIR) / "policy_benchmark.json").open(encoding="utf-8") as file:
        benchmark = json.load(file)
    assert benchmark["all_required_models_present"]
    assert benchmark["models_evaluated"] == len(MODEL_MATRIX)
    for row in benchmark["rows"]:
        threshold = 0.90 if row["level"] in (4, 5) else 1.0
        assert row["victory_rate"] >= threshold


def test_comparison_summaries_contain_required_behavioral_evidence():
    with (Path(LOGS_DIR) / "level1_algorithm_comparison_summary.json").open(
        encoding="utf-8"
    ) as file:
        level1 = json.load(file)
    assert level1["greedy_rollouts"]["qlearning"]["mean_steps"] == 5.0
    assert level1["greedy_rollouts"]["qlearning"]["mean_hazard_adjacent_steps"] == 4.0
    assert level1["greedy_rollouts"]["sarsa"]["mean_steps"] == 7.0
    assert level1["greedy_rollouts"]["sarsa"]["mean_hazard_adjacent_steps"] == 0.0

    with (Path(LOGS_DIR) / "level6_intrinsic_comparison_summary.json").open(
        encoding="utf-8"
    ) as file:
        level6 = json.load(file)
    assert level6["environment_rewards_unchanged"] is True
    assert level6["observed_result"]["first_success_episode_improvement"] > 0
    assert level6["variants"]["baseline"]["greedy_victory_rate"] == 1.0
    assert level6["variants"]["intrinsic"]["greedy_victory_rate"] == 1.0
