import json
from pathlib import Path

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.benchmark import MODEL_MATRIX
from gridworld.build_evidence import expected_artifacts
from gridworld.compare import greedy_rollout
from gridworld.environment import GridWorldEnv, STATE_SCHEMA
from gridworld.train import LOGS_DIR, MODELS_DIR, load_config


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
        assert metadata["layout_fingerprint"] == GridWorldEnv(
            level, seed=1
        ).layout_fingerprint
        assert initial_state in agent.q_table


def test_final_policy_benchmark_meets_acceptance_thresholds():
    with (Path(LOGS_DIR) / "policy_benchmark.json").open(encoding="utf-8") as file:
        benchmark = json.load(file)
    assert benchmark["all_required_models_present"]
    assert benchmark["models_evaluated"] == len(MODEL_MATRIX)
    for row in benchmark["rows"]:
        threshold = 0.90 if row["level"] in (4, 5) else 1.0
        assert row["victory_rate"] >= threshold
        assert "blocked_action_rate" in row
        assert row["representative_blocked_actions"] == 0

    rows = {
        (row["level"], row["algorithm"], row["intrinsic"]): row
        for row in benchmark["rows"]
    }
    assert rows[(2, "qlearning", 0)]["representative_steps"] == 30
    assert rows[(4, "sarsa", 0)]["representative_steps"] == 29
    assert rows[(4, "sarsa", 0)]["victory_rate"] >= 0.99


def test_level4_sarsa_goes_straight_after_lower_right_apple_when_safe():
    agent = SARSAAgent()
    agent.load(Path(MODELS_DIR) / "level4_sarsa.pkl")
    rollout = greedy_rollout(
        4,
        agent,
        load_config(),
        seed=24_099,
        max_steps=350,
    )
    apple_index = rollout["path"].index((8, 8))
    assert rollout["path"][apple_index + 1 : apple_index + 4] == [
        (7, 8),
        (6, 8),
        (5, 8),
    ]
    assert rollout["steps"] == 29
    assert rollout["blocked_actions"] == 0


def test_comparison_summaries_contain_required_behavioral_evidence():
    with (Path(LOGS_DIR) / "level1_algorithm_comparison_summary.json").open(
        encoding="utf-8"
    ) as file:
        level1 = json.load(file)
    qlearning = level1["greedy_rollouts"]["qlearning"]
    sarsa = level1["greedy_rollouts"]["sarsa"]
    assert qlearning["mean_steps"] == 7.0
    assert qlearning["mean_hazard_adjacent_steps"] == 6.0
    assert sarsa["mean_steps"] > qlearning["mean_steps"]
    assert sarsa["mean_hazard_adjacent_steps"] < qlearning["mean_hazard_adjacent_steps"]

    with (Path(LOGS_DIR) / "level6_intrinsic_comparison_summary.json").open(
        encoding="utf-8"
    ) as file:
        level6 = json.load(file)
    assert level6["environment_rewards_unchanged"] is True
    assert level6["observed_result"]["first_success_episode_improvement"] > 0
    assert level6["variants"]["baseline"]["greedy_victory_rate"] == 1.0
    assert level6["variants"]["intrinsic"]["greedy_victory_rate"] == 1.0
