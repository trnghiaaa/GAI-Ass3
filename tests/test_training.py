import numpy as np
import pytest

from gridworld.compare import greedy_rollout
from gridworld.train import (
    load_config,
    make_agent,
    make_environment,
    resolve_training_profile,
    train,
)
from gridworld.optimize import candidate_score, parse_seeds


def test_champion_selection_prioritizes_reliable_held_out_completion():
    strong = {
        "victory_rate": 0.97,
        "timeouts": 2,
        "monster_deaths": 28,
        "mean_steps_on_victory": 35.0,
    }
    shorter_but_weaker = {
        "victory_rate": 0.96,
        "timeouts": 0,
        "monster_deaths": 40,
        "mean_steps_on_victory": 20.0,
    }
    assert candidate_score(strong) > candidate_score(shorter_but_weaker)
    assert parse_seeds("3, 3, 5", [1]) == [3, 5]


def _trained(level_id, kind, seed, episodes=None, intrinsic=False, max_steps=None):
    config = load_config()
    profile = resolve_training_profile(
        config,
        level_id,
        kind,
        {"seed": seed, "episodes": episodes, "max_steps": max_steps},
    )
    agent = make_agent(
        kind, config, use_intrinsic=intrinsic, level_id=level_id,
        seed=seed, profile=profile,
    )
    env = make_environment(level_id, config, seed)
    metrics = train(
        env, agent, profile["episodes"], max_steps=profile["max_steps"],
        base_seed=seed, return_metrics=True, verbose=False,
    )
    return config, profile, agent, metrics


def test_training_records_truncation_and_reward_components():
    config = load_config()
    agent = make_agent("qlearning", config, 1, use_intrinsic=True,
                       level_id=0, seed=8)
    env = make_environment(0, config, 8)
    metrics = train(
        env, agent, 1, max_steps=1, base_seed=8,
        bootstrap_on_truncation=False, return_metrics=True, verbose=False,
    )
    row = metrics[0]
    assert row["status"] == "truncated" and row["truncated"] == 1
    assert row["extrinsic_reward"] == row["episode_reward"] == 0.0
    assert row["intrinsic_reward"] > 0
    assert row["learning_reward"] == pytest.approx(row["intrinsic_reward"])
    assert row["unique_states"] >= 1 and row["unique_positions"] >= 1
    assert row["visited_positions"]


@pytest.mark.acceptance
def test_level0_q_learning_reaches_verified_16_step_optimum():
    config, profile, agent, metrics = _trained(0, "qlearning", 1729)
    rollout = greedy_rollout(
        0, agent, config, seed=9001, max_steps=profile["max_steps"]
    )
    assert rollout["victory"] == 1
    assert rollout["environment_reward"] == 3.0
    assert rollout["steps"] == 16
    assert np.mean([row["victory"] for row in metrics[-100:]]) >= 0.95


@pytest.mark.acceptance
def test_level1_q_learning_and_sarsa_are_reliably_different():
    config, q_profile, q_agent, _ = _trained(1, "qlearning", 101)
    _, s_profile, sarsa_agent, _ = _trained(1, "sarsa", 101)
    q_rollout = greedy_rollout(
        1, q_agent, config, seed=777, max_steps=q_profile["max_steps"]
    )
    sarsa_rollout = greedy_rollout(
        1, sarsa_agent, config, seed=777, max_steps=s_profile["max_steps"]
    )
    assert q_rollout["victory"] == sarsa_rollout["victory"] == 1
    assert q_rollout["steps"] == 5
    assert q_rollout["hazard_adjacent_steps"] == 4
    assert sarsa_rollout["steps"] == 7
    assert sarsa_rollout["hazard_adjacent_steps"] == 0


@pytest.mark.acceptance
def test_level6_intrinsic_ab_converges_and_improves_mean_first_discovery():
    seeds = [111, 222, 333, 444, 555]
    first_success = {False: [], True: []}
    for intrinsic in (False, True):
        for seed in seeds:
            config, profile, agent, metrics = _trained(
                6, "qlearning", seed, intrinsic=intrinsic
            )
            first_success[intrinsic].append(next(
                row["episode"] for row in metrics if row["victory"]
            ))
            rollout = greedy_rollout(
                6, agent, config, seed=9900 + seed,
                max_steps=profile["max_steps"],
            )
            assert rollout["victory"] == 1
            assert rollout["steps"] == 22
            assert np.mean([row["victory"] for row in metrics[-100:]]) >= 0.99
    assert np.mean(first_success[True]) < np.mean(first_success[False])
