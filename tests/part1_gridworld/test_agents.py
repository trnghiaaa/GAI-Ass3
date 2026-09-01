import pickle

import numpy as np
import pytest

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent


def test_q_learning_uses_off_policy_maximum():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon_start=0,
                           epsilon_end=0, num_episodes=2, seed=1)
    state, next_state = ("s",), ("next",)
    agent.q_table[state][2] = 1.0
    agent.q_table[next_state][:] = [2.0, 5.0, 3.0, 4.0]
    agent.begin_episode(state)
    stats = agent.update(state, 2, 2.0, next_state, False)
    expected_target = 2.0 + 0.9 * 5.0
    assert stats["td_target"] == pytest.approx(expected_target)
    assert agent.q_table[state][2] == pytest.approx(1.0 + 0.5 * (expected_target - 1.0))


def test_q_learning_terminal_target_does_not_bootstrap():
    agent = QLearningAgent(alpha=1.0, gamma=0.99, use_intrinsic=False)
    agent.q_table[("terminal",)][:] = 99
    agent.begin_episode(("s",))
    stats = agent.update(("s",), 0, 1.0, ("terminal",), True)
    assert stats["td_target"] == 1.0
    assert agent.q_table[("s",)][0] == 1.0


def test_sarsa_uses_actual_next_action_not_maximum():
    agent = SARSAAgent(alpha=1.0, gamma=0.5, epsilon_start=0,
                       epsilon_end=0, num_episodes=2)
    state, next_state = ("s",), ("next",)
    agent.q_table[next_state][:] = [100.0, 4.0, 3.0, 2.0]
    agent.begin_episode(state)
    stats = agent.update(state, 1, 2.0, next_state, False, next_action=2)
    assert stats["td_target"] == pytest.approx(2.0 + 0.5 * 3.0)
    assert agent.q_table[state][1] == pytest.approx(3.5)


def test_sarsa_requires_next_action_on_nonterminal_transition():
    agent = SARSAAgent()
    agent.begin_episode(("s",))
    with pytest.raises(ValueError, match="next_action"):
        agent.update(("s",), 0, 0.0, ("next",), False)


def test_epsilon_greedy_randomly_breaks_exact_ties_only():
    agent = QLearningAgent(epsilon_start=0, epsilon_end=0,
                           num_episodes=2, seed=42)
    state = ("tie",)
    agent.q_table[state][:] = [1.0, 1.0, 0.0, -1.0]
    actions = {agent.choose_action(state) for _ in range(200)}
    assert actions == {0, 1}

    agent.q_table[state][:] = [1.0, 1.0 - 1e-12, 0.0, -1.0]
    assert {agent.choose_action(state) for _ in range(50)} == {0}


def test_full_exploration_samples_all_four_actions():
    agent = QLearningAgent(epsilon_start=1, epsilon_end=1,
                           num_episodes=2, seed=7)
    assert {agent.choose_action(("s",)) for _ in range(400)} == {0, 1, 2, 3}


def test_linear_epsilon_schedule_reaches_configured_end_on_last_episode():
    agent = QLearningAgent(epsilon_start=1.0, epsilon_end=0.0,
                           num_episodes=5)
    values = [agent.epsilon]
    for _ in range(4):
        values.append(agent.decay_epsilon())
    assert values == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    assert agent.decay_epsilon() == 0.0


@pytest.mark.parametrize("agent_class", [QLearningAgent, SARSAAgent])
def test_intrinsic_bonus_uses_destination_prior_visits_exactly(agent_class):
    agent = agent_class(alpha=1.0, gamma=0.0, use_intrinsic=True,
                        intrinsic_strength=1.0)
    start, destination = ("start",), ("destination",)
    agent.begin_episode(start)
    kwargs = {"next_action": None} if agent_class is SARSAAgent else {}
    first = agent.update(start, 0, 2.0, destination, True, **kwargs)
    second = agent.update(start, 1, 2.0, destination, True, **kwargs)

    assert first["destination_visits_before"] == 0
    assert first["intrinsic_reward"] == pytest.approx(1.0)
    assert first["extrinsic_reward"] == 2.0
    assert first["learning_reward"] == pytest.approx(3.0)
    assert second["destination_visits_before"] == 1
    assert second["intrinsic_reward"] == pytest.approx(1 / np.sqrt(2))
    assert second["learning_reward"] == pytest.approx(2 + 1 / np.sqrt(2))


def test_visit_counts_reset_for_each_episode():
    agent = QLearningAgent(use_intrinsic=True, intrinsic_strength=0.25)
    agent.begin_episode(("s",))
    first = agent.update(("s",), 0, 0, ("new",), True)
    agent.begin_episode(("s",))
    reset_first = agent.update(("s",), 0, 0, ("new",), True)
    assert first["intrinsic_reward"] == reset_first["intrinsic_reward"] == 0.25
    assert agent.visit_counts == {("s",): 1, ("new",): 1}


def test_independent_agent_rng_is_reproducible():
    left = QLearningAgent(epsilon_start=1, epsilon_end=1, seed=123)
    right = QLearningAgent(epsilon_start=1, epsilon_end=1, seed=123)
    assert [left.choose_action(("s",)) for _ in range(50)] == [
        right.choose_action(("s",)) for _ in range(50)
    ]


def test_metadata_rich_save_load_round_trip(tmp_path):
    path = tmp_path / "agent.pkl"
    agent = QLearningAgent(seed=12, use_intrinsic=True, intrinsic_strength=0.1)
    agent.q_table[("s",)][:] = [1, 2, 3, 4]
    agent.save(path, metadata={"level_id": 6, "state_schema": ["example"]})

    loaded = QLearningAgent()
    metadata = loaded.load(path)
    assert np.array_equal(loaded.q_table[("s",)], [1, 2, 3, 4])
    assert loaded.use_intrinsic and loaded.intrinsic_strength == 0.1
    assert metadata["level_id"] == 6


def test_legacy_model_load_is_backward_compatible(tmp_path):
    path = tmp_path / "legacy.pkl"
    payload = {
        "q_table": {("s",): np.asarray([4.0, 3.0, 2.0, 1.0])},
        "epsilon": 0.01,
        "alpha": 0.2,
        "gamma": 0.8,
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)
    agent = QLearningAgent()
    assert agent.load(path) == {}
    assert agent.alpha == 0.2 and agent.gamma == 0.8
    assert np.array_equal(agent.q_table[("s",)], payload["q_table"][("s",)])
