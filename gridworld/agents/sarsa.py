"""Tabular SARSA for the visual Gridworld.

On-policy update::

    Q(s, a) <- Q(s, a) + alpha *
               [r_total + gamma * Q(s', a') - Q(s, a)]

The next action is the action actually selected by the current epsilon-greedy
policy and is reused by the training loop on the following step.
"""

from __future__ import annotations

from typing import Any

from gridworld.agents.base import NUM_ACTIONS, TabularAgentBase


class SARSAAgent(TabularAgentBase):
    """On-policy SARSA with the same exploration schedule as Q-learning."""

    algorithm_name = "sarsa"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        num_episodes: int = 1000,
        use_intrinsic: bool = False,
        intrinsic_strength: float = 0.0001,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            num_episodes=num_episodes,
            use_intrinsic=use_intrinsic,
            intrinsic_strength=intrinsic_strength,
            seed=seed,
        )

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        next_action: int | None = None,
    ) -> dict[str, float | int | bool]:
        """Apply one on-policy update and return auditable reward components."""
        stats = self.reward_components(reward, next_state)
        learning_reward = float(stats["learning_reward"])

        if done:
            td_target = learning_reward
        else:
            if next_action is None:
                raise ValueError("SARSA requires next_action when done is False")
            td_target = learning_reward + self.gamma * float(
                self.q_table[next_state][next_action]
            )

        old_value = float(self.q_table[state][action])
        td_error = td_target - old_value
        self.q_table[state][action] = old_value + self.alpha * td_error

        stats.update(
            {
                "terminal_update": bool(done),
                "td_target": float(td_target),
                "td_error": float(td_error),
                "updated_q_value": float(self.q_table[state][action]),
            }
        )
        self.last_update = stats
        return stats
