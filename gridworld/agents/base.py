"""Shared behavior for the tabular Gridworld agents.

The assignment uses two update rules, but exploration, epsilon scheduling,
count-based intrinsic reward, and model persistence must behave identically for
Q-learning and SARSA.  Keeping those concerns here prevents small differences
between the two agents from invalidating an experiment.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


NUM_ACTIONS = 4
MODEL_FORMAT_VERSION = 2


def _zero_action_values() -> np.ndarray:
    return np.zeros(NUM_ACTIONS, dtype=float)


class TabularAgentBase:
    """Common epsilon-greedy, intrinsic-reward, and persistence behavior."""

    algorithm_name = "tabular"

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
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if intrinsic_strength < 0.0:
            raise ValueError("intrinsic_strength cannot be negative")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.num_episodes = int(num_episodes)
        self.epsilon = self.epsilon_start
        self.episodes_completed = 0
        self.epsilon_decay = (
            (self.epsilon_start - self.epsilon_end)
            / max(self.num_episodes - 1, 1)
        )

        self.q_table: defaultdict[Any, np.ndarray] = defaultdict(_zero_action_values)

        self.use_intrinsic = bool(use_intrinsic)
        self.intrinsic_strength = float(intrinsic_strength)
        self.visit_counts: dict[Any, int] = {}
        self.last_update: dict[str, float | int | bool] = {}

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model_metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Exploration and reproducibility
    # ------------------------------------------------------------------

    def set_seed(self, seed: int | None) -> None:
        """Replace this agent's independent random-number stream."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def best_actions(self, state: Any) -> np.ndarray:
        """Return exactly maximal actions; near-equal values are not ties."""
        q_values = self.q_table[state]
        return np.flatnonzero(q_values == np.max(q_values))

    def choose_action(self, state: Any) -> int:
        """Select an action using epsilon-greedy exploration.

        All randomness comes from the agent-owned generator.  This makes two
        agents reproducible without coupling either one to NumPy's global RNG.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(NUM_ACTIONS))
        return int(self.rng.choice(self.best_actions(state)))

    def decay_epsilon(self) -> float:
        """Advance the config-driven linear schedule by one episode."""
        self.episodes_completed += 1
        if self.num_episodes == 1:
            self.epsilon = self.epsilon_end
            return self.epsilon

        progress = min(self.episodes_completed / (self.num_episodes - 1), 1.0)
        self.epsilon = self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )
        return self.epsilon

    # ------------------------------------------------------------------
    # Per-episode state counts and intrinsic reward
    # ------------------------------------------------------------------

    def reset_episode_visits(self) -> None:
        """Clear visit counts at the start of every episode."""
        self.visit_counts = {}

    def record_visit(self, state: Any) -> int:
        """Record a state visit and return its new count.

        Kept as a public compatibility helper.  The training loop uses
        :meth:`begin_episode` for the initial state; transition destinations are
        counted automatically by :meth:`reward_components`.
        """
        count = self.visit_counts.get(state, 0) + 1
        self.visit_counts[state] = count
        return count

    def begin_episode(self, initial_state: Any | None = None) -> None:
        """Reset counters and record the reset state as already visited."""
        self.reset_episode_visits()
        if initial_state is not None:
            self.record_visit(initial_state)

    def reward_components(self, env_reward: float, next_state: Any) -> dict[str, float | int]:
        """Return and record the reward used by the tabular update.

        The count is the number of *prior* visits to the transition destination.
        Therefore the required formula is applied exactly as::

            r_i = intrinsic_strength / sqrt(n(s) + 1)

        where ``s`` is the reached state (``next_state`` in the transition).

        The environment reward object/value is never modified.  The destination
        is recorded after calculating the bonus, including on terminal steps.
        """
        prior_visits = self.visit_counts.get(next_state, 0)
        intrinsic_reward = 0.0
        if self.use_intrinsic:
            intrinsic_reward = self.intrinsic_strength / np.sqrt(prior_visits + 1)
        self.visit_counts[next_state] = prior_visits + 1

        extrinsic_reward = float(env_reward)
        learning_reward = extrinsic_reward + float(intrinsic_reward)
        return {
            "extrinsic_reward": extrinsic_reward,
            "intrinsic_reward": float(intrinsic_reward),
            "learning_reward": learning_reward,
            "destination_visits_before": prior_visits,
            "destination_visits_after": prior_visits + 1,
        }

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _model_payload(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged_metadata = dict(self.model_metadata)
        if metadata:
            merged_metadata.update(dict(metadata))
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "algorithm": self.algorithm_name,
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "episodes_completed": self.episodes_completed,
            "num_episodes": self.num_episodes,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "use_intrinsic": self.use_intrinsic,
            "intrinsic_strength": self.intrinsic_strength,
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state,
            "metadata": merged_metadata,
        }

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        """Save a metadata-rich model while retaining the historical file format."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file:
            pickle.dump(self._model_payload(metadata), file)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load both current models and the original four-field model files."""
        with Path(path).open("rb") as file:
            data = pickle.load(file)

        if not isinstance(data, dict) or "q_table" not in data:
            raise ValueError(f"Unsupported model payload in {path}")

        table = {
            state: np.asarray(values, dtype=float).copy()
            for state, values in data["q_table"].items()
        }
        if any(values.shape != (NUM_ACTIONS,) for values in table.values()):
            raise ValueError("Model Q-table contains an invalid action-vector shape")
        self.q_table = defaultdict(_zero_action_values, table)

        self.alpha = float(data.get("alpha", self.alpha))
        self.gamma = float(data.get("gamma", self.gamma))
        self.epsilon_start = float(data.get("epsilon_start", self.epsilon_start))
        self.epsilon_end = float(data.get("epsilon_end", self.epsilon_end))
        self.num_episodes = int(data.get("num_episodes", self.num_episodes))
        self.episodes_completed = int(data.get("episodes_completed", 0))
        self.epsilon_decay = float(
            data.get(
                "epsilon_decay",
                (self.epsilon_start - self.epsilon_end)
                / max(self.num_episodes - 1, 1),
            )
        )
        self.epsilon = float(data.get("epsilon", self.epsilon_end))
        self.use_intrinsic = bool(data.get("use_intrinsic", self.use_intrinsic))
        self.intrinsic_strength = float(
            data.get("intrinsic_strength", self.intrinsic_strength)
        )
        self.model_metadata = dict(data.get("metadata", {}))

        self.set_seed(data.get("seed", self.seed))
        rng_state = data.get("rng_state")
        if rng_state is not None:
            try:
                self.rng.bit_generator.state = rng_state
            except (TypeError, ValueError):
                # A future NumPy bit generator may not accept an older state.
                # The stored seed still gives a valid deterministic fallback.
                pass

        self.reset_episode_visits()
        self.last_update = {}
        return self.model_metadata
