"""Compatibility adapter for the four-value Gym API shown in the rubric.

Stable-Baselines3 2.x uses Gymnasium and should train directly with
``ArenaEnv``.  This adapter exists for demonstrations or marking code that
expects ``reset() -> observation`` and ``step() -> observation, reward, done,
info``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arena.environment import ArenaEnv


class LegacyArenaEnv:
    """Expose an :class:`ArenaEnv` through the assignment's legacy API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.env = ArenaEnv(*args, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
        self.last_reset_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reset and return only the initial observation."""

        observation, info = self.env.reset(seed=seed, options=options)
        self.last_reset_info = info
        return observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply an action and combine Gymnasium's two end flags as ``done``."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        compatibility_info = dict(info)
        compatibility_info.update(
            {
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        return observation, reward, done, compatibility_info

    def render(self) -> np.ndarray | None:
        """Display or return the same scene rendered by the wrapped environment."""

        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self) -> ArenaEnv:
        return self.env


__all__ = ["LegacyArenaEnv"]
