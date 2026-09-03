"""Observation-only teacher used to initialise the Rotation DQN.

The teacher is deliberately kept out of evaluation and gameplay.  It turns the
same fixed numeric observation seen by the agent into demonstrations so the
network can learn the difficult turn-then-thrust sequence before reinforcement
learning refines it against environment rewards.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arena.core.environment import ObservationIndex as I, ROTATION_ACTIONS


class RotationTeacher:
    """Provide safe, attacking demonstrations for rotation/thrust control."""

    def __init__(self, safety_threshold: float = 0.12) -> None:
        self.safety_threshold = float(safety_threshold)

    @staticmethod
    def _turn_action(turn: float) -> int:
        return (
            ROTATION_ACTIONS["ROTATE_RIGHT"]
            if turn > 0.0
            else ROTATION_ACTIONS["ROTATE_LEFT"]
        )

    def action(self, observation: np.ndarray) -> int:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        urgency = float(values[I.SAFETY_URGENCY])
        if urgency > self.safety_threshold:
            alignment = float(values[I.SAFETY_ESCAPE_ALIGNMENT])
            if alignment < 0.72:
                return self._turn_action(float(values[I.SAFETY_ESCAPE_TURN]))
            return ROTATION_ACTIONS["THRUST"]

        target_exists = (
            float(values[I.ENEMY_COUNT]) > 0.0
            or float(values[I.SPAWNER_COUNT]) > 0.0
        )
        if not target_exists:
            return ROTATION_ACTIONS["NOOP"]

        alignment = float(values[I.ACTIVE_TARGET_AIM_ALIGNMENT])
        turn = float(values[I.ACTIVE_TARGET_TURN_DIRECTION])
        distance = float(values[I.ACTIVE_TARGET_DISTANCE])
        weapon_ready = float(values[I.WEAPON_READY]) > 0.99
        if weapon_ready and alignment > 0.92:
            return ROTATION_ACTIONS["SHOOT"]
        if alignment < 0.78:
            return self._turn_action(turn)
        if distance > 0.26:
            return ROTATION_ACTIONS["THRUST"]
        if weapon_ready:
            return ROTATION_ACTIONS["SHOOT"]
        return self._turn_action(turn)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[Any, None]:
        """Match the Stable-Baselines3 prediction interface for benchmarks."""

        del deterministic
        values = np.asarray(observation)
        if values.ndim == 1:
            return np.asarray(self.action(values)), None
        return np.asarray([self.action(row) for row in values]), None


__all__ = ["RotationTeacher"]
