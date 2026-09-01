"""Gymnasium wrappers used consistently during training and evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


SUM_EVENT_KEYS = (
    "projectiles_fired",
    "projectile_hits",
    "enemies_spawned",
    "enemies_destroyed",
    "spawners_destroyed",
    "damage_dealt_enemy",
    "damage_dealt_spawner",
    "damage_taken",
    "spawner_progress",
    "aim_improvement",
    "hazard_escape_improvement",
    "hazard_exposure",
    "crowd_escape",
    "contact_events",
    "sustain_healed",
    "xp_gained",
    "levels_gained",
    "minibosses_spawned",
    "minibosses_destroyed",
    "miniboss_caches",
    "boss_skills_cast",
    "boss_skills_dodged",
    "boss_skill_hits",
    "boss_defenders_spawned",
    "boss_defenders_destroyed",
    "boss_immune_hits",
    "boss_health_regenerated",
    "missiles_fired",
    "missiles_evaded",
    "missile_hits",
    "missile_escape_improvement",
    "barrier_blocks",
    "drone_shots",
    "drone_hits",
)


class ActionRepeatWrapper(gym.Wrapper):
    """Hold one discrete decision for several fixed simulation frames.

    Action repeat makes rotation/thrust decisions meaningful at 60 Hz while the
    underlying environment and renderer still advance smoothly one frame at a
    time. Rewards and event counters are accumulated across the repeated frames.
    """

    def __init__(self, env: gym.Env, repeat: int = 4) -> None:
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        super().__init__(env)
        self.repeat = int(repeat)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation: np.ndarray | None = None
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        totals = {key: 0.0 for key in SUM_EVENT_KEYS}
        reward_breakdown: dict[str, float] = {}
        frames = 0
        any_shot = False
        any_phase = False
        any_boss_clear = False

        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            frames += 1
            last_info = dict(info)
            any_shot = any_shot or bool(info.get("shot_fired"))
            any_phase = any_phase or bool(info.get("phase_advanced"))
            any_boss_clear = any_boss_clear or bool(info.get("boss_phase_cleared"))
            for key in SUM_EVENT_KEYS:
                totals[key] += float(info.get(key, 0.0))
            for key, value in info.get("reward_breakdown", {}).items():
                reward_breakdown[key] = reward_breakdown.get(key, 0.0) + float(value)
            if terminated or truncated:
                break

        assert observation is not None
        for key, value in totals.items():
            if key in (
                "projectile_hits",
                "projectiles_fired",
                "enemies_spawned",
                "enemies_destroyed",
                "spawners_destroyed",
                "levels_gained",
                "minibosses_spawned",
                "minibosses_destroyed",
                "miniboss_caches",
                "boss_skills_cast",
                "boss_skills_dodged",
                "boss_skill_hits",
                "boss_defenders_spawned",
                "boss_defenders_destroyed",
                "boss_immune_hits",
                "missiles_fired",
                "missiles_evaded",
                "missile_hits",
                "barrier_blocks",
                "drone_shots",
                "drone_hits",
            ):
                last_info[key] = int(value)
            else:
                last_info[key] = value
        last_info["shot_fired"] = any_shot
        last_info["phase_advanced"] = any_phase
        last_info["boss_phase_cleared"] = any_boss_clear
        last_info["reward_breakdown"] = reward_breakdown
        last_info["action_repeat_frames"] = frames
        return observation, total_reward, terminated, truncated, last_info


class BossCurriculumWrapper(gym.Wrapper):
    """Start some training episodes at a boss without changing evaluation.

    The wrapped environment keeps its normal dynamics, observations, rewards,
    controls, and termination. Only the initial phase is sampled, making rare
    boss telegraphs common enough for DQN to learn from them.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        probability: float = 0.65,
        phases: tuple[int, ...] = (3,),
        seed: int = 0,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not phases or any(int(phase) < 1 for phase in phases):
            raise ValueError("phases must contain positive phase numbers")
        super().__init__(env)
        self.probability = float(probability)
        self.phases = tuple(int(phase) for phase in phases)
        self._curriculum_seed = int(seed)
        self._rng = np.random.default_rng(self._curriculum_seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + self._curriculum_seed)
        reset_options = dict(options or {})
        if "start_phase" not in reset_options and self._rng.random() < self.probability:
            reset_options["start_phase"] = int(self._rng.choice(self.phases))
        observation, info = self.env.reset(seed=seed, options=reset_options)
        info = dict(info)
        info["curriculum_start_phase"] = int(self.env.unwrapped.phase)
        return observation, info


__all__ = ["ActionRepeatWrapper", "BossCurriculumWrapper", "SUM_EVENT_KEYS"]
