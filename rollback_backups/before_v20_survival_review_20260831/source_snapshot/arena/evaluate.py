"""Evaluate and optionally render a trained Stable-Baselines3 DQN agent."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pygame
from stable_baselines3 import DQN

from arena.cooldown import CooldownAwarePolicy

from arena.environment import (
    ArenaEnv,
    OBSERVATION_NAMES,
    ObservationIndex,
    ROTATION_ACTIONS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Policy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]: ...


class ObservationPrefixPolicy:
    """Adapt a legacy policy by passing only its original observation prefix."""

    def __init__(self, policy: Policy, observation_size: int) -> None:
        if observation_size <= 0:
            raise ValueError("observation_size must be positive")
        self.policy = policy
        self.observation_size = observation_size

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[Any, Any]:
        vector = np.asarray(observation)
        return self.policy.predict(
            vector[..., : self.observation_size],
            deterministic=deterministic,
        )


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _weighted_segment_mean(
    episodes: list[dict[str, Any]], value_key: str, step_key: str
) -> float | None:
    weighted_total = 0.0
    total_steps = 0
    for episode in episodes:
        value = episode.get(value_key)
        steps = int(episode.get(step_key, 0))
        if value is None or steps <= 0:
            continue
        weighted_total += float(value) * steps
        total_steps += steps
    return float(weighted_total / total_steps) if total_steps else None


def _aggregate_phase_diagnostics(
    episode_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize phase-entry health and pre/post-Phase-2 survival telemetry."""

    phase2_cohort = [
        episode
        for episode in episode_results
        if episode.get("health_on_first_entry_phase2") is not None
    ]
    phase2_health = [
        float(episode["health_on_first_entry_phase2"])
        for episode in phase2_cohort
    ]
    phase3_health = [
        float(episode["health_on_first_entry_phase3"])
        for episode in episode_results
        if episode.get("health_on_first_entry_phase3") is not None
    ]
    deaths = [
        episode
        for episode in episode_results
        if episode.get("end_reason") == "player_destroyed"
    ]
    phase1_deaths = sum(int(episode["final_phase"]) == 1 for episode in deaths)
    phase2_deaths = sum(int(episode["final_phase"]) == 2 for episode in deaths)
    phase3plus_deaths = sum(int(episode["final_phase"]) >= 3 for episode in deaths)
    episode_count = max(1, len(episode_results))

    def cohort_values(key: str) -> list[float]:
        return [
            float(episode[key])
            for episode in phase2_cohort
            if episode.get(key) is not None
        ]

    return {
        "phase2_reached_cohort_episodes": len(phase2_cohort),
        "phase3_reached_cohort_episodes": len(phase3_health),
        "mean_health_on_first_entry_phase2": _mean_or_none(phase2_health),
        "median_health_on_first_entry_phase2": _median_or_none(phase2_health),
        "mean_health_on_first_entry_phase3": _mean_or_none(phase3_health),
        "median_health_on_first_entry_phase3": _median_or_none(phase3_health),
        "mean_damage_taken_before_phase2_transition": _mean_or_none(
            cohort_values("damage_taken_before_phase2")
        ),
        "mean_damage_taken_after_phase2_transition": _mean_or_none(
            cohort_values("damage_taken_after_phase2")
        ),
        "mean_contact_events_before_phase2": _mean_or_none(
            cohort_values("contact_events_before_phase2")
        ),
        "mean_contact_events_after_phase2": _mean_or_none(
            cohort_values("contact_events_after_phase2")
        ),
        "mean_steps_survived_after_first_entering_phase2": _mean_or_none(
            cohort_values("steps_survived_after_first_entering_phase2")
        ),
        "death_count_phase1": phase1_deaths,
        "death_count_phase2": phase2_deaths,
        "death_count_phase3_or_higher": phase3plus_deaths,
        "death_rate_phase1": float(phase1_deaths / episode_count),
        "death_rate_phase2": float(phase2_deaths / episode_count),
        "death_rate_phase3_or_higher": float(phase3plus_deaths / episode_count),
        "danger_step_fraction_before_phase2": _weighted_segment_mean(
            phase2_cohort,
            "danger_step_fraction_before_phase2",
            "steps_before_phase2",
        ),
        "danger_step_fraction_after_phase2": _weighted_segment_mean(
            phase2_cohort,
            "danger_step_fraction_after_phase2",
            "steps_after_phase2",
        ),
        "mean_crowd_pressure_before_phase2": _weighted_segment_mean(
            phase2_cohort,
            "mean_crowd_pressure_before_phase2",
            "steps_before_phase2",
        ),
        "mean_crowd_pressure_after_phase2": _weighted_segment_mean(
            phase2_cohort,
            "mean_crowd_pressure_after_phase2",
            "steps_after_phase2",
        ),
        "mean_minimum_enemy_distance_before_phase2": _mean_or_none(
            cohort_values("minimum_enemy_distance_before_phase2")
        ),
        "mean_minimum_enemy_distance_after_phase2": _mean_or_none(
            cohort_values("minimum_enemy_distance_after_phase2")
        ),
    }


class SafetyShieldPolicy:
    """Veto only immediately unsafe rotation-control commands.

    The DQN remains in control during ordinary navigation and combat. Near a
    boundary or under imminent enemy pressure, this layer converts an unsafe
    command into rotation/thrust toward the environment's safe corridor.
    """

    def __init__(
        self,
        policy: Policy,
        boundary_threshold: float = 0.12,
        emergency_threshold: float = 0.65,
    ) -> None:
        self.policy = policy
        self.boundary_threshold = boundary_threshold
        self.emergency_threshold = emergency_threshold
        self.override_count = 0
        self.boundary_override_count = 0
        self.emergency_override_count = 0

    @property
    def last_raw_action(self) -> int | None:
        return getattr(self.policy, "last_raw_action", None)

    @staticmethod
    def _escape_action(alignment: float, turn: float) -> int:
        if alignment >= 0.45:
            return ROTATION_ACTIONS["THRUST"]
        if abs(turn) < 0.04:
            return ROTATION_ACTIONS["ROTATE_RIGHT"]
        return (
            ROTATION_ACTIONS["ROTATE_RIGHT"]
            if turn > 0.0
            else ROTATION_ACTIONS["ROTATE_LEFT"]
        )

    @staticmethod
    def _forward_wall_clearance(vector: np.ndarray) -> float:
        """Recover heading-ray wall distance from normalized wall features."""

        left, right, top, bottom = (
            float(value) * 160.0
            for value in vector[
                ObservationIndex.WALL_LEFT_CLEARANCE
                : ObservationIndex.WALL_BOTTOM_CLEARANCE + 1
            ]
        )
        heading_x = float(vector[ObservationIndex.PLAYER_HEADING_COS])
        heading_y = float(vector[ObservationIndex.PLAYER_HEADING_SIN])
        distances: list[float] = []
        if heading_x < -1e-6:
            distances.append(left / -heading_x)
        elif heading_x > 1e-6:
            distances.append(right / heading_x)
        if heading_y < -1e-6:
            distances.append(top / -heading_y)
        elif heading_y > 1e-6:
            distances.append(bottom / heading_y)
        return min(distances, default=float("inf"))

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[Any, Any]:
        predicted, state = self.policy.predict(observation, deterministic=deterministic)
        proposed = int(np.asarray(predicted).item())
        vector = np.asarray(observation).reshape(-1)

        boundary_alignment = float(
            vector[ObservationIndex.BOUNDARY_INWARD_ALIGNMENT]
        )
        if (
            self._forward_wall_clearance(vector) < 75.0
            and proposed
            not in (
                ROTATION_ACTIONS["ROTATE_LEFT"],
                ROTATION_ACTIONS["ROTATE_RIGHT"],
            )
        ):
            safe_action = self._escape_action(
                boundary_alignment,
                float(vector[ObservationIndex.BOUNDARY_INWARD_TURN_DIRECTION]),
            )
            if safe_action != proposed:
                self.override_count += 1
                self.boundary_override_count += 1
            return np.asarray(safe_action), state

        emergency = float(vector[ObservationIndex.EMERGENCY_THREAT])
        normalized_speed = math.hypot(
            float(vector[ObservationIndex.PLAYER_VELOCITY_X]),
            float(vector[ObservationIndex.PLAYER_VELOCITY_Y]),
        )
        if (
            emergency >= 0.52
            and normalized_speed < 0.32
            and proposed in (ROTATION_ACTIONS["NOOP"], ROTATION_ACTIONS["SHOOT"])
        ):
            safe_action = self._escape_action(
                float(vector[ObservationIndex.SAFE_CORRIDOR_ALIGNMENT]),
                float(vector[ObservationIndex.SAFE_CORRIDOR_TURN_DIRECTION]),
            )
            if safe_action != proposed:
                self.override_count += 1
                self.emergency_override_count += 1
            return np.asarray(safe_action), state

        if (
            proposed == ROTATION_ACTIONS["SHOOT"]
            and emergency >= self.emergency_threshold
            and float(vector[ObservationIndex.SPAWNER_FORWARD_ALIGNMENT]) > 0.75
            and float(vector[ObservationIndex.ENEMY_FORWARD_ALIGNMENT]) < 0.2
        ):
            safe_action = self._escape_action(
                float(vector[ObservationIndex.SAFE_CORRIDOR_ALIGNMENT]),
                float(vector[ObservationIndex.SAFE_CORRIDOR_TURN_DIRECTION]),
            )
            if safe_action != proposed:
                self.override_count += 1
                self.emergency_override_count += 1
            return np.asarray(safe_action), state

        return predicted, state


def _evaluate_policy(
    policy: Policy | None,
    control_style: str,
    episodes: int,
    seed: int,
    render: bool,
    playback_speed: float | None = None,
) -> dict[str, Any]:
    """Run complete episodes and return JSON-serializable diagnostics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if playback_speed is not None and playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")

    env = ArenaEnv(
        control_style=control_style,
        render_mode="human" if render else None,
    )
    env.action_space.seed(seed)
    render_clock = pygame.time.Clock() if render else None
    effective_playback_speed = (
        env.playback_speed if playback_speed is None else playback_speed
    )
    playback_fps = max(1, round(env.fps * effective_playback_speed))
    episode_results: list[dict[str, Any]] = []
    action_counts: Counter[int] = Counter()
    raw_action_counts: Counter[int] = Counter()
    raw_cooldown_shoots = 0
    immediate_danger_decisions = 0
    danger_noop_decisions = 0
    danger_thrust_decisions = 0
    near_wall_decisions = 0
    near_wall_action_counts: Counter[int] = Counter()

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            maximum_phase = env.phase
            destroyed_enemies = 0
            destroyed_spawners = 0
            projectile_hits = 0
            shots_fired = 0
            invalid_shots = 0
            enemy_damage_dealt = 0.0
            spawner_damage_dealt = 0.0
            total_damage = 0.0
            contact_events = 0
            danger_steps = 0
            crowd_pressure_sum = 0.0
            wall_steps = 0
            outward_thrust_steps = 0
            emergency_steps = 0
            emergency_shots = 0
            dangerous_spawner_damage = 0.0
            minimum_enemy_distance = float("inf")
            health_on_first_entry_phase2: float | None = None
            health_on_first_entry_phase3: float | None = None
            phase2_entry_step: int | None = None
            steps_before_phase2 = 0
            steps_after_phase2 = 0
            damage_before_phase2 = 0.0
            damage_after_phase2 = 0.0
            contacts_before_phase2 = 0
            contacts_after_phase2 = 0
            danger_steps_before_phase2 = 0
            danger_steps_after_phase2 = 0
            crowd_pressure_before_phase2 = 0.0
            crowd_pressure_after_phase2 = 0.0
            minimum_enemy_distance_before_phase2 = float("inf")
            minimum_enemy_distance_after_phase2 = float("inf")

            while True:
                entered_phase2_before_step = phase2_entry_step is not None
                if policy is None:
                    action = int(env.action_space.sample())
                else:
                    predicted_action, _ = policy.predict(observation, deterministic=True)
                    action = int(np.asarray(predicted_action).item())
                reported_raw_action = getattr(policy, "last_raw_action", None)
                raw_action = int(
                    action if reported_raw_action is None else reported_raw_action
                )
                action_counts[action] += 1
                raw_action_counts[raw_action] += 1

                if control_style == "rotation":
                    weapon_ready = float(observation[ObservationIndex.WEAPON_READY])
                    if (
                        raw_action == ROTATION_ACTIONS["SHOOT"]
                        and weapon_ready < 0.999
                    ):
                        raw_cooldown_shoots += 1
                    immediate_danger = (
                        float(observation[ObservationIndex.EMERGENCY_THREAT]) >= 0.35
                    )
                    if immediate_danger:
                        immediate_danger_decisions += 1
                        danger_noop_decisions += int(
                            action == ROTATION_ACTIONS["NOOP"]
                        )
                        danger_thrust_decisions += int(
                            action == ROTATION_ACTIONS["THRUST"]
                        )
                    near_wall = bool(
                        np.min(
                            observation[
                                ObservationIndex.WALL_LEFT_CLEARANCE
                                : ObservationIndex.WALL_BOTTOM_CLEARANCE + 1
                            ]
                        )
                        <= 0.25
                    )
                    if near_wall:
                        near_wall_decisions += 1
                        near_wall_action_counts[action] += 1

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                maximum_phase = max(maximum_phase, int(info["phase"]))
                destroyed_enemies += int(info["enemies_destroyed"])
                destroyed_spawners += int(info["spawners_destroyed"])
                projectile_hits += int(info["projectile_hits"])
                shots_fired += int(bool(info["shot_fired"]))
                invalid_shots += int(bool(info["invalid_shot"]))
                enemy_damage_dealt += float(info["enemy_damage_dealt"])
                spawner_damage_dealt += float(info["spawner_damage_dealt"])
                total_damage += float(info["damage_taken"])
                contact_events += int(float(info["damage_taken"]) > 0.0)
                crowd_pressure = float(info["crowd_pressure"])
                crowd_pressure_sum += crowd_pressure
                danger_steps += int(crowd_pressure > 0.0)
                wall_steps += int(float(info["wall_proximity"]) >= 0.75)
                outward_thrust_steps += int(float(info["wall_outward_thrust"]) > 0.05)
                emergency_steps += int(float(info["emergency_threat"]) >= 0.35)
                emergency_shots += int(float(info["emergency_attack"]) >= 0.35)
                dangerous_spawner_damage += float(info["dangerous_spawner_damage"])
                step_damage = float(info["damage_taken"])
                step_contact = int(step_damage > 0.0)
                active_enemy = int(info["active_enemies"]) > 0
                nearest_enemy_distance = float(info["nearest_enemy_distance"])
                if entered_phase2_before_step:
                    steps_after_phase2 += 1
                    damage_after_phase2 += step_damage
                    contacts_after_phase2 += step_contact
                    danger_steps_after_phase2 += int(crowd_pressure > 0.0)
                    crowd_pressure_after_phase2 += crowd_pressure
                    if active_enemy:
                        minimum_enemy_distance_after_phase2 = min(
                            minimum_enemy_distance_after_phase2,
                            nearest_enemy_distance,
                        )
                else:
                    # Damage and contacts on the transition step occur before
                    # the environment advances the phase, so they belong to
                    # the pre-Phase-2 segment.
                    steps_before_phase2 += 1
                    damage_before_phase2 += step_damage
                    contacts_before_phase2 += step_contact
                    danger_steps_before_phase2 += int(crowd_pressure > 0.0)
                    crowd_pressure_before_phase2 += crowd_pressure
                    if active_enemy:
                        minimum_enemy_distance_before_phase2 = min(
                            minimum_enemy_distance_before_phase2,
                            nearest_enemy_distance,
                        )
                if bool(info["phase_advanced"]):
                    entered_phase = int(info["phase"])
                    if entered_phase >= 2 and phase2_entry_step is None:
                        phase2_entry_step = int(info["step"])
                        health_on_first_entry_phase2 = float(info["player_health"])
                    if entered_phase >= 3 and health_on_first_entry_phase3 is None:
                        health_on_first_entry_phase3 = float(info["player_health"])
                if int(info["active_enemies"]) > 0:
                    minimum_enemy_distance = min(
                        minimum_enemy_distance,
                        float(info["nearest_enemy_distance"]),
                    )

                if render:
                    env.render()
                    if env.window_closed:
                        raise KeyboardInterrupt
                    assert render_clock is not None
                    render_clock.tick(playback_fps)
                if terminated or truncated:
                    episode_results.append(
                        {
                            "episode": episode + 1,
                            "reward": float(total_reward),
                            "steps": int(info["step"]),
                            "maximum_phase": maximum_phase,
                            "final_phase": int(info["phase"]),
                            "final_health": float(info["player_health"]),
                            "enemies_destroyed": destroyed_enemies,
                            "spawners_destroyed": destroyed_spawners,
                            "projectile_hits": projectile_hits,
                            "shots_fired": shots_fired,
                            "invalid_shots": invalid_shots,
                            "enemy_damage_dealt": enemy_damage_dealt,
                            "spawner_damage_dealt": spawner_damage_dealt,
                            "damage_taken": total_damage,
                            "contact_events": contact_events,
                            "danger_step_fraction": danger_steps / max(1, int(info["step"])),
                            "mean_crowd_pressure": crowd_pressure_sum
                            / max(1, int(info["step"])),
                            "wall_step_fraction": wall_steps / max(1, int(info["step"])),
                            "outward_thrust_step_fraction": outward_thrust_steps
                            / max(1, int(info["step"])),
                            "emergency_step_fraction": emergency_steps
                            / max(1, int(info["step"])),
                            "emergency_shots": emergency_shots,
                            "dangerous_spawner_damage": dangerous_spawner_damage,
                            "minimum_enemy_distance": (
                                minimum_enemy_distance
                                if np.isfinite(minimum_enemy_distance)
                                else None
                            ),
                            "health_on_first_entry_phase2": health_on_first_entry_phase2,
                            "health_on_first_entry_phase3": health_on_first_entry_phase3,
                            "damage_taken_before_phase2": damage_before_phase2,
                            "damage_taken_after_phase2": damage_after_phase2,
                            "contact_events_before_phase2": contacts_before_phase2,
                            "contact_events_after_phase2": contacts_after_phase2,
                            "steps_before_phase2": steps_before_phase2,
                            "steps_after_phase2": steps_after_phase2,
                            "steps_survived_after_first_entering_phase2": (
                                int(info["step"]) - phase2_entry_step
                                if phase2_entry_step is not None
                                else None
                            ),
                            "danger_step_fraction_before_phase2": (
                                danger_steps_before_phase2 / max(1, steps_before_phase2)
                            ),
                            "danger_step_fraction_after_phase2": (
                                danger_steps_after_phase2 / steps_after_phase2
                                if steps_after_phase2 > 0
                                else None
                            ),
                            "mean_crowd_pressure_before_phase2": (
                                crowd_pressure_before_phase2 / max(1, steps_before_phase2)
                            ),
                            "mean_crowd_pressure_after_phase2": (
                                crowd_pressure_after_phase2 / steps_after_phase2
                                if steps_after_phase2 > 0
                                else None
                            ),
                            "minimum_enemy_distance_before_phase2": (
                                minimum_enemy_distance_before_phase2
                                if np.isfinite(minimum_enemy_distance_before_phase2)
                                else None
                            ),
                            "minimum_enemy_distance_after_phase2": (
                                minimum_enemy_distance_after_phase2
                                if np.isfinite(minimum_enemy_distance_after_phase2)
                                else None
                            ),
                            "end_reason": str(info["episode_end"]),
                        }
                    )
                    break
    finally:
        env.close()

    rewards = np.asarray([item["reward"] for item in episode_results], dtype=float)
    phases = np.asarray([item["maximum_phase"] for item in episode_results], dtype=float)
    lengths = np.asarray([item["steps"] for item in episode_results], dtype=float)
    action_names = {value: key for key, value in env.action_names.items()}
    named_action_counts = {
        action_names[action]: int(action_counts.get(action, 0))
        for action in range(env.action_space.n)
    }
    named_raw_action_counts = {
        action_names[action]: int(raw_action_counts.get(action, 0))
        for action in range(env.action_space.n)
    }
    named_near_wall_action_counts = {
        action_names[action]: int(near_wall_action_counts.get(action, 0))
        for action in range(env.action_space.n)
    }
    total_actions = max(1, sum(action_counts.values()))

    return {
        "episodes": episodes,
        "control_style": control_style,
        "playback_speed": effective_playback_speed,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_episode_steps": float(lengths.mean()),
        "mean_max_phase": float(phases.mean()),
        "maximum_phase_reached": int(phases.max()),
        "phase_2_or_higher_rate": float(np.mean(phases >= 2.0)),
        "death_rate": float(
            np.mean([item["end_reason"] == "player_destroyed" for item in episode_results])
        ),
        "mean_enemies_destroyed": float(
            np.mean([item["enemies_destroyed"] for item in episode_results])
        ),
        "mean_spawners_destroyed": float(
            np.mean([item["spawners_destroyed"] for item in episode_results])
        ),
        "mean_projectile_hits": float(
            np.mean([item["projectile_hits"] for item in episode_results])
        ),
        "mean_shots_fired": float(
            np.mean([item["shots_fired"] for item in episode_results])
        ),
        "mean_invalid_shots": float(
            np.mean([item["invalid_shots"] for item in episode_results])
        ),
        "shot_hit_rate": float(
            sum(item["projectile_hits"] for item in episode_results)
            / max(1, sum(item["shots_fired"] for item in episode_results))
        ),
        "mean_enemy_damage_dealt": float(
            np.mean([item["enemy_damage_dealt"] for item in episode_results])
        ),
        "mean_spawner_damage_dealt": float(
            np.mean([item["spawner_damage_dealt"] for item in episode_results])
        ),
        "mean_damage_taken": float(
            np.mean([item["damage_taken"] for item in episode_results])
        ),
        "mean_contact_events": float(
            np.mean([item["contact_events"] for item in episode_results])
        ),
        "mean_damage_per_1000_steps": float(
            1000.0
            * sum(item["damage_taken"] for item in episode_results)
            / max(1, sum(item["steps"] for item in episode_results))
        ),
        "mean_danger_step_fraction": float(
            np.mean([item["danger_step_fraction"] for item in episode_results])
        ),
        "mean_crowd_pressure": float(
            np.mean([item["mean_crowd_pressure"] for item in episode_results])
        ),
        "mean_wall_step_fraction": float(
            np.mean([item["wall_step_fraction"] for item in episode_results])
        ),
        "mean_outward_thrust_step_fraction": float(
            np.mean([item["outward_thrust_step_fraction"] for item in episode_results])
        ),
        "mean_emergency_step_fraction": float(
            np.mean([item["emergency_step_fraction"] for item in episode_results])
        ),
        "mean_emergency_shots": float(
            np.mean([item["emergency_shots"] for item in episode_results])
        ),
        "mean_dangerous_spawner_damage": float(
            np.mean([item["dangerous_spawner_damage"] for item in episode_results])
        ),
        "mean_minimum_enemy_distance": float(
            np.mean(
                [
                    item["minimum_enemy_distance"]
                    for item in episode_results
                    if item["minimum_enemy_distance"] is not None
                ]
            )
        ),
        "action_counts": named_action_counts,
        "raw_model_action_counts": named_raw_action_counts,
        "executed_action_counts": named_action_counts,
        "raw_cooldown_shoot_attempts": int(raw_cooldown_shoots),
        "raw_cooldown_shoot_attempt_rate": float(raw_cooldown_shoots / total_actions),
        "cooldown_replacement_rate": float(raw_cooldown_shoots / total_actions),
        "immediate_danger_decisions": int(immediate_danger_decisions),
        "danger_noop_fraction": float(
            danger_noop_decisions / max(1, immediate_danger_decisions)
        ),
        "danger_thrust_fraction": float(
            danger_thrust_decisions / max(1, immediate_danger_decisions)
        ),
        "near_wall_decisions": int(near_wall_decisions),
        "near_wall_action_counts": named_near_wall_action_counts,
        "phase_transition_diagnostics": _aggregate_phase_diagnostics(
            episode_results
        ),
        "episode_results": episode_results,
    }


def evaluate_dqn(
    model: DQN,
    control_style: str = "rotation",
    episodes: int = 10,
    seed: int = 42,
    render: bool = False,
    playback_speed: float | None = None,
) -> dict[str, Any]:
    return _evaluate_policy(
        model,
        control_style,
        episodes,
        seed,
        render,
        playback_speed,
    )


def evaluate_random_policy(
    control_style: str = "rotation",
    episodes: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    return _evaluate_policy(None, control_style, episodes, seed, False)


def _latest_model(control_style: str) -> Path:
    model_root = PROJECT_ROOT / "models" / f"dqn_{control_style}"
    polished = model_root / "control_style_1_dqn_polished" / "final_model.zip"
    if control_style == "rotation" and polished.is_file():
        return polished
    progression = (
        model_root / "control_style_1_dqn_progression" / "final_model.zip"
    )
    if control_style == "rotation" and progression.is_file():
        return progression
    improved = model_root / "control_style_1_dqn_improved" / "final_model.zip"
    if control_style == "rotation" and improved.is_file():
        return improved
    promoted = model_root / f"control_style_1_dqn_current" / "final_model.zip"
    if control_style == "rotation" and promoted.is_file():
        return promoted
    candidates = list(model_root.glob("**/final_model.zip"))
    if not candidates:
        candidates = list(model_root.glob("**/best_model.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained DQN model found below {model_root}. Run python -m arena.train first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--observation-prefix",
        type=int,
        default=None,
        help="Pass only the first N features to a legacy model.",
    )
    parser.add_argument("--control-style", choices=("rotation", "direct"), default="rotation")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--cooldown-aware",
        action="store_true",
        help="Explicitly enable the default rotation-control cooldown mask.",
    )
    parser.add_argument(
        "--allow-invalid-shots",
        action="store_true",
        help="Disable cooldown-aware replacement for comparison purposes.",
    )
    parser.add_argument(
        "--safety-shield",
        action="store_true",
        help="Veto outward-boundary and attack-during-emergency commands.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=None,
        help="Visual speed multiplier; defaults to arena/config.json (currently 0.75).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or _latest_model(args.control_style)
    model = DQN.load(model_path, device="auto")
    model_observation_size = int(np.prod(model.observation_space.shape))
    observation_prefix = args.observation_prefix
    if observation_prefix is None and model_observation_size < len(OBSERVATION_NAMES):
        observation_prefix = model_observation_size
    cooldown_policy: CooldownAwarePolicy | None = None
    use_cooldown_mask = (
        args.control_style == "rotation" and not args.allow_invalid_shots
    )
    if use_cooldown_mask:
        cooldown_policy = CooldownAwarePolicy(model)
        policy: Policy = cooldown_policy
    else:
        policy = (
            ObservationPrefixPolicy(model, observation_prefix)
            if observation_prefix is not None
            else model
        )
    safety_policy: SafetyShieldPolicy | None = None
    if args.safety_shield:
        if args.control_style != "rotation":
            raise ValueError("--safety-shield currently supports rotation controls only")
        safety_policy = SafetyShieldPolicy(policy)
        policy = safety_policy
    metrics = evaluate_dqn(
        policy,
        control_style=args.control_style,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        playback_speed=args.playback_speed,
    )
    metrics["safety_shield"] = bool(safety_policy is not None)
    if safety_policy is not None:
        metrics["safety_overrides"] = safety_policy.override_count
        metrics["boundary_safety_overrides"] = safety_policy.boundary_override_count
        metrics["emergency_safety_overrides"] = safety_policy.emergency_override_count
    metrics["cooldown_aware_actions"] = bool(cooldown_policy is not None)
    if cooldown_policy is not None:
        metrics["cooldown_shoot_replacements"] = cooldown_policy.replacement_count
    rendered = json.dumps(metrics, indent=2)
    print(f"Model: {model_path}")
    if observation_prefix is not None:
        print(f"Observation compatibility prefix: {observation_prefix}")
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
