"""Deterministic, seed-controlled evaluation for trained arena policies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from arena.environment import ArenaEnv
from arena.wrappers import ActionRepeatWrapper


ROW_FIELDS = (
    "episode", "seed", "reward", "phase", "phases_advanced",
    "simulation_steps", "decisions", "end_reason", "survived_time_limit",
    "enemies_destroyed", "spawners_destroyed", "damage_dealt", "damage_taken",
    "shots_fired", "projectile_hits", "accuracy", "player_level", "xp_earned",
    "weapon_name", "upgrades_chosen", "phase_rewards_chosen", "bosses_destroyed",
    "build_summary",
)


def evaluate_model(
    model: Any,
    control_style: str,
    *,
    episodes: int = 20,
    action_repeat: int = 4,
    seed: int = 9000,
    deterministic: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one model and return per-episode rows plus aggregate metrics."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rows: list[dict[str, Any]] = []

    for episode in range(episodes):
        env = ActionRepeatWrapper(
            ArenaEnv(control_style=control_style), repeat=action_repeat
        )
        observation, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        decisions = 0
        final_info: dict[str, Any] = {}
        terminated = truncated = False
        try:
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, final_info = env.step(
                    int(np.asarray(action).item())
                )
                episode_reward += float(reward)
                decisions += 1
        finally:
            env.close()

        stats = final_info.get("episode_stats", {})
        shots = int(stats.get("shots_fired", 0))
        hits = int(stats.get("projectile_hits", 0))
        rows.append(
            {
                "episode": episode + 1,
                "seed": seed + episode,
                "reward": round(episode_reward, 6),
                "phase": int(final_info.get("phase", 1)),
                "phases_advanced": int(stats.get("phases_advanced", 0)),
                "simulation_steps": int(final_info.get("step", 0)),
                "decisions": decisions,
                "end_reason": str(final_info.get("episode_end", "unknown")),
                "survived_time_limit": bool(truncated),
                "enemies_destroyed": int(stats.get("enemies_destroyed", 0)),
                "spawners_destroyed": int(stats.get("spawners_destroyed", 0)),
                "damage_dealt": round(
                    float(stats.get("damage_dealt_enemy", 0.0))
                    + float(stats.get("damage_dealt_spawner", 0.0)), 4
                ),
                "damage_taken": round(float(stats.get("damage_taken", 0.0)), 4),
                "shots_fired": shots,
                "projectile_hits": hits,
                "accuracy": round(hits / shots, 6) if shots else 0.0,
                "player_level": int(final_info.get("player_level", 1)),
                "xp_earned": round(float(stats.get("xp_earned", 0.0)), 4),
                "weapon_name": str(final_info.get("weapon_name", "Pulse Cannon")),
                "upgrades_chosen": int(stats.get("upgrades_chosen", 0)),
                "phase_rewards_chosen": int(stats.get("phase_rewards_chosen", 0)),
                "bosses_destroyed": int(stats.get("bosses_destroyed", 0)),
                "build_summary": ";".join(
                    f"{key}:{value}"
                    for key, value in final_info.get("upgrade_stacks", {}).items()
                    if int(value) > 0
                ),
            }
        )

    rewards = [float(row["reward"]) for row in rows]
    phases = [int(row["phase"]) for row in rows]
    levels = [int(row["player_level"]) for row in rows]
    aggregate = {
        "control_style": control_style,
        "episodes": episodes,
        "action_repeat": action_repeat,
        "seed_start": seed,
        "deterministic": deterministic,
        "mean_reward": round(fmean(rewards), 6),
        "reward_std": round(pstdev(rewards), 6),
        "mean_phase": round(fmean(phases), 4),
        "max_phase": max(phases),
        "phase_progression_rate": round(
            sum(int(row["phases_advanced"]) > 0 for row in rows) / episodes, 6
        ),
        "survival_rate": round(
            sum(bool(row["survived_time_limit"]) for row in rows) / episodes, 6
        ),
        "mean_enemies_destroyed": round(
            fmean(int(row["enemies_destroyed"]) for row in rows), 4
        ),
        "mean_spawners_destroyed": round(
            fmean(int(row["spawners_destroyed"]) for row in rows), 4
        ),
        "mean_damage_taken": round(
            fmean(float(row["damage_taken"]) for row in rows), 4
        ),
        "mean_accuracy": round(fmean(float(row["accuracy"]) for row in rows), 6),
        "mean_player_level": round(fmean(levels), 4),
        "max_player_level": max(levels),
        "mean_xp_earned": round(
            fmean(float(row["xp_earned"]) for row in rows), 4
        ),
        "mean_bosses_destroyed": round(
            fmean(int(row["bosses_destroyed"]) for row in rows), 4
        ),
        "mean_upgrades_chosen": round(
            fmean(int(row["upgrades_chosen"]) for row in rows), 4
        ),
    }
    return rows, aggregate


def write_benchmark(
    rows: list[dict[str, Any]], aggregate: dict[str, Any], output_stem: Path
) -> None:
    """Write marker-friendly CSV and JSON evidence for a benchmark run."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with output_stem.with_suffix(".json").open("w", encoding="utf-8") as output:
        json.dump({"aggregate": aggregate, "episodes": rows}, output, indent=2)


__all__ = ["ROW_FIELDS", "evaluate_model", "write_benchmark"]
