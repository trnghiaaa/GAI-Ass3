"""Deterministic, seed-controlled evaluation for trained arena policies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from arena.core.environment import ArenaEnv
from arena.learning.wrappers import ActionRepeatWrapper


ROW_FIELDS = (
    "episode", "seed", "reward", "phase", "phases_advanced",
    "simulation_steps", "decisions", "end_reason", "survived_time_limit",
    "enemies_destroyed", "spawners_destroyed", "damage_dealt", "damage_taken",
    "shots_fired", "projectile_hits", "accuracy", "hits_per_projectile", "player_level", "xp_earned",
    "weapon_name", "upgrades_chosen", "phase_rewards_chosen", "bosses_destroyed",
    "minibosses_destroyed", "boss_skills_dodged", "boss_skill_hits",
    "boss_defenders_destroyed", "boss_immune_hits", "missiles_evaded", "missile_hits",
    "boss_rewards_chosen", "drone_level", "barrier_charges",
    "build_summary",
    "contact_events", "crowd_fraction", "wall_fraction", "mean_enemy_clearance",
    "phase_diagnostics",
)


def evaluate_model(
    model: Any,
    control_style: str,
    *,
    episodes: int = 20,
    action_repeat: int = 4,
    seed: int = 9000,
    deterministic: bool = True,
    reset_options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one model and return per-episode rows plus aggregate metrics."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rows: list[dict[str, Any]] = []

    for episode in range(episodes):
        env = ActionRepeatWrapper(
            ArenaEnv(control_style=control_style), repeat=action_repeat
        )
        observation, _ = env.reset(seed=seed + episode, options=reset_options)
        episode_reward = 0.0
        decisions = 0
        contacts = 0
        crowd_steps = wall_steps = samples = 0
        clearance_total = 0.0
        enemy_samples = 0
        phase_diagnostics: dict[str, dict[str, float]] = {}
        final_info: dict[str, Any] = {}
        terminated = truncated = False
        try:
            while not (terminated or truncated):
                phase_before = str(env.unwrapped.phase)
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, final_info = env.step(
                    int(np.asarray(action).item())
                )
                episode_reward += float(reward)
                decisions += 1
                contacts += int(final_info.get('contact_events', 0))
                phase_data = phase_diagnostics.setdefault(phase_before, {
                    'frames': 0, 'damage': 0.0, 'contacts': 0,
                    'boss_casts': 0, 'boss_hits': 0, 'boss_dodges': 0,
                })
                phase_data['frames'] += int(final_info.get('action_repeat_frames', action_repeat))
                phase_data['damage'] += float(final_info.get('damage_taken', 0.0))
                phase_data['contacts'] += int(final_info.get('contact_events', 0))
                for output_key, event_key in [('boss_casts','boss_skills_cast'), ('boss_hits','boss_skill_hits'), ('boss_dodges','boss_skills_dodged')]:
                    phase_data[output_key] += int(final_info.get(event_key, 0))
                core = env.unwrapped
                crowd_steps += int(core._crowd_metrics()[1] > 0.25)
                p = core.player
                wall_steps += int(min(p.x-p.radius, core.width-p.radius-p.x,
                                      p.y-core.playfield_top-p.radius, core.height-p.radius-p.y) < 40)
                samples += 1
                if core.enemies:
                    clearance_total += min(max(0.0, np.hypot(e.x-p.x, e.y-p.y)-p.radius-e.radius) for e in core.enemies)
                    enemy_samples += 1
        finally:
            env.close()

        stats = final_info.get("episode_stats", {})
        shots = int(stats.get("shots_fired", 0))
        hits = int(stats.get("projectile_hits", 0))
        rows.append(
            {
                "episode": episode + 1,
                'contact_events': contacts,
                'crowd_fraction': crowd_steps / max(1,samples),
                'wall_fraction': wall_steps / max(1,samples),
                'mean_enemy_clearance': clearance_total / max(1,enemy_samples),
                'phase_diagnostics': phase_diagnostics,
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
                "accuracy": round(min(1.0, hits / shots), 6) if shots else 0.0,
                "hits_per_projectile": round(hits / shots, 6) if shots else 0.0,
                "player_level": int(final_info.get("player_level", 1)),
                "xp_earned": round(float(stats.get("xp_earned", 0.0)), 4),
                "weapon_name": str(final_info.get("weapon_name", "Pulse Cannon")),
                "upgrades_chosen": int(stats.get("upgrades_chosen", 0)),
                "phase_rewards_chosen": int(stats.get("phase_rewards_chosen", 0)),
                "bosses_destroyed": int(stats.get("bosses_destroyed", 0)),
                "minibosses_destroyed": int(stats.get("minibosses_destroyed", 0)),
                "boss_skills_dodged": int(stats.get("boss_skills_dodged", 0)),
                "boss_skill_hits": int(stats.get("boss_skill_hits", 0)),
                "boss_defenders_destroyed": int(stats.get("boss_defenders_destroyed", 0)),
                "boss_immune_hits": int(stats.get("boss_immune_hits", 0)),
                "missiles_evaded": int(stats.get("missiles_evaded", 0)),
                "missile_hits": int(stats.get("missile_hits", 0)),
                "boss_rewards_chosen": int(stats.get("boss_rewards_chosen", 0)),
                "drone_level": int(final_info.get("drone_level", 0)),
                "barrier_charges": int(final_info.get("barrier_charges", 0)),
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
        'damage_per_1000_frames': 1000 * sum(r['damage_taken'] for r in rows) / max(1,sum(r['simulation_steps'] for r in rows)),
        'contacts_per_1000_frames': 1000 * sum(r['contact_events'] for r in rows) / max(1,sum(r['simulation_steps'] for r in rows)),
        'mean_crowd_fraction': fmean(r['crowd_fraction'] for r in rows),
        'mean_wall_fraction': fmean(r['wall_fraction'] for r in rows),
        'mean_enemy_clearance': fmean(r['mean_enemy_clearance'] for r in rows),
        'mean_simulation_steps': fmean(r['simulation_steps'] for r in rows),
        "episodes": episodes,
        "action_repeat": action_repeat,
        "seed_start": seed,
        "deterministic": deterministic,
        "reset_options": dict(reset_options or {}),
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
        "mean_hits_per_projectile": round(
            fmean(float(row["hits_per_projectile"]) for row in rows), 6
        ),
        "mean_player_level": round(fmean(levels), 4),
        "max_player_level": max(levels),
        "mean_xp_earned": round(
            fmean(float(row["xp_earned"]) for row in rows), 4
        ),
        "mean_bosses_destroyed": round(
            fmean(int(row["bosses_destroyed"]) for row in rows), 4
        ),
        "mean_minibosses_destroyed": round(
            fmean(int(row["minibosses_destroyed"]) for row in rows), 4
        ),
        "mean_boss_skills_dodged": round(
            fmean(int(row["boss_skills_dodged"]) for row in rows), 4
        ),
        "mean_boss_skill_hits": round(
            fmean(int(row["boss_skill_hits"]) for row in rows), 4
        ),
        "mean_boss_defenders_destroyed": round(
            fmean(int(row["boss_defenders_destroyed"]) for row in rows), 4
        ),
        "mean_boss_immune_hits": round(
            fmean(int(row["boss_immune_hits"]) for row in rows), 4
        ),
        "mean_missiles_evaded": round(
            fmean(int(row["missiles_evaded"]) for row in rows), 4
        ),
        "mean_missile_hits": round(
            fmean(int(row["missile_hits"]) for row in rows), 4
        ),
        "mean_drone_level": round(
            fmean(int(row["drone_level"]) for row in rows), 4
        ),
        "mean_upgrades_chosen": round(
            fmean(int(row["upgrades_chosen"]) for row in rows), 4
        ),
    }
    phase_totals: dict[str, dict[str, float]] = {}
    for row in rows:
        for phase, values in row['phase_diagnostics'].items():
            totals = phase_totals.setdefault(phase, {key:0.0 for key in values})
            for key, value in values.items():
                totals[key] += value
    for totals in phase_totals.values():
        totals['damage_per_1000_frames'] = 1000 * totals['damage'] / max(1, totals['frames'])
        totals['contacts_per_1000_frames'] = 1000 * totals['contacts'] / max(1, totals['frames'])
    aggregate['phase_diagnostics'] = phase_totals
    return rows, aggregate


def write_benchmark(
    rows: list[dict[str, Any]], aggregate: dict[str, Any], output_stem: Path
) -> None:
    """Write marker-friendly CSV and JSON evidence for a benchmark run."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows([
            {key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
             for key, value in row.items()} for row in rows
        ])
    with output_stem.with_suffix(".json").open("w", encoding="utf-8") as output:
        json.dump({"aggregate": aggregate, "episodes": rows}, output, indent=2)


__all__ = ["ROW_FIELDS", "evaluate_model", "write_benchmark"]
