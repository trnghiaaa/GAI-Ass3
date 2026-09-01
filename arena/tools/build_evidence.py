"""Build report-ready Part II comparison files and verify submission artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame

from arena.evaluation.benchmark import evaluate_model, write_benchmark
from arena.core.environment import (
    ArenaEnv,
    DIRECT_ACTIONS,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_NAMES,
)
from arena.presentation.renderer import ArenaRenderer
from arena.settings import (
    ARENA_EVIDENCE_DIR,
    ARENA_LOG_DIR,
    ARENA_TRAINING_DIR,
    PROJECT_ROOT,
    TENSORBOARD_DIR,
    ensure_artifact_directories,
    metadata_path,
    model_path,
)


TENSORBOARD_RUN_STEMS = (
    "tune_direct_fast_exploration",
    "tune_direct_balanced",
    "tune_direct_long_exploration",
    "tune_rotation_fast_exploration",
    "tune_rotation_balanced",
    "tune_rotation_long_exploration",
)


def _selected_run_stem(style: str) -> str:
    path = metadata_path(style)
    if path.exists():
        metadata = json.loads(path.read_text(encoding='utf-8'))
        return str(metadata.get('run_name', f'dqn_{style}'))
    return f'dqn_{style}'


def _selected_run_dir(style: str) -> Path:
    return ARENA_TRAINING_DIR / _selected_run_stem(style)


def _final_benchmark_paths(style: str) -> tuple[Path, Path]:
    """Find the current revalidated benchmark named in final metadata."""

    with metadata_path(style).open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    recorded = metadata.get("holdout_evaluation")
    if recorded:
        json_path = Path(str(recorded))
        if not json_path.is_absolute():
            json_path = PROJECT_ROOT / json_path
        return json_path.with_suffix(".csv"), json_path
    stem = _selected_run_dir(style) / "benchmark"
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def _boss_focus_paths(style: str) -> tuple[Path, Path] | None:
    """Return the optional fixed-boss evaluation recorded by final metadata."""

    with metadata_path(style).open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    recorded = metadata.get("boss_focus_evaluation")
    if not recorded:
        return None
    json_path = Path(str(recorded))
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    return json_path.with_suffix(".csv"), json_path


def required_artifacts() -> list[Path]:
    paths: list[Path] = []
    for style in ("direct", "rotation"):
        run_dir = _selected_run_dir(style)
        benchmark_csv, benchmark_json = _final_benchmark_paths(style)
        paths.extend(
            [
                model_path(style),
                metadata_path(style),
                run_dir / "training.monitor.csv",
                run_dir / "training_curve.png",
                benchmark_csv,
                benchmark_json,
                run_dir / "summary.json",
                run_dir / "model_selection.json",
            ]
        )
        boss_focus = _boss_focus_paths(style)
        if boss_focus:
            paths.extend(boss_focus)
    paths.extend(
        [
            ARENA_LOG_DIR / "tuning" / "hyperparameter_results.csv",
            ARENA_LOG_DIR / "tuning" / "hyperparameter_results.json",
            ARENA_LOG_DIR / "tuning" / "hyperparameter_comparison.png",
        ]
    )
    return paths


def _latest_tensorboard_events() -> list[Path]:
    """Return one reproducible final TensorBoard run for each trained variant."""

    events: list[Path] = []
    stems = [_selected_run_stem(style) for style in ('direct', 'rotation')]
    stems.extend(stem for stem in TENSORBOARD_RUN_STEMS if stem.startswith('tune_'))
    for stem in stems:
        candidates: list[tuple[int, Path]] = []
        for directory in TENSORBOARD_DIR.glob(f"{stem}_*"):
            try:
                run_number = int(directory.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((run_number, directory))
        if not candidates:
            continue
        latest = max(candidates, key=lambda item: item[0])[1]
        events.extend(sorted(latest.glob("events.out.tfevents.*")))
    return events


def _load_final_metadata() -> list[dict[str, Any]]:
    rows = []
    for style in ("direct", "rotation"):
        with metadata_path(style).open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        if metadata.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(f"{style} model metadata uses a stale arena schema")
        if metadata.get("observation_names") != list(OBSERVATION_NAMES):
            raise ValueError(f"{style} model metadata has a stale observation contract")
        benchmark = metadata["benchmark"]
        rows.append(
            {
                "control_style": style,
                "episodes": benchmark['episodes'],
                "seed_start": benchmark['seed_start'],
                "action_repeat": benchmark['action_repeat'],
                "algorithm": metadata["algorithm"],
                "timesteps": metadata["total_timesteps"],
                "network": "x".join(str(value) for value in metadata["network"]),
                "mean_reward": benchmark["mean_reward"],
                "reward_std": benchmark["reward_std"],
                "mean_phase": benchmark["mean_phase"],
                "max_phase": benchmark["max_phase"],
                "phase_progression_rate": benchmark["phase_progression_rate"],
                "survival_rate": benchmark["survival_rate"],
                "mean_enemies_destroyed": benchmark["mean_enemies_destroyed"],
                "mean_spawners_destroyed": benchmark["mean_spawners_destroyed"],
                "mean_accuracy": benchmark["mean_accuracy"],
                "mean_hits_per_projectile": benchmark.get(
                    "mean_hits_per_projectile", benchmark["mean_accuracy"]
                ),
                "mean_player_level": benchmark.get("mean_player_level", 1.0),
                "max_player_level": benchmark.get("max_player_level", 1),
                "mean_xp_earned": benchmark.get("mean_xp_earned", 0.0),
                "mean_bosses_destroyed": benchmark.get("mean_bosses_destroyed", 0.0),
                "mean_minibosses_destroyed": benchmark.get(
                    "mean_minibosses_destroyed", 0.0
                ),
                "mean_boss_skills_dodged": benchmark.get(
                    "mean_boss_skills_dodged", 0.0
                ),
                "mean_boss_skill_hits": benchmark.get(
                    "mean_boss_skill_hits", 0.0
                ),
                "mean_drone_level": benchmark.get("mean_drone_level", 0.0),
                "mean_upgrades_chosen": benchmark.get("mean_upgrades_chosen", 0.0),
            }
        )
    return rows


def _write_control_comparison(rows: list[dict[str, Any]]) -> None:
    csv_path = ARENA_EVIDENCE_DIR / "control_style_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ARENA_EVIDENCE_DIR / "control_style_comparison.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump({"control_styles": rows}, output, indent=2)

    figure, axes_grid = plt.subplots(2, 2, figsize=(12, 8.2))
    axes = axes_grid.flat
    labels = [str(row["control_style"]).title() for row in rows]
    colors = ["#35cfff", "#b552ff"]
    metrics = (
        ("mean_reward", "Mean reward"),
        ("phase_progression_rate", "Phase progression rate"),
        ("mean_enemies_destroyed", "Mean enemies destroyed"),
        ("mean_player_level", "Mean ship level"),
    )
    for axis, (key, label) in zip(axes, metrics):
        values = [float(row[key]) for row in rows]
        axis.bar(labels, values, color=colors)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
        if key == "phase_progression_rate":
            axis.set_ylim(0, 1.05)
            axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    figure.suptitle("Final Part II deterministic policy comparison")
    figure.tight_layout()
    figure.savefig(ARENA_EVIDENCE_DIR / "control_style_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    positions = np.arange(len(labels))
    width = 0.34
    dodges = [float(row["mean_boss_skills_dodged"]) for row in rows]
    hits = [float(row["mean_boss_skill_hits"]) for row in rows]
    axis.bar(
        positions - width / 2,
        dodges,
        width,
        label="Barrages dodged",
        color="#35d6c4",
    )
    axis.bar(
        positions + width / 2,
        hits,
        width,
        label="Barrage hits",
        color="#ff4f78",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Mean events per held-out episode")
    axis.set_title("Learned boss-hazard interaction")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(ARENA_EVIDENCE_DIR / "boss_avoidance_comparison.png", dpi=180)
    plt.close(figure)


class _SeededRandomPolicy:
    def __init__(self, action_count: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.action_count = action_count

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
        del observation, deterministic
        return int(self.rng.integers(self.action_count)), None


def _write_learning_baseline(final_rows: list[dict[str, Any]]) -> None:
    comparisons: list[dict[str, Any]] = []
    for index, style in enumerate(("direct", "rotation")):
        action_count = 6 if style == "direct" else 5
        trained = next(row for row in final_rows if row["control_style"] == style)
        random_rows, random_aggregate = evaluate_model(
            _SeededRandomPolicy(action_count, trained['seed_start'] + index),
            style,
            episodes=trained['episodes'],
            action_repeat=trained['action_repeat'],
            seed=trained['seed_start'],
            deterministic=False,
        )
        write_benchmark(
            random_rows,
            random_aggregate,
            ARENA_EVIDENCE_DIR / f"random_baseline_{style}",
        )
        trained = next(row for row in final_rows if row["control_style"] == style)
        comparisons.append(
            {
                "control_style": style,
                "random_mean_reward": random_aggregate["mean_reward"],
                "trained_mean_reward": trained["mean_reward"],
                "random_phase_progression_rate": random_aggregate["phase_progression_rate"],
                "trained_phase_progression_rate": trained["phase_progression_rate"],
            }
        )
    with (ARENA_EVIDENCE_DIR / "learning_vs_random.json").open("w", encoding="utf-8") as output:
        json.dump({"comparisons": comparisons}, output, indent=2)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    labels = [str(row["control_style"]).title() for row in comparisons]
    positions = np.arange(len(labels))
    width = 0.34
    for axis, random_key, trained_key, title in (
        (axes[0], "random_mean_reward", "trained_mean_reward", "Mean episode reward"),
        (axes[1], "random_phase_progression_rate", "trained_phase_progression_rate", "Phase progression rate"),
    ):
        axis.bar(positions - width / 2, [row[random_key] for row in comparisons], width, label="Random", color="#667085")
        axis.bar(positions + width / 2, [row[trained_key] for row in comparisons], width, label="Trained DQN", color="#35cfff")
        axis.set_xticks(positions, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axes[0].legend()
    figure.suptitle("Learned policies versus seeded random-action baselines")
    figure.tight_layout()
    figure.savefig(ARENA_EVIDENCE_DIR / "learning_vs_random.png", dpi=180)
    plt.close(figure)


def _capture_environment_preview() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    env = ArenaEnv(control_style="direct")
    env.reset(seed=8)
    env.player.level = 5
    env.player.xp = env.xp_threshold_for_level(5) + 20.0
    env.upgrade_stacks.update(
        {
            "multishot": 2,
            "laser": 1,
            "damage": 2,
            "range": 1,
            "shield": 1,
            "drone": 4,
            "homing": 1,
            "regen": 1,
        }
    )
    env.last_upgrade_name = str(env.weapon_profile()["name"])
    env.spawners[0].spawn_cooldown_steps = 1
    env.player.angle = math.atan2(
        env.spawners[0].y - env.player.y,
        env.spawners[0].x - env.player.x,
    )
    renderer = ArenaRenderer(env, mode="rgb_array")
    try:
        for frame in range(95):
            if env.done:
                break
            action = DIRECT_ACTIONS["SHOOT"] if frame % 14 == 0 else DIRECT_ACTIONS["NOOP"]
            env.step(action)
        # The seeded showcase can clear phase one quickly. Re-stage a live phase
        # so the report image demonstrates combat rather than an empty hand-off.
        env.phase_transition_steps = 0
        if not env.spawners:
            env._spawn_phase_spawners()
        if not env.enemies:
            for spawner in env.spawners[:2]:
                env._spawn_enemy(spawner)
        env.player.angle = math.atan2(
            env.spawners[0].y - env.player.y,
            env.spawners[0].x - env.player.x,
        )
        for frame in range(12):
            action = DIRECT_ACTIONS["SHOOT"] if frame in (0, 7) else DIRECT_ACTIONS["NOOP"]
            env.step(action)
        renderer.render(
            footer_text="PART II ENVIRONMENT  •  continuous motion  •  live collision and health systems"
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "environment_showcase.png")
        renderer.render(
            footer_text="GAME PAUSED  •  P or RESUME continues",
            paused=True,
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "pause_showcase.png")
        env.upgrade_banner_steps = max(
            1,
            int(float(env.progression_cfg["upgrade_banner_seconds"]) * env.fps),
        )
        renderer.render(
            footer_text="COMBAT XP  •  composed build upgrades  •  required action sets unchanged"
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "upgrade_showcase.png")
        env.upgrade_banner_steps = 0
        env.phase = 2
        env.phase_step_count = 0
        env.phase_max_steps = env._phase_step_budget()
        env.phase_transition_steps = 0
        env.player.health = env.player.max_health
        env.spawners = []
        env.enemies = []
        env.projectiles = []
        env._spawn_phase_spawners()
        for spawner in env.spawners:
            env._spawn_enemy(spawner)
        renderer.render(
            footer_text="THREAT SURGE  •  modest pressure while hull and time are comfortable"
        )
        pygame.image.save(
            renderer.surface, ARENA_EVIDENCE_DIR / "adaptive_pressure_showcase.png"
        )
        env.phase = 3
        env.spawners = []
        env.enemies = []
        env.projectiles = []
        env.danger_zones = []
        env.phase_transition_steps = max(
            1, int(float(env.phase_cfg["transition_seconds"]) * env.fps * 0.62)
        )
        renderer.render(
            footer_text="BOSS TRANSITION  •  animated threat warning before deployment"
        )
        pygame.image.save(
            renderer.surface, ARENA_EVIDENCE_DIR / "boss_transition_showcase.png"
        )
        env.phase_transition_steps = 0
        env.manual_choices = True
        env.pending_choice_kind = "phase_reward"
        support_catalog = {
            str(item["id"]): dict(item)
            for item in env.progression_cfg["phase_reward_catalog"]
        }
        env.pending_choices = [
            support_catalog["wingman"],
            support_catalog["overdrive"],
            support_catalog["aegis"],
        ]
        renderer.render(
            footer_text="CLEAR UPGRADE PREVIEW  •  current build compared with the exact result"
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "choice_showcase.png")
        env.pending_choice_kind = None
        env.pending_choices = []
        env.upgrade_banner_steps = 0
        env.phase = 3
        env.spawners = []
        env.enemies = []
        env.projectiles = []
        env.phase_transition_steps = 0
        env._spawn_phase_spawners()
        env._spawn_enemy(env.spawners[0])
        env.boss_skill_index = 2
        env._cast_boss_skill({"boss_skills_cast": 0})
        renderer.render(
            footer_text="BOSS PHASE  •  readable diagonal warning  •  move before impact"
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "boss_showcase.png")

        env.danger_zones = []
        boss = next(spawner for spawner in env.spawners if spawner.is_boss)
        # Stage the report frame away from HUD/arena edges so every telegraph,
        # sentry and health plate is visible without changing live gameplay.
        boss.x, boss.y = 630.0, 245.0
        env.player.x, env.player.y = 310.0, 410.0
        boss.shield = 0.0
        boss.health = boss.max_health * 0.38
        defender_events = {"boss_defenders_spawned": 0, "enemies_spawned": 0}
        env._spawn_boss_defenders(boss, defender_events)
        env._fire_enemy_missile(
            env.boss_defenders[0], {"missiles_fired": 0}
        )
        renderer.render(
            footer_text="AEGIS INTERMISSION  •  destroy sentries  •  dodge telegraphed missiles"
        )
        pygame.image.save(
            renderer.surface,
            ARENA_EVIDENCE_DIR / "boss_intermission_showcase.png",
        )

        env.phase = 4
        env.spawners = []
        env.enemies = []
        env.projectiles = []
        env.danger_zones = []
        env._spawn_phase_spawners()
        env._spawn_miniboss(env.spawners[0])
        renderer.render(
            footer_text="RIFT HUNTER  •  optional miniboss  •  XP + repair + Aegis cache"
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "miniboss_showcase.png")

        boss_catalog = [dict(item) for item in env.progression_cfg["boss_reward_catalog"]]
        env.pending_choice_kind = "boss_reward"
        env.pending_choices = boss_catalog[:3]
        renderer.render(
            footer_text="BOSS RELIC DRAFT  •  lasting rewards match the encounter risk"
        )
        pygame.image.save(renderer.surface, ARENA_EVIDENCE_DIR / "boss_reward_showcase.png")
        env.pending_choice_kind = None
        env.pending_choices = []
        env.done = True
        env.last_end_reason = "phase_timeout"
        env.episode_stats["reward"] = 684.2
        env.episode_stats["enemies_destroyed"] = 73
        env.episode_stats["spawners_destroyed"] = 15
        env.episode_stats["bosses_destroyed"] = 2
        renderer.episode_end_has_next = False
        renderer.render(
            footer_text="MISSION SUMMARY  •  replay or return when you are ready"
        )
        pygame.image.save(
            renderer.surface, ARENA_EVIDENCE_DIR / "ai_mission_summary_showcase.png"
        )
    finally:
        renderer.close()
        env.close()


def build() -> None:
    ensure_artifact_directories()
    missing = [path for path in required_artifacts() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required Part II artifacts are missing:\n{formatted}")
    tensorboard_events = _latest_tensorboard_events()
    if not tensorboard_events:
        raise FileNotFoundError(f"No TensorBoard event files found below {TENSORBOARD_DIR}")

    rows = _load_final_metadata()
    _write_control_comparison(rows)
    _write_learning_baseline(rows)
    _capture_environment_preview()
    manifest = {
        "environment_schema": ENVIRONMENT_SCHEMA_VERSION,
        "historical_hyperparameter_sweep_schema": 5,
        "note": "Final schema-10 evidence includes revalidated models, a measured gradual normal-phase pressure director, independent visible boss tiers, a persistent AI mission summary, finite-guidance sentry missiles, explicit missile-escape observations, a boss-intermission curriculum, and fixed-seed normal plus Phase-3 checkpoint selection. The compact tuning sweep is retained as hyperparameter evidence.",
        "verified_files": [str(path.relative_to(PROJECT_ROOT)) for path in required_artifacts()],
        "tensorboard_event_files": [
            str(path.relative_to(PROJECT_ROOT)) for path in tensorboard_events
        ],
        "generated": [
            "control_style_comparison.csv",
            "control_style_comparison.json",
            "control_style_comparison.png",
            "boss_avoidance_comparison.png",
            "environment_showcase.png",
            "pause_showcase.png",
            "upgrade_showcase.png",
            "adaptive_pressure_showcase.png",
            "boss_transition_showcase.png",
            "choice_showcase.png",
            "boss_showcase.png",
            "boss_intermission_showcase.png",
            "miniboss_showcase.png",
            "boss_reward_showcase.png",
            "ai_mission_summary_showcase.png",
            "random_baseline_direct.csv",
            "random_baseline_direct.json",
            "random_baseline_rotation.csv",
            "random_baseline_rotation.json",
            "learning_vs_random.json",
            "learning_vs_random.png",
        ],
    }
    with (ARENA_EVIDENCE_DIR / "evidence_manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    print(
        f"Part II evidence ready: {len(required_artifacts())} required files and "
        f"{len(tensorboard_events)} TensorBoard event file(s) verified."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
