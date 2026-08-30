"""Build report-ready Part II comparison files and verify submission artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygame

from arena.environment import ArenaEnv, DIRECT_ACTIONS
from arena.renderer import ArenaRenderer
from arena.settings import (
    ARENA_LOG_DIR,
    PROJECT_ROOT,
    TENSORBOARD_DIR,
    ensure_artifact_directories,
    metadata_path,
    model_path,
)


def required_artifacts() -> list[Path]:
    paths: list[Path] = []
    for style in ("direct", "rotation"):
        run_dir = ARENA_LOG_DIR / "runs" / f"dqn_{style}"
        paths.extend(
            [
                model_path(style),
                metadata_path(style),
                run_dir / "training.monitor.csv",
                run_dir / "training_curve.png",
                run_dir / "benchmark.csv",
                run_dir / "benchmark.json",
                run_dir / "summary.json",
                run_dir / "model_selection.json",
            ]
        )
    paths.extend(
        [
            ARENA_LOG_DIR / "tuning" / "hyperparameter_results.csv",
            ARENA_LOG_DIR / "tuning" / "hyperparameter_results.json",
            ARENA_LOG_DIR / "tuning" / "hyperparameter_comparison.png",
        ]
    )
    return paths


def _load_final_metadata() -> list[dict[str, Any]]:
    rows = []
    for style in ("direct", "rotation"):
        with metadata_path(style).open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        benchmark = metadata["benchmark"]
        rows.append(
            {
                "control_style": style,
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
                "mean_player_level": benchmark.get("mean_player_level", 1.0),
                "max_player_level": benchmark.get("max_player_level", 1),
                "mean_xp_earned": benchmark.get("mean_xp_earned", 0.0),
            }
        )
    return rows


def _write_control_comparison(rows: list[dict[str, Any]]) -> None:
    csv_path = ARENA_LOG_DIR / "control_style_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ARENA_LOG_DIR / "control_style_comparison.json").open(
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
    figure.savefig(ARENA_LOG_DIR / "control_style_comparison.png", dpi=180)
    plt.close(figure)


def _capture_environment_preview() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    env = ArenaEnv(control_style="direct")
    env.reset(seed=8)
    env.player.level = 4
    env.player.xp = 270.0
    env.last_upgrade_name = str(env.weapon_profile()["name"])
    env.spawners[0].spawn_cooldown_steps = 1
    renderer = ArenaRenderer(env, mode="rgb_array")
    try:
        for frame in range(95):
            if env.done:
                break
            action = DIRECT_ACTIONS["SHOOT"] if frame % 14 == 0 else DIRECT_ACTIONS["NOOP"]
            env.step(action)
        renderer.render(
            footer_text="PART II ENVIRONMENT  •  continuous motion  •  live collision and health systems"
        )
        pygame.image.save(renderer.surface, ARENA_LOG_DIR / "environment_showcase.png")
        env.upgrade_banner_steps = max(
            1,
            int(float(env.progression_cfg["upgrade_banner_seconds"]) * env.fps),
        )
        renderer.render(
            footer_text="COMBAT XP  •  automatic upgrades  •  required action sets unchanged"
        )
        pygame.image.save(renderer.surface, ARENA_LOG_DIR / "upgrade_showcase.png")
    finally:
        renderer.close()
        env.close()


def build() -> None:
    ensure_artifact_directories()
    missing = [path for path in required_artifacts() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required Part II artifacts are missing:\n{formatted}")
    tensorboard_events = list(TENSORBOARD_DIR.rglob("events.out.tfevents.*"))
    if not tensorboard_events:
        raise FileNotFoundError(f"No TensorBoard event files found below {TENSORBOARD_DIR}")

    rows = _load_final_metadata()
    _write_control_comparison(rows)
    _capture_environment_preview()
    manifest = {
        "verified_files": [str(path.relative_to(PROJECT_ROOT)) for path in required_artifacts()],
        "tensorboard_event_files": [
            str(path.relative_to(PROJECT_ROOT)) for path in tensorboard_events
        ],
        "generated": [
            "control_style_comparison.csv",
            "control_style_comparison.json",
            "control_style_comparison.png",
            "environment_showcase.png",
            "upgrade_showcase.png",
        ],
    }
    with (ARENA_LOG_DIR / "evidence_manifest.json").open("w", encoding="utf-8") as output:
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
