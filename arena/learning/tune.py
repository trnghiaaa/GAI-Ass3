"""Run reproducible DQN hyperparameter comparisons for report evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arena.settings import ARENA_LOG_DIR, training_settings
from arena.learning.train import train_control_style


PROFILES = ("fast_exploration", "balanced", "long_exploration")


def run_tuning(
    styles: tuple[str, ...],
    *,
    timesteps: int,
    benchmark_episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Train every style/profile pair and write a compact comparison bundle."""

    rows: list[dict[str, Any]] = []
    for style_index, style in enumerate(styles):
        for profile_index, profile in enumerate(PROFILES):
            run_name = f"tune_{style}_{profile}"
            metadata = train_control_style(
                style,
                timesteps=timesteps,
                seed=seed + style_index * 100 + profile_index,
                profile=profile,
                run_name=run_name,
                benchmark_episodes=benchmark_episodes,
                verbose=0,
            )
            settings = training_settings(style, profile)
            benchmark = metadata["benchmark"]
            rows.append(
                {
                    "control_style": style,
                    "profile": profile,
                    "timesteps": timesteps,
                    "learning_rate": settings["learning_rate"],
                    "exploration_fraction": settings["exploration_fraction"],
                    "network": "x".join(str(size) for size in settings["net_arch"]),
                    "mean_reward": benchmark["mean_reward"],
                    "reward_std": benchmark["reward_std"],
                    "mean_phase": benchmark["mean_phase"],
                    "phase_progression_rate": benchmark["phase_progression_rate"],
                    "survival_rate": benchmark["survival_rate"],
                    "mean_enemies_destroyed": benchmark["mean_enemies_destroyed"],
                    "mean_spawners_destroyed": benchmark["mean_spawners_destroyed"],
                    "mean_player_level": benchmark["mean_player_level"],
                    "max_player_level": benchmark["max_player_level"],
                    "mean_xp_earned": benchmark["mean_xp_earned"],
                }
            )

    output_dir = ARENA_LOG_DIR / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "hyperparameter_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "hyperparameter_results.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump({"profiles": rows}, output, indent=2)
    _plot_results(rows, output_dir / "hyperparameter_comparison.png")
    return rows


def _plot_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len({row["control_style"] for row in rows}), figsize=(12, 5), squeeze=False)
    for axis, style in zip(axes[0], sorted({row["control_style"] for row in rows})):
        selected = [row for row in rows if row["control_style"] == style]
        labels = [str(row["profile"]).replace("_", "\n") for row in selected]
        rewards = [float(row["mean_reward"]) for row in selected]
        colors = ["#ff9f43", "#35d6c4", "#b76cff"]
        bars = axis.bar(labels, rewards, color=colors[: len(selected)])
        axis.axhline(0, color="#667085", linewidth=1)
        axis.set_title(f"{style.title()} controls")
        axis.set_ylabel("Deterministic mean reward")
        axis.grid(axis="y", alpha=0.2)
        for bar, row in zip(bars, selected):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{float(row['phase_progression_rate']):.0%} phase",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8,
            )
    figure.suptitle("Part II DQN hyperparameter exploration")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare three non-default DQN configurations"
    )
    parser.add_argument(
        "--control-style", choices=("direct", "rotation", "both"), default="both"
    )
    parser.add_argument("--timesteps", type=int, default=40000)
    parser.add_argument("--benchmark-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    styles = (
        ("direct", "rotation")
        if args.control_style == "both"
        else (args.control_style,)
    )
    rows = run_tuning(
        styles,
        timesteps=args.timesteps,
        benchmark_episodes=args.benchmark_episodes,
        seed=args.seed,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
