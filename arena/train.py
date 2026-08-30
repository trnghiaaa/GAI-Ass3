"""Train separate Stable-Baselines3 DQN agents for both required controls.

Examples
--------
python -m arena.train --control-style both
python -m arena.train --control-style rotation --timesteps 300000
python -m arena.train --control-style direct --profile fast_exploration
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
import torch

from arena.benchmark import evaluate_model, write_benchmark
from arena.environment import ArenaEnv, OBSERVATION_NAMES
from arena.settings import (
    ARENA_LOG_DIR,
    TENSORBOARD_DIR,
    ensure_artifact_directories,
    metadata_path,
    model_path,
    training_settings,
)
from arena.wrappers import ActionRepeatWrapper


MONITOR_INFO_KEYS = (
    "phase",
    "player_health",
    "episode_enemies_destroyed",
    "episode_spawners_destroyed",
    "episode_phases_advanced",
    "episode_damage_taken",
    "episode_xp_earned",
    "episode_max_level",
)


class ArenaTelemetryCallback(BaseCallback):
    """Expose arena-specific diagnostics in the TensorBoard run."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        self.logger.record(
            "arena/phase",
            float(np.mean([float(info.get("phase", 1)) for info in infos])),
        )
        self.logger.record(
            "arena/player_health",
            float(np.mean([float(info.get("player_health", 0)) for info in infos])),
        )
        self.logger.record(
            "arena/player_level",
            float(np.mean([float(info.get("player_level", 1)) for info in infos])),
        )
        self.logger.record(
            "arena/xp_progress",
            float(np.mean([float(info.get("xp_progress", 0)) for info in infos])),
        )
        for name in ("enemies_destroyed", "spawners_destroyed", "damage_taken"):
            self.logger.record(
                f"arena/event_{name}",
                float(np.mean([float(info.get(name, 0)) for info in infos])),
            )
        return True


def _make_env(
    control_style: str,
    action_repeat: int,
    *,
    monitor_file: Path | None = None,
) -> Monitor:
    env = ActionRepeatWrapper(
        ArenaEnv(control_style=control_style), repeat=action_repeat
    )
    filename = str(monitor_file) if monitor_file is not None else None
    return Monitor(env, filename=filename, info_keywords=MONITOR_INFO_KEYS)


def _plot_monitor(monitor_file: Path, output_path: Path, title: str) -> None:
    if not monitor_file.exists():
        return
    with monitor_file.open("r", encoding="utf-8") as source:
        reader = csv.DictReader(line for line in source if not line.startswith("#"))
        rows = list(reader)
    if not rows:
        return
    rewards = [float(row["r"]) for row in rows]
    phases = [float(row["phase"]) for row in rows]
    episodes = np.arange(1, len(rows) + 1)
    window = max(1, min(50, len(rows) // 8 or 1))
    smooth = [
        float(np.mean(rewards[max(0, index - window + 1) : index + 1]))
        for index in range(len(rewards))
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        episodes,
        rewards,
        color="#5b76a8",
        alpha=0.28,
        label="episode",
    )
    axes[0].plot(
        episodes,
        smooth,
        color="#35d6c4",
        linewidth=2.2,
        label=f"rolling {window}",
    )
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()
    axes[0].grid(alpha=0.18)
    axes[1].plot(
        episodes, phases, color="#b76cff", linewidth=1.4
    )
    axes[1].set_ylabel("Final phase")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.18)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train_control_style(
    control_style: str,
    *,
    timesteps: int | None = None,
    seed: int | None = None,
    profile: str = "balanced",
    run_name: str | None = None,
    benchmark_episodes: int = 12,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train, save, and independently benchmark one control-style policy."""

    ensure_artifact_directories()
    settings = training_settings(control_style, profile)
    total_timesteps = int(timesteps or settings["total_timesteps"])
    if total_timesteps < 1:
        raise ValueError("timesteps must be positive")
    if benchmark_episodes < 1:
        raise ValueError("benchmark_episodes must be positive")
    training_seed = int(settings["seed"] if seed is None else seed)
    action_repeat = int(settings["action_repeat"])
    stem = run_name or f"dqn_{control_style}"
    run_dir = ARENA_LOG_DIR / "runs" / stem
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = run_dir / "training.monitor.csv"

    train_env = _make_env(control_style, action_repeat, monitor_file=monitor_file)
    eval_env = _make_env(control_style, action_repeat)
    learning_starts = min(
        int(settings["learning_starts"]), max(0, total_timesteps // 5)
    )
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=float(settings["learning_rate"]),
        buffer_size=int(settings["buffer_size"]),
        learning_starts=learning_starts,
        batch_size=int(settings["batch_size"]),
        gamma=float(settings["gamma"]),
        train_freq=int(settings["train_freq"]),
        gradient_steps=int(settings["gradient_steps"]),
        target_update_interval=int(settings["target_update_interval"]),
        exploration_fraction=float(settings["exploration_fraction"]),
        exploration_initial_eps=float(settings["exploration_initial_eps"]),
        exploration_final_eps=float(settings["exploration_final_eps"]),
        policy_kwargs={"net_arch": list(settings["net_arch"])},
        tensorboard_log=str(TENSORBOARD_DIR),
        seed=training_seed,
        device="auto",
        verbose=verbose,
    )

    callbacks = CallbackList(
        [
            ArenaTelemetryCallback(),
            CheckpointCallback(
                save_freq=max(1, int(settings["checkpoint_frequency"])),
                save_path=str(checkpoint_dir),
                name_prefix=stem,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(run_dir / "evaluations"),
                eval_freq=max(1, int(settings["evaluation_frequency"])),
                n_eval_episodes=int(settings["evaluation_episodes"]),
                deterministic=True,
                render=False,
            ),
        ]
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name=stem,
            progress_bar=False,
        )
        last_model_path = run_dir / "last_model.zip"
        model.save(str(last_model_path))
    finally:
        train_env.close()
        eval_env.close()

    candidates: list[tuple[str, DQN]] = [("last", model)]
    callback_best_path = best_dir / "best_model.zip"
    if callback_best_path.exists():
        candidates.append(("eval_callback_best", DQN.load(str(callback_best_path))))

    candidate_results: list[dict[str, Any]] = []
    selected_name = ""
    selected_model = model
    selected_rows: list[dict[str, Any]] = []
    benchmark: dict[str, Any] = {}
    selected_score = float("-inf")
    for candidate_name, candidate_model in candidates:
        rows, candidate_benchmark = evaluate_model(
            candidate_model,
            control_style,
            episodes=benchmark_episodes,
            action_repeat=action_repeat,
            seed=training_seed + 10000,
        )
        write_benchmark(
            rows, candidate_benchmark, run_dir / f"benchmark_{candidate_name}"
        )
        score = (
            float(candidate_benchmark["mean_reward"])
            + 40.0 * float(candidate_benchmark["phase_progression_rate"])
            + 8.0 * float(candidate_benchmark["survival_rate"])
        )
        candidate_results.append(
            {"candidate": candidate_name, "selection_score": score, **candidate_benchmark}
        )
        if score > selected_score:
            selected_score = score
            selected_name = candidate_name
            selected_model = candidate_model
            selected_rows = rows
            benchmark = candidate_benchmark

    destination = model_path(control_style, run_name)
    selected_model.save(str(destination))
    write_benchmark(selected_rows, benchmark, run_dir / "benchmark")
    with (run_dir / "model_selection.json").open("w", encoding="utf-8") as output:
        json.dump(
            {"selected": selected_name, "candidates": candidate_results},
            output,
            indent=2,
        )
    _plot_monitor(
        monitor_file,
        run_dir / "training_curve.png",
        f"DQN learning — {control_style.title()} controls",
    )

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "DQN",
        "control_style": control_style,
        "profile": profile,
        "total_timesteps": total_timesteps,
        "seed": training_seed,
        "action_repeat": action_repeat,
        "observation_names": list(OBSERVATION_NAMES),
        "observation_size": len(OBSERVATION_NAMES),
        "network": list(settings["net_arch"]),
        "selected_checkpoint": selected_name,
        "checkpoint_candidates": candidate_results,
        "hyperparameters": {
            key: settings[key]
            for key in (
                "learning_rate",
                "buffer_size",
                "learning_starts",
                "batch_size",
                "gamma",
                "train_freq",
                "gradient_steps",
                "target_update_interval",
                "exploration_fraction",
                "exploration_initial_eps",
                "exploration_final_eps",
            )
        },
        "benchmark": benchmark,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }
    with metadata_path(control_style, run_name).open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    print(
        f"Saved {control_style} model to {model_path(control_style, run_name)}\n"
        f"Mean benchmark reward: {benchmark['mean_reward']:.2f}; "
        f"phase progression: {benchmark['phase_progression_rate']:.0%}"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stable-Baselines3 DQN agents for the Part II arena"
    )
    parser.add_argument(
        "--control-style",
        choices=("direct", "rotation", "both"),
        default="both",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--profile",
        choices=("balanced", "fast_exploration", "long_exploration"),
        default="balanced",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional artifact stem; only valid when training one style",
    )
    parser.add_argument("--benchmark-episodes", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_name and args.control_style == "both":
        raise SystemExit("--run-name requires one --control-style")
    styles = (
        ("direct", "rotation")
        if args.control_style == "both"
        else (args.control_style,)
    )
    for index, style in enumerate(styles):
        training_seed = None if args.seed is None else args.seed + index
        train_control_style(
            style,
            timesteps=args.timesteps,
            seed=training_seed,
            profile=args.profile,
            run_name=args.run_name,
            benchmark_episodes=args.benchmark_episodes,
            verbose=0 if args.quiet else 1,
        )


if __name__ == "__main__":
    main()
