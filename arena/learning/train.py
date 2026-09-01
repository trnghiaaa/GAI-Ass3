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

from arena.evaluation.benchmark import evaluate_model, write_benchmark
from arena.core.environment import ArenaEnv, ENVIRONMENT_SCHEMA_VERSION, OBSERVATION_NAMES
from arena.settings import (
    ARENA_LOG_DIR,
    ARENA_TRAINING_DIR,
    TENSORBOARD_DIR,
    ensure_artifact_directories,
    metadata_path,
    model_path,
    training_settings,
)
from arena.learning.wrappers import ActionRepeatWrapper, BossCurriculumWrapper
from arena.learning.cooldown import CooldownAwareDQN, load_dqn


MONITOR_INFO_KEYS = (
    "phase",
    "player_health",
    "episode_enemies_destroyed",
    "episode_spawners_destroyed",
    "episode_phases_advanced",
    "episode_damage_taken",
    "episode_xp_earned",
    "episode_max_level",
    "active_minibosses",
    "active_boss_hazards",
    "active_boss_defenders",
    "active_enemy_missiles",
    "drone_level",
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
        for name in (
            "enemies_destroyed",
            "spawners_destroyed",
            "minibosses_destroyed",
            "boss_skills_dodged",
            "boss_skill_hits",
            "boss_defenders_destroyed",
            "boss_immune_hits",
            "missiles_evaded",
            "missile_hits",
            "hazard_exposure",
            "damage_taken",
        ):
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
    boss_curriculum: float = 0.0,
    curriculum_phases: tuple[int, ...] = (3,),
    curriculum_seed: int = 0,
) -> Monitor:
    base_env: gymnasium.Env = ArenaEnv(control_style=control_style)
    if boss_curriculum > 0.0:
        base_env = BossCurriculumWrapper(
            base_env,
            probability=boss_curriculum,
            phases=curriculum_phases,
            seed=curriculum_seed,
        )
    env = ActionRepeatWrapper(base_env, repeat=action_repeat)
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


def transfer_prefix_policy(source: DQN, destination: DQN) -> None:
    """Preserve every old Q value, initially ignoring appended observations.

    Replay/optimizer state is intentionally fresh for the new environment.
    This is transfer learning, not a claim of additional training from scratch.
    """
    old_state = source.policy.state_dict()
    new_state = destination.policy.state_dict()
    old_size = int(source.observation_space.shape[0])
    new_size = int(destination.observation_space.shape[0])
    for name, value in new_state.items():
        old = old_state[name]
        if old.shape == value.shape:
            value.copy_(old)
        elif name in ('q_net.q_net.0.weight', 'q_net_target.q_net.0.weight') and new_size > old_size:
            value.zero_()
            value[:, :old_size].copy_(old)
        else:
            raise ValueError(f'Cannot transfer incompatible architecture: {name}')
    destination.policy.load_state_dict(new_state)


def restrict_to_appended_inputs(model: DQN, original_size: int) -> None:
    """Train only the new feature connections to preserve the original network."""
    if original_size >= model.observation_space.shape[0]:
        raise ValueError('Appended-input adaptation requires genuinely new features')
    for name, parameter in model.q_net.named_parameters():
        parameter.requires_grad_(name == 'q_net.0.weight')
        if name == 'q_net.0.weight':
            mask = torch.zeros_like(parameter)
            mask[:, original_size:] = 1.0
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)


def train_control_style(
    control_style: str,
    *,
    timesteps: int | None = None,
    seed: int | None = None,
    profile: str = "balanced",
    run_name: str | None = None,
    benchmark_episodes: int = 12,
    verbose: int = 1,
    init_model: Path | None = None,
    cooldown_mask: bool = False,
    new_inputs_only: bool = False,
    boss_curriculum: float = 0.0,
    curriculum_phases: tuple[int, ...] = (3,),
    action_repeat_override: int | None = None,
) -> dict[str, Any]:
    """Train, save, and independently benchmark one control-style policy."""

    ensure_artifact_directories()
    if cooldown_mask and control_style != 'rotation':
        raise ValueError('Cooldown masking currently requires rotation controls')
    settings = training_settings(control_style, profile)
    total_timesteps = int(timesteps or settings["total_timesteps"])
    if total_timesteps < 1:
        raise ValueError("timesteps must be positive")
    if benchmark_episodes < 1:
        raise ValueError("benchmark_episodes must be positive")
    if not 0.0 <= boss_curriculum <= 1.0:
        raise ValueError("boss_curriculum must be between 0 and 1")
    if not curriculum_phases:
        raise ValueError("curriculum_phases cannot be empty")
    training_seed = int(settings["seed"] if seed is None else seed)
    action_repeat = int(
        settings["action_repeat"]
        if action_repeat_override is None
        else action_repeat_override
    )
    if action_repeat < 1:
        raise ValueError("action repeat must be positive")
    stem = run_name or f"dqn_{control_style}"
    run_dir = ARENA_TRAINING_DIR / stem
    if model_path(control_style, run_name).exists() or run_dir.exists():
        raise FileExistsError(f'Preserve existing artifacts: choose a new --run-name ({stem})')
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = run_dir / "training.monitor.csv"

    train_env = _make_env(
        control_style,
        action_repeat,
        monitor_file=monitor_file,
        boss_curriculum=boss_curriculum,
        curriculum_phases=curriculum_phases,
        curriculum_seed=training_seed,
    )
    eval_env = _make_env(control_style, action_repeat)
    learning_starts = min(
        int(settings["learning_starts"]), max(0, total_timesteps // 5)
    )
    model_class = CooldownAwareDQN if cooldown_mask else DQN
    model = model_class(
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
    model.cooldown_mask_enabled = cooldown_mask
    if init_model is not None:
        source = DQN.load(str(init_model), device='cpu')
        with init_model.with_suffix('.metadata.json').open(encoding='utf-8') as handle:
            source_meta = json.load(handle)
        source_names = source_meta['observation_names']
        if source_meta['control_style'] != control_style or source_names != list(OBSERVATION_NAMES[:len(source_names)]):
            raise ValueError('Transfer requires the same control style and an exact observation prefix')
        transfer_prefix_policy(source, model)
        if new_inputs_only:
            restrict_to_appended_inputs(model, int(source.observation_space.shape[0]))
    elif new_inputs_only:
        raise ValueError('--new-inputs-only requires --init-model')

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
        candidates.append(("eval_callback_best", load_dqn(str(callback_best_path))))

    candidate_results: list[dict[str, Any]] = []
    selected_name = ""
    selected_model = model
    selected_rows: list[dict[str, Any]] = []
    selected_boss_rows: list[dict[str, Any]] = []
    benchmark: dict[str, Any] = {}
    boss_focus_benchmark: dict[str, Any] = {}
    selected_score = float("-inf")
    for candidate_name, candidate_model in candidates:
        rows, candidate_benchmark = evaluate_model(
            candidate_model,
            control_style,
            episodes=benchmark_episodes,
            action_repeat=action_repeat,
            seed=training_seed + 10000,
        )
        boss_rows, boss_benchmark = evaluate_model(
            candidate_model,
            control_style,
            episodes=max(6, benchmark_episodes // 2),
            action_repeat=action_repeat,
            seed=training_seed + 20000,
            reset_options={"start_phase": 3},
        )
        write_benchmark(
            rows, candidate_benchmark, run_dir / f"benchmark_{candidate_name}"
        )
        write_benchmark(
            boss_rows,
            boss_benchmark,
            run_dir / f"boss_benchmark_{candidate_name}",
        )
        score = (
            float(candidate_benchmark["mean_reward"])
            + 40.0 * float(candidate_benchmark["phase_progression_rate"])
            + 8.0 * float(candidate_benchmark["survival_rate"])
            + 6.0 * float(candidate_benchmark["mean_boss_skills_dodged"])
            - 8.0 * float(candidate_benchmark["mean_boss_skill_hits"])
            + 10.0 * float(candidate_benchmark["mean_bosses_destroyed"])
            + 0.02 * float(candidate_benchmark["mean_simulation_steps"])
            - 0.35 * float(candidate_benchmark["damage_per_1000_frames"])
            - 3.0 * float(candidate_benchmark["contacts_per_1000_frames"])
            - 20.0 * float(candidate_benchmark["mean_crowd_fraction"])
            + 0.08 * float(candidate_benchmark["mean_enemy_clearance"])
            + 12.0 * float(boss_benchmark["mean_boss_skills_dodged"])
            - 18.0 * float(boss_benchmark["mean_boss_skill_hits"])
            + 8.0 * float(boss_benchmark["mean_boss_defenders_destroyed"])
            + 2.0 * float(boss_benchmark["mean_missiles_evaded"])
            - 10.0 * float(boss_benchmark["mean_missile_hits"])
            - 0.5 * float(boss_benchmark["mean_boss_immune_hits"])
            + 12.0 * float(boss_benchmark["mean_bosses_destroyed"])
        )
        candidate_results.append(
            {
                "candidate": candidate_name,
                "selection_score": score,
                **candidate_benchmark,
                "boss_focus": boss_benchmark,
            }
        )
        if score > selected_score:
            selected_score = score
            selected_name = candidate_name
            selected_model = candidate_model
            selected_rows = rows
            selected_boss_rows = boss_rows
            benchmark = candidate_benchmark
            boss_focus_benchmark = boss_benchmark

    destination = model_path(control_style, run_name)
    selected_model.save(str(destination))
    write_benchmark(selected_rows, benchmark, run_dir / "benchmark")
    write_benchmark(
        selected_boss_rows,
        boss_focus_benchmark,
        run_dir / "boss_benchmark",
    )
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
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "DQN",
        "initialization_model": str(init_model) if init_model else None,
        "cooldown_mask": cooldown_mask,
        "new_inputs_only": new_inputs_only,
        "boss_curriculum_probability": boss_curriculum,
        "boss_curriculum_phases": list(curriculum_phases),
        "run_name": stem,
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
        "boss_focus_benchmark": boss_focus_benchmark,
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
        choices=(
            "balanced",
            "fast_exploration",
            "long_exploration",
            "safety_aware",
            "safety_consolidation",
            "safety_exploration",
            "threat_aware",
            "boss_dodge",
            "boss_intermission",
        ),
        default="balanced",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional artifact stem; only valid when training one style",
    )
    parser.add_argument("--benchmark-episodes", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--init-model', type=Path, default=None)
    parser.add_argument('--cooldown-mask', action='store_true')
    parser.add_argument('--new-inputs-only', action='store_true')
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=None,
        help="Optional held-action cadence for a documented control ablation",
    )
    parser.add_argument(
        "--boss-curriculum",
        type=float,
        default=0.0,
        help="Fraction of training episodes that start at a boss phase",
    )
    parser.add_argument(
        "--curriculum-phases",
        default="3",
        help="Comma-separated boss phases sampled by the training curriculum",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(1)
    if args.init_model and args.control_style == 'both':
        raise SystemExit('--init-model requires a single control style')
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
            init_model=args.init_model,
            cooldown_mask=args.cooldown_mask,
            new_inputs_only=args.new_inputs_only,
            boss_curriculum=args.boss_curriculum,
            curriculum_phases=tuple(
                int(value.strip())
                for value in args.curriculum_phases.split(",")
                if value.strip()
            ),
            action_repeat_override=args.action_repeat,
        )


if __name__ == "__main__":
    main()
