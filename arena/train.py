"""Train a Stable-Baselines3 DQN agent in the Part II arena.

Control style 1 is the ``rotation`` action set:

0. no action
1. thrust forward
2. rotate left
3. rotate right
4. shoot

The arena is kept headless while learning. TensorBoard data, Monitor episode
data, checkpoints, the best evaluation model, a final model, and a compact
JSON training summary are written below ``logs/`` and ``models/``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "logs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from arena.environment import ArenaEnv
from arena.cooldown import CooldownAwareDQN
from arena.evaluate import evaluate_dqn, evaluate_random_policy


CONFIG_PATH = Path(__file__).with_name("config.json")


class PhaseCurriculumWrapper(gym.Wrapper):
    """Start training episodes across early phases to expose dense crowds.

    This wrapper is used only by the training environment. Evaluation and
    playback still begin at phase 1, so reported progression remains genuine.
    """

    def __init__(self, env: gym.Env, maximum_start_phase: int, seed: int) -> None:
        super().__init__(env)
        if maximum_start_phase < 1:
            raise ValueError("maximum_start_phase must be at least 1")
        self.maximum_start_phase = maximum_start_phase
        self._curriculum_rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._curriculum_rng = np.random.default_rng(seed)
        start_phase = int(self._curriculum_rng.integers(1, self.maximum_start_phase + 1))
        reset_options = dict(options or {})
        reset_options["start_phase"] = start_phase
        observation, info = self.env.reset(seed=seed, options=reset_options)
        info["curriculum_start_phase"] = start_phase
        return observation, info


def load_training_config() -> dict[str, Any]:
    """Return the DQN section of the checked-in arena configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)["training"]


def make_monitored_env(
    control_style: str,
    seed: int,
    monitor_file: Path | None = None,
    curriculum_max_phase: int = 1,
) -> Monitor:
    """Create a seeded, headless arena wrapped with SB3's Monitor."""

    env: gym.Env = ArenaEnv(control_style=control_style, render_mode=None)
    env.action_space.seed(seed)
    if curriculum_max_phase > 1:
        env = PhaseCurriculumWrapper(env, curriculum_max_phase, seed)
    filename = str(monitor_file) if monitor_file is not None else None
    return Monitor(
        env,
        filename=filename,
        info_keywords=("phase", "player_health"),
    )


def build_dqn(
    env: Monitor,
    config: dict[str, Any],
    tensorboard_dir: Path,
    seed: int,
    device: str,
    verbose: int = 1,
    cooldown_aware_actions: bool = False,
) -> DQN:
    """Build the config-driven neural-network Q-learning agent."""

    model_class = CooldownAwareDQN if cooldown_aware_actions else DQN
    return model_class(
        policy=str(config.get("policy", "MlpPolicy")),
        env=env,
        learning_rate=float(config["learning_rate"]),
        buffer_size=int(config["buffer_size"]),
        learning_starts=int(config["learning_starts"]),
        batch_size=int(config["batch_size"]),
        tau=float(config["tau"]),
        gamma=float(config["gamma"]),
        train_freq=int(config["train_freq"]),
        gradient_steps=int(config["gradient_steps"]),
        target_update_interval=int(config["target_update_interval"]),
        exploration_fraction=float(config["exploration_fraction"]),
        exploration_initial_eps=float(config["exploration_initial_eps"]),
        exploration_final_eps=float(config["exploration_final_eps"]),
        max_grad_norm=float(config["max_grad_norm"]),
        policy_kwargs={"net_arch": list(config["network_architecture"])},
        tensorboard_log=str(tensorboard_dir),
        seed=seed,
        device=device,
        verbose=verbose,
    )


def transfer_dqn_policy(source_path: Path, target: DQN) -> tuple[int, int]:
    """Expand a trained DQN to a larger observation vector without forgetting it.

    Every compatible layer is copied exactly. Extra columns in the first layer
    are initialized to zero, so the expanded policy initially produces the same
    Q-values as the source policy and can learn the new features gradually.
    """

    source = DQN.load(source_path, device=target.device)
    source_observations = int(np.prod(source.observation_space.shape))
    target_observations = int(np.prod(target.observation_space.shape))
    if source.action_space.n != target.action_space.n:
        raise ValueError(
            "Cannot transfer DQN policy with a different number of actions: "
            f"{source.action_space.n} != {target.action_space.n}"
        )
    if source_observations > target_observations:
        raise ValueError(
            "Source observation vector is larger than target: "
            f"{source_observations} > {target_observations}"
        )

    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    expanded_keys: list[str] = []
    with torch.no_grad():
        for key, target_tensor in target_state.items():
            source_tensor = source_state.get(key)
            if source_tensor is None:
                raise ValueError(f"Source policy is missing parameter {key}")
            if source_tensor.shape == target_tensor.shape:
                target_state[key] = source_tensor.detach().clone()
                continue
            if (
                key.endswith("q_net.0.weight")
                and source_tensor.ndim == 2
                and target_tensor.ndim == 2
                and source_tensor.shape[0] == target_tensor.shape[0]
                and source_tensor.shape[1] <= target_tensor.shape[1]
            ):
                expanded = torch.zeros_like(target_tensor)
                expanded[:, : source_tensor.shape[1]] = source_tensor
                target_state[key] = expanded
                expanded_keys.append(key)
                continue
            raise ValueError(
                f"Cannot transfer parameter {key}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
    if source_observations < target_observations and not expanded_keys:
        raise ValueError("Transfer did not find an expanded observation input layer")
    target.policy.load_state_dict(target_state)
    return source_observations, target_observations


def protect_transferred_policy(target: DQN, source_observations: int) -> None:
    """Freeze the source policy and train only newly appended input columns."""

    first_layer = target.q_net.q_net[0]
    if not isinstance(first_layer, torch.nn.Linear):
        raise TypeError("Expected the DQN Q-network to start with a linear layer")
    if not 0 < source_observations < first_layer.in_features:
        raise ValueError(
            "source_observations must identify a non-empty appended feature suffix"
        )
    for parameter in target.q_net.parameters():
        parameter.requires_grad_(False)
    first_layer.weight.requires_grad_(True)

    gradient_mask = torch.zeros_like(first_layer.weight)
    gradient_mask[:, source_observations:] = 1.0
    first_layer.weight.register_hook(lambda gradient: gradient * gradient_mask)


class ArenaMetricsCallback(BaseCallback):
    """Expose arena-specific phase and health diagnostics to TensorBoard."""

    def __init__(self, log_interval: int = 1_000) -> None:
        super().__init__(verbose=0)
        self.log_interval = log_interval
        self.maximum_phase_seen = 1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            phase = int(info.get("phase", 1))
            self.maximum_phase_seen = max(self.maximum_phase_seen, phase)

        if self.num_timesteps % self.log_interval == 0 and infos:
            self.logger.record("arena/maximum_phase_seen", self.maximum_phase_seen)
            self.logger.record_mean(
                "arena/player_health", float(infos[0].get("player_health", 0.0))
            )
        return True


def save_training_curve(monitor_dir: Path, destination: Path) -> bool:
    """Plot raw episode returns and a 20-episode moving average."""

    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    for monitor_file in sorted(monitor_dir.glob("*.monitor.csv")):
        with monitor_file.open("r", encoding="utf-8") as input_file:
            first_line = input_file.readline()
            if not first_line.startswith("#"):
                input_file.seek(0)
            for row in csv.DictReader(input_file):
                episode_lengths.append(int(row["l"]))
                episode_rewards.append(float(row["r"]))

    if not episode_rewards:
        return False

    x = np.cumsum(np.asarray(episode_lengths, dtype=int))
    rewards = np.asarray(episode_rewards, dtype=float)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x, rewards, color="#7c89ff", alpha=0.28, linewidth=1, label="Episode reward")

    window = min(20, len(rewards))
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axis.plot(
            x[window - 1 :],
            smoothed,
            color="#ff4fb3",
            linewidth=2,
            label=f"{window}-episode moving average",
        )

    axis.set_title("DQN Control Style 1 Training")
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Episode reward")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return True


def parse_args() -> argparse.Namespace:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-style",
        choices=("rotation", "direct"),
        default="rotation",
        help="Control style 1 is 'rotation' (the default).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(config["total_timesteps"]),
        help="Training timesteps (assignment range: 100000-600000).",
    )
    parser.add_argument("--seed", type=int, default=int(config["seed"]))
    parser.add_argument("--device", default="auto", help="SB3 device, e.g. auto/cpu/cuda")
    parser.add_argument("--eval-episodes", type=int, default=int(config["evaluation_episodes"]))
    parser.add_argument("--eval-freq", type=int, default=int(config["evaluation_frequency"]))
    parser.add_argument(
        "--exploration-initial-eps",
        type=float,
        default=None,
        help="Optional DQN initial epsilon override (useful for policy fine-tuning).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional DQN learning-rate override for conservative fine-tuning.",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=None,
        help="Optional fraction of training over which epsilon decays.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Artifact directory name (defaults to a timestamped name).",
    )
    parser.add_argument(
        "--skip-random-baseline",
        action="store_true",
        help="Skip the pre-training random-policy evaluation.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help=(
            "Optional trained DQN whose policy is copied into this model; newly "
            "added observation inputs are zero-initialized."
        ),
    )
    parser.add_argument(
        "--train-new-inputs-only",
        action="store_true",
        help="Freeze transferred combat weights and train only appended input columns.",
    )
    parser.add_argument(
        "--cooldown-aware-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mask SHOOT during cooldown in action selection and Bellman targets "
            "(enabled by default for rotation control)."
        ),
    )
    parser.add_argument(
        "--curriculum-max-phase",
        type=int,
        default=1,
        help=(
            "Randomly begin training episodes in phases 1 through this value. "
            "Evaluation always starts at phase 1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.eval_episodes <= 0 or args.eval_freq <= 0:
        raise ValueError("--eval-episodes and --eval-freq must be positive")
    if args.curriculum_max_phase < 1:
        raise ValueError("--curriculum-max-phase must be at least 1")

    config = load_training_config()
    cooldown_aware_actions = bool(
        args.cooldown_aware_actions and args.control_style == "rotation"
    )
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise ValueError("--learning-rate must be positive")
        config["learning_rate"] = args.learning_rate
    if args.exploration_initial_eps is not None:
        if not 0.0 <= args.exploration_initial_eps <= 1.0:
            raise ValueError("--exploration-initial-eps must be between 0 and 1")
        config["exploration_initial_eps"] = args.exploration_initial_eps
    if args.exploration_fraction is not None:
        if not 0.0 < args.exploration_fraction <= 1.0:
            raise ValueError("--exploration-fraction must be in (0, 1]")
        config["exploration_fraction"] = args.exploration_fraction
    set_random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"dqn_{args.control_style}_seed{args.seed}_{timestamp}"
    model_dir = PROJECT_ROOT / "models" / f"dqn_{args.control_style}" / run_name
    log_dir = PROJECT_ROOT / "logs" / f"dqn_{args.control_style}" / run_name
    monitor_dir = log_dir / "monitor"
    evaluation_dir = log_dir / "evaluation"
    tensorboard_dir = log_dir / "tensorboard"
    checkpoint_dir = model_dir / "checkpoints"
    for directory in (
        model_dir,
        monitor_dir,
        evaluation_dir,
        tensorboard_dir,
        checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Training DQN control style: {args.control_style}")
    print(f"Timesteps: {args.timesteps:,} | Seed: {args.seed} | Device: {args.device}")
    print(f"Models: {model_dir}")
    print(f"Logs:   {log_dir}")

    random_metrics: dict[str, Any] | None = None
    if not args.skip_random_baseline:
        print("Evaluating random-policy baseline...")
        random_metrics = evaluate_random_policy(
            control_style=args.control_style,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
        )
        print(
            "Random baseline: "
            f"mean reward={random_metrics['mean_reward']:.3f}, "
            f"mean max phase={random_metrics['mean_max_phase']:.2f}"
        )

    train_env = make_monitored_env(
        args.control_style,
        args.seed,
        monitor_dir / "train",
        curriculum_max_phase=args.curriculum_max_phase,
    )
    eval_env = make_monitored_env(args.control_style, args.seed + 1)
    model = build_dqn(
        train_env,
        config,
        tensorboard_dir,
        args.seed,
        args.device,
        cooldown_aware_actions=cooldown_aware_actions,
    )
    transferred_observations: tuple[int, int] | None = None
    if args.initialize_from is not None:
        source_path = args.initialize_from.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Initialization model not found: {source_path}")
        transferred_observations = transfer_dqn_policy(source_path, model)
        print(
            "Transferred policy: "
            f"{source_path} "
            f"({transferred_observations[0]} -> {transferred_observations[1]} observations)"
        )
        if args.train_new_inputs_only:
            protect_transferred_policy(model, transferred_observations[0])
            print(
                "Protected source policy; training only observation columns "
                f"{transferred_observations[0]}:{transferred_observations[1]}"
            )
    elif args.train_new_inputs_only:
        raise ValueError("--train-new-inputs-only requires --initialize-from")

    checkpoint_callback = CheckpointCallback(
        save_freq=int(config["checkpoint_frequency"]),
        save_path=str(checkpoint_dir),
        name_prefix=f"dqn_{args.control_style}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    evaluation_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(evaluation_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        warn=True,
    )
    metrics_callback = ArenaMetricsCallback()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, evaluation_callback, metrics_callback],
            tb_log_name="DQN",
            progress_bar=False,
        )
        final_model = model_dir / "final_model"
        model.save(final_model)

        print("Evaluating trained deterministic policy...")
        trained_metrics = evaluate_dqn(
            model,
            control_style=args.control_style,
            episodes=args.eval_episodes,
            seed=args.seed + 20_000,
        )
        curve_path = log_dir / "training_curve.png"
        curve_created = save_training_curve(monitor_dir, curve_path)

        summary = {
            "algorithm": "DQN",
            "policy": str(config.get("policy", "MlpPolicy")),
            "control_style": args.control_style,
            "evaluation_playback_speed": float(train_env.unwrapped.playback_speed),
            "actions": ["NOOP", "THRUST", "ROTATE_LEFT", "ROTATE_RIGHT", "SHOOT"]
            if args.control_style == "rotation"
            else ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "SHOOT"],
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "trained_timesteps": int(model.num_timesteps),
            "initialization_model": (
                str(args.initialize_from.resolve())
                if args.initialize_from is not None
                else None
            ),
            "transferred_observation_dimensions": (
                list(transferred_observations)
                if transferred_observations is not None
                else None
            ),
            "trained_new_observation_inputs_only": bool(args.train_new_inputs_only),
            "cooldown_aware_action_selection": cooldown_aware_actions,
            "cooldown_shoot_replacements_during_training": int(
                getattr(model, "cooldown_replacements", 0)
            ),
            "training_curriculum_max_start_phase": args.curriculum_max_phase,
            "hyperparameters": {
                key: config[key]
                for key in (
                    "learning_rate",
                    "buffer_size",
                    "learning_starts",
                    "batch_size",
                    "tau",
                    "gamma",
                    "train_freq",
                    "gradient_steps",
                    "target_update_interval",
                    "exploration_fraction",
                    "exploration_initial_eps",
                    "exploration_final_eps",
                    "network_architecture",
                )
            },
            "random_baseline": random_metrics,
            "trained_policy": trained_metrics,
            "maximum_phase_seen_during_training": metrics_callback.maximum_phase_seen,
            "artifacts": {
                "final_model": str(final_model.with_suffix(".zip")),
                "best_model": str(model_dir / "best_model.zip"),
                "tensorboard": str(tensorboard_dir),
                "monitor": str(monitor_dir / "train.monitor.csv"),
                "training_curve": str(curve_path) if curve_created else None,
            },
        }
        summary_path = log_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(
            "Training complete: "
            f"mean reward={trained_metrics['mean_reward']:.3f}, "
            f"mean max phase={trained_metrics['mean_max_phase']:.2f}, "
            f"best max phase={trained_metrics['maximum_phase_reached']}"
        )
        print(f"Final model: {final_model.with_suffix('.zip')}")
        print(f"Summary:     {summary_path}")
        if curve_created:
            print(f"Curve:       {curve_path}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
