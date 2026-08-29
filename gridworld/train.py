"""Config-driven training and evidence capture for Gridworld agents.

Examples::

    python -m gridworld.train --level 0 --agent qlearning
    python -m gridworld.train --level 1 --agent sarsa --seed 202
    python -m gridworld.train --level 6 --agent qlearning --intrinsic

The public ``train`` and ``make_agent`` functions retain their original calling
style.  Rich metrics are opt-in through ``return_metrics=True``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.environment import GridWorldEnv


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "gridworld")
LOGS_DIR = os.path.join(ROOT_DIR, "logs", "gridworld")


def load_config(path: str | os.PathLike[str] = CONFIG_PATH) -> dict[str, Any]:
    """Load the Gridworld JSON configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_training_profile(
    config: Mapping[str, Any],
    level_id: int,
    agent_type: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve defaults -> agent -> level -> level+agent -> CLI overrides."""
    if agent_type not in {"qlearning", "sarsa"}:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    profile = dict(config.get("training", {}))
    profile.update(config.get("agent_profiles", {}).get(agent_type, {}))
    profile.update(config.get("level_profiles", {}).get(str(level_id), {}))
    profile.update(
        config.get("level_agent_profiles", {})
        .get(str(level_id), {})
        .get(agent_type, {})
    )
    if overrides:
        profile.update({key: value for key, value in overrides.items() if value is not None})

    profile.setdefault(
        "intrinsic_strength",
        config.get("intrinsic", {}).get("reward_strength", 0.0001),
    )
    required = ("episodes", "alpha", "gamma", "epsilon_start", "epsilon_end")
    missing = [key for key in required if key not in profile]
    if missing:
        raise KeyError(f"Training profile is missing required keys: {missing}")

    profile["episodes"] = int(profile["episodes"])
    profile["max_steps"] = int(profile.get("max_steps", 500))
    profile["seed"] = int(profile.get("seed", 1729))
    profile["rolling_window"] = int(profile.get("rolling_window", 50))
    profile["bootstrap_on_truncation"] = bool(
        profile.get("bootstrap_on_truncation", True)
    )
    if profile["episodes"] <= 0 or profile["max_steps"] <= 0:
        raise ValueError("episodes and max_steps must be positive")
    return profile


def make_agent(
    agent_type: str,
    config: Mapping[str, Any],
    num_episodes: int | None = None,
    use_intrinsic: bool = False,
    *,
    level_id: int = 0,
    seed: int | None = None,
    profile: Mapping[str, Any] | None = None,
):
    """Instantiate an agent from a resolved profile.

    ``num_episodes`` remains the third positional argument for compatibility
    with the original scripts.
    """
    resolved = dict(profile or resolve_training_profile(config, level_id, agent_type))
    if num_episodes is not None:
        resolved["episodes"] = int(num_episodes)
    if seed is not None:
        resolved["seed"] = int(seed)

    agent_class = QLearningAgent if agent_type == "qlearning" else SARSAAgent
    if agent_type not in {"qlearning", "sarsa"}:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    return agent_class(
        alpha=float(resolved["alpha"]),
        gamma=float(resolved["gamma"]),
        epsilon_start=float(resolved["epsilon_start"]),
        epsilon_end=float(resolved["epsilon_end"]),
        num_episodes=int(resolved["episodes"]),
        use_intrinsic=use_intrinsic,
        intrinsic_strength=float(resolved.get("intrinsic_strength", 0.0001)),
        seed=int(resolved["seed"]) if resolved.get("seed") is not None else None,
    )


def make_environment(
    level_id: int,
    config: Mapping[str, Any],
    seed: int | None = None,
) -> GridWorldEnv:
    """Create either the seeded environment API or the original compatible API."""
    monster_chance = float(config.get("monster", {}).get("move_chance", 0.4))
    try:
        return GridWorldEnv(
            level_id,
            monster_move_chance=monster_chance,
            seed=seed,
        )
    except TypeError as error:
        if "seed" not in str(error):
            raise
        return GridWorldEnv(level_id, monster_move_chance=monster_chance)


def reset_environment(env: GridWorldEnv, seed: int | None = None):
    """Reset across both the new seeded API and the original reset() API."""
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError as error:
            if "seed" not in str(error):
                raise
            result = env.reset()
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0]
    return result


def step_environment(env: GridWorldEnv, action: int):
    """Normalize assignment-style and Gymnasium-style step results."""
    result = env.step(action)
    if len(result) == 4:
        next_state, reward, done, info = result
        return next_state, float(reward), bool(done), dict(info)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        normalized_info = dict(info)
        if truncated:
            normalized_info["environment_truncated"] = True
        return next_state, float(reward), bool(terminated or truncated), normalized_info
    raise ValueError("GridWorldEnv.step must return four or five values")


def train(
    env: GridWorldEnv,
    agent,
    num_episodes: int,
    renderer=None,
    max_steps: int = 500,
    *,
    base_seed: int | None = None,
    bootstrap_on_truncation: bool = True,
    return_metrics: bool = False,
    verbose: bool = True,
):
    """Train one agent with explicit time-limit and reward-component logging.

    Artificial time limits are recorded as truncations, never victories or
    deaths.  By default they bootstrap because the underlying environment did
    not reach a terminal state; this behavior is config-driven.
    """
    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError("num_episodes and max_steps must be positive")

    metrics: list[dict[str, Any]] = []
    log_interval = max(1, num_episodes // 20)

    for episode_index in range(num_episodes):
        episode_seed = None if base_seed is None else int(base_seed) + episode_index
        state = reset_environment(env, seed=episode_seed)
        if hasattr(agent, "begin_episode"):
            agent.begin_episode(state)
        else:
            agent.reset_episode_visits()
            agent.record_visit(state)

        epsilon_used = float(agent.epsilon)
        action = agent.choose_action(state)
        extrinsic_total = 0.0
        intrinsic_total = 0.0
        learning_total = 0.0
        step_count = 0
        last_info: dict[str, Any] = {}
        env_done = False
        truncated = False

        for step_count in range(1, max_steps + 1):
            next_state, env_reward, env_done, info = step_environment(env, action)
            last_info = info
            environment_truncated = bool(info.get("environment_truncated", False))
            time_limit_reached = step_count == max_steps and not env_done
            truncated = environment_truncated or time_limit_reached
            terminal_update = env_done or (
                truncated and not bootstrap_on_truncation
            )

            if isinstance(agent, SARSAAgent):
                next_action = (
                    agent.choose_action(next_state) if not terminal_update else None
                )
                update_stats = agent.update(
                    state,
                    action,
                    env_reward,
                    next_state,
                    terminal_update,
                    next_action=next_action,
                )
            else:
                update_stats = agent.update(
                    state,
                    action,
                    env_reward,
                    next_state,
                    terminal_update,
                )
                next_action = (
                    agent.choose_action(next_state) if not terminal_update else None
                )

            extrinsic_total += float(update_stats["extrinsic_reward"])
            intrinsic_total += float(update_stats["intrinsic_reward"])
            learning_total += float(update_stats["learning_reward"])

            if renderer is not None:
                renderer.render(
                    episode=episode_index + 1,
                    step=step_count,
                    total_reward=extrinsic_total,
                )
                if hasattr(renderer, "quit_requested") and renderer.quit_requested():
                    raise KeyboardInterrupt("Rendered training stopped by user")

            state = next_state
            action = next_action
            if env_done or truncated:
                break

        victory = bool(last_info.get("victory", False))
        death_cause = last_info.get("death")
        if victory:
            status = "victory"
        elif death_cause:
            status = "death"
        elif truncated:
            status = "truncated"
        elif env_done:
            status = "terminated"
        else:
            status = "unknown"

        visited_positions = sorted(
            {
                (int(visited_state[0]), int(visited_state[1]))
                for visited_state in getattr(agent, "visit_counts", {})
                if isinstance(visited_state, tuple) and len(visited_state) >= 2
            }
        )
        metric = {
            "episode": episode_index + 1,
            "seed": "" if episode_seed is None else episode_seed,
            "epsilon": epsilon_used,
            # Legacy column retained for consumers of the original CSV files.
            "episode_reward": extrinsic_total,
            "extrinsic_reward": extrinsic_total,
            "intrinsic_reward": intrinsic_total,
            "learning_reward": learning_total,
            "steps": step_count,
            "status": status,
            "victory": int(victory),
            "death": int(bool(death_cause)),
            "death_cause": death_cause or "",
            "truncated": int(truncated),
            "unique_states": len(getattr(agent, "visit_counts", {})),
            "unique_positions": len(visited_positions),
            "visited_positions": "|".join(
                f"{row}:{column}" for row, column in visited_positions
            ),
            "q_table_states": len(agent.q_table),
        }
        metrics.append(metric)
        agent.decay_epsilon()

        if verbose and (
            episode_index == 0
            or (episode_index + 1) % log_interval == 0
            or episode_index + 1 == num_episodes
        ):
            recent = metrics[-min(50, len(metrics)) :]
            avg_reward = float(np.mean([row["extrinsic_reward"] for row in recent]))
            avg_success = float(np.mean([row["victory"] for row in recent]))
            print(
                f"  Episode {episode_index + 1:>5}/{num_episodes} | "
                f"EnvR {extrinsic_total:>5.2f} | IntR {intrinsic_total:>5.2f} | "
                f"Avg50 {avg_reward:>4.2f} | Win50 {avg_success:>5.1%} | "
                f"Steps {step_count:>3} | Eps {epsilon_used:.3f} | {status}"
            )

    if return_metrics:
        return metrics
    return (
        [float(row["extrinsic_reward"]) for row in metrics],
        [int(row["steps"]) for row in metrics],
    )


def rolling_mean(values: Iterable[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates for a trailing moving average."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return np.asarray([]), np.asarray([])
    effective_window = max(1, min(int(window), array.size))
    mean = np.convolve(
        array,
        np.ones(effective_window, dtype=float) / effective_window,
        mode="valid",
    )
    x_values = np.arange(effective_window, array.size + 1)
    return x_values, mean


def save_training_plot(
    rewards,
    path: str | os.PathLike[str],
    title: str = "Training Curve",
    window: int = 50,
    *,
    metrics: list[Mapping[str, Any]] | None = None,
) -> None:
    """Save the historical reward plot or a four-panel evidence dashboard."""
    if metrics is None and rewards and isinstance(rewards[0], Mapping):
        metrics = list(rewards)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        reward_values = np.asarray(rewards, dtype=float)
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.plot(reward_values, alpha=0.25, color="#1976D2", label="Episode reward")
        x_values, mean = rolling_mean(reward_values, window)
        axis.plot(x_values, mean, color="#0D47A1", linewidth=2, label="Moving average")
        axis.set_xlabel("Episode")
        axis.set_ylabel("Environment reward")
        axis.set_title(title)
        axis.legend(loc="best")
        axis.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        episode = [int(row["episode"]) for row in metrics]

        for key, color, label in (
            ("extrinsic_reward", "#1565C0", "Environment reward"),
            ("learning_reward", "#7B1FA2", "Learning reward"),
        ):
            x_values, mean = rolling_mean((row[key] for row in metrics), window)
            axes[0, 0].plot(x_values, mean, color=color, linewidth=2, label=label)
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend(loc="best")

        x_values, success = rolling_mean((row["victory"] for row in metrics), window)
        axes[0, 1].plot(x_values, success, color="#2E7D32", linewidth=2)
        axes[0, 1].set_ylabel("Victory rate")
        axes[0, 1].set_ylim(-0.03, 1.03)

        x_values, steps = rolling_mean((row["steps"] for row in metrics), window)
        axes[1, 0].plot(x_values, steps, color="#EF6C00", linewidth=2)
        axes[1, 0].set_ylabel("Steps per episode")
        axes[1, 0].set_xlabel("Episode")

        x_values, coverage = rolling_mean(
            (row.get("unique_positions", row["unique_states"]) for row in metrics), window
        )
        axes[1, 1].plot(
            x_values, coverage, color="#00838F", linewidth=2, label="Unique positions"
        )
        intrinsic_values = [float(row["intrinsic_reward"]) for row in metrics]
        if any(value != 0.0 for value in intrinsic_values):
            intrinsic_axis = axes[1, 1].twinx()
            x_intrinsic, intrinsic = rolling_mean(intrinsic_values, window)
            intrinsic_axis.plot(
                x_intrinsic,
                intrinsic,
                color="#AD1457",
                linewidth=1.5,
                alpha=0.8,
                label="Intrinsic return",
            )
            intrinsic_axis.set_ylabel("Intrinsic return")
        axes[1, 1].set_ylabel("Unique grid positions")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].legend(loc="best")

        for axis in axes.flat:
            axis.grid(True, alpha=0.25)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        # Keep the full episode range visible for very short smoke runs.
        axes[1, 0].set_xlim(1, max(episode) if episode else 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metrics_csv(metrics: list[Mapping[str, Any]], path: str | os.PathLike[str]) -> None:
    """Write one complete, auditable row per episode."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        raise ValueError("Cannot write an empty metrics collection")
    fields = list(metrics[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def summarize_metrics(
    metrics: list[Mapping[str, Any]],
    tail_window: int = 100,
) -> dict[str, Any]:
    """Produce report-ready aggregate metrics without hiding failures."""
    if not metrics:
        raise ValueError("Cannot summarize empty metrics")
    tail = metrics[-min(tail_window, len(metrics)) :]
    successful_steps = [int(row["steps"]) for row in tail if int(row["victory"])]
    first_success = next(
        (int(row["episode"]) for row in metrics if int(row["victory"])),
        None,
    )
    return {
        "episodes": len(metrics),
        "first_success_episode": first_success,
        "overall": {
            "mean_extrinsic_reward": float(
                np.mean([float(row["extrinsic_reward"]) for row in metrics])
            ),
            "mean_intrinsic_reward": float(
                np.mean([float(row["intrinsic_reward"]) for row in metrics])
            ),
            "victory_rate": float(np.mean([int(row["victory"]) for row in metrics])),
            "death_rate": float(np.mean([int(row["death"]) for row in metrics])),
            "truncation_rate": float(
                np.mean([int(row["truncated"]) for row in metrics])
            ),
        },
        "tail_window": len(tail),
        "tail": {
            "mean_extrinsic_reward": float(
                np.mean([float(row["extrinsic_reward"]) for row in tail])
            ),
            "mean_intrinsic_reward": float(
                np.mean([float(row["intrinsic_reward"]) for row in tail])
            ),
            "victory_rate": float(np.mean([int(row["victory"]) for row in tail])),
            "death_rate": float(np.mean([int(row["death"]) for row in tail])),
            "truncation_rate": float(np.mean([int(row["truncated"]) for row in tail])),
            "mean_steps_on_victory": (
                float(np.mean(successful_steps)) if successful_steps else None
            ),
        },
    }


def save_json(data: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _argument_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "intrinsic_strength": args.intrinsic_strength,
        "bootstrap_on_truncation": args.bootstrap_on_truncation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Gridworld tabular RL agent")
    parser.add_argument("--level", type=int, default=0, help="Level ID (0-6)")
    parser.add_argument(
        "--agent",
        choices=["qlearning", "sarsa"],
        default="qlearning",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--intrinsic-strength", type=float, default=None)
    parser.add_argument(
        "--intrinsic",
        action="store_true",
        help="Use the Task 5 destination-state exploration bonus",
    )
    parser.add_argument("--render", action="store_true", help="Render training (slow)")
    parser.add_argument("--quiet", action="store_true")
    truncation_group = parser.add_mutually_exclusive_group()
    truncation_group.add_argument(
        "--bootstrap-on-truncation",
        dest="bootstrap_on_truncation",
        action="store_true",
    )
    truncation_group.add_argument(
        "--no-bootstrap-on-truncation",
        dest="bootstrap_on_truncation",
        action="store_false",
    )
    parser.set_defaults(bootstrap_on_truncation=None)
    args = parser.parse_args()

    config = load_config()
    profile = resolve_training_profile(
        config,
        args.level,
        args.agent,
        overrides=_argument_overrides(args),
    )
    seed = int(profile["seed"])
    agent = make_agent(
        args.agent,
        config,
        use_intrinsic=args.intrinsic,
        level_id=args.level,
        seed=seed,
        profile=profile,
    )
    env = make_environment(args.level, config, seed=seed)

    print(f"=== Training {args.agent.upper()} on Level {args.level} ===")
    print(
        f"    Episodes {profile['episodes']} | alpha {profile['alpha']} | "
        f"gamma {profile['gamma']} | epsilon {profile['epsilon_start']} -> "
        f"{profile['epsilon_end']} | seed {seed}"
    )
    if args.intrinsic:
        print(
            "    Intrinsic: beta / sqrt(n(s) + 1) for reached state s, "
            f"beta={profile['intrinsic_strength']}"
        )

    renderer = None
    if args.render:
        from gridworld.renderer import GridWorldRenderer

        renderer = GridWorldRenderer(
            env,
            cell_size=int(config["rendering"]["cell_size"]),
            fps=int(config["rendering"]["fps"]),
            title=f"Training Level {args.level} ({args.agent})",
        )

    try:
        metrics = train(
            env,
            agent,
            int(profile["episodes"]),
            renderer=renderer,
            max_steps=int(profile["max_steps"]),
            base_seed=seed,
            bootstrap_on_truncation=bool(profile["bootstrap_on_truncation"]),
            return_metrics=True,
            verbose=not args.quiet,
        )
    finally:
        if renderer is not None:
            renderer.close()

    suffix = "_intrinsic" if args.intrinsic else ""
    stem = f"level{args.level}_{args.agent}{suffix}"
    model_path = Path(MODELS_DIR) / f"{stem}.pkl"
    csv_path = Path(LOGS_DIR) / f"{stem}_rewards.csv"
    plot_path = Path(LOGS_DIR) / f"{stem}.png"
    summary_path = Path(LOGS_DIR) / f"{stem}_summary.json"

    summary = summarize_metrics(metrics)
    metadata = {
        "level_id": args.level,
        "algorithm": args.agent,
        "seed": seed,
        "intrinsic_enabled": args.intrinsic,
        "profile": profile,
        "summary": summary,
        "state_schema": getattr(env, "state_schema", None),
    }
    agent.save(model_path, metadata=metadata)
    save_metrics_csv(metrics, csv_path)
    save_training_plot(
        metrics,
        plot_path,
        title=(
            f"Level {args.level} - {args.agent.upper()}"
            + (" with intrinsic reward" if args.intrinsic else "")
        ),
        window=int(profile["rolling_window"]),
    )
    save_json(metadata, summary_path)

    print(f"  [+] Model:  {model_path}")
    print(f"  [+] Metrics: {csv_path}")
    print(f"  [+] Plot:    {plot_path}")
    print(f"  [+] Summary: {summary_path}")
    print("=== Training complete ===")


if __name__ == "__main__":
    main()
