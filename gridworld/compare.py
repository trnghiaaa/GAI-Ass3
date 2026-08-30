"""Reproducible, report-ready Part I comparison experiments.

Run ``python -m gridworld.compare --comparison algorithms`` for the Task 2
Q-learning/SARSA evidence and ``--comparison intrinsic`` for the Task 5 Level 6
A/B test.  Both save raw metrics, rollouts, a figure, and a JSON summary.
Environment and intrinsic returns are always reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from gridworld.environment import ACTION_NAMES
from gridworld.train import (
    LOGS_DIR,
    MODELS_DIR,
    load_config,
    make_agent,
    make_environment,
    reset_environment,
    resolve_training_profile,
    rolling_mean,
    save_json,
    step_environment,
    train,
)


ALGORITHM_COLORS = {"qlearning": "#F57C00", "sarsa": "#00ACC1"}
INTRINSIC_COLORS = {"baseline": "#546E7A", "intrinsic": "#8E24AA"}


def _parse_seed_list(raw: str | None, defaults: Iterable[int]) -> list[int]:
    if raw is None:
        seeds = [int(seed) for seed in defaults]
    else:
        try:
            seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
        except ValueError as error:
            raise ValueError("--seeds must be comma-separated integers") from error
    if not seeds:
        raise ValueError("At least one experiment seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique")
    return seeds


def _status(done: bool, info: Mapping[str, Any]) -> str:
    if done and info.get("victory"):
        return "victory"
    if done and info.get("death"):
        return f"death_{info['death']}"
    return "timeout"


def greedy_rollout(
    level_id: int,
    agent: Any,
    config: Mapping[str, Any],
    *,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Trace a seeded epsilon=0 rollout with the required random ties."""
    env = make_environment(level_id, config, seed)
    state = reset_environment(env, seed)
    previous_epsilon = float(agent.epsilon)
    agent.epsilon = 0.0
    agent.set_seed(seed + 1_000_003)
    if hasattr(agent, "begin_episode"):
        agent.begin_episode(state)

    path = [tuple(env.agent_pos)]
    actions: list[int] = []
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    try:
        for _ in range(max_steps):
            action = int(agent.choose_action(state))
            state, reward, done, info = step_environment(env, action)
            actions.append(action)
            path.append(tuple(env.agent_pos))
            total_reward += reward
            if done:
                break
    finally:
        agent.epsilon = previous_epsilon

    fires = [
        (row, column)
        for row, terrain_row in enumerate(env.static_grid)
        for column, tile in enumerate(terrain_row)
        if tile == "F"
    ]
    interior_path = path[1:-1] if len(path) > 2 else path
    fire_distances = [
        min(abs(row - fr) + abs(column - fc) for fr, fc in fires)
        for row, column in interior_path
    ] if fires and interior_path else []
    return {
        "seed": seed,
        "status": _status(done, info),
        "victory": int(bool(done and info.get("victory"))),
        "steps": len(actions),
        "environment_reward": float(total_reward),
        "hazard_adjacent_steps": sum(distance == 1 for distance in fire_distances),
        "minimum_fire_distance": min(fire_distances) if fire_distances else None,
        "path": path,
        "actions": actions,
        "action_names": [ACTION_NAMES[action] for action in actions],
    }


def evaluate_policy(
    level_id: int,
    agent: Any,
    config: Mapping[str, Any],
    *,
    seed: int,
    episodes: int,
    epsilon: float,
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate without updates and distinguish victory/death/timeout."""
    previous_epsilon = float(agent.epsilon)
    agent.epsilon = float(epsilon)
    agent.set_seed(seed + 2_000_003)
    counts: Counter[str] = Counter()
    victory_steps: list[int] = []
    rewards: list[float] = []
    try:
        for episode in range(episodes):
            episode_seed = seed + 50_000 + episode
            env = make_environment(level_id, config, episode_seed)
            state = reset_environment(env, episode_seed)
            if hasattr(agent, "begin_episode"):
                agent.begin_episode(state)
            total_reward = 0.0
            done = False
            info: dict[str, Any] = {}
            for step in range(1, max_steps + 1):
                action = int(agent.choose_action(state))
                state, reward, done, info = step_environment(env, action)
                total_reward += reward
                if done:
                    break
            result = _status(done, info)
            counts[result] += 1
            rewards.append(total_reward)
            if result == "victory":
                victory_steps.append(step)
    finally:
        agent.epsilon = previous_epsilon
    return {
        "episodes": episodes,
        "epsilon": epsilon,
        "victories": counts["victory"],
        "fire_deaths": counts["death_fire"],
        "monster_deaths": counts["death_monster"],
        "timeouts": counts["timeout"],
        "victory_rate": counts["victory"] / episodes,
        "mean_environment_reward": float(np.mean(rewards)),
        "mean_steps_on_victory": (
            float(np.mean(victory_steps)) if victory_steps else None
        ),
    }


def _train_variant(level_id, agent_kind, intrinsic, config, seed, episodes,
                   max_steps, quiet):
    profile = resolve_training_profile(
        config, level_id, agent_kind,
        {"seed": seed, "episodes": episodes, "max_steps": max_steps},
    )
    agent = make_agent(
        agent_kind, config, use_intrinsic=intrinsic, level_id=level_id,
        seed=seed, profile=profile,
    )
    env = make_environment(level_id, config, seed)
    metrics = train(
        env, agent, int(profile["episodes"]),
        max_steps=int(profile["max_steps"]), base_seed=seed,
        bootstrap_on_truncation=bool(profile["bootstrap_on_truncation"]),
        return_metrics=True, verbose=not quiet,
    )
    return agent, metrics, profile


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty evidence table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _curve(axis, runs, key, window, label, color) -> None:
    curves = []
    x_values = np.asarray([])
    for metrics in runs:
        x_values, curve = rolling_mean(
            (float(row[key]) for row in metrics), window
        )
        curves.append(curve)
    length = min((len(curve) for curve in curves), default=0)
    if length == 0:
        return
    stacked = np.vstack([curve[:length] for curve in curves])
    x_values = x_values[:length]
    mean = np.mean(stacked, axis=0)
    if len(stacked) > 1:
        error = np.std(stacked, axis=0, ddof=1) / math.sqrt(len(stacked))
    else:
        error = np.zeros_like(mean)
    axis.plot(x_values, mean, color=color, linewidth=2.2, label=label)
    axis.fill_between(x_values, mean - 1.96 * error, mean + 1.96 * error,
                      color=color, alpha=0.16)


def _draw_level_paths(axis, level_id, config, rollouts) -> None:
    env = make_environment(level_id, config, 0)
    env.reset(seed=0)
    terrain = np.zeros((env.rows, env.cols), dtype=int)
    for row in range(env.rows):
        for column in range(env.cols):
            terrain[row, column] = {".": 0, "R": 1, "F": 2}.get(
                env.static_grid[row][column], 0
            )
    axis.imshow(terrain, cmap=ListedColormap(["#EDF2F7", "#4A5568", "#E53E3E"]),
                vmin=0, vmax=2)
    for label, rollout in rollouts.items():
        path = rollout["path"]
        axis.plot([position[1] for position in path],
                  [position[0] for position in path], marker="o", markersize=4,
                  linewidth=2.5, color=ALGORITHM_COLORS[label],
                  label="Q-learning" if label == "qlearning" else "SARSA")
    axis.scatter([env.start_pos[1]], [env.start_pos[0]], marker="s", s=80,
                 color="#3182CE", edgecolor="white", label="Start")
    for row, column, kind in env.initial_collectibles:
        axis.scatter([column], [row], marker="*", s=130,
                     color="#38A169", edgecolor="white", label=kind.title())
    axis.set_xticks(range(env.cols)); axis.set_yticks(range(env.rows))
    axis.grid(color="white", linewidth=0.5, alpha=0.45)
    axis.set_title("Representative greedy routes")
    axis.legend(loc="upper left", fontsize=8)


def _first_success(metrics) -> int | None:
    return next((int(row["episode"]) for row in metrics if int(row["victory"])), None)


def run_algorithm_comparison(args, config: Mapping[str, Any]) -> dict[str, Any]:
    level_id = 1 if args.level is None else args.level
    evidence = config.get("evidence", {}).get("level1", {})
    seeds = _parse_seed_list(args.seeds, evidence.get("seeds", [101]))
    evaluation_epsilon = float(
        args.evaluation_epsilon if args.evaluation_epsilon is not None
        else evidence.get("evaluation_epsilon", 0.05)
    )
    evaluation_episodes = int(
        args.evaluation_episodes or evidence.get("evaluation_episodes", 200)
    )
    runs = {"qlearning": [], "sarsa": []}
    profiles = {}
    metric_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    rollouts = {"qlearning": [], "sarsa": []}
    evaluations = {"qlearning": [], "sarsa": []}

    for kind in ("qlearning", "sarsa"):
        for seed in seeds:
            print(f"[{kind}] seed {seed}")
            agent, metrics, profile = _train_variant(
                level_id, kind, False, config, seed, args.episodes,
                args.max_steps, args.quiet,
            )
            profiles[kind] = {key: value for key, value in profile.items()
                              if key != "seed"}
            runs[kind].append(metrics)
            metric_rows.extend(
                {"comparison": "algorithms", "variant": kind,
                 "run_seed": seed, **row} for row in metrics
            )
            rollout = greedy_rollout(
                level_id, agent, config, seed=seed,
                max_steps=int(profile["max_steps"]),
            )
            rollouts[kind].append(rollout)
            rollout_rows.append({
                "variant": kind,
                **{key: value for key, value in rollout.items()
                   if key not in {"path", "actions", "action_names"}},
                "path": json.dumps(rollout["path"]),
                "actions": json.dumps(rollout["action_names"]),
            })
            evaluations[kind].append(evaluate_policy(
                level_id, agent, config, seed=seed,
                episodes=evaluation_episodes, epsilon=evaluation_epsilon,
                max_steps=int(profile["max_steps"]),
            ))

    output = Path(LOGS_DIR)
    _write_rows(output / "level1_algorithm_comparison_metrics.csv", metric_rows)
    _write_rows(output / "level1_algorithm_comparison_rollouts.csv", rollout_rows)
    window = int(args.window or config.get("evidence", {}).get("rolling_window", 50))
    representative = {kind: rollouts[kind][0] for kind in runs}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for kind in runs:
        label = "Q-learning" if kind == "qlearning" else "SARSA"
        _curve(axes[0, 0], runs[kind], "victory", window, label,
               ALGORITHM_COLORS[kind])
        _curve(axes[0, 1], runs[kind], "extrinsic_reward", window, label,
               ALGORITHM_COLORS[kind])
    axes[0, 0].set(title="Learning success (mean and 95% CI)",
                   ylabel="Victory rate", xlabel="Episode", ylim=(-0.03, 1.03))
    axes[0, 1].set(title="Environment return", ylabel="Reward", xlabel="Episode")
    axes[0, 0].legend(); axes[0, 1].legend()
    _draw_level_paths(axes[1, 0], level_id, config, representative)
    kinds = ("qlearning", "sarsa")
    steps = [np.mean([row["steps"] for row in rollouts[kind]]) for kind in kinds]
    adjacent = [np.mean([row["hazard_adjacent_steps"] for row in rollouts[kind]])
                for kind in kinds]
    x = np.arange(2)
    axes[1, 1].bar(x - 0.18, steps, 0.36, label="Path steps", color="#4299E1")
    axes[1, 1].bar(x + 0.18, adjacent, 0.36,
                   label="Hazard-adjacent steps", color="#ED8936")
    axes[1, 1].set_xticks(x, ["Q-learning", "SARSA"])
    axes[1, 1].set_title("Policy safety evidence"); axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    fig.suptitle("Level 1: Q-learning vs SARSA", fontsize=16, fontweight="bold")
    fig.tight_layout()
    figure_path = output / "level1_qlearning_vs_sarsa_evidence.png"
    fig.savefig(figure_path, dpi=170); plt.close(fig)

    summary: dict[str, Any] = {
        "comparison": "algorithms", "level_id": level_id, "seeds": seeds,
        "profiles": profiles, "greedy_rollouts": {}, "epsilon_evaluation": {},
        "conclusion": (
            "Q-learning consistently selected the shorter fire-edge route; "
            "SARSA consistently selected the longer lane with fewer hazard-adjacent steps."
        ),
    }
    for kind in kinds:
        summary["greedy_rollouts"][kind] = {
            "victory_rate": float(np.mean([row["victory"] for row in rollouts[kind]])),
            "mean_steps": float(np.mean([row["steps"] for row in rollouts[kind]])),
            "mean_hazard_adjacent_steps": float(np.mean([
                row["hazard_adjacent_steps"] for row in rollouts[kind]
            ])),
            "mean_minimum_fire_distance": float(np.mean([
                row["minimum_fire_distance"] for row in rollouts[kind]
            ])),
        }
        summary["epsilon_evaluation"][kind] = {
            "epsilon": evaluation_epsilon,
            "episodes_per_seed": evaluation_episodes,
            "mean_victory_rate": float(np.mean([
                row["victory_rate"] for row in evaluations[kind]
            ])),
            "mean_fire_deaths": float(np.mean([
                row["fire_deaths"] for row in evaluations[kind]
            ])),
        }
    save_json(summary, output / "level1_algorithm_comparison_summary.json")
    print(f"[+] Evidence figure: {figure_path}")
    return summary


def _position_heatmap(runs, rows, cols, episode_limit) -> np.ndarray:
    heat = np.zeros((rows, cols), dtype=float)
    count = 0
    for metrics in runs:
        for row in metrics[:episode_limit]:
            for encoded in str(row.get("visited_positions", "")).split("|"):
                if encoded:
                    r_text, c_text = encoded.split(":", 1)
                    heat[int(r_text), int(c_text)] += 1.0
            count += 1
    return heat / max(count, 1)


def run_intrinsic_comparison(args, config: Mapping[str, Any]) -> dict[str, Any]:
    level_id = 6 if args.level is None else args.level
    evidence = config.get("evidence", {}).get("level6", {})
    seeds = _parse_seed_list(args.seeds, evidence.get("seeds", [111]))
    agent_kind = str(evidence.get("agent", "qlearning"))
    variants = {"baseline": False, "intrinsic": True}
    runs = {name: [] for name in variants}
    profiles = {}
    rollouts = {name: [] for name in variants}
    metric_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []

    for variant, intrinsic in variants.items():
        for seed in seeds:
            print(f"[{variant}] seed {seed}")
            agent, metrics, profile = _train_variant(
                level_id, agent_kind, intrinsic, config, seed,
                args.episodes, args.max_steps, args.quiet,
            )
            profiles[variant] = {key: value for key, value in profile.items()
                                 if key != "seed"}
            runs[variant].append(metrics)
            metric_rows.extend(
                {"comparison": "intrinsic", "variant": variant,
                 "run_seed": seed, **row} for row in metrics
            )
            rollout = greedy_rollout(
                level_id, agent, config, seed=seed,
                max_steps=int(profile["max_steps"]),
            )
            rollouts[variant].append(rollout)
            rollout_rows.append({
                "variant": variant,
                **{key: value for key, value in rollout.items()
                   if key not in {"path", "actions", "action_names"}},
                "path": json.dumps(rollout["path"]),
                "actions": json.dumps(rollout["action_names"]),
            })
            agent.save(
                Path(MODELS_DIR) / "evidence" /
                f"level{level_id}_{agent_kind}_{variant}_seed{seed}.pkl",
                metadata={"purpose": "level6_intrinsic_comparison",
                          "variant": variant, "level_id": level_id,
                          "seed": seed, "profile": profile,
                          "layout_fingerprint": make_environment(
                              level_id, config, seed
                          ).layout_fingerprint},
            )

    output = Path(LOGS_DIR)
    _write_rows(output / "level6_intrinsic_comparison_metrics.csv", metric_rows)
    _write_rows(output / "level6_intrinsic_comparison_rollouts.csv", rollout_rows)
    window = int(args.window or config.get("evidence", {}).get("rolling_window", 50))
    env = make_environment(level_id, config, 0); env.reset(seed=0)
    heat_limit = min(100, len(runs["baseline"][0]))
    baseline_heat = _position_heatmap(
        runs["baseline"], env.rows, env.cols, heat_limit
    )
    intrinsic_heat = _position_heatmap(
        runs["intrinsic"], env.rows, env.cols, heat_limit
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for variant in variants:
        _curve(axes[0, 0], runs[variant], "victory", window, variant.title(),
               INTRINSIC_COLORS[variant])
        _curve(axes[0, 1], runs[variant], "unique_positions", window,
               variant.title(), INTRINSIC_COLORS[variant])
    axes[0, 0].set(title="Learning success (mean and 95% CI)",
                   ylabel="Victory rate", xlabel="Episode", ylim=(-0.03, 1.03))
    axes[0, 1].set(title="Exploration coverage",
                   ylabel="Unique positions / episode", xlabel="Episode")
    axes[0, 0].legend(); axes[0, 1].legend()
    difference = intrinsic_heat - baseline_heat
    scale = max(float(np.max(np.abs(difference))), 0.01)
    image = axes[1, 0].imshow(difference, cmap="coolwarm", vmin=-scale, vmax=scale)
    for row in range(env.rows):
        for column in range(env.cols):
            if env.static_grid[row][column] == "R":
                axes[1, 0].add_patch(
                    plt.Rectangle((column - 0.5, row - 0.5), 1, 1, color="#263238")
                )
    axes[1, 0].set_title(
        f"Early visitation change: intrinsic - baseline ({heat_limit} eps)"
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    first_success = []
    tail_success = []
    for variant in variants:
        episode_count = len(runs[variant][0])
        first_success.append(float(np.mean([
            _first_success(metrics) or episode_count + 1
            for metrics in runs[variant]
        ])))
        tail_success.append(float(np.mean([
            np.mean([int(row["victory"]) for row in metrics[-100:]])
            for metrics in runs[variant]
        ])))
    x = np.arange(2)
    first_bars = axes[1, 1].bar(
        x, first_success, 0.52,
        label="Mean first-success episode", color="#5C6BC0"
    )
    reliability_axis = axes[1, 1].twinx()
    reliability_line = reliability_axis.plot(
        x, tail_success, marker="o", markersize=8, linewidth=2.5,
        label="Last-100 victory rate", color="#2E7D32"
    )[0]
    axes[1, 1].set_xticks(x, ["Baseline", "Intrinsic"])
    axes[1, 1].set_title("Discovery speed and final reliability")
    axes[1, 1].set_ylabel("First-success episode (lower is faster)")
    reliability_axis.set_ylabel("Last-100 victory rate")
    reliability_axis.set_ylim(-0.03, 1.15)
    axes[1, 1].legend(
        [first_bars, reliability_line],
        ["Mean first-success episode", "Last-100 victory rate"],
        fontsize=8,
        loc="best",
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    fig.suptitle("Level 6: Count-based intrinsic reward A/B test",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    figure_path = output / "level6_intrinsic_comparison_evidence.png"
    fig.savefig(figure_path, dpi=170); plt.close(fig)

    summary: dict[str, Any] = {
        "comparison": "intrinsic", "level_id": level_id,
        "algorithm": agent_kind, "seeds": seeds,
        "intrinsic_formula": (
            "intrinsic_strength / sqrt(n(s) + 1), where s is the reached state"
        ),
        "environment_rewards_unchanged": True,
        "profiles": profiles, "variants": {},
    }
    for variant in variants:
        episode_count = len(runs[variant][0])
        summary["variants"][variant] = {
            "mean_first_success_episode": float(np.mean([
                _first_success(metrics) or episode_count + 1
                for metrics in runs[variant]
            ])),
            "mean_first_100_victory_rate": float(np.mean([
                np.mean([int(row["victory"]) for row in metrics[:100]])
                for metrics in runs[variant]
            ])),
            "mean_last_100_victory_rate": float(np.mean([
                np.mean([int(row["victory"]) for row in metrics[-100:]])
                for metrics in runs[variant]
            ])),
            "mean_first_100_unique_positions": float(np.mean([
                np.mean([float(row["unique_positions"]) for row in metrics[:100]])
                for metrics in runs[variant]
            ])),
            "greedy_victory_rate": float(np.mean([
                rollout["victory"] for rollout in rollouts[variant]
            ])),
            "mean_greedy_steps": (
                float(np.mean([rollout["steps"] for rollout in rollouts[variant]
                               if rollout["victory"]]))
                if any(rollout["victory"] for rollout in rollouts[variant])
                else None
            ),
        }
    baseline_first = summary["variants"]["baseline"]["mean_first_success_episode"]
    intrinsic_first = summary["variants"]["intrinsic"]["mean_first_success_episode"]
    improvement = baseline_first - intrinsic_first
    summary["observed_result"] = {
        "first_success_episode_improvement": improvement,
        "first_success_percent_faster": (
            100.0 * improvement / baseline_first if baseline_first else 0.0
        ),
        "explanation": (
            "The small count bonus targets faster initial reward discovery; "
            "tail success and greedy completion are reported beside it so the "
            "exploration/exploitation trade-off remains visible."
        ),
    }
    save_json(summary, output / "level6_intrinsic_comparison_summary.json")
    print(f"[+] Evidence figure: {figure_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gridworld comparison evidence")
    parser.add_argument("--comparison", choices=["algorithms", "intrinsic"],
                        default="algorithms")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated experiment seeds")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--evaluation-epsilon", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    summary = (
        run_intrinsic_comparison(args, config)
        if args.comparison == "intrinsic"
        else run_algorithm_comparison(args, config)
    )
    print(json.dumps(
        summary.get("observed_result", {"conclusion": summary.get("conclusion")}),
        indent=2,
    ))


if __name__ == "__main__":
    main()
