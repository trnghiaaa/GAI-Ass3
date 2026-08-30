"""Evaluate every required saved Part I policy with one reproducible command."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.compare import evaluate_policy, greedy_rollout
from gridworld.environment import GridWorldEnv
from gridworld.train import LOGS_DIR, MODELS_DIR, load_config, resolve_training_profile


MODEL_MATRIX = [
    (0, "qlearning", False),
    (1, "qlearning", False), (1, "sarsa", False),
    (2, "qlearning", False), (2, "sarsa", False),
    (3, "qlearning", False), (3, "sarsa", False),
    (4, "qlearning", False), (4, "sarsa", False),
    (5, "qlearning", False), (5, "sarsa", False),
    (6, "qlearning", False), (6, "qlearning", True),
]


def _load(kind: str, path: Path, seed: int):
    agent_class = QLearningAgent if kind == "qlearning" else SARSAAgent
    agent = agent_class(seed=seed)
    metadata = agent.load(path)
    return agent, metadata


def run_benchmark(episodes: int = 100, monster_episodes: int = 300,
                  seed: int = 24000) -> list[dict[str, Any]]:
    config = load_config()
    rows: list[dict[str, Any]] = []
    for index, (level, kind, intrinsic) in enumerate(MODEL_MATRIX):
        suffix = "_intrinsic" if intrinsic else ""
        path = Path(MODELS_DIR) / f"level{level}_{kind}{suffix}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Required saved model is missing: {path}")
        run_seed = seed + index * 1000
        agent, metadata = _load(kind, path, run_seed)
        saved_q_table_states = len(agent.q_table)
        profile = resolve_training_profile(config, level, kind)
        environment = GridWorldEnv(level, seed=run_seed)
        if metadata.get("layout_fingerprint") != environment.layout_fingerprint:
            raise ValueError(
                f"Saved model layout does not match current Level {level}: {path}"
            )
        count = monster_episodes if level in (4, 5) else episodes
        evaluation = evaluate_policy(
            level, agent, config, seed=run_seed, episodes=count, epsilon=0.0,
            max_steps=int(profile["max_steps"]),
        )
        rollout = greedy_rollout(
            level, agent, config, seed=run_seed + 99,
            max_steps=int(profile["max_steps"]),
        )
        row = {
            "level": level,
            "algorithm": kind,
            "intrinsic": int(intrinsic),
            "model": path.name,
            "model_bytes": path.stat().st_size,
            # Evaluation can encounter unseen stochastic monster states.  The
            # defaultdict then gains zero-valued rows in memory, so capture the
            # model's actual saved table size before running any rollouts.
            "q_table_states": saved_q_table_states,
            "episodes": count,
            "victories": evaluation["victories"],
            "victory_rate": evaluation["victory_rate"],
            "fire_deaths": evaluation["fire_deaths"],
            "monster_deaths": evaluation["monster_deaths"],
            "timeouts": evaluation["timeouts"],
            "mean_steps_on_victory": evaluation["mean_steps_on_victory"],
            "representative_status": rollout["status"],
            "representative_steps": rollout["steps"],
            "representative_reward": rollout["environment_reward"],
            "training_seed": metadata.get("seed", ""),
            "layout_fingerprint": metadata.get("layout_fingerprint", ""),
            "state_schema": "|".join(metadata.get("state_schema", [])),
        }
        rows.append(row)
        label = kind + (" + intrinsic" if intrinsic else "")
        print(
            f"L{level} {label:<22} "
            f"{evaluation['victory_rate']:>6.1%} wins | "
            f"deaths {evaluation['fire_deaths'] + evaluation['monster_deaths']:>3} | "
            f"timeouts {evaluation['timeouts']:>3}"
        )
    return rows


def save_benchmark(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    output = Path(LOGS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "policy_benchmark.csv"
    json_path = output / "policy_benchmark.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "models_evaluated": len(rows),
        "all_required_models_present": len(rows) == len(MODEL_MATRIX),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark every saved Part I model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--monster-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=24000)
    args = parser.parse_args()
    rows = run_benchmark(args.episodes, args.monster_episodes, args.seed)
    csv_path, json_path = save_benchmark(rows)
    print(f"[+] CSV:  {csv_path}\n[+] JSON: {json_path}")


if __name__ == "__main__":
    main()
