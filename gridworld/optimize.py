"""Held-out champion selection for robust stochastic Gridworld policies.

The assignment's update rules and environment rewards remain untouched.  This
module simply trains multiple seeded Q-learning or SARSA candidates, evaluates
them on the same unseen monster transitions, and persists the strongest table.

Example::

    python -m gridworld.optimize --level 4 --agent qlearning
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.compare import evaluate_policy
from gridworld.environment import STATE_SCHEMA
from gridworld.train import (
    LOGS_DIR,
    MODELS_DIR,
    load_config,
    make_agent,
    make_environment,
    resolve_training_profile,
    save_json,
    save_metrics_csv,
    save_training_plot,
    summarize_metrics,
    train,
)


def parse_seeds(value: str | None, defaults: list[int]) -> list[int]:
    """Parse a comma-separated seed list while rejecting empty selections."""

    seeds = defaults if value is None else [
        int(item.strip()) for item in value.split(",") if item.strip()
    ]
    if not seeds:
        raise ValueError("At least one candidate seed is required")
    return list(dict.fromkeys(int(seed) for seed in seeds))


def candidate_score(evaluation: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank completion first, then timeouts, deaths, and successful path length."""

    mean_steps = evaluation.get("mean_steps_on_victory")
    return (
        float(evaluation["victory_rate"]),
        -float(evaluation.get("timeouts", 0)),
        -float(evaluation.get("monster_deaths", 0)),
        -float(mean_steps if mean_steps is not None else float("inf")),
    )


def _load_existing(kind: str, path: Path, seed: int):
    agent_class = QLearningAgent if kind == "qlearning" else SARSAAgent
    agent = agent_class(seed=seed)
    metadata = agent.load(path)
    return agent, metadata


def _evaluate_candidate(
    level: int,
    agent: Any,
    config: Mapping[str, Any],
    *,
    seed: int,
    episodes: int,
    max_steps: int,
) -> dict[str, Any]:
    # Evaluation can encounter unseen states and the Q-table is a defaultdict.
    # Evaluate a copy so the saved champion contains training knowledge only.
    return evaluate_policy(
        level,
        copy.deepcopy(agent),
        config,
        seed=seed,
        episodes=episodes,
        epsilon=0.0,
        max_steps=max_steps,
    )


def optimize_model(
    level: int,
    kind: str,
    config: Mapping[str, Any],
    *,
    seeds: list[int],
    intrinsic: bool = False,
    episodes: int | None = None,
    max_steps: int | None = None,
    validation_episodes: int = 1000,
    validation_seed: int = 73_000,
    include_current: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train, compare, and save the strongest held-out candidate."""

    if level not in (4, 5):
        raise ValueError("Champion selection is intended for monster Levels 4 and 5")
    if validation_episodes <= 0:
        raise ValueError("validation_episodes must be positive")

    suffix = "_intrinsic" if intrinsic else ""
    stem = f"level{level}_{kind}{suffix}"
    model_path = Path(MODELS_DIR) / f"{stem}.pkl"
    candidates: list[dict[str, Any]] = []

    if include_current and model_path.exists():
        existing, existing_metadata = _load_existing(kind, model_path, validation_seed)
        existing_profile = dict(existing_metadata.get("profile", {}))
        existing_max_steps = int(
            max_steps
            or existing_profile.get("max_steps")
            or resolve_training_profile(config, level, kind)["max_steps"]
        )
        evaluation = _evaluate_candidate(
            level,
            existing,
            config,
            seed=validation_seed,
            episodes=validation_episodes,
            max_steps=existing_max_steps,
        )
        candidates.append({
            "source": "current_saved_model",
            "seed": existing_metadata.get("seed"),
            "agent": existing,
            "metrics": None,
            "profile": existing_profile,
            "evaluation": evaluation,
        })

    for seed in seeds:
        profile = resolve_training_profile(
            config,
            level,
            kind,
            {
                "seed": seed,
                "episodes": episodes,
                "max_steps": max_steps,
            },
        )
        agent = make_agent(
            kind,
            config,
            use_intrinsic=intrinsic,
            level_id=level,
            seed=seed,
            profile=profile,
        )
        env = make_environment(level, config, seed)
        if not quiet:
            print(
                f"Training candidate seed {seed}: {profile['episodes']} episodes"
            )
        metrics = train(
            env,
            agent,
            int(profile["episodes"]),
            max_steps=int(profile["max_steps"]),
            base_seed=seed,
            bootstrap_on_truncation=bool(profile["bootstrap_on_truncation"]),
            return_metrics=True,
            verbose=False,
        )
        evaluation = _evaluate_candidate(
            level,
            agent,
            config,
            seed=validation_seed,
            episodes=validation_episodes,
            max_steps=int(profile["max_steps"]),
        )
        candidates.append({
            "source": "trained_candidate",
            "seed": seed,
            "agent": agent,
            "metrics": metrics,
            "profile": profile,
            "evaluation": evaluation,
        })
        print(
            f"  seed {seed}: {evaluation['victory_rate']:.2%} victories, "
            f"{evaluation['monster_deaths']} deaths, {evaluation['timeouts']} timeouts"
        )

    champion = max(candidates, key=lambda item: candidate_score(item["evaluation"]))
    selection_rows = [
        {
            "source": item["source"],
            "seed": item["seed"],
            "score": list(candidate_score(item["evaluation"])),
            **item["evaluation"],
        }
        for item in candidates
    ]
    result = {
        "level_id": level,
        "algorithm": kind,
        "intrinsic_enabled": intrinsic,
        "validation_seed": validation_seed,
        "validation_episodes": validation_episodes,
        "selection_rule": (
            "highest victory rate; then fewer timeouts, fewer monster deaths, "
            "and shorter successful paths"
        ),
        "selected_source": champion["source"],
        "selected_seed": champion["seed"],
        "candidates": selection_rows,
    }
    selection_path = Path(LOGS_DIR) / f"{stem}_selection.json"
    save_json(result, selection_path)

    if champion["source"] == "current_saved_model":
        print(
            f"Current {model_path.name} remains champion at "
            f"{champion['evaluation']['victory_rate']:.2%}; no artifact overwritten."
        )
        print(f"Selection evidence: {selection_path}")
        return result

    metrics = champion["metrics"]
    profile = champion["profile"]
    summary = summarize_metrics(metrics)
    metadata = {
        "level_id": level,
        "algorithm": kind,
        "seed": champion["seed"],
        "intrinsic_enabled": intrinsic,
        "profile": profile,
        "summary": summary,
        "state_schema": STATE_SCHEMA,
        "champion_selection": result,
    }
    csv_path = Path(LOGS_DIR) / f"{stem}_rewards.csv"
    plot_path = Path(LOGS_DIR) / f"{stem}.png"
    summary_path = Path(LOGS_DIR) / f"{stem}_summary.json"
    champion["agent"].save(model_path, metadata=metadata)
    save_metrics_csv(metrics, csv_path)
    save_training_plot(
        metrics,
        plot_path,
        title=f"Level {level} - {kind.upper()} selected champion",
        window=int(profile["rolling_window"]),
    )
    save_json(metadata, summary_path)
    print(
        f"Selected seed {champion['seed']} with "
        f"{champion['evaluation']['victory_rate']:.2%} held-out victories."
    )
    print(f"Selection evidence: {selection_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a robust held-out champion for a monster level"
    )
    parser.add_argument("--level", type=int, choices=(4, 5), required=True)
    parser.add_argument("--agent", choices=("qlearning", "sarsa"), required=True)
    parser.add_argument("--seeds", default=None, help="Comma-separated candidates")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--intrinsic", action="store_true")
    parser.add_argument("--exclude-current", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config()
    selection = config.get("model_selection", {})
    seeds = parse_seeds(
        args.seeds,
        list(selection.get("monster_candidate_seeds", [1729, 2718, 3141])),
    )
    optimize_model(
        args.level,
        args.agent,
        config,
        seeds=seeds,
        intrinsic=args.intrinsic,
        episodes=args.episodes,
        max_steps=args.max_steps,
        validation_episodes=int(
            args.validation_episodes
            or selection.get("validation_episodes", 1000)
        ),
        validation_seed=int(
            args.validation_seed or selection.get("validation_seed", 73_000)
        ),
        include_current=not args.exclude_current,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
