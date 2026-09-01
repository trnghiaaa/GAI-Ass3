"""Choose a saved DQN checkpoint using normal and boss-start holdouts.

This is an evaluation utility only. It never alters policy actions, rewards,
or the environment; promotion remains an explicit separate command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from arena.benchmark import evaluate_model
from arena.cooldown import load_dqn
from arena.environment import ENVIRONMENT_SCHEMA_VERSION, OBSERVATION_NAMES


def _candidate_paths(run_dir: Path, reference: Path) -> list[Path]:
    """Return every meaningful checkpoint once, preserving a readable order."""

    paths = [reference, run_dir / "last_model.zip", run_dir / "best" / "best_model.zip"]
    paths.extend(sorted((run_dir / "checkpoints").glob("*_steps.zip")))
    unique: list[Path] = []
    for path in paths:
        if path.is_file() and path not in unique:
            unique.append(path)
    if not unique:
        raise FileNotFoundError("No model files found for checkpoint selection")
    return unique


def _score(normal: dict[str, Any], boss: dict[str, Any]) -> float:
    """Reward progression and active boss evasion, not passive stalling."""

    return (
        float(normal["mean_reward"])
        + 30.0 * float(normal["mean_phase"])
        + 60.0 * float(normal["phase_progression_rate"])
        + 30.0 * float(normal["mean_bosses_destroyed"])
        + 6.0 * float(normal["mean_boss_skills_dodged"])
        - 12.0 * float(normal["mean_boss_skill_hits"])
        + 30.0 * float(boss["phase_progression_rate"])
        + 35.0 * float(boss["mean_bosses_destroyed"])
        + 10.0 * float(boss["mean_boss_skills_dodged"])
        - 24.0 * float(boss["mean_boss_skill_hits"])
        + 10.0 * float(boss["mean_boss_defenders_destroyed"])
        + 2.0 * float(boss["mean_missiles_evaded"])
        - 12.0 * float(boss["mean_missile_hits"])
        - 0.75 * float(boss["mean_boss_immune_hits"])
    )


def evaluate_candidates(
    *,
    run_dir: Path,
    reference: Path,
    control_style: str,
    output: Path,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate candidates on fixed normal and Phase-3 boss-start episodes."""

    if output.with_suffix(".json").exists():
        raise FileExistsError("Choose a fresh --output path; existing evidence is protected")
    if episodes < 1:
        raise ValueError("episodes must be positive")

    rows: list[dict[str, Any]] = []
    for path in _candidate_paths(run_dir, reference):
        model = load_dqn(str(path), device="cpu")
        if tuple(model.observation_space.shape) != (len(OBSERVATION_NAMES),):
            raise ValueError(f"{path} has an incompatible observation size")
        normal_rows, normal = evaluate_model(
            model,
            control_style,
            episodes=episodes,
            action_repeat=4,
            seed=seed,
        )
        boss_rows, boss = evaluate_model(
            model,
            control_style,
            episodes=episodes,
            action_repeat=4,
            seed=seed + 10_000,
            reset_options={"start_phase": 3},
        )
        del normal_rows, boss_rows
        rows.append(
            {
                "candidate": str(path),
                "is_reference": path.resolve() == reference.resolve(),
                "normal": normal,
                "boss_focus": boss,
                "selection_score": round(_score(normal, boss), 6),
            }
        )

    winner = max(rows, key=lambda row: float(row["selection_score"]))
    result = {
        "environment_schema": ENVIRONMENT_SCHEMA_VERSION,
        "control_style": control_style,
        "episodes_per_condition": episodes,
        "normal_seed_start": seed,
        "boss_seed_start": seed + 10_000,
        "winner": winner,
        "candidates": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control-style", choices=("direct", "rotation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=71_000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate_candidates(
        run_dir=args.run_dir,
        reference=args.reference,
        control_style=args.control_style,
        output=args.output,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
