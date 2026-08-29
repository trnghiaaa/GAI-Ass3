"""One-command builder for every required Part I model and evidence artifact.

Usage::

    python -m gridworld.build_evidence

The command intentionally delegates to the public training/comparison CLIs so
the exact commands remain visible in the terminal and reproducible by markers.
Use ``--quick`` only for a smoke run; it is not submission-quality evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from gridworld.train import LOGS_DIR, MODELS_DIR


MODEL_RUNS = [
    (0, "qlearning", False),
    (1, "qlearning", False),
    (1, "sarsa", False),
    (2, "qlearning", False),
    (2, "sarsa", False),
    (3, "qlearning", False),
    (3, "sarsa", False),
    (4, "qlearning", False),
    (4, "sarsa", False),
    (5, "qlearning", False),
    (5, "sarsa", False),
    (6, "qlearning", False),
    (6, "qlearning", True),
]


def _run(arguments: list[str]) -> None:
    command = [sys.executable, "-m", *arguments]
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def expected_artifacts() -> list[Path]:
    artifacts: list[Path] = []
    for level, agent, intrinsic in MODEL_RUNS:
        suffix = "_intrinsic" if intrinsic else ""
        stem = f"level{level}_{agent}{suffix}"
        artifacts.extend([
            Path(MODELS_DIR) / f"{stem}.pkl",
            Path(LOGS_DIR) / f"{stem}.png",
            Path(LOGS_DIR) / f"{stem}_rewards.csv",
            Path(LOGS_DIR) / f"{stem}_summary.json",
        ])
    artifacts.extend([
        Path(LOGS_DIR) / "level1_qlearning_vs_sarsa_evidence.png",
        Path(LOGS_DIR) / "level1_algorithm_comparison_metrics.csv",
        Path(LOGS_DIR) / "level1_algorithm_comparison_rollouts.csv",
        Path(LOGS_DIR) / "level1_algorithm_comparison_summary.json",
        Path(LOGS_DIR) / "level6_intrinsic_comparison_evidence.png",
        Path(LOGS_DIR) / "level6_intrinsic_comparison_metrics.csv",
        Path(LOGS_DIR) / "level6_intrinsic_comparison_rollouts.csv",
        Path(LOGS_DIR) / "level6_intrinsic_comparison_summary.json",
        Path(LOGS_DIR) / "policy_benchmark.csv",
        Path(LOGS_DIR) / "policy_benchmark.json",
    ])
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all Part I evidence")
    parser.add_argument("--quick", action="store_true",
                        help="20-episode smoke run; not submission evidence")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--comparisons-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.models_only and args.comparisons_only:
        parser.error("--models-only and --comparisons-only are mutually exclusive")

    if not args.comparisons_only:
        for level, agent, intrinsic in MODEL_RUNS:
            command = ["gridworld.train", "--level", str(level), "--agent", agent]
            if intrinsic:
                command.append("--intrinsic")
            if args.quick:
                command.extend(["--episodes", "20", "--max-steps", "50"])
            if args.quiet:
                command.append("--quiet")
            _run(command)

    if not args.models_only:
        for comparison in ("algorithms", "intrinsic"):
            command = ["gridworld.compare", "--comparison", comparison]
            if args.quick:
                command.extend(["--episodes", "20", "--seeds", "1"])
                if comparison == "algorithms":
                    command.extend(["--evaluation-episodes", "5"])
            if args.quiet:
                command.append("--quiet")
            _run(command)

    if not args.quick:
        _run(["gridworld.benchmark"])
        missing = [path for path in expected_artifacts() if not path.exists()]
        if missing:
            print("\nBuild finished, but required artifacts are missing:")
            for path in missing:
                print(f"  - {path}")
            raise SystemExit(1)
        print(f"\nPart I evidence complete: {len(expected_artifacts())} artifacts verified.")
    else:
        print("\nQuick smoke build complete. Re-run without --quick for submission artifacts.")


if __name__ == "__main__":
    main()
