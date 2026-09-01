"""Unified launcher for both Assignment 3 reinforcement-learning projects.

Examples::

    python main.py --part 1
    python main.py --part 2

Part II remains the default when ``--part`` is omitted. The package entry
points ``python -m gridworld`` and ``python -m arena`` remain available.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_python() -> Path | None:
    """Return this checkout's virtual-environment interpreter when available."""

    candidates = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _same_interpreter(first: str | Path, second: str | Path) -> bool:
    """Compare interpreter paths using the host platform's path semantics."""

    return os.path.normcase(os.path.realpath(first)) == os.path.normcase(
        os.path.realpath(second)
    )


def _ensure_project_runtime() -> None:
    """Re-launch through ``.venv`` if the selected Python lacks Pygame."""

    if importlib.util.find_spec("pygame") is not None:
        return

    project_python = _project_python()
    if project_python is not None and not _same_interpreter(
        sys.executable, project_python
    ):
        completed = subprocess.run(
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(completed.returncode)

    raise SystemExit(
        "Pygame is not installed and no usable project .venv was found.\n"
        "Create the environment and install the requirements with:\n"
        "  python -m venv .venv\n"
        "  .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Part I Gridworld or Part II Neon Rift Arena."
    )
    parser.add_argument(
        "--part",
        choices=("1", "2"),
        default="2",
        help="project to launch (default: 2)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Launch the selected project and forward remaining options to Part I."""

    args, remaining = build_parser().parse_known_args(() if argv is None else argv)
    _ensure_project_runtime()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    if args.part == "1":
        from gridworld.app import main as launch_gridworld

        launch_gridworld(remaining)
        return

    if remaining:
        build_parser().error(
            f"Part II does not accept launcher arguments: {' '.join(remaining)}"
        )

    try:
        from arena.app import main as launch_arena
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Cannot start Part II because {missing!r} is unavailable.\n"
            "Install the project dependencies with:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    launch_arena()


if __name__ == "__main__":
    main(sys.argv[1:])
