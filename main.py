"""Friendly root launcher for Assignment 3 Part I.

Run the complete Gridworld application from the repository root with::

    python main.py

The package entry point (``python -m gridworld``) remains equivalent for users
who prefer module-style execution.

On a repository checkout that contains ``.venv``, this launcher automatically
uses that environment when the currently selected Python does not have Pygame.
The hand-off is performed by the operating system rather than a shell, so a
workspace path containing PowerShell characters such as ``&`` remains safe.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


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


_ensure_project_runtime()

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from gridworld.app import main


if __name__ == "__main__":
    main()
