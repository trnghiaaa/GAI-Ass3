"""Canonical one-click launcher for Assignment 3 Part II.

Run this file with the VS Code Run button or with ``python main.py``.
The ``python -m arena`` module entry point remains equivalent.
"""

from __future__ import annotations


def main() -> None:
    """Open the polished Neon Rift Arena mode-selection menu."""

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
    main()
