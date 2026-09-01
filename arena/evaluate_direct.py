"""Convenience entry point for the trained direct-control agent."""

import sys

from arena.evaluate import main as evaluate_main


def main() -> None:
    """Forward all evaluation options while locking direct controls."""

    evaluate_main(sys.argv[1:], forced_control_style="direct")


if __name__ == "__main__":
    main()
