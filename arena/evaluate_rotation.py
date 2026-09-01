"""Convenience entry point for the trained rotation/thrust agent."""

import sys

from arena.evaluate import main as evaluate_main


def main() -> None:
    """Forward all evaluation options while locking rotation controls."""

    evaluate_main(sys.argv[1:], forced_control_style="rotation")


if __name__ == "__main__":
    main()
