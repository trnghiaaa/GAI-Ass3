"""Regression tests for the one-click Part II entry point."""

import unittest
from unittest.mock import patch

import main as root_launcher


class MainLauncherTests(unittest.TestCase):
    def test_main_opens_the_part_two_visual_launcher(self) -> None:
        with patch("arena.app.main") as launch_arena:
            root_launcher.main()

        launch_arena.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

