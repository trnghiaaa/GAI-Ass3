"""Regression tests for the combined Part I/Part II launcher."""

import unittest
from unittest.mock import patch

import main as root_launcher


class MainLauncherTests(unittest.TestCase):
    def test_main_opens_the_part_two_visual_launcher(self) -> None:
        with patch("arena.app.main") as launch_arena:
            root_launcher.main()

        launch_arena.assert_called_once_with()

    def test_part_one_arguments_are_forwarded_to_gridworld(self) -> None:
        with patch("gridworld.app.main") as launch_gridworld:
            root_launcher.main(["--part", "1", "--mode", "manual", "--level", "4"])

        launch_gridworld.assert_called_once_with(
            ["--mode", "manual", "--level", "4"]
        )


if __name__ == "__main__":
    unittest.main()
