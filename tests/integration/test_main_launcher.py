"""Regression tests for the combined Part I/Part II launcher."""

import unittest
from unittest.mock import patch

import main as root_launcher


class MainLauncherTests(unittest.TestCase):
    def test_main_opens_the_visual_master_hub_by_default(self) -> None:
        with patch("main.launch_master_hub", return_value="quit") as launch_hub:
            root_launcher.main()

        launch_hub.assert_called_once_with()

    def test_master_hub_selection_dispatches_to_arena(self) -> None:
        with patch("main.launch_master_hub", side_effect=["2", "quit"]), patch(
            "arena.app.main"
        ) as launch_arena:
            root_launcher.main()

        launch_arena.assert_called_once_with()

    def test_master_hub_selection_dispatches_to_gridworld(self) -> None:
        with patch("main.launch_master_hub", side_effect=["1", "quit"]), patch(
            "gridworld.app.main"
        ) as launch_gridworld:
            root_launcher.main()

        launch_gridworld.assert_called_once_with([])

    def test_part_two_direct_cli_flag(self) -> None:
        with patch("arena.app.main") as launch_arena:
            root_launcher.main(["--part", "2"])

        launch_arena.assert_called_once_with()

    def test_part_one_arguments_are_forwarded_to_gridworld(self) -> None:
        with patch("gridworld.app.main") as launch_gridworld:
            root_launcher.main(["--part", "1", "--mode", "manual", "--level", "4"])

        launch_gridworld.assert_called_once_with(
            ["--mode", "manual", "--level", "4"]
        )

    def test_main_rejects_invalid_parts_clearly(self) -> None:
        with self.assertRaises(SystemExit):
            root_launcher.main(["--part", "invalid_option"])


if __name__ == "__main__":
    unittest.main()
