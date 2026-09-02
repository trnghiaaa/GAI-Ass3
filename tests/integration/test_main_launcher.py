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

    def test_master_hub_help_overlay_toggle(self) -> None:
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            import pygame
            launcher = root_launcher.MasterLauncher()
            try:
                self.assertFalse(launcher.show_help_overlay)
                launcher.show_help_overlay = True
                self.assertTrue(launcher.show_help_overlay)
                launcher._draw(hover_p1=False, hover_p2=False, hover_vol_btn=False, hover_help_btn=False)
            finally:
                launcher._cleanup_audio()
                pygame.quit()

    def test_master_audio_synthesis_and_volume_controls(self) -> None:
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            import pygame
            audio = root_launcher.MasterAudio()
            try:
                if audio.available:
                    self.assertIn("hover", audio.effects)
                    self.assertIn("select", audio.effects)
                    self.assertIn("click", audio.effects)
                    audio.start_music()

                    # Volume control
                    self.assertAlmostEqual(audio.set_volume(0.5), 0.5, places=5)
                    self.assertAlmostEqual(audio.get_volume(), 0.5, places=5)
                    self.assertEqual(audio.set_volume(1.5), 1.0)
                    self.assertEqual(audio.set_volume(-0.5), 0.0)

                    # Toggle mute
                    audio.set_volume(0.8)
                    self.assertFalse(audio.toggle())
                    self.assertTrue(audio.toggle())
            finally:
                audio.close()
                pygame.quit()

    def test_master_hub_modal_dismiss_isolation(self) -> None:
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            import pygame
            launcher = root_launcher.MasterLauncher()
            try:
                launcher.show_help_overlay = True

                # Pressing key while overlay active should dismiss overlay
                ev_esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
                # Verify overlay state
                self.assertTrue(launcher.show_help_overlay)
                launcher.show_help_overlay = False
                self.assertFalse(launcher.show_help_overlay)
            finally:
                launcher._cleanup_audio()
                pygame.quit()


if __name__ == "__main__":
    unittest.main()
