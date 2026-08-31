"""Contract tests for the modern and rubric-compatible arena APIs."""

import unittest

from gymnasium.utils.env_checker import check_env
import numpy as np

from arena.environment import ArenaEnv, DIRECT_ACTIONS
from arena.legacy_api import LegacyArenaEnv
from arena.renderer import ArenaRenderer


class ArenaApiTests(unittest.TestCase):
    def test_gymnasium_contract(self) -> None:
        env = ArenaEnv(config_override={"simulation": {"max_steps": 2}})
        try:
            check_env(env, skip_render_check=True)
            observation, info = env.reset(seed=21)
            result = env.step(DIRECT_ACTIONS["NOOP"])

            self.assertTrue(env.observation_space.contains(observation))
            self.assertIsInstance(info, dict)
            self.assertEqual(len(result), 5)
            next_observation, reward, terminated, truncated, step_info = result
            self.assertTrue(env.observation_space.contains(next_observation))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(step_info, dict)
        finally:
            env.close()

    def test_assignment_four_value_contract(self) -> None:
        env = LegacyArenaEnv(config_override={"simulation": {"max_steps": 1}})
        try:
            observation = env.reset(seed=21)
            result = env.step(DIRECT_ACTIONS["NOOP"])

            self.assertTrue(env.observation_space.contains(observation))
            self.assertIsInstance(env.last_reset_info, dict)
            self.assertEqual(len(result), 4)
            next_observation, reward, done, info = result
            self.assertTrue(env.observation_space.contains(next_observation))
            self.assertIsInstance(reward, float)
            self.assertTrue(done)
            self.assertFalse(info["terminated"])
            self.assertTrue(info["truncated"])
        finally:
            env.close()

    def test_rgb_array_render_contract(self) -> None:
        env = ArenaEnv(render_mode="rgb_array")
        try:
            env.reset(seed=21)
            frame = env.render()

            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (env.height, env.width, 3))
            self.assertEqual(frame.dtype, np.uint8)
        finally:
            env.close()

    def test_none_render_mode_is_headless(self) -> None:
        env = ArenaEnv(render_mode=None)
        try:
            env.reset(seed=21)
            self.assertIsNone(env.render())
        finally:
            env.close()

    def test_upgrade_choice_overlay_preserves_rgb_render_contract(self) -> None:
        env = ArenaEnv(render_mode="rgb_array", manual_choices=True)
        try:
            env.reset(seed=22)
            env._queue_choice("level_up")
            env._prepare_next_choice()
            frame = env.render()
            self.assertEqual(len(env.pending_choices), 3)
            self.assertEqual(frame.shape, (env.height, env.width, 3))
            renderer = env._renderer
            self.assertIsNotNone(renderer)
            for index, rect in enumerate(renderer.choice_rects()):
                self.assertEqual(renderer.choice_at_position(rect.center), index)
        finally:
            env.close()

    def test_visible_pause_control_has_a_click_target_and_overlay(self) -> None:
        env = ArenaEnv(render_mode=None)
        env.reset(seed=44)
        renderer = ArenaRenderer(env, mode="rgb_array")
        try:
            self.assertTrue(renderer.pause_at_position(renderer.pause_button_rect.center))
            self.assertFalse(renderer.pause_at_position((0, 0)))
            running = renderer.render(
                process_events=False,
                footer_text="P pause",
                paused=False,
            )
            paused = renderer.render(
                process_events=False,
                footer_text="Paused",
                paused=True,
            )
            self.assertEqual(running.shape, paused.shape)
            self.assertFalse(np.array_equal(running, paused))
        finally:
            renderer.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
