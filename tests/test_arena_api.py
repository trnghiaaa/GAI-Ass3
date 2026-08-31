"""Contract tests for the modern and rubric-compatible arena APIs."""

import unittest

from gymnasium.utils.env_checker import check_env
import numpy as np

from arena.environment import ArenaEnv, DIRECT_ACTIONS
from arena.legacy_api import LegacyArenaEnv


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

    def test_playback_rate_is_slower_than_simulation_rate(self) -> None:
        env = ArenaEnv()
        try:
            self.assertEqual(env.fps, 60)
            self.assertEqual(env.playback_speed, 0.75)
            self.assertEqual(env.render_fps, 45)
            self.assertEqual(env.metadata["render_fps"], 45)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
