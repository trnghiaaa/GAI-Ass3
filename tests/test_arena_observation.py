"""Tests for the arena's normalized, fixed-size feature observation."""

import math
import unittest

import numpy as np

from arena.entities import Enemy, Spawner
from arena.environment import (
    ArenaEnv,
    DIRECT_ACTIONS,
    OBSERVATION_NAMES,
    ObservationIndex,
)


class ArenaObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ArenaEnv(control_style="direct")
        self.env.reset(seed=13)

    def tearDown(self) -> None:
        self.env.close()

    def test_observation_is_fixed_numeric_vector_not_pixels(self) -> None:
        observation = self.env._get_observation()

        self.assertEqual(observation.shape, (20,))
        self.assertEqual(observation.ndim, 1)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(len(OBSERVATION_NAMES), 20)
        self.assertTrue(self.env.observation_space.contains(observation))

        # Removing every variable-length entity list must not alter the shape.
        self.env.enemies = []
        self.env.spawners = []
        self.env.projectiles = []
        self.assertEqual(self.env._get_observation().shape, (20,))

    def test_vectors_remain_in_bounds_during_both_control_styles(self) -> None:
        for style in ("direct", "rotation"):
            env = ArenaEnv(control_style=style)
            try:
                observation, _ = env.reset(seed=13)
                self.assertTrue(env.observation_space.contains(observation))
                for _ in range(120):
                    action = env.action_space.sample()
                    observation, _, terminated, truncated, _ = env.step(action)
                    self.assertTrue(env.observation_space.contains(observation))
                    if terminated or truncated:
                        observation, _ = env.reset()
                        self.assertTrue(env.observation_space.contains(observation))
            finally:
                env.close()

    def test_player_features_encode_position_velocity_orientation_and_health(self) -> None:
        self.env.player.x = self.env.width * 0.75
        playfield_height = self.env.height - self.env.playfield_top
        self.env.player.y = self.env.playfield_top + playfield_height * 0.25
        self.env.player.vx = float(self.env.player_cfg["direct_speed"])
        self.env.player.vy = -float(self.env.player_cfg["direct_speed"]) / 2.0
        self.env.player.angle = math.pi
        self.env.player.health = self.env.player.max_health * 0.4

        observation = self.env._get_observation()

        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_X], 0.5)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_Y], -0.5)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_VELOCITY_X], 250 / 280)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_VELOCITY_Y], -125 / 280)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_HEADING_COS], -1.0)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_HEADING_SIN], 0.0, places=6)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_HEALTH], 0.4)

    def test_nearest_enemy_uses_unit_relative_direction_and_distance(self) -> None:
        near_enemy = Enemy(
            x=self.env.player.x + 30.0,
            y=self.env.player.y + 40.0,
            radius=15.0,
            entity_id=900,
            max_health=50.0,
            health=25.0,
            speed=0.0,
        )
        far_enemy = Enemy(
            x=self.env.player.x - 200.0,
            y=self.env.player.y,
            radius=15.0,
            entity_id=901,
            max_health=50.0,
            health=50.0,
            speed=0.0,
        )
        self.env.enemies = [far_enemy, near_enemy]

        observation = self.env._get_observation()
        diagonal = math.hypot(self.env.width, self.env.height - self.env.playfield_top)

        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_ENEMY_DIRECTION_X], 0.6
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_ENEMY_DIRECTION_Y], 0.8
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_ENEMY_DISTANCE], 50.0 / diagonal
        )
        self.assertAlmostEqual(observation[ObservationIndex.NEAREST_ENEMY_HEALTH], 0.5)

    def test_nearest_spawner_and_phase_are_encoded(self) -> None:
        self.env.spawners = [
            Spawner(
                x=self.env.player.x - 60.0,
                y=self.env.player.y,
                radius=28.0,
                entity_id=902,
                max_health=100.0,
                health=75.0,
                spawn_cooldown_steps=100,
            )
        ]
        self.env.phase = 4

        observation = self.env._get_observation()

        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_SPAWNER_DIRECTION_X], -1.0
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_SPAWNER_DIRECTION_Y], 0.0
        )
        self.assertGreater(observation[ObservationIndex.NEAREST_SPAWNER_DISTANCE], 0.0)
        self.assertAlmostEqual(observation[ObservationIndex.NEAREST_SPAWNER_HEALTH], 0.75)
        self.assertAlmostEqual(observation[ObservationIndex.PHASE], 0.4)

    def test_missing_target_sentinel_is_unambiguous(self) -> None:
        self.env.enemies = []
        self.env.spawners = []

        observation = self.env._get_observation()

        for start in (
            ObservationIndex.NEAREST_ENEMY_DIRECTION_X,
            ObservationIndex.NEAREST_SPAWNER_DIRECTION_X,
        ):
            self.assertEqual(observation[start], 0.0)
            self.assertEqual(observation[start + 1], 0.0)
            self.assertEqual(observation[start + 2], 1.0)
            self.assertEqual(observation[start + 3], 0.0)

    def test_named_observation_view_matches_vector_indices(self) -> None:
        observation = self.env._get_observation()
        named = self.env.observation_as_dict(observation)

        self.assertEqual(tuple(named), OBSERVATION_NAMES)
        self.assertEqual(named["player_health"], observation[ObservationIndex.PLAYER_HEALTH])
        self.assertEqual(named["phase"], observation[ObservationIndex.PHASE])

        with self.assertRaises(ValueError):
            self.env.observation_as_dict(np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
