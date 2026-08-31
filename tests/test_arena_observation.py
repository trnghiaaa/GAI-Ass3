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

        self.assertEqual(observation.shape, (47,))
        self.assertEqual(observation.ndim, 1)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(len(OBSERVATION_NAMES), 47)
        self.assertTrue(self.env.observation_space.contains(observation))

        # Removing every variable-length entity list must not alter the shape.
        self.env.enemies = []
        self.env.spawners = []
        self.env.projectiles = []
        self.assertEqual(self.env._get_observation().shape, (47,))

    def test_boundary_and_emergency_features_are_explicit(self) -> None:
        self.env.player.x = self.env.player.radius
        self.env.player.y = self.env.height / 2.0
        self.env.player.angle = math.pi
        self.env.enemies = [
            Enemy(
                x=self.env.player.x + 45.0,
                y=self.env.player.y,
                radius=15.0,
                entity_id=999,
                max_health=50.0,
                health=50.0,
                speed=68.0,
                vx=-68.0,
            )
        ]

        observation = self.env._get_observation()

        self.assertEqual(observation[ObservationIndex.WALL_LEFT_CLEARANCE], 0.0)
        self.assertEqual(observation[ObservationIndex.WALL_RIGHT_CLEARANCE], 1.0)
        self.assertLess(observation[ObservationIndex.BOUNDARY_INWARD_ALIGNMENT], 0.0)
        self.assertGreater(observation[ObservationIndex.EMERGENCY_THREAT], 0.5)

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
        self.env.player.angular_velocity = math.radians(110.0)

        observation = self.env._get_observation()

        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_X], 0.5)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_Y], -0.5)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_VELOCITY_X], 250 / 280)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_VELOCITY_Y], -125 / 280)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_HEADING_COS], -1.0)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_HEADING_SIN], 0.0, places=6)
        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_HEALTH], 0.4)
        self.assertAlmostEqual(
            observation[ObservationIndex.PLAYER_ANGULAR_VELOCITY], 0.5
        )

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
        self.env.player.angle = 0.0

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
        self.assertAlmostEqual(
            observation[ObservationIndex.ENEMY_FORWARD_ALIGNMENT], 0.6
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.ENEMY_TURN_DIRECTION], 0.8
        )

    def test_closing_speed_and_body_velocity_are_encoded(self) -> None:
        self.env.player.angle = 0.0
        self.env.player.vx = 70.0
        self.env.player.vy = 28.0
        enemy = Enemy(
            x=self.env.player.x + 100.0,
            y=self.env.player.y,
            radius=15.0,
            entity_id=903,
            max_health=50.0,
            health=50.0,
            speed=68.0,
            vx=-68.0,
            vy=0.0,
        )
        self.env.enemies = [enemy]

        observation = self.env._get_observation()

        expected_closing = (68.0 + 70.0) / (68.0 + 280.0)
        self.assertAlmostEqual(
            observation[ObservationIndex.ENEMY_CLOSING_SPEED], expected_closing
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.PLAYER_FORWARD_VELOCITY], 70.0 / 280.0
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.PLAYER_LATERAL_VELOCITY], 28.0 / 280.0
        )

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

    def test_second_enemy_and_crowd_escape_are_encoded(self) -> None:
        self.env.player.angle = 0.0
        right_enemy = Enemy(
            x=self.env.player.x + 100.0,
            y=self.env.player.y,
            radius=15.0,
            entity_id=904,
            max_health=50.0,
            health=50.0,
            speed=0.0,
        )
        lower_enemy = Enemy(
            x=self.env.player.x,
            y=self.env.player.y + 120.0,
            radius=15.0,
            entity_id=905,
            max_health=50.0,
            health=50.0,
            speed=0.0,
        )
        self.env.enemies = [lower_enemy, right_enemy]

        observation = self.env._get_observation()

        self.assertAlmostEqual(
            observation[ObservationIndex.SECOND_ENEMY_DIRECTION_X], 0.0
        )
        self.assertAlmostEqual(
            observation[ObservationIndex.SECOND_ENEMY_DIRECTION_Y], 1.0
        )
        self.assertGreater(observation[ObservationIndex.CROWD_PRESSURE], 0.0)
        self.assertAlmostEqual(
            observation[ObservationIndex.DANGER_ENEMY_COUNT], 2.0 / 14.0
        )
        self.assertLess(observation[ObservationIndex.CROWD_ESCAPE_DIRECTION_X], 0.0)
        self.assertLess(observation[ObservationIndex.CROWD_ESCAPE_DIRECTION_Y], 0.0)
        self.assertLess(observation[ObservationIndex.ESCAPE_FORWARD_ALIGNMENT], 0.0)

    def test_escape_corridor_does_not_point_out_through_a_nearby_wall(self) -> None:
        self.env.player.x = self.env.player.radius + 8.0
        self.env.player.y = self.env.height / 2.0
        self.env.enemies = [
            Enemy(
                x=self.env.player.x + 100.0,
                y=self.env.player.y,
                radius=15.0,
                entity_id=906,
                max_health=50.0,
                health=50.0,
                speed=0.0,
            )
        ]

        crowd_features = self.env._crowd_threat_features()
        safe_x, safe_y = self.env._safe_escape_direction(
            self.env.enemies,
            float(self.env.reward_cfg["crowd_danger_distance"]),
            crowd_features[4],
            crowd_features[5],
        )
        projected_x = self.env.player.x + 160.0 * safe_x
        projected_y = self.env.player.y + 160.0 * safe_y

        self.assertGreaterEqual(projected_x, self.env.player.radius)
        self.assertLessEqual(projected_x, self.env.width - self.env.player.radius)
        self.assertGreaterEqual(projected_y, self.env.playfield_top + self.env.player.radius)
        self.assertLessEqual(projected_y, self.env.height - self.env.player.radius)

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

        for index in (
            ObservationIndex.ENEMY_FORWARD_ALIGNMENT,
            ObservationIndex.ENEMY_TURN_DIRECTION,
            ObservationIndex.SPAWNER_FORWARD_ALIGNMENT,
            ObservationIndex.SPAWNER_TURN_DIRECTION,
            ObservationIndex.ENEMY_CLOSING_SPEED,
            ObservationIndex.SECOND_ENEMY_DIRECTION_X,
            ObservationIndex.SECOND_ENEMY_DIRECTION_Y,
            ObservationIndex.SECOND_ENEMY_CLOSING_SPEED,
            ObservationIndex.CROWD_ESCAPE_DIRECTION_X,
            ObservationIndex.CROWD_ESCAPE_DIRECTION_Y,
            ObservationIndex.CROWD_PRESSURE,
            ObservationIndex.DANGER_ENEMY_COUNT,
            ObservationIndex.ESCAPE_FORWARD_ALIGNMENT,
            ObservationIndex.ESCAPE_TURN_DIRECTION,
            ObservationIndex.SAFE_CORRIDOR_ALIGNMENT,
            ObservationIndex.SAFE_CORRIDOR_TURN_DIRECTION,
            ObservationIndex.EMERGENCY_THREAT,
        ):
            self.assertEqual(observation[index], 0.0)
        self.assertEqual(observation[ObservationIndex.SECOND_ENEMY_DISTANCE], 1.0)

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
