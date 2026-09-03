"""Tests for the arena's normalized, fixed-size feature observation."""

import math
import unittest

import numpy as np

from arena.entities import DangerZone, Enemy, Projectile, Spawner
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

        self.assertEqual(observation.shape, (len(OBSERVATION_NAMES),))
        self.assertEqual(observation.ndim, 1)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(len(OBSERVATION_NAMES), 126)
        self.assertTrue(self.env.observation_space.contains(observation))

        # Removing every variable-length entity list must not alter the shape.
        self.env.enemies = []
        self.env.spawners = []
        self.env.projectiles = []
        self.assertEqual(
            self.env._get_observation().shape, (len(OBSERVATION_NAMES),)
        )

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
        self.assertAlmostEqual(observation[ObservationIndex.PHASE], 0.2)

    def test_explicit_aim_alignment_supports_rotation_learning(self) -> None:
        self.env.enemies = [
            Enemy(
                x=self.env.player.x + 100.0,
                y=self.env.player.y,
                radius=15.0,
                entity_id=903,
                max_health=50.0,
                health=50.0,
                speed=0.0,
            )
        ]
        self.env.player.angle = 0.0
        observation = self.env._get_observation()
        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_ENEMY_AIM_ALIGNMENT], 1.0
        )

        self.env.player.angle = math.pi
        observation = self.env._get_observation()
        self.assertAlmostEqual(
            observation[ObservationIndex.NEAREST_ENEMY_AIM_ALIGNMENT], -1.0
        )

    def test_active_target_exposes_signed_turn_direction(self) -> None:
        self.env.enemies = [
            Enemy(
                x=self.env.player.x + 100.0,
                y=self.env.player.y,
                radius=15.0,
                entity_id=904,
                max_health=50.0,
                health=50.0,
                speed=0.0,
            )
        ]
        self.env.spawners = []
        self.env.player.angle = -math.pi / 2
        observation = self.env._get_observation()
        self.assertAlmostEqual(
            observation[ObservationIndex.ACTIVE_TARGET_AIM_ALIGNMENT], 0.0, places=6
        )
        self.assertGreater(
            observation[ObservationIndex.ACTIVE_TARGET_TURN_DIRECTION], 0.0
        )
        self.assertEqual(observation[ObservationIndex.ACTIVE_TARGET_IS_SPAWNER], 0.0)

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

    def test_combat_level_and_weapon_upgrade_state_are_encoded(self) -> None:
        self.env.player.level = 4
        self.env.player.xp = self.env.xp_threshold_for_level(4) + 10.0
        self.env.upgrade_stacks.update(
            {"laser": 1, "fire_rate": 2, "damage": 2, "range": 1,
             "piercing": 1, "splash": 2, "engine": 1, "shield": 1}
        )
        self.env.player.max_health += 25.0
        self.env.support_drone_phase = self.env.phase
        self.env.nova_bomb_armed = True

        observation = self.env._get_observation()

        self.assertAlmostEqual(observation[ObservationIndex.PLAYER_LEVEL], 3 / 24)
        self.assertGreater(observation[ObservationIndex.XP_PROGRESS], 0.0)
        self.assertGreater(observation[ObservationIndex.WEAPON_FIRE_RATE], 0.0)
        self.assertEqual(observation[ObservationIndex.WEAPON_IS_LASER], 1.0)
        self.assertGreater(observation[ObservationIndex.MAX_HEALTH_BONUS], 0.0)
        self.assertGreater(observation[ObservationIndex.DAMAGE_RESISTANCE], 0.0)
        self.assertGreater(observation[ObservationIndex.PROJECTILE_RANGE], 0.0)
        self.assertGreater(observation[ObservationIndex.PROJECTILE_PIERCE], 0.0)
        self.assertGreater(observation[ObservationIndex.PROJECTILE_SPLASH], 0.0)
        self.assertGreater(observation[ObservationIndex.ENGINE_POWER], 0.0)
        self.assertEqual(observation[ObservationIndex.SUPPORT_DRONE_ACTIVE], 1.0)
        self.assertEqual(observation[ObservationIndex.NOVA_BOMB_ARMED], 1.0)

    def test_multi_lane_boss_hazards_expose_combined_and_secondary_escape(self) -> None:
        self.env.danger_zones = [
            DangerZone(
                kind="line",
                x=self.env.player.x,
                y=self.env.player.y,
                angle=0.0,
                half_width=30,
                half_length=500,
                telegraph_steps=30,
                maximum_telegraph_steps=60,
                active_steps=15,
                attack_id=5,
            ),
            DangerZone(
                kind="line",
                x=self.env.player.x,
                y=self.env.player.y,
                angle=math.pi / 2,
                half_width=30,
                half_length=500,
                telegraph_steps=30,
                maximum_telegraph_steps=60,
                active_steps=15,
                attack_id=5,
            ),
        ]

        observation = self.env._get_observation()

        self.assertAlmostEqual(observation[ObservationIndex.HAZARD_COUNT], 0.5)
        self.assertNotEqual(
            (
                observation[ObservationIndex.HAZARD_COMBINED_ESCAPE_X],
                observation[ObservationIndex.HAZARD_COMBINED_ESCAPE_Y],
            ),
            (0.0, 0.0),
        )
        self.assertGreater(
            observation[ObservationIndex.SECONDARY_HAZARD_DISTANCE_TO_SAFETY],
            0.0,
        )

    def test_hazard_escape_vector_never_points_through_top_boundary(self) -> None:
        self.env.player.y = self.env.playfield_top + self.env.player.radius
        self.env.danger_zones = [
            DangerZone(
                kind="line",
                x=self.env.player.x,
                y=self.env.player.y + 20.0,
                angle=0.0,
                half_width=45.0,
                half_length=500.0,
                telegraph_steps=30,
                maximum_telegraph_steps=60,
                active_steps=15,
                attack_id=77,
            )
        ]

        observation = self.env._get_observation()

        self.assertGreater(observation[ObservationIndex.HAZARD_ESCAPE_Y], 0.0)
        self.assertGreater(
            observation[ObservationIndex.HAZARD_COMBINED_ESCAPE_Y], 0.0
        )

    def test_sentry_volley_exposes_multi_missile_escape_and_rotation_turn(self) -> None:
        self.env.player.x = self.env.width / 2.0
        self.env.player.y = self.env.playfield_top + self.env.player.radius + 2.0
        self.env.player.angle = 0.0
        self.env.projectiles = [
            Projectile(
                x=self.env.player.x - 180.0,
                y=self.env.player.y,
                radius=7.0,
                entity_id=8101,
                vx=170.0,
                vy=0.0,
                damage=10.0,
                lifetime_steps=180,
                weapon_kind="enemy_missile",
                owner="enemy",
            ),
            Projectile(
                x=self.env.player.x + 210.0,
                y=self.env.player.y,
                radius=7.0,
                entity_id=8102,
                vx=-170.0,
                vy=0.0,
                damage=10.0,
                lifetime_steps=180,
                weapon_kind="enemy_missile",
                owner="enemy",
            ),
        ]

        observation = self.env._get_observation()

        self.assertEqual(observation[ObservationIndex.MISSILE_COUNT], 0.5)
        self.assertGreater(observation[ObservationIndex.MISSILE_PRESSURE], 0.0)
        self.assertGreater(
            observation[ObservationIndex.MISSILE_COMBINED_ESCAPE_Y], 0.0
        )
        self.assertGreater(observation[ObservationIndex.MISSILE_ESCAPE_TURN], 0.0)
        self.assertLess(observation[ObservationIndex.SECOND_MISSILE_DISTANCE], 1.0)
        self.assertGreater(observation[ObservationIndex.SECOND_MISSILE_RISK], 0.0)
        self.assertGreater(observation[ObservationIndex.SAFETY_URGENCY], 0.0)
        self.assertGreater(observation[ObservationIndex.SAFETY_ESCAPE_Y], 0.0)
        self.assertGreater(observation[ObservationIndex.SAFETY_ESCAPE_TURN], 0.0)
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_unified_safety_signal_reconciles_crowd_hazard_and_wall(self) -> None:
        self.env.player.x = self.env.player.radius + 2.0
        self.env.player.y = self.env.playfield_top + self.env.player.radius + 2.0
        self.env.player.angle = math.pi
        self.env.enemies = [
            Enemy(
                x=self.env.player.x + 35.0,
                y=self.env.player.y + 20.0,
                radius=15.0,
                entity_id=8401,
                max_health=50.0,
                health=50.0,
                speed=72.0,
            )
        ]
        self.env.danger_zones = [
            DangerZone(
                kind="circle",
                x=self.env.player.x,
                y=self.env.player.y,
                radius=65.0,
                telegraph_steps=20,
                maximum_telegraph_steps=60,
                active_steps=15,
                attack_id=8402,
            )
        ]

        observation = self.env._get_observation()

        self.assertGreater(observation[ObservationIndex.SAFETY_ESCAPE_X], 0.0)
        self.assertGreater(observation[ObservationIndex.SAFETY_ESCAPE_Y], 0.0)
        self.assertGreater(observation[ObservationIndex.SAFETY_URGENCY], 0.8)
        self.assertNotEqual(observation[ObservationIndex.SAFETY_ESCAPE_TURN], 0.0)

    def test_new_upgrade_state_is_explicitly_encoded(self) -> None:
        self.env.upgrade_stacks.update({"critical": 3, "leech": 2, "riftbreaker": 4})
        observation = self.env._get_observation()
        self.assertAlmostEqual(observation[ObservationIndex.CRITICAL_CHANCE], 0.6)
        self.assertAlmostEqual(observation[ObservationIndex.LEECH_STRENGTH], 0.5)
        self.assertAlmostEqual(observation[ObservationIndex.RIFTBREAKER_POWER], 4 / 6)


if __name__ == "__main__":
    unittest.main()
