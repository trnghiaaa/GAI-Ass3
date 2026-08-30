"""Focused tests for the Part II arena mechanics."""

import math
import unittest

from arena.entities import Enemy, Spawner
from arena.environment import ArenaEnv, DIRECT_ACTIONS, ROTATION_ACTIONS


class ArenaEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ArenaEnv(control_style="direct")
        self.env.reset(seed=7)

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_creates_player_spawners_and_valid_observation(self) -> None:
        observation, info = self.env.reset(seed=7)

        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(observation.shape, (34,))
        self.assertEqual(self.env.player.health, self.env.player.max_health)
        self.assertEqual(info["phase"], 1)
        self.assertEqual(len(self.env.spawners), 2)

    def test_direct_movement_uses_continuous_coordinates(self) -> None:
        start_x = self.env.player.x

        self.env.step(DIRECT_ACTIONS["RIGHT"])

        self.assertGreater(self.env.player.x, start_x)
        self.assertAlmostEqual(
            self.env.player.x - start_x,
            float(self.env.player_cfg["direct_speed"]) / self.env.fps,
        )

    def test_rotation_controls_turn_and_thrust(self) -> None:
        env = ArenaEnv(control_style="rotation")
        try:
            env.reset(seed=7)
            initial_angle = env.player.angle
            env.step(ROTATION_ACTIONS["ROTATE_RIGHT"])
            self.assertGreater(env.player.angle % math.tau, initial_angle % math.tau)
            env.step(ROTATION_ACTIONS["THRUST"])
            self.assertGreater(math.hypot(env.player.vx, env.player.vy), 0.0)
        finally:
            env.close()

    def test_spawner_periodically_creates_an_enemy(self) -> None:
        self.env.spawners[0].spawn_cooldown_steps = 1

        _, _, _, _, info = self.env.step(DIRECT_ACTIONS["NOOP"])

        self.assertEqual(info["enemies_spawned"], 1)
        self.assertEqual(len(self.env.enemies), 1)

    def test_enemy_navigates_toward_player(self) -> None:
        enemy = Enemy(
            x=100.0,
            y=100.0,
            radius=15.0,
            entity_id=500,
            max_health=50.0,
            health=50.0,
            speed=72.0,
        )
        self.env.enemies = [enemy]
        before = math.hypot(enemy.x - self.env.player.x, enemy.y - self.env.player.y)

        self.env.step(DIRECT_ACTIONS["NOOP"])

        after = math.hypot(enemy.x - self.env.player.x, enemy.y - self.env.player.y)
        self.assertLess(after, before)

    def test_projectile_damages_and_destroys_enemy(self) -> None:
        enemy = Enemy(
            x=self.env.player.x + 80.0,
            y=self.env.player.y,
            radius=15.0,
            entity_id=501,
            max_health=25.0,
            health=25.0,
            speed=0.0,
        )
        self.env.enemies = [enemy]
        self.env.step(DIRECT_ACTIONS["SHOOT"])

        destroyed = 0
        damage_reward_seen = False
        for _ in range(12):
            _, _, _, _, info = self.env.step(DIRECT_ACTIONS["NOOP"])
            destroyed += info["enemies_destroyed"]
            damage_reward_seen = damage_reward_seen or (
                info["reward_breakdown"]["enemy_damage"] > 0
            )
            if destroyed:
                break

        self.assertEqual(destroyed, 1)
        self.assertTrue(damage_reward_seen)
        self.assertEqual(self.env.enemies, [])
        self.assertEqual(self.env.episode_stats["enemies_destroyed"], 1)
        self.assertEqual(self.env.episode_stats["projectile_hits"], 1)

    def test_destroying_last_spawner_advances_phase(self) -> None:
        spawner = Spawner(
            x=self.env.player.x + 80.0,
            y=self.env.player.y,
            radius=28.0,
            entity_id=502,
            max_health=25.0,
            health=25.0,
            spawn_cooldown_steps=999,
        )
        self.env.enemies = []
        self.env.spawners = [spawner]
        self.env.step(DIRECT_ACTIONS["SHOOT"])

        phase_advanced = False
        for _ in range(12):
            _, _, _, _, info = self.env.step(DIRECT_ACTIONS["NOOP"])
            phase_advanced = phase_advanced or info["phase_advanced"]
            if phase_advanced:
                break

        self.assertTrue(phase_advanced)
        self.assertEqual(self.env.phase, 2)
        self.assertEqual(self.env.spawners, [])
        self.assertGreater(self.env.phase_transition_steps, 0)

    def test_enemy_contact_can_end_episode(self) -> None:
        self.env.player.health = 10.0
        self.env.enemies = [
            Enemy(
                x=self.env.player.x,
                y=self.env.player.y,
                radius=15.0,
                entity_id=503,
                max_health=50.0,
                health=50.0,
                speed=0.0,
            )
        ]

        _, _, terminated, truncated, info = self.env.step(DIRECT_ACTIONS["NOOP"])

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["episode_end"], "player_destroyed")
        self.assertEqual(self.env.player.health, 0.0)

    def test_maximum_step_count_truncates_episode(self) -> None:
        env = ArenaEnv(
            control_style="direct",
            config_override={"simulation": {"max_steps": 2}},
        )
        try:
            env.reset(seed=1)
            _, _, terminated, truncated, _ = env.step(DIRECT_ACTIONS["NOOP"])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            _, _, terminated, truncated, info = env.step(DIRECT_ACTIONS["NOOP"])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["episode_end"], "time_limit")
        finally:
            env.close()

    def test_xp_unlocks_rapid_twin_laser_and_nova_weapon_tiers(self) -> None:
        expected = (
            (1, "Pulse Cannon", 1, "pulse"),
            (2, "Rapid Loader", 1, "rapid"),
            (3, "Twin Pulse", 2, "twin"),
            (4, "Laser Array", 1, "laser"),
            (5, "Nova Tri-Beam", 3, "nova"),
        )

        for level, name, shot_count, kind in expected:
            with self.subTest(level=level):
                self.env.player.level = level
                self.env.player.fire_cooldown_steps = 0
                self.env.projectiles = []
                fired = self.env._fire_projectile(auto_aim=False)
                self.assertEqual(self.env.weapon_profile()["name"], name)
                self.assertEqual(fired, shot_count)
                self.assertEqual(len(self.env.projectiles), shot_count)
                self.assertTrue(
                    all(projectile.weapon_kind == kind for projectile in self.env.projectiles)
                )

    def test_combat_xp_levels_up_without_changing_environment_reward(self) -> None:
        events = {
            "enemies_destroyed": 4,
            "spawners_destroyed": 0,
            "phase_advanced": False,
        }
        self.env._apply_combat_progression(events)

        self.assertEqual(events["xp_gained"], 40.0)
        self.assertEqual(events["levels_gained"], 1)
        self.assertEqual(events["upgrade_unlocked"], "Rapid Loader")
        self.assertEqual(self.env.player.level, 2)
        self.assertEqual(self.env.player.xp, 40.0)

        reward_events = {
            "damage_dealt_enemy": 0.0,
            "damage_dealt_spawner": 0.0,
            "enemies_destroyed": 0,
            "spawners_destroyed": 0,
            "phase_advanced": False,
            "damage_taken": 0.0,
            "spawner_progress": 0.0,
            "aim_improvement": 0.0,
            "shot_fired": False,
            "shot_alignment": 0.0,
            "xp_gained": 999.0,
        }
        _, breakdown = self.env._calculate_reward(reward_events, terminated=False)
        self.assertNotIn("xp", breakdown)


if __name__ == "__main__":
    unittest.main()
