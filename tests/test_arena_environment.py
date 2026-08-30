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
        self.assertEqual(observation.shape, (43,))
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
        self.env.player.angle = 0.0
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

    def test_direct_shooting_uses_nearest_target_assist(self) -> None:
        self.env.enemies = [
            Enemy(
                x=self.env.player.x + 100.0,
                y=self.env.player.y,
                radius=15.0,
                entity_id=991,
                max_health=50.0,
                health=50.0,
                speed=0.0,
            )
        ]
        self.env.player.angle = -math.pi / 2.0

        _, _, _, _, info = self.env.step(DIRECT_ACTIONS["SHOOT"])

        self.assertTrue(info["shot_fired"])
        self.assertAlmostEqual(self.env.player.angle, 0.0)
        self.assertGreater(self.env.projectiles[0].vx, 0.0)
        self.assertAlmostEqual(self.env.projectiles[0].vy, 0.0)

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
        self.env.player.angle = 0.0
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
            self.assertEqual(info["episode_end"], "safety_limit")
        finally:
            env.close()

    def test_each_phase_has_its_own_deadline(self) -> None:
        env = ArenaEnv(
            control_style="direct",
            config_override={
                "simulation": {
                    "phase_time_limit_seconds": 2 / 60,
                    "max_steps": 20,
                }
            },
        )
        try:
            env.reset(seed=2)
            env.step(DIRECT_ACTIONS["NOOP"])
            _, _, terminated, truncated, info = env.step(DIRECT_ACTIONS["NOOP"])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["episode_end"], "phase_timeout")
            self.assertEqual(info["phase_step"], 2)
            self.assertEqual(info["phase_time_remaining"], 0.0)
        finally:
            env.close()

    def test_phase_clear_resets_timer_and_removes_leftover_combatants(self) -> None:
        enemy = Enemy(
            x=100.0,
            y=100.0,
            radius=15.0,
            entity_id=992,
            max_health=50.0,
            health=50.0,
            speed=0.0,
        )
        self.env.enemies = [enemy]
        self.env.spawners = []
        self.env._fire_projectile(auto_aim=False)
        self.env.phase_step_count = 120
        xp_before = self.env.player.xp

        _, _, _, _, info = self.env.step(DIRECT_ACTIONS["NOOP"])

        self.assertTrue(info["phase_advanced"])
        self.assertEqual(info["enemies_dispersed"], 1)
        self.assertGreaterEqual(info["projectiles_cleared"], 1)
        self.assertEqual(self.env.enemies, [])
        self.assertEqual(self.env.projectiles, [])
        self.assertEqual(self.env.phase_step_count, 0)
        self.assertEqual(
            self.env.player.xp,
            xp_before + float(self.env.progression_cfg["phase_xp"]),
        )
        self.assertEqual(self.env.episode_stats["enemies_destroyed"], 0)

        transition_before = self.env.phase_transition_steps
        self.env.step(DIRECT_ACTIONS["NOOP"])
        self.assertEqual(self.env.phase_step_count, 0)
        self.assertEqual(self.env.phase_transition_steps, transition_before - 1)

    def test_selected_upgrades_compose_a_diverse_weapon_build(self) -> None:
        self.env.upgrade_stacks.update(
            {"multishot": 3, "laser": 1, "damage": 2, "range": 2,
             "piercing": 1, "splash": 2, "fire_rate": 2}
        )
        profile = self.env.weapon_profile()
        fired = self.env._fire_projectile(auto_aim=False)

        self.assertEqual(profile["name"], "Prism Laser")
        self.assertEqual(profile["kind"], "laser")
        self.assertEqual(fired, 4)
        self.assertGreater(profile["damage_multiplier"], 1.0)
        self.assertGreater(profile["lifetime_multiplier"], 1.0)
        self.assertEqual(profile["pierces"], 1)
        self.assertGreater(profile["splash_radius"], 0.0)
        self.assertTrue(all(item.weapon_kind == "laser" for item in self.env.projectiles))

    def test_combat_xp_levels_up_without_changing_environment_reward(self) -> None:
        events = {
            "enemies_destroyed": 4,
            "spawners_destroyed": 0,
            "phase_advanced": False,
        }
        self.env._apply_combat_progression(events)
        self.env._prepare_next_choice(events)

        self.assertEqual(events["xp_gained"], 40.0)
        self.assertEqual(events["levels_gained"], 1)
        self.assertIsNotNone(events["upgrade_unlocked"])
        self.assertIsNotNone(events["choice_selected"])
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

    def test_manual_level_up_pauses_until_one_of_three_cards_is_chosen(self) -> None:
        env = ArenaEnv(control_style="direct", manual_choices=True)
        try:
            env.reset(seed=8)
            events = {"choice_offered": None, "choice_selected": None, "upgrade_unlocked": None}
            env._queue_choice("level_up")
            env._prepare_next_choice(events)
            self.assertEqual(env.pending_choice_kind, "level_up")
            self.assertEqual(len(env.pending_choices), 3)
            with self.assertRaises(RuntimeError):
                env.step(DIRECT_ACTIONS["NOOP"])
            selected = env.choose_pending_choice(0)
            self.assertEqual(env.pending_choice_kind, None)
            self.assertEqual(env.last_upgrade_name, selected["name"])
        finally:
            env.close()

    def test_phase_reward_can_repair_arm_bomb_or_activate_wingman(self) -> None:
        env = ArenaEnv(control_style="direct", manual_choices=True)
        try:
            env.reset(seed=9)
            env.player.health = 20.0
            env._queue_choice("phase_reward")
            env._prepare_next_choice()
            by_id = {item["id"]: index for index, item in enumerate(env.pending_choices)}
            env.choose_pending_choice(by_id["repair_cache"])
            self.assertGreater(env.player.health, 20.0)

            env._queue_choice("phase_reward")
            env._prepare_next_choice()
            by_id = {item["id"]: index for index, item in enumerate(env.pending_choices)}
            env.choose_pending_choice(by_id["nova_bomb"])
            self.assertTrue(env.nova_bomb_armed)

            env._queue_choice("phase_reward")
            env._prepare_next_choice()
            by_id = {item["id"]: index for index, item in enumerate(env.pending_choices)}
            env.choose_pending_choice(by_id["wingman"])
            self.assertTrue(env.support_drone_active)
        finally:
            env.close()

    def test_manual_phase_clear_opens_support_draft_before_next_assault(self) -> None:
        env = ArenaEnv(control_style="direct", manual_choices=True)
        try:
            env.reset(seed=10)
            env.spawners = []
            _, _, _, _, info = env.step(DIRECT_ACTIONS["NOOP"])
            self.assertTrue(info["phase_advanced"])
            self.assertEqual(env.phase, 2)
            self.assertEqual(env.pending_choice_kind, "phase_reward")
            self.assertEqual(len(env.pending_choices), 3)
            transition_before = env.phase_transition_steps
            with self.assertRaises(RuntimeError):
                env.step(DIRECT_ACTIONS["NOOP"])
            self.assertEqual(env.phase_transition_steps, transition_before)
        finally:
            env.close()

    def test_every_third_phase_spawns_a_boss_rift_and_elite_minions(self) -> None:
        self.env.phase = 3
        self.env.spawners = []
        self.env._spawn_phase_spawners()
        self.assertEqual(len(self.env.spawners), 1)
        boss = self.env.spawners[0]
        self.assertTrue(boss.is_boss)
        self.assertGreater(boss.radius, float(self.env.spawner_cfg["radius"]))
        self.env._spawn_enemy(boss)
        self.assertTrue(self.env.enemies[-1].is_elite)

    def test_hull_shield_and_engine_choices_have_real_gameplay_effects(self) -> None:
        base_speed = float(self.env.player_cfg["direct_speed"])
        self.env.upgrade_stacks["engine"] = 2
        before_x = self.env.player.x
        self.env.step(DIRECT_ACTIONS["RIGHT"])
        self.assertAlmostEqual(
            self.env.player.x - before_x, base_speed * 1.2 / self.env.fps
        )

        self.env.upgrade_stacks["shield"] = 2
        self.env.player.health = self.env.player.max_health
        self.env.enemies = [
            Enemy(
                x=self.env.player.x,
                y=self.env.player.y,
                radius=15,
                entity_id=990,
                max_health=50,
                health=50,
                speed=0,
            )
        ]
        _, _, _, _, info = self.env.step(DIRECT_ACTIONS["NOOP"])
        unshielded = float(self.env.enemy_cfg["contact_damage"])
        self.assertAlmostEqual(info["damage_taken"], unshielded * 0.76)

    def test_nova_bomb_damages_a_new_phase_without_changing_phase_rule(self) -> None:
        events = {
            "damage_dealt_enemy": 0.0,
            "damage_dealt_spawner": 0.0,
            "projectile_hits": 0,
            "drone_hits": 0,
            "impacts": [],
            "enemies_destroyed": 0,
            "spawners_destroyed": 0,
            "nova_bomb_detonated": False,
        }
        self.env.phase = 2
        self.env.spawners = []
        self.env.nova_bomb_armed = True
        self.env._spawn_phase_spawners(events)

        self.assertTrue(events["nova_bomb_detonated"])
        self.assertFalse(self.env.nova_bomb_armed)
        self.assertEqual(len(self.env.spawners), 3)
        self.assertTrue(all(item.health < item.max_health for item in self.env.spawners))


if __name__ == "__main__":
    unittest.main()
