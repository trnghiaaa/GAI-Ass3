"""Schema-6 difficulty, fairness, observation and transfer regressions."""
from collections import defaultdict
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3 import DQN
import torch

from arena.entities import DangerZone, Enemy
from arena.environment import ArenaEnv, ObservationIndex as I
from arena.train import transfer_prefix_policy, restrict_to_appended_inputs
from gymnasium.spaces import Box
from stable_baselines3.common.logger import configure
from arena.cooldown import CooldownAwareDQN, load_dqn
from arena.benchmark import evaluate_model


class ThreatTests(unittest.TestCase):
    def setUp(self):
        self.env = ArenaEnv(control_style='rotation')
        self.env.reset(seed=81)

    def tearDown(self):
        self.env.close()

    def boss(self, phase=3):
        self.env.phase = phase
        self.env.phase_max_steps = self.env._phase_step_budget()
        self.env.spawners.clear()
        self.env._spawn_phase_spawners()
        return self.env.spawners[0]

    def test_upgrades_arrive_faster_but_late_thresholds_still_scale(self):
        self.assertEqual(self.env.xp_threshold_for_level(2), 52)
        self.assertGreater(self.env.xp_threshold_for_level(5), 52 * 4**1.8)
        self.assertEqual(self.env.progression_cfg["enemy_xp"], 7)
        phase_one_objective_xp = (
            2 * self.env.progression_cfg["spawner_xp"]
            + self.env.progression_cfg["phase_xp"]
        )
        self.assertGreaterEqual(phase_one_objective_xp, self.env.xp_threshold_for_level(2))

    def test_rift_count_ramps_every_two_phases_instead_of_spiking(self):
        expected_counts = {1: 2, 2: 2, 4: 3, 5: 4, 7: 5}
        for phase, expected in expected_counts.items():
            self.env.phase = phase
            self.env.spawners.clear()
            self.env.enemies.clear()
            self.env._spawn_phase_spawners()
            self.assertEqual(len(self.env.spawners), expected, msg=f"phase {phase}")

    def test_difficulty_still_rises_but_respects_fairness_caps(self):
        base_health = float(self.env.enemy_cfg["max_health"])
        previous_health = 0.0
        previous_interval = float("inf")
        for phase in (1, 5, 10, 20):
            self.env.phase = phase
            self.env.spawners.clear()
            self.env.enemies.clear()
            self.env._spawn_phase_spawners()
            self.env._spawn_enemy(self.env.spawners[0])
            enemy = self.env.enemies[-1]
            interval = self.env._spawn_interval_steps(self.env.spawners[0])
            self.assertGreater(enemy.max_health, previous_health)
            self.assertLessEqual(interval, previous_interval)
            self.assertLessEqual(
                enemy.speed,
                float(self.env.enemy_cfg["speed"])
                * float(self.env.phase_cfg["enemy_speed_max_multiplier"]),
            )
            self.assertLessEqual(
                self.env.maximum_active_enemies(),
                int(self.env.phase_cfg["maximum_enemy_absolute"]),
            )
            self.assertGreaterEqual(
                interval,
                int(float(self.env.phase_cfg["minimum_spawn_interval_seconds"]) * self.env.fps),
            )
            previous_health = enemy.max_health
            previous_interval = interval
        self.assertLess(previous_health, base_health * 3.3)

    def test_adaptive_pressure_adds_activity_then_releases_cleanly(self):
        self.env.phase = 2
        self.env.phase_transition_steps = 0
        self.env.phase_step_count = 0
        self.env.phase_max_steps = self.env._phase_step_budget()
        self.env.player.health = self.env.player.max_health

        pressure = self.env.adaptive_pressure()
        surged_limit = self.env.maximum_active_enemies()
        surged_interval = self.env._spawn_interval_steps(self.env.spawners[0])
        self.assertEqual(pressure, 1.0)
        self.assertEqual(surged_limit, 19)

        self.env.player.health = self.env.player.max_health * 0.5
        self.assertEqual(self.env.adaptive_pressure(), 0.0)
        self.assertEqual(self.env.maximum_active_enemies(), 17)
        self.assertGreater(self.env._spawn_interval_steps(self.env.spawners[0]), surged_interval)

        self.env.player.health = self.env.player.max_health
        self.env.phase_step_count = round(self.env.phase_max_steps * 0.8)
        self.assertEqual(self.env.adaptive_pressure(), 0.0)

    def test_adaptive_pressure_never_changes_boss_encounters(self):
        self.boss(3)
        self.env.phase_transition_steps = 0
        self.env.phase_step_count = 0
        self.env.player.health = self.env.player.max_health
        self.assertEqual(self.env.adaptive_pressure(), 0.0)
        self.assertEqual(self.env.maximum_active_enemies(), 7)

    def test_shield_absorbs_then_overflows_without_regeneration(self):
        boss = self.boss()
        shield = boss.shield
        health = boss.health
        events = defaultdict(float, impacts=[])
        self.env._damage_target(boss, shield + 10, events, set(), set(), source='player', count_hit=True)
        self.assertEqual(boss.shield, 0)
        self.assertAlmostEqual(boss.health, health - 10)
        self.assertAlmostEqual(events['damage_dealt_spawner'], shield + 10)

    def test_summons_are_finite_and_have_a_cooldown(self):
        boss = self.boss(9)
        self.env.player.x = self.env.width - boss.x
        self.env.player.y = self.env.height - boss.y + self.env.playfield_top
        boss.health = boss.max_health * 0.2
        events = defaultdict(int)
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 1)
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 1)
        boss.summon_cooldown_steps = 0
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 2)
        # Only two summoned hunters may be active simultaneously. Defeating
        # one frees a slot for the final member of the finite three-summon budget.
        self.env.enemies.pop()
        boss.summon_cooldown_steps = 0
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 3)
        self.env.enemies.clear()
        boss.summon_cooldown_steps = 0
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 3)

    def test_first_boss_is_fairer_then_reinforcement_budget_scales_gradually(self):
        first = self.boss(3)
        first_effective_health = first.health + first.shield
        first_spawn_interval = self.env._spawn_interval_steps(first)
        self.env._spawn_enemy(first)
        first_minion_health = self.env.enemies[-1].max_health
        self.assertEqual(self.env.boss_summon_limit_for_phase(), 1)
        self.assertEqual(self.env.boss_active_summon_limit_for_phase(), 1)
        self.assertEqual(self.env.maximum_active_enemies(), 7)
        self.assertEqual(self.env.phase_max_steps, 90 * self.env.fps)

        second = self.boss(6)
        second_spawn_interval = self.env._spawn_interval_steps(second)
        self.env._spawn_enemy(second)
        self.assertGreater(second.health + second.shield, first_effective_health)
        self.assertGreater(first_spawn_interval, second_spawn_interval)
        self.assertGreater(self.env.enemies[-1].max_health, first_minion_health)
        self.assertEqual(self.env.boss_summon_limit_for_phase(), 2)
        self.assertEqual(self.env.boss_active_summon_limit_for_phase(), 2)
        self.assertGreater(self.env.maximum_active_enemies(), 7)
        self.assertEqual(self.env.phase_max_steps, 70 * self.env.fps)

        self.boss(12)
        self.assertEqual(self.env.boss_summon_limit_for_phase(), 4)
        self.assertEqual(self.env.boss_active_summon_limit_for_phase(), 2)

    def test_late_boss_tiers_remain_stronger_than_the_first_boss(self):
        snapshots = []
        for phase in (3, 6, 30):
            self.env.phase = phase
            self.env.spawners.clear()
            self.env.enemies.clear()
            self.env._spawn_phase_spawners()
            boss = self.env.spawners[0]
            self.env._spawn_enemy(boss)
            minion = self.env.enemies[-1]
            snapshots.append(
                (
                    boss.health + boss.shield,
                    minion.max_health,
                    minion.speed,
                    self.env.boss_summon_limit_for_phase(),
                    self.env.boss_defender_count_for_phase(),
                    self.env.boss_threat_tier,
                )
            )

        first, second, late = snapshots
        self.assertGreater(second[0], first[0])
        self.assertGreater(late[0], second[0])
        self.assertGreater(second[1], first[1])
        self.assertGreater(late[1], second[1])
        self.assertGreater(second[2], first[2])
        self.assertGreaterEqual(late[2], second[2])
        self.assertEqual((first[3], second[3], late[3]), (1, 2, 4))
        self.assertEqual((first[4], second[4], late[4]), (2, 3, 4))
        self.assertEqual((first[5], second[5], late[5]), (1, 2, 10))

    def test_early_boss_power_rises_in_gradual_adaptable_steps(self):
        effective_health = []
        minion_health = []
        for phase in (3, 6, 9, 12):
            self.env.phase = phase
            self.env.spawners.clear()
            self.env.enemies.clear()
            self.env._spawn_phase_spawners()
            boss = self.env.spawners[0]
            self.env._spawn_enemy(boss)
            effective_health.append(boss.health + boss.shield)
            minion_health.append(self.env.enemies[-1].max_health)

        for values in (effective_health, minion_health):
            ratios = [later / earlier for earlier, later in zip(values, values[1:])]
            self.assertTrue(all(1.10 <= ratio <= 1.50 for ratio in ratios))

    def test_miniboss_frequency_rises_after_phase_five_with_a_cap(self):
        self.env.phase = 4
        early = self.env.miniboss_chance_for_phase()
        self.env.phase = 6
        growing = self.env.miniboss_chance_for_phase()
        self.env.phase = 99
        capped = self.env.miniboss_chance_for_phase()
        self.assertEqual(early, 0.24)
        self.assertGreater(growing, early)
        self.assertEqual(capped, 0.66)

    def test_summons_do_not_appear_on_a_close_player(self):
        boss = self.boss()
        self.env.player.x, self.env.player.y = boss.x, boss.y
        boss.health *= .2
        self.env._update_boss_summons(defaultdict(int))
        self.assertEqual(boss.summons_used, 0)

    def test_crowd_and_closing_speed_are_explicit(self):
        p = self.env.player
        self.env.enemies = [Enemy(p.x+60, p.y, 15, 91, 50, 50, 72, vx=-72),
                            Enemy(p.x+90, p.y+10, 15, 92, 50, 50, 72)]
        obs = self.env._get_observation()
        self.assertGreater(obs[I.NEAREST_ENEMY_CLOSING], 0)
        self.assertGreater(obs[I.CROWD_PRESSURE], 0)
        self.assertLess(obs[I.CROWD_ESCAPE_X], 0)
        self.assertGreater(obs[I.SECOND_ENEMY_DISTANCE], 0)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_standoff_rewards_a_safe_range_instead_of_touching_spawner(self):
        s = self.env.spawners[0]
        self.env.player.x, self.env.player.y = s.x+100, s.y
        too_close = self.env._shaping_snapshot()['spawner_distance']
        self.env.player.x = s.x + s.radius + self.env.player.radius + float(
            self.env.reward_cfg["spawner_standoff"]
        )
        preferred = self.env._shaping_snapshot()['spawner_distance']
        self.assertGreater(too_close, preferred)
        self.assertAlmostEqual(preferred, 0.0)

    def test_boss_warning_zone_has_dense_exposure_and_hit_penalties(self):
        player = self.env.player
        self.env.danger_zones = [
            DangerZone(
                kind="circle",
                x=player.x,
                y=player.y,
                radius=90,
                telegraph_steps=10,
                active_steps=10,
                maximum_telegraph_steps=10,
                damage=20,
                attack_id=999,
            )
        ]
        _, _, _, _, info = self.env.step(0)
        self.assertGreater(info["hazard_exposure"], 0.0)
        self.assertLess(info["reward_breakdown"]["hazard_exposure"], 0.0)

        events = defaultdict(float, boss_skill_hits=1)
        events.update(
            damage_dealt_enemy=0.0,
            damage_dealt_spawner=0.0,
            enemies_destroyed=0,
            spawners_destroyed=0,
            phase_advanced=False,
            minibosses_destroyed=0,
            boss_phase_cleared=False,
            boss_skills_dodged=0,
            hazard_escape_improvement=0.0,
            crowd_escape=0.0,
            crowd_pressure=0.0,
            phase_timeout=False,
            damage_taken=0.0,
            spawner_progress=0.0,
            aim_improvement=0.0,
            shot_fired=False,
            shot_alignment=0.0,
        )
        _, breakdown = self.env._calculate_reward(events, terminated=False)
        self.assertEqual(breakdown["boss_skill_hit"], -8.0)

    def test_shield_break_makes_boss_move_slowly(self):
        boss = self.boss()
        boss.shield = 0.0
        boss.health = boss.max_health
        before = (boss.x, boss.y)
        self.env._update_boss_movement_and_defenders(defaultdict(float))
        self.assertNotEqual((boss.x, boss.y), before)
        self.assertLessEqual(
            np.hypot(boss.vx, boss.vy),
            float(self.env.phase_cfg["boss_move_speed"]) * 1.01,
        )

    def test_low_health_defender_wave_blocks_boss_then_restores_priority(self):
        boss = self.boss()
        boss.shield = 0.0
        boss.health = boss.max_health * 0.40
        events = defaultdict(float, impacts=[])
        self.env._update_boss_movement_and_defenders(events)
        self.assertTrue(boss.defender_wave_started)
        self.assertEqual(
            len(self.env.boss_defenders), self.env.boss_defender_count_for_phase()
        )
        self.assertFalse(self.env.boss_is_vulnerable(boss))
        self.assertIn(self.env._nearest_target(), self.env.boss_defenders)

        health_before = boss.health
        self.env._damage_target(
            boss, 100.0, events, set(), set(), source="player", count_hit=True
        )
        self.assertEqual(boss.health, health_before)
        self.assertEqual(events["boss_immune_hits"], 1)

        self.env.enemies = [
            enemy for enemy in self.env.enemies if not enemy.is_boss_defender
        ]
        self.assertTrue(self.env.boss_is_vulnerable(boss))
        self.env._damage_target(
            boss, 10.0, events, set(), set(), source="player", count_hit=True
        )
        self.assertLess(boss.health, health_before)

    def test_defenders_fire_telegraphed_missiles_and_observation_exposes_them(self):
        boss = self.boss()
        boss.shield = 0.0
        boss.health = boss.max_health * 0.40
        events = defaultdict(float, impacts=[])
        self.env._update_boss_movement_and_defenders(events)
        defender = self.env.boss_defenders[0]
        defender.missile_cooldown_steps = 0
        self.env._update_boss_movement_and_defenders(events)
        missile = next(
            projectile
            for projectile in self.env.projectiles
            if projectile.weapon_kind == "enemy_missile"
        )
        self.assertGreater(missile.telegraph_steps, 0)
        observation = self.env._get_observation()
        self.assertGreater(observation[I.BOSS_DEFENDER_COUNT], 0.0)
        self.assertEqual(observation[I.BOSS_VULNERABLE], 0.0)
        self.assertLess(observation[I.NEAREST_MISSILE_DISTANCE], 1.0)
        self.assertGreater(observation[I.MISSILE_LOCK_ON], 0.0)
        self.assertGreater(
            np.hypot(
                observation[I.MISSILE_ESCAPE_X],
                observation[I.MISSILE_ESCAPE_Y],
            ),
            0.9,
        )
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_sentry_missile_has_finite_guidance_and_can_be_dodged(self):
        self.env.enemies.clear()
        self.env.spawners.clear()
        self.env.projectiles.clear()
        self.env.player.x, self.env.player.y = 430.0, 300.0
        self.env.player.health = self.env.player.max_health
        defender = Enemy(
            130.0,
            300.0,
            17.0,
            8001,
            50.0,
            50.0,
            0.0,
            is_boss_defender=True,
            defender_kind="turret",
        )
        events = defaultdict(float, impacts=[])
        self.env._fire_enemy_missile(defender, events)
        missile = self.env.projectiles[0]
        telegraph_steps = missile.telegraph_steps

        for _ in range(telegraph_steps):
            self.env._update_projectiles(events)
        self.assertEqual(missile.telegraph_steps, 0)

        # Move perpendicular after launch. Guidance expires quickly, so the
        # missile crosses the old line rather than turning forever.
        for _ in range(240):
            self.env.player.y = min(
                self.env.height - self.env.player.radius,
                self.env.player.y + float(self.env.player_cfg["direct_speed"]) * self.env.dt,
            )
            self.env._update_projectiles(events)
            if not self.env.projectiles:
                break
        self.assertEqual(self.env.player.health, self.env.player.max_health)
        self.assertEqual(missile.guidance_steps, 0)
        self.assertGreaterEqual(events["missiles_evaded"], 1)

    def test_defender_repair_is_capped_and_wave_cannot_repeat(self):
        boss = self.boss()
        boss.shield = 0.0
        boss.health = boss.max_health * 0.40
        events = defaultdict(float, impacts=[])
        self.env._update_boss_movement_and_defenders(events)
        repair_cap = boss.defender_regen_cap

        for _ in range(self.env.fps * 30):
            self.env._update_boss_movement_and_defenders(events)
        self.assertLessEqual(boss.health, repair_cap)
        self.assertGreater(events["boss_health_regenerated"], 0.0)

        self.env.enemies = [
            enemy for enemy in self.env.enemies if not enemy.is_boss_defender
        ]
        spawned_before = events["boss_defenders_spawned"]
        self.env._update_boss_movement_and_defenders(events)
        self.assertEqual(events["boss_defenders_spawned"], spawned_before)
        self.assertFalse(self.env.boss_intermission_active)

    def test_crowd_reward_does_not_credit_enemy_removal(self):
        p = self.env.player
        self.env.enemies = [Enemy(p.x+60, p.y, 15, 91, 50, 50, 72)]
        before = self.env._shaping_snapshot()
        self.env.enemies.clear()
        events = {}
        self.env._apply_shaping_delta(events, before)
        self.assertEqual(events['crowd_escape'], 0)

    def test_crowd_reward_measures_separation_not_action_choice(self):
        p = self.env.player
        self.env.enemies = [Enemy(p.x+60, p.y, 15, 91, 50, 50, 72)]
        before = self.env._shaping_snapshot()
        p.x -= 20
        events = {}
        self.env._apply_shaping_delta(events, before)
        self.assertGreater(events['crowd_escape'], 0)

    def test_crowd_reward_survives_an_unrelated_new_spawn(self):
        p = self.env.player
        original = Enemy(p.x+60, p.y, 15, 91, 50, 50, 72)
        self.env.enemies = [original]
        before = self.env._shaping_snapshot()
        p.x -= 20
        self.env.enemies.append(Enemy(p.x-300, p.y, 15, 92, 50, 50, 72))
        events = {}
        self.env._apply_shaping_delta(events, before)
        self.assertGreater(events['crowd_escape'], 0)

    def test_summons_respect_enemy_cap(self):
        boss = self.boss()
        self.env.player.x, self.env.player.y = self.env.width-boss.x, self.env.height-boss.y+self.env.playfield_top
        boss.health *= .3
        cap = int(self.env.enemy_cfg['maximum_active']) + 2 * int(self.env.phase_cfg['maximum_enemy_growth_per_phase'])
        for _ in range(cap):
            self.env._spawn_enemy(boss)
        self.env._update_boss_summons(defaultdict(int))
        self.assertEqual(len(self.env.enemies), cap)
        self.assertEqual(boss.summons_used, 0)

    def test_normal_spawns_also_respect_absolute_late_phase_cap(self):
        self.env.phase = 40
        cap = int(self.env.phase_cfg['maximum_enemy_absolute'])
        for _ in range(cap):
            self.env._spawn_enemy(self.env.spawners[0])
        for spawner in self.env.spawners:
            spawner.spawn_cooldown_steps = 0
        self.env._update_spawners(defaultdict(int))
        self.assertEqual(len(self.env.enemies), cap)

    def test_transfer_preserves_q_values_for_original_prefix(self):
        torch.set_num_threads(1)
        old_env = ArenaEnv(control_style='rotation')
        old_env.observation_space = Box(-1, 1, (70,), dtype=np.float32)
        old = DQN('MlpPolicy', old_env, policy_kwargs={'net_arch':[256,256]}, buffer_size=10)
        new = DQN('MlpPolicy', self.env, policy_kwargs={'net_arch':[256,256]}, buffer_size=10)
        transfer_prefix_policy(old, new)
        sample = torch.randn(3, len(self.env.observation_names))
        with torch.no_grad():
            torch.testing.assert_close(old.q_net(sample[:,:70]), new.q_net(sample))
            torch.testing.assert_close(old.q_net_target(sample[:,:70]), new.q_net_target(sample))
        old_env.close()

    def test_cooldown_mask_excludes_unavailable_target_and_survives_save(self):
        model = CooldownAwareDQN('MlpPolicy', self.env, buffer_size=8,
                                 learning_rate=0.0, policy_kwargs={'net_arch':[16]})
        model.cooldown_mask_enabled = True
        model.set_logger(configure(folder=None, format_strings=[]))
        with torch.no_grad():
            for p in model.policy.parameters():
                p.zero_()
            model.q_net_target.q_net[-1].bias[4] = 1000.0
            model.q_net.q_net[-1].bias[4] = 1000.0
        obs = np.zeros(len(self.env.observation_names), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        self.assertNotEqual(int(action), 4)
        model.replay_buffer.add(obs, obs, np.array([0]), np.array([0.0]), np.array([False]), [{}])
        model.train(gradient_steps=1, batch_size=1)
        self.assertAlmostEqual(model.logger.name_to_value['train/loss'], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'masked.zip')
            model.save(path)
            restored = load_dqn(path, device='cpu')
            self.assertIsInstance(restored, CooldownAwareDQN)
            self.assertNotEqual(int(restored.predict(obs, deterministic=True)[0]), 4)

    def test_appended_input_training_cannot_change_old_connections(self):
        model = DQN('MlpPolicy', self.env, buffer_size=16, learning_starts=0,
                    batch_size=2, policy_kwargs={'net_arch':[16]})
        restrict_to_appended_inputs(model, 70)
        before = {name: p.detach().clone() for name,p in model.q_net.named_parameters()}
        model.learn(12)
        for name, parameter in model.q_net.named_parameters():
            if name == 'q_net.0.weight':
                torch.testing.assert_close(parameter[:,:70], before[name][:,:70], rtol=0, atol=0)
                self.assertFalse(torch.equal(parameter[:,70:], before[name][:,70:]))
            else:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_phase_diagnostics_count_actual_frames(self):
        class Noop:
            def predict(self, observation, deterministic=True):
                return np.array(0), None
        with patch('arena.benchmark.ArenaEnv', side_effect=lambda **kwargs: ArenaEnv(
            **kwargs, config_override={'simulation':{'max_steps':7}}
        )):
            rows, aggregate = evaluate_model(Noop(), 'rotation', episodes=1)
        self.assertEqual(rows[0]['simulation_steps'], 7)
        self.assertEqual(aggregate['phase_diagnostics']['1']['frames'], 7)


if __name__ == '__main__':
    unittest.main()
