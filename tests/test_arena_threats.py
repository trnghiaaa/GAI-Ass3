"""Schema-6 difficulty, fairness, observation and transfer regressions."""
from collections import defaultdict
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3 import DQN
import torch

from arena.entities import Enemy
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

    def boss(self):
        self.env.phase = 3
        self.env.spawners.clear()
        self.env._spawn_phase_spawners()
        return self.env.spawners[0]

    def test_later_xp_thresholds_are_harder_but_first_upgrade_unchanged(self):
        self.assertEqual(self.env.xp_threshold_for_level(2), 55)
        self.assertGreater(self.env.xp_threshold_for_level(5), 55 * 4**1.72)

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
        boss = self.boss()
        self.env.player.x = self.env.width - boss.x
        self.env.player.y = self.env.height - boss.y + self.env.playfield_top
        boss.health = boss.max_health * 0.3
        events = defaultdict(int)
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 1)
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 1)
        boss.summon_cooldown_steps = 0
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 2)
        self.env.enemies.clear()
        boss.summon_cooldown_steps = 0
        self.env._update_boss_summons(events)
        self.assertEqual(boss.summons_used, 2)

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

    def test_standoff_removes_incentive_to_touch_spawner(self):
        s = self.env.spawners[0]
        self.env.player.x, self.env.player.y = s.x+100, s.y
        self.assertEqual(self.env._shaping_snapshot()['spawner_distance'], 0)

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
        sample = torch.randn(3,89)
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
        obs = np.zeros(89, dtype=np.float32)
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
