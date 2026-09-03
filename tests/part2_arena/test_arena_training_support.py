"""Tests for Part II reward diagnostics and training wrappers."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

import arena.evaluate_direct as evaluate_direct
import arena.evaluate_rotation as evaluate_rotation
from arena.learning.rotation_expert import RotationTeacher
from arena.environment import (
    ArenaEnv,
    OBSERVATION_NAMES,
    ObservationIndex,
    ROTATION_ACTIONS,
)
from arena.settings import training_settings
from arena.wrappers import ActionRepeatWrapper, BossCurriculumWrapper


class ArenaTrainingSupportTests(unittest.TestCase):
    def test_rotation_teacher_prioritises_safety_turn_before_attack(self) -> None:
        env = ArenaEnv(control_style="rotation")
        try:
            observation, _ = env.reset(seed=2)
            observation[:] = 0.0
            observation[ObservationIndex.ENEMY_COUNT] = 1.0
            observation[ObservationIndex.WEAPON_READY] = 1.0
            observation[ObservationIndex.ACTIVE_TARGET_AIM_ALIGNMENT] = 1.0
            observation[ObservationIndex.SAFETY_URGENCY] = 0.8
            observation[ObservationIndex.SAFETY_ESCAPE_ALIGNMENT] = -0.5
            observation[ObservationIndex.SAFETY_ESCAPE_TURN] = 1.0

            action = RotationTeacher().action(observation)

            self.assertEqual(action, ROTATION_ACTIONS["ROTATE_RIGHT"])
        finally:
            env.close()

    def test_rotation_teacher_fires_only_when_ready_and_aligned(self) -> None:
        env = ArenaEnv(control_style="rotation")
        try:
            observation, _ = env.reset(seed=3)
            observation[:] = 0.0
            observation[ObservationIndex.SPAWNER_COUNT] = 1.0
            observation[ObservationIndex.WEAPON_READY] = 1.0
            observation[ObservationIndex.ACTIVE_TARGET_AIM_ALIGNMENT] = 0.95

            teacher = RotationTeacher()
            self.assertEqual(teacher.action(observation), ROTATION_ACTIONS["SHOOT"])
            observation[ObservationIndex.WEAPON_READY] = 0.0
            self.assertNotEqual(teacher.action(observation), ROTATION_ACTIONS["SHOOT"])
        finally:
            env.close()

    def test_rotation_teacher_keeps_attacking_through_moderate_pressure(self) -> None:
        observation = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        observation[ObservationIndex.SPAWNER_COUNT] = 1.0
        observation[ObservationIndex.WEAPON_READY] = 1.0
        observation[ObservationIndex.ACTIVE_TARGET_AIM_ALIGNMENT] = 0.96
        observation[ObservationIndex.SAFETY_URGENCY] = 0.35

        self.assertEqual(
            RotationTeacher().action(observation), ROTATION_ACTIONS["SHOOT"]
        )

    def test_rotation_teacher_reacts_during_boss_telegraph_not_after_it(self) -> None:
        observation = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        observation[ObservationIndex.SPAWNER_COUNT] = 1.0
        observation[ObservationIndex.WEAPON_READY] = 1.0
        observation[ObservationIndex.ACTIVE_TARGET_AIM_ALIGNMENT] = 1.0
        observation[ObservationIndex.HAZARD_DISTANCE_TO_SAFETY] = 0.5
        observation[ObservationIndex.HAZARD_TIME_TO_IMPACT] = 0.8
        observation[ObservationIndex.SAFETY_ESCAPE_ALIGNMENT] = -0.5
        observation[ObservationIndex.SAFETY_ESCAPE_TURN] = -1.0

        self.assertEqual(
            RotationTeacher().action(observation), ROTATION_ACTIONS["ROTATE_LEFT"]
        )

    def test_rotation_teacher_escapes_a_close_closing_enemy(self) -> None:
        observation = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        observation[ObservationIndex.ENEMY_COUNT] = 1.0
        observation[ObservationIndex.NEAREST_ENEMY_DISTANCE] = 0.1
        observation[ObservationIndex.NEAREST_ENEMY_CLOSING] = 0.3
        observation[ObservationIndex.SAFETY_ESCAPE_ALIGNMENT] = -0.4
        observation[ObservationIndex.SAFETY_ESCAPE_TURN] = 1.0

        self.assertEqual(
            RotationTeacher().action(observation), ROTATION_ACTIONS["ROTATE_RIGHT"]
        )

    def test_boss_curriculum_only_changes_training_reset_phase(self) -> None:
        base = ArenaEnv(control_style="direct")
        env = BossCurriculumWrapper(base, probability=1.0, phases=(3,), seed=7)
        try:
            observation, info = env.reset(seed=12)
            self.assertEqual(base.phase, 3)
            self.assertTrue(base.is_boss_phase)
            self.assertEqual(info["curriculum_start_phase"], 3)
            self.assertEqual(base.player.level, 3)
            self.assertGreaterEqual(info["curriculum_bootstrap_choices"], 4)
            self.assertGreater(sum(base.upgrade_stacks.values()), 0)
            self.assertGreater(base.barrier_charges, 0)
            self.assertEqual(observation.shape, base.observation_space.shape)
        finally:
            env.close()

    def test_standard_reset_still_starts_at_phase_one(self) -> None:
        env = ArenaEnv(control_style="direct")
        try:
            env.reset(seed=12)
            self.assertEqual(env.phase, 1)
        finally:
            env.close()

    def test_action_repeat_accumulates_rewards_events_and_simulation_frames(self) -> None:
        base = ArenaEnv(
            control_style="rotation",
            config_override={"simulation": {"max_steps": 20}},
        )
        env = ActionRepeatWrapper(base, repeat=4)
        try:
            env.reset(seed=3)
            _, reward, terminated, truncated, info = env.step(
                ROTATION_ACTIONS["ROTATE_RIGHT"]
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(base.step_count, 4)
            self.assertEqual(info["action_repeat_frames"], 4)
            self.assertAlmostEqual(reward, sum(info["reward_breakdown"].values()))
            self.assertNotEqual(info["aim_improvement"], 0.0)
        finally:
            env.close()

    def test_action_repeat_stops_immediately_at_episode_boundary(self) -> None:
        base = ArenaEnv(config_override={"simulation": {"max_steps": 2}})
        env = ActionRepeatWrapper(base, repeat=4)
        try:
            env.reset(seed=4)
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["action_repeat_frames"], 2)
            self.assertEqual(base.step_count, 2)
        finally:
            env.close()

    def test_sentry_curriculum_can_start_at_the_defensive_wave(self) -> None:
        base = ArenaEnv(control_style="rotation")
        env = BossCurriculumWrapper(
            base,
            probability=1.0,
            phases=(3,),
            sentry_probability=1.0,
            seed=8,
        )
        try:
            observation, info = env.reset(seed=14)
            boss = next(item for item in base.spawners if item.is_boss)
            self.assertEqual(boss.shield, 0.0)
            self.assertTrue(base.boss_intermission_active)
            self.assertTrue(info["curriculum_sentry_wave"])
            self.assertGreater(len(base.boss_defenders), 0)
            self.assertEqual(observation.shape, base.observation_space.shape)
        finally:
            env.close()

    def test_explicit_evaluation_reset_is_never_changed_by_curriculum(self) -> None:
        base = ArenaEnv(control_style="direct")
        env = BossCurriculumWrapper(
            base,
            probability=1.0,
            phases=(3,),
            sentry_probability=1.0,
            seed=9,
        )
        try:
            _, info = env.reset(seed=15, options={"start_phase": 1})
            self.assertEqual(base.phase, 1)
            self.assertEqual(base.player.level, 1)
            self.assertEqual(info["curriculum_bootstrap_choices"], 0)
            self.assertFalse(info["curriculum_sentry_wave"])
        finally:
            env.close()

    def test_action_repeat_accumulates_wall_contacts(self) -> None:
        base = ArenaEnv(control_style="rotation")
        env = ActionRepeatWrapper(base, repeat=4)
        try:
            env.reset(seed=5)
            base.player.x = base.player.radius
            base.player.vx = -120.0
            _, _, _, _, info = env.step(ROTATION_ACTIONS["THRUST"])
            self.assertGreaterEqual(info["wall_contacts"], 1)
            self.assertAlmostEqual(base.player.vx, 0.0)
        finally:
            env.close()

    def test_named_profiles_and_style_overrides_are_non_default_tuning(self) -> None:
        direct = training_settings("direct", "balanced")
        rotation = training_settings("rotation", "balanced")
        fast = training_settings("direct", "fast_exploration")

        self.assertEqual(direct["algorithm"], "DQN")
        self.assertEqual(direct["net_arch"], [256, 256])
        self.assertGreater(rotation["exploration_fraction"], direct["exploration_fraction"])
        self.assertNotEqual(fast["learning_rate"], direct["learning_rate"])
        self.assertNotEqual(fast["net_arch"], direct["net_arch"])

    def test_dedicated_evaluators_forward_cli_options_and_lock_control_style(self) -> None:
        options = ["--headless", "--episodes", "2", "--seed", "91"]
        cases = ((evaluate_direct, "direct"), (evaluate_rotation, "rotation"))

        for module, control_style in cases:
            with self.subTest(control_style=control_style):
                with patch.object(module, "evaluate_main") as evaluator:
                    with patch.object(sys, "argv", [module.__name__, *options]):
                        module.main()
                evaluator.assert_called_once_with(
                    options, forced_control_style=control_style
                )


if __name__ == "__main__":
    unittest.main()
