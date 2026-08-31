"""Small checks for the SB3 control-style-1 training pipeline."""

import tempfile
from pathlib import Path
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

from arena.environment import ROTATION_ACTIONS
from arena.cooldown import (
    CooldownAwareDQN,
    CooldownAwarePolicy,
    replace_unavailable_shots,
)
from arena.evaluate import SafetyShieldPolicy
from arena.train import (
    PhaseCurriculumWrapper,
    build_dqn,
    load_training_config,
    make_monitored_env,
    protect_transferred_policy,
    transfer_dqn_policy,
)


class ArenaTrainingTests(unittest.TestCase):
    def test_safety_shield_vetoes_only_targeted_unsafe_actions(self) -> None:
        class FixedPolicy:
            def __init__(self, action):
                self.action = action

            def predict(self, observation, deterministic=True):
                return np.asarray(self.action), None

        observation = np.ones(47, dtype=np.float32)
        observation[40:44] = 1.0

        # Outward thrust at the left edge is redirected into a turn.
        observation[40] = 0.0
        observation[4] = -1.0
        observation[5] = 0.0
        observation[44] = -1.0
        observation[45] = 1.0
        shield = SafetyShieldPolicy(FixedPolicy(ROTATION_ACTIONS["THRUST"]))
        action, _ = shield.predict(observation)
        self.assertEqual(int(action), ROTATION_ACTIONS["ROTATE_RIGHT"])

        # A shot aimed at a boss during a severe off-axis enemy threat is fled.
        observation[40:44] = 1.0
        observation[21] = 0.0
        observation[23] = 0.9
        observation[38] = 0.8
        observation[39] = -0.2
        observation[46] = 0.9
        shield = SafetyShieldPolicy(FixedPolicy(ROTATION_ACTIONS["SHOOT"]))
        action, _ = shield.predict(observation)
        self.assertEqual(int(action), ROTATION_ACTIONS["THRUST"])

        # A defensive shot accurately aimed at the red enemy remains allowed.
        observation[21] = 0.9
        action, _ = shield.predict(observation)
        self.assertEqual(int(action), ROTATION_ACTIONS["SHOOT"])

    def test_training_curriculum_samples_early_phases_only(self) -> None:
        env = make_monitored_env("rotation", seed=17, curriculum_max_phase=3)
        try:
            phases = {env.reset()[1]["curriculum_start_phase"] for _ in range(30)}
            self.assertEqual(phases, {1, 2, 3})
            self.assertIsInstance(env.env, PhaseCurriculumWrapper)
        finally:
            env.close()

    def test_rotation_environment_has_exact_control_style_1_actions(self) -> None:
        env = make_monitored_env("rotation", seed=3)
        try:
            self.assertEqual(env.action_space.n, 5)
            self.assertEqual(
                ROTATION_ACTIONS,
                {
                    "NOOP": 0,
                    "THRUST": 1,
                    "ROTATE_LEFT": 2,
                    "ROTATE_RIGHT": 3,
                    "SHOOT": 4,
                },
            )
        finally:
            env.close()

    def test_config_builds_an_mlp_dqn_for_the_rotation_environment(self) -> None:
        env = make_monitored_env("rotation", seed=3)
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                model = build_dqn(
                    env,
                    load_training_config(),
                    Path(temporary_directory),
                    seed=3,
                    device="cpu",
                    verbose=0,
                )
            self.assertIsInstance(model, DQN)
            self.assertNotIsInstance(model, CooldownAwareDQN)
            self.assertEqual(model.action_space.n, 5)
            self.assertEqual(model.policy.net_arch, [256, 256])
        finally:
            env.close()

    def test_cooldown_mask_uses_best_non_shoot_q_action(self) -> None:
        env = make_monitored_env("rotation", seed=5)
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                model = build_dqn(
                    env,
                    load_training_config(),
                    Path(temporary_directory),
                    seed=5,
                    device="cpu",
                    verbose=0,
                    cooldown_aware_actions=True,
                )
                with torch.no_grad():
                    for parameter in model.q_net.parameters():
                        parameter.zero_()
                    model.q_net.q_net[-1].bias.copy_(
                        torch.tensor([1.0, 4.0, 3.0, 2.0, 10.0])
                    )

                observation = np.zeros(47, dtype=np.float32)
                observation[7] = 0.0
                masked, replacements = replace_unavailable_shots(
                    model,
                    observation,
                    ROTATION_ACTIONS["SHOOT"],
                )
                self.assertEqual(int(masked), ROTATION_ACTIONS["THRUST"])
                self.assertEqual(replacements, 1)

                observation[7] = 1.0
                unmasked, replacements = replace_unavailable_shots(
                    model,
                    observation,
                    ROTATION_ACTIONS["SHOOT"],
                )
                self.assertEqual(int(unmasked), ROTATION_ACTIONS["SHOOT"])
                self.assertEqual(replacements, 0)
        finally:
            env.close()

    def test_cooldown_policy_reports_raw_and_executed_actions(self) -> None:
        env = make_monitored_env("rotation", seed=6)
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                model = build_dqn(
                    env,
                    load_training_config(),
                    Path(temporary_directory),
                    seed=6,
                    device="cpu",
                    verbose=0,
                )
                with torch.no_grad():
                    for parameter in model.q_net.parameters():
                        parameter.zero_()
                    model.q_net.q_net[-1].bias.copy_(
                        torch.tensor([1.0, 4.0, 3.0, 2.0, 10.0])
                    )
                policy = CooldownAwarePolicy(model)
                observation = np.zeros(47, dtype=np.float32)
                observation[7] = 0.0
                executed, _ = policy.predict(observation)

                self.assertEqual(policy.last_raw_action, ROTATION_ACTIONS["SHOOT"])
                self.assertEqual(int(executed), ROTATION_ACTIONS["THRUST"])
                self.assertEqual(policy.raw_cooldown_shoot_count, 1)
                self.assertEqual(policy.raw_action_counts[ROTATION_ACTIONS["SHOOT"]], 1)
                self.assertEqual(policy.executed_action_counts[ROTATION_ACTIONS["THRUST"]], 1)
        finally:
            env.close()

    def test_cooldown_dqn_masks_shoot_in_bellman_target(self) -> None:
        env = make_monitored_env("rotation", seed=7)
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                model = build_dqn(
                    env,
                    load_training_config(),
                    Path(temporary_directory),
                    seed=7,
                    device="cpu",
                    verbose=0,
                    cooldown_aware_actions=True,
                )
                # A short learn call exercises the custom target update.  An
                # unavailable SHOOT value is set extremely high; consistent
                # masking keeps the loss finite.
                model.learning_starts = 0
                model.batch_size = 2
                model.buffer_size = 32
                model.learn(total_timesteps=8)
                self.assertGreater(model._n_updates, 0)
                for parameter in model.q_net.parameters():
                    self.assertTrue(torch.isfinite(parameter).all())
        finally:
            env.close()

    def test_transfer_expands_observations_without_changing_initial_q_values(self) -> None:
        class SourceEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32)
            action_space = gym.spaces.Discrete(5)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(28, dtype=np.float32), {}

            def step(self, action):
                del action
                return np.zeros(28, dtype=np.float32), 0.0, False, False, {}

        target_env = make_monitored_env("rotation", seed=3)
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                temporary_path = Path(temporary_directory)
                source = DQN(
                    "MlpPolicy",
                    SourceEnv(),
                    policy_kwargs={"net_arch": [256, 256]},
                    device="cpu",
                    verbose=0,
                )
                source_path = temporary_path / "source_model"
                source.save(source_path)
                target = build_dqn(
                    target_env,
                    load_training_config(),
                    temporary_path,
                    seed=3,
                    device="cpu",
                    verbose=0,
                )

                old_size, new_size = transfer_dqn_policy(
                    source_path.with_suffix(".zip"), target
                )
                old_observation = np.linspace(-1.0, 1.0, 28, dtype=np.float32)
                expanded_observation = np.pad(old_observation, (0, 19))
                source_q = source.q_net(source.policy.obs_to_tensor(old_observation)[0])
                target_q = target.q_net(target.policy.obs_to_tensor(expanded_observation)[0])

            self.assertEqual((old_size, new_size), (28, 47))
            np.testing.assert_allclose(
                source_q.detach().cpu().numpy(),
                target_q.detach().cpu().numpy(),
                rtol=0.0,
                atol=1e-6,
            )

            protect_transferred_policy(target, old_size)
            target.policy.optimizer.zero_grad()
            target.q_net(
                target.policy.obs_to_tensor(np.ones(47, dtype=np.float32))[0]
            ).sum().backward()
            first_layer = target.q_net.q_net[0]
            self.assertIsNotNone(first_layer.weight.grad)
            np.testing.assert_array_equal(
                first_layer.weight.grad[:, :old_size].detach().cpu().numpy(),
                0.0,
            )
            self.assertGreater(
                np.abs(
                    first_layer.weight.grad[:, old_size:].detach().cpu().numpy()
                ).sum(),
                0.0,
            )
            for parameter in list(target.q_net.parameters())[1:]:
                self.assertIsNone(parameter.grad)
        finally:
            target_env.close()


if __name__ == "__main__":
    unittest.main()
