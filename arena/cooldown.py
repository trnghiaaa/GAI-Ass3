"""Cooldown-aware action selection for rotation-control DQN agents."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN

from arena.environment import ObservationIndex, ROTATION_ACTIONS


def load_dqn(path: str, **kwargs: Any) -> DQN:
    """Restore the trained action/target semantics, including checkpoint files."""
    model = DQN.load(path, **kwargs)
    if getattr(model, 'cooldown_mask_enabled', False):
        return CooldownAwareDQN.load(path, **kwargs)
    return model


def _best_ready_actions(model: DQN, observations: np.ndarray) -> np.ndarray:
    """Return each observation's highest-Q non-shoot action."""

    model_observations = np.asarray(observations, dtype=np.float32)
    expected_size = int(np.prod(model.observation_space.shape))
    model_observations = model_observations[..., :expected_size]
    observation_tensor, _ = model.policy.obs_to_tensor(model_observations)
    with torch.no_grad():
        q_values = model.q_net(observation_tensor).detach().cpu().numpy()
    q_values[..., ROTATION_ACTIONS["SHOOT"]] = -np.inf
    return np.argmax(q_values, axis=-1).astype(np.int64)


def replace_unavailable_shots(
    model: DQN,
    observations: np.ndarray,
    actions: np.ndarray | int,
) -> tuple[np.ndarray, int]:
    """Replace SHOOT during cooldown with the network's best valid action."""

    observation_array = np.asarray(observations, dtype=np.float32)
    single_observation = observation_array.ndim == 1
    observation_batch = (
        observation_array.reshape(1, -1) if single_observation else observation_array
    )
    action_batch = np.asarray(actions, dtype=np.int64).reshape(-1).copy()
    unavailable = (
        observation_batch[:, ObservationIndex.WEAPON_READY] < 0.999
    ) & (action_batch == ROTATION_ACTIONS["SHOOT"])
    replacement_count = int(np.count_nonzero(unavailable))
    if replacement_count:
        alternatives = _best_ready_actions(model, observation_batch[unavailable])
        action_batch[unavailable] = alternatives
    if single_observation:
        return np.asarray(action_batch[0]), replacement_count
    return action_batch, replacement_count


class CooldownAwareDQN(DQN):
    """DQN with a consistent state-dependent mask for the SHOOT action.

    Masking only action selection is not sufficient for Q-learning: the
    Bellman target must also exclude SHOOT when it is unavailable in the next
    state.  Otherwise an impossible action can dominate ``max Q(s', a)`` and
    teach every preceding state an inflated target.
    """

    cooldown_replacements: int = 0

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        actions, buffer_actions = super()._sample_action(
            learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )
        if self._last_obs is None or isinstance(self._last_obs, dict):
            return actions, buffer_actions
        actions, replacements = replace_unavailable_shots(
            self,
            self._last_obs,
            actions,
        )
        self.cooldown_replacements += replacements
        # Discrete DQN actions are stored without scaling.
        return actions, actions.copy()

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        actions, state = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        if isinstance(observation, dict):
            return actions, state
        actions, replacements = replace_unavailable_shots(self, observation, actions)
        self.cooldown_replacements += replacements
        return actions, state

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Run the SB3 DQN update with cooldown-aware next-state targets."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                weapon_unavailable = (
                    replay_data.next_observations[:, ObservationIndex.WEAPON_READY]
                    < 0.999
                )
                next_q_values[weapon_unavailable, ROTATION_ACTIONS["SHOOT"]] = (
                    -torch.inf
                )
                next_q_values = next_q_values.max(dim=1).values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(
                current_q_values,
                dim=1,
                index=replay_data.actions.long(),
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


class CooldownAwarePolicy:
    """Apply cooldown masking to an existing saved DQN during evaluation."""

    def __init__(self, model: DQN) -> None:
        self.model = model
        self.replacement_count = 0
        self.prediction_count = 0
        self.raw_cooldown_shoot_count = 0
        self.raw_action_counts: Counter[int] = Counter()
        self.executed_action_counts: Counter[int] = Counter()
        self.last_raw_action: int | None = None

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[Any, Any]:
        expected_size = int(np.prod(self.model.observation_space.shape))
        model_observation = np.asarray(observation)[..., :expected_size]
        actions, state = self.model.predict(
            model_observation,
            deterministic=deterministic,
        )
        raw_action = int(np.asarray(actions).item())
        self.last_raw_action = raw_action
        self.prediction_count += 1
        self.raw_action_counts[raw_action] += 1
        observation_vector = np.asarray(observation).reshape(-1)
        if (
            raw_action == ROTATION_ACTIONS["SHOOT"]
            and observation_vector[ObservationIndex.WEAPON_READY] < 0.999
        ):
            self.raw_cooldown_shoot_count += 1
        actions, replacements = replace_unavailable_shots(
            self.model,
            observation,
            actions,
        )
        self.replacement_count += replacements
        self.executed_action_counts[int(np.asarray(actions).item())] += 1
        return actions, state


__all__ = [
    "load_dqn",
    "CooldownAwareDQN",
    "CooldownAwarePolicy",
    "replace_unavailable_shots",
]
