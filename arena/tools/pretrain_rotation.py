"""Pretrain the Rotation DQN from safe demonstration trajectories.

This is an initialisation step, not a runtime controller.  The output remains a
normal Stable-Baselines3 DQN checkpoint and is intended for subsequent online
reinforcement learning with ``arena.train --init-model``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from arena.core.environment import ArenaEnv, OBSERVATION_NAMES
from arena.learning.cooldown import load_dqn
from arena.learning.rotation_expert import RotationTeacher
from arena.learning.wrappers import ActionRepeatWrapper, BossCurriculumWrapper
from arena.settings import TENSORBOARD_DIR


def _collect(
    teacher: RotationTeacher,
    *,
    episodes: int,
    seed: int,
    action_repeat: int,
    boss_starts: bool,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    for episode in range(episodes):
        base = ArenaEnv(control_style="rotation")
        wrapped = (
            BossCurriculumWrapper(
                base,
                probability=1.0,
                phases=(3, 6),
                seed=seed,
                sentry_probability=0.45,
            )
            if boss_starts
            else base
        )
        env = ActionRepeatWrapper(wrapped, repeat=action_repeat)
        observation, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        try:
            while not (terminated or truncated):
                action = teacher.action(observation)
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                actions.append(action)
                observation, _, terminated, truncated, _ = env.step(action)
        finally:
            env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-episodes", type=int, default=32)
    parser.add_argument("--boss-episodes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=550000)
    parser.add_argument("--action-repeat", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        raise FileExistsError("Choose a new output path; artifacts are protected")

    torch.set_num_threads(1)
    teacher = RotationTeacher()
    normal_x, normal_y = _collect(
        teacher,
        episodes=args.normal_episodes,
        seed=args.seed,
        action_repeat=args.action_repeat,
        boss_starts=False,
    )
    boss_x, boss_y = _collect(
        teacher,
        episodes=args.boss_episodes,
        seed=args.seed + 10000,
        action_repeat=args.action_repeat,
        boss_starts=True,
    )
    features = np.concatenate((normal_x, boss_x))
    labels = np.concatenate((normal_y, boss_y))
    generator = np.random.default_rng(args.seed)
    order = generator.permutation(len(labels))
    split = max(1, int(len(order) * 0.9))
    train_indices, validation_indices = order[:split], order[split:]

    model = load_dqn(str(args.source), device="cpu")
    optimizer = torch.optim.Adam(
        model.q_net.parameters(), lr=float(args.learning_rate)
    )
    counts = np.bincount(labels[train_indices], minlength=5).astype(np.float32)
    # A mild fourth-root balance preserves overall imitation fidelity while
    # still protecting the rarer SHOOT and NOOP decisions from being ignored.
    weights = (counts.sum() / np.maximum(1.0, counts)) ** 0.25
    weights /= weights.mean()
    weights = np.clip(weights, 0.65, 1.8)
    class_weights = torch.as_tensor(weights, dtype=torch.float32)
    writer = SummaryWriter(
        str(TENSORBOARD_DIR / f"rotation_teacher_{args.seed}")
    )
    batch_size = 1024
    try:
        for epoch in range(args.epochs):
            generator.shuffle(train_indices)
            losses: list[float] = []
            for start in range(0, len(train_indices), batch_size):
                batch = train_indices[start : start + batch_size]
                inputs = torch.as_tensor(features[batch], dtype=torch.float32)
                targets = torch.as_tensor(labels[batch], dtype=torch.long)
                logits = model.q_net(inputs)
                loss = F.cross_entropy(logits, targets, weight=class_weights)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.item()))
            with torch.no_grad():
                validation_logits = model.q_net(
                    torch.as_tensor(features[validation_indices], dtype=torch.float32)
                )
                validation_accuracy = float(
                    (
                        validation_logits.argmax(dim=1).cpu().numpy()
                        == labels[validation_indices]
                    ).mean()
                )
            writer.add_scalar("teacher/loss", float(np.mean(losses)), epoch)
            writer.add_scalar("teacher/validation_accuracy", validation_accuracy, epoch)
    finally:
        writer.close()

    model.q_net_target.load_state_dict(model.q_net.state_dict())
    model.cooldown_mask_enabled = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "DQN",
        "control_style": "rotation",
        "training_stage": "demonstration_initialisation",
        "runtime_teacher": False,
        "source_model": str(args.source),
        "normal_episodes": args.normal_episodes,
        "boss_episodes": args.boss_episodes,
        "demonstrations": int(len(labels)),
        "action_counts": dict(Counter(int(value) for value in labels)),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "validation_accuracy": validation_accuracy,
        "action_repeat": args.action_repeat,
        "cooldown_mask": True,
        "observation_names": list(OBSERVATION_NAMES),
        "observation_size": len(OBSERVATION_NAMES),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
