"""Bounded, seeded evaluation for saved Gridworld policies.

The default command renders actual Pygame gameplay.  ``--headless`` is useful
for automated acceptance checks; both modes use the same rollout code and
distinguish victory, death, and a safe runner-level timeout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.train import (
    MODELS_DIR,
    load_config,
    make_environment,
    reset_environment,
    resolve_training_profile,
    step_environment,
)


def load_agent(agent_kind: str, model_path: str | os.PathLike[str], seed: int):
    agent_class = QLearningAgent if agent_kind == "qlearning" else SARSAAgent
    agent = agent_class(seed=seed)
    metadata = agent.load(model_path)
    agent.set_seed(seed)
    return agent, metadata


def evaluate_agent(
    level_id: int,
    agent: Any,
    config: Mapping[str, Any],
    *,
    episodes: int,
    max_steps: int,
    base_seed: int,
    epsilon: float = 0.0,
    renderer: Any = None,
) -> list[dict[str, Any]]:
    """Evaluate without learning and return one auditable row per episode."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    pygame = None
    if renderer is not None:
        import pygame as pygame_module
        pygame = pygame_module

    agent.epsilon = float(epsilon)
    agent.set_seed(base_seed + 7_000_003)
    rows: list[dict[str, Any]] = []
    quit_requested = False

    for episode in range(episodes):
        seed = base_seed + episode
        env = make_environment(level_id, config, seed)
        state = reset_environment(env, seed)
        if hasattr(agent, "begin_episode"):
            agent.begin_episode(state)
        if renderer is not None:
            renderer.env = env

        total_reward = 0.0
        path = [tuple(env.agent_pos)]
        done = False
        info: dict[str, Any] = {}
        for step in range(1, max_steps + 1):
            if renderer is not None:
                renderer.render(
                    episode=episode + 1,
                    step=step - 1,
                    total_reward=total_reward,
                    trail=path,
                    message=f"Greedy policy (epsilon={epsilon:g})",
                )
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        quit_requested = True
                if quit_requested:
                    break

            action = int(agent.choose_action(state))
            state, reward, done, info = step_environment(env, action)
            total_reward += reward
            path.append(tuple(env.agent_pos))
            if done:
                if renderer is not None:
                    renderer.render(
                        episode=episode + 1,
                        step=step,
                        total_reward=total_reward,
                        trail=path,
                        message=("VICTORY" if info.get("victory") else
                                 f"DEATH: {info.get('death', 'hazard')}").upper(),
                    )
                break

        if quit_requested:
            break
        if done and info.get("victory"):
            status = "victory"
        elif done and info.get("death"):
            status = f"death_{info['death']}"
        else:
            status = "timeout"
        rows.append({
            "episode": episode + 1,
            "seed": seed,
            "status": status,
            "victory": int(status == "victory"),
            "death": int(status.startswith("death_")),
            "timeout": int(status == "timeout"),
            "steps": step,
            "environment_reward": total_reward,
            "path": path,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Gridworld policy")
    parser.add_argument("--level", type=int, default=0, choices=range(7))
    parser.add_argument("--agent", choices=["qlearning", "sarsa"],
                        default="qlearning")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="0 gives greedy evaluation with random exact ties")
    parser.add_argument("--intrinsic", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable episode metrics")
    args = parser.parse_args()

    config = load_config()
    evaluation = config.get("evaluation", {})
    profile = resolve_training_profile(config, args.level, args.agent)
    episodes = int(args.episodes or evaluation.get("episodes", 3))
    max_steps = int(args.max_steps or evaluation.get("max_steps", profile["max_steps"]))
    fps = int(args.fps or evaluation.get("fps", 8))
    seed = int(args.seed if args.seed is not None else evaluation.get("seed", 9001))
    suffix = "_intrinsic" if args.intrinsic else ""
    model_path = Path(MODELS_DIR) / f"level{args.level}_{args.agent}{suffix}.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            f"Train it with: python -m gridworld.train --level {args.level} "
            f"--agent {args.agent}{' --intrinsic' if args.intrinsic else ''}"
        )

    agent, metadata = load_agent(args.agent, model_path, seed)
    expected_schema = metadata.get("state_schema") if metadata else None
    probe_env = make_environment(args.level, config, seed)
    if expected_schema and list(expected_schema) != list(probe_env.state_schema):
        raise SystemExit("Saved model state schema differs from the current environment; retrain it.")

    renderer = None
    if not args.headless:
        from gridworld.renderer import GridWorldRenderer
        renderer = GridWorldRenderer(
            probe_env,
            cell_size=int(config["rendering"]["cell_size"]),
            fps=fps,
            title=f"Level {args.level} - {args.agent.upper()} evaluation",
        )
    try:
        rows = evaluate_agent(
            args.level, agent, config, episodes=episodes, max_steps=max_steps,
            base_seed=seed, epsilon=args.epsilon, renderer=renderer,
        )
    finally:
        if renderer is not None:
            renderer.close()

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(
                f"Episode {row['episode']:>2}: {row['status']:<14} | "
                f"steps {row['steps']:>3} | env reward {row['environment_reward']:.1f}"
            )
        if rows:
            print(
                f"Summary: {sum(row['victory'] for row in rows)}/{len(rows)} victories, "
                f"{sum(row['death'] for row in rows)} deaths, "
                f"{sum(row['timeout'] for row in rows)} timeouts."
            )


if __name__ == "__main__":
    main()
