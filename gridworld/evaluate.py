"""
Evaluation script to visually replay and verify trained agents in Pygame.

Usage:
    python -m gridworld.evaluate --level 0 --agent qlearning
    python -m gridworld.evaluate --level 0 --agent qlearning --episodes 5 --fps 6
"""

import argparse
import json
import os
import sys
import time
import pygame

from gridworld.environment import GridWorldEnv
from gridworld.agents.q_learning import QLearningAgent
from gridworld.renderer import GridWorldRenderer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "gridworld")


def load_config():
    """Load configuration from config.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Gridworld Agent")
    parser.add_argument("--level", type=int, default=0, help="Level ID (0-6)")
    parser.add_argument("--agent", type=str, default="qlearning",
                        choices=["qlearning", "sarsa"], help="Agent type")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes to replay")
    parser.add_argument("--fps", type=int, default=6,
                        help="Visual playback speed (frames per second)")
    parser.add_argument("--intrinsic", action="store_true",
                        help="Load intrinsic curiosity model")
    args = parser.parse_args()

    config = load_config()
    monster_chance = config.get("monster", {}).get("move_chance", 0.4)

    suffix = "_intrinsic" if args.intrinsic else ""
    model_filename = f"level{args.level}_{args.agent}{suffix}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"[!] Error: Trained model not found at {model_path}")
        print(f"    Train the agent first using: python -m gridworld.train --level {args.level} --agent {args.agent}")
        sys.exit(1)

    # Initialize environment and load agent
    env = GridWorldEnv(args.level, monster_move_chance=monster_chance)

    if args.agent == "qlearning":
        agent = QLearningAgent()
    else:
        from gridworld.agents.sarsa import SARSAAgent
        agent = SARSAAgent()

    agent.load(model_path)
    # Set epsilon = 0 for deterministic greedy evaluation
    agent.epsilon = 0.0

    renderer = GridWorldRenderer(
        env,
        cell_size=config["rendering"]["cell_size"],
        fps=args.fps,
        title=f"Evaluation - Level {args.level} ({args.agent.upper()})",
    )

    print(f"=== Evaluating {args.agent.upper()} on Level {args.level} ===")
    print(f"    Model: {model_path}")
    print(f"    Replay Episodes: {args.episodes} | Playback FPS: {args.fps}")
    print("    (Close Pygame window or press ESC/Q to exit)")
    print()

    for ep in range(args.episodes):
        state = env.reset()
        total_reward = 0.0
        step_count = 0

        # Render initial frame
        renderer.render(episode=ep + 1, step=0, total_reward=0.0)
        time.sleep(0.3)

        done = False
        info = {}

        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            renderer.render(episode=ep + 1, step=step_count, total_reward=total_reward)

            # Check if user requested window exit during evaluation
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    renderer.close()
                    sys.exit(0)

        result = "VICTORY" if info.get("victory") else f"DIED ({info.get('death', 'hazard')})"
        print(f"  Episode {ep + 1:>2}: {result:<10} | Steps: {step_count:>3} | Total Reward: {total_reward:.1f}")
        time.sleep(0.8)

    print()
    print("=== Evaluation Complete. Window will remain open until closed. ===")

    # Keep window open for inspection
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
        renderer.clock.tick(10)

    renderer.close()


if __name__ == "__main__":
    main()
