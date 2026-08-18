"""
Manual play mode -- control the agent with arrow keys to verify
the environment and renderer work correctly.

Usage:
    python -m gridworld.play --level 0
    python -m gridworld.play --level 2
    python -m gridworld.play --level 4

Controls:
    Arrow keys  = Move agent (UP / DOWN / LEFT / RIGHT)
    R           = Reset the level
    ESC / Q     = Quit
"""

import argparse
import json
import os
import sys
import pygame

from gridworld.environment import GridWorldEnv, UP, DOWN, LEFT, RIGHT
from gridworld.renderer import GridWorldRenderer


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Manually play a gridworld level")
    parser.add_argument("--level", type=int, required=True, help="Level id (0-6)")
    args = parser.parse_args()

    config = load_config()
    monster_chance = config["monster"]["move_chance"]

    env = GridWorldEnv(args.level, monster_move_chance=monster_chance)
    renderer = GridWorldRenderer(
        env,
        cell_size=config["rendering"]["cell_size"],
        fps=config["rendering"]["fps"],
        title=f"Manual Play -- Level {args.level}",
    )

    state = env.reset()
    total_reward = 0.0
    step_count = 0

    print(f"=== Manual Play -- Level {args.level} ===")
    print(f"    Grid: {env.rows}x{env.cols}")
    print(f"    Collectibles: {len(env.initial_collectibles)}")
    print(f"    Monsters: {len(env.initial_monster_positions)}")
    print(f"    Key: {'Yes' if env.initial_key_pos else 'No'}")
    print()
    print("Controls: Arrow keys = move, R = reset, ESC/Q = quit")
    print()

    running = True
    while running:
        renderer.render(episode=None, step=step_count, total_reward=total_reward)

        # Wait for a key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False

                elif event.type == pygame.KEYDOWN:
                    action = None

                    if event.key == pygame.K_UP:
                        action = UP
                    elif event.key == pygame.K_DOWN:
                        action = DOWN
                    elif event.key == pygame.K_LEFT:
                        action = LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = RIGHT
                    elif event.key == pygame.K_r:
                        # Reset
                        state = env.reset()
                        total_reward = 0.0
                        step_count = 0
                        print("  [RESET]")
                        waiting = False
                        continue
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        waiting = False
                        continue

                    if action is not None:
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                        step_count += 1

                        # Print step info
                        parts = [f"Step {step_count}"]
                        if reward != 0:
                            parts.append(f"reward={reward:+.1f}")
                        if "picked_up" in info:
                            parts.append(f"picked up {info['picked_up']}")
                        if "death" in info:
                            parts.append(f"DIED ({info['death']})")
                        if "victory" in info:
                            parts.append("VICTORY!")
                        print("  " + "  |  ".join(parts))

                        if done:
                            print(f"\n  Episode ended. Total reward: {total_reward}")
                            print("  Press R to reset or ESC to quit.\n")

                        waiting = False

            renderer.clock.tick(30)

    renderer.close()


if __name__ == "__main__":
    main()
