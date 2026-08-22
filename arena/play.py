"""Interactive Pygame demonstration for the Part II arena.

Examples
--------
python -m arena.play --control-style direct
python -m arena.play --control-style rotation
"""

from __future__ import annotations

import argparse

import pygame

from arena.environment import ArenaEnv, DIRECT_ACTIONS, ROTATION_ACTIONS
from arena.renderer import ArenaRenderer


def _direct_action(keys: pygame.key.ScancodeWrapper) -> int:
    if keys[pygame.K_SPACE]:
        return DIRECT_ACTIONS["SHOOT"]
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return DIRECT_ACTIONS["UP"]
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return DIRECT_ACTIONS["DOWN"]
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return DIRECT_ACTIONS["LEFT"]
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return DIRECT_ACTIONS["RIGHT"]
    return DIRECT_ACTIONS["NOOP"]


def _rotation_action(keys: pygame.key.ScancodeWrapper) -> int:
    if keys[pygame.K_SPACE]:
        return ROTATION_ACTIONS["SHOOT"]
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return ROTATION_ACTIONS["THRUST"]
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return ROTATION_ACTIONS["ROTATE_LEFT"]
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return ROTATION_ACTIONS["ROTATE_RIGHT"]
    return ROTATION_ACTIONS["NOOP"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively play the Part II arena")
    parser.add_argument(
        "--control-style",
        choices=("direct", "rotation"),
        default="direct",
        help="Ship control scheme",
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    args = parser.parse_args()

    env = ArenaEnv(control_style=args.control_style)
    env.reset(seed=args.seed)
    renderer = ArenaRenderer(env, mode="human")

    if args.control_style == "direct":
        footer = "WASD / ARROWS move   •   SPACE auto-aim fire   •   R restart   •   ESC quit"
        choose_action = _direct_action
    else:
        footer = "W / UP thrust   •   A/D rotate   •   SPACE fire   •   R restart   •   ESC quit"
        choose_action = _rotation_action

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    env.reset(seed=args.seed)

        if running and not env.done:
            action = choose_action(pygame.key.get_pressed())
            env.step(action)

        renderer.render(process_events=False, footer_text=footer)
        renderer.clock.tick(env.fps)

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
