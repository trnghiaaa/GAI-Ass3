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


def play_manual(control_style: str = "direct", seed: int = 42) -> None:
    """Run one interactive manual session and return when its window closes."""

    env = ArenaEnv(control_style=control_style, manual_choices=True)
    env.reset(seed=seed)
    renderer = ArenaRenderer(env, mode="human")

    if control_style == "direct":
        footer = "WASD / ARROWS move   •   SPACE fire (close target assist)   •   R restart   •   ESC quit"
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
                    env.reset(seed=seed)
                elif event.key == pygame.K_TAB and env.pending_choice_kind is None:
                    renderer.show_build_panel = not renderer.show_build_panel
                elif env.pending_choice_kind is not None and event.key in (
                    pygame.K_1,
                    pygame.K_2,
                    pygame.K_3,
                ):
                    index = event.key - pygame.K_1
                    if index < len(env.pending_choices):
                        env.choose_pending_choice(index)
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and env.pending_choice_kind is not None
            ):
                index = renderer.choice_at_position(event.pos)
                if index is not None:
                    env.choose_pending_choice(index)

        if (
            running
            and not env.done
            and env.pending_choice_kind is None
            and not renderer.show_build_panel
        ):
            action = choose_action(pygame.key.get_pressed())
            env.step(action)

        active_footer = (
            "Choose one card with the mouse or keys 1–3   •   The battle timer is paused"
            if env.pending_choice_kind is not None
            else (
                "SHIP BUILD  •  TAB closes this panel  •  The battle timer is paused"
                if renderer.show_build_panel
                else footer + "   •   TAB build"
            )
        )
        renderer.render(process_events=False, footer_text=active_footer)
        renderer.clock.tick(env.fps)

    renderer.close()
    env.close()


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
    play_manual(args.control_style, args.seed)


if __name__ == "__main__":
    main()
