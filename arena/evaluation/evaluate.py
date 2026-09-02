"""Load and visually evaluate either trained Part II DQN policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from typing import TYPE_CHECKING, Any

import numpy as np
import pygame

if TYPE_CHECKING:
    from stable_baselines3 import DQN

from arena.evaluation.benchmark import evaluate_model, write_benchmark
from arena.core.environment import ArenaEnv, ENVIRONMENT_SCHEMA_VERSION, OBSERVATION_NAMES
from arena.presentation.renderer import ArenaRenderer
from arena.settings import ARENA_EVIDENCE_DIR, metadata_path, model_path


def policy_readiness(control_style: str) -> tuple[bool, str]:
    """Return whether the default model and metadata match the live arena."""

    path = model_path(control_style)
    metadata_file = metadata_path(control_style)
    if not path.exists():
        return False, "MODEL MISSING"
    if not metadata_file.exists():
        return False, "METADATA MISSING"
    try:
        with metadata_file.open("r", encoding="utf-8") as source:
            metadata = json.load(source)
    except (OSError, json.JSONDecodeError):
        return False, "METADATA INVALID"
    if metadata.get("control_style") != control_style:
        return False, "WRONG CONTROL MODEL"
    if metadata.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        return False, "RETRAIN REQUIRED"
    if metadata.get("observation_names") != list(OBSERVATION_NAMES):
        return False, "RETRAIN REQUIRED"
    return True, "MODEL READY"


def load_policy(
    control_style: str, model_file: Path | None = None
) -> tuple[DQN, dict]:
    """Load a model and validate its saved environment contract."""

    path = model_file or model_path(control_style)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained {control_style} model found at {path}. "
            f"Train it with: python -m arena.train --control-style {control_style}"
        )
    metadata_file = (
        path.with_suffix(".metadata.json")
        if model_file is not None
        else metadata_path(control_style)
    )
    metadata = {}
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        if metadata.get("control_style") != control_style:
            raise ValueError(
                f"Model metadata is for {metadata.get('control_style')!r}, "
                f"not {control_style!r}"
            )
        if metadata.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError("Model was trained for an older arena version; retraining is required")
        if metadata.get("observation_names") != list(OBSERVATION_NAMES):
            raise ValueError("Model observation schema does not match the current arena")
    from arena.learning.cooldown import load_dqn

    return load_dqn(str(path), device="auto"), metadata


def watch_policy(
    model: DQN,
    control_style: str,
    *,
    episodes: int = 3,
    seed: int = 42,
    action_repeat: int = 4,
) -> None:
    """Render deterministic gameplay at smooth simulation-frame granularity."""

    env = ArenaEnv(control_style=control_style)
    observation, _ = env.reset(seed=seed)
    renderer = ArenaRenderer(env, mode="human")
    episode = 1
    frames_left = 0
    current_action = 0
    paused = False
    speed_options = (0.5, 1.0, 2.0, 4.0)
    speed_index = 1
    running = True

    try:
        while running:
            single_step = False
            for event in pygame.event.get():
                if renderer.handle_volume_event(event):
                    continue
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if env.done:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q, pygame.K_m):
                            running = False
                        elif event.key in (pygame.K_r, pygame.K_RETURN):
                            observation, _ = env.reset(seed=seed + episode - 1)
                            frames_left = 0
                            paused = False
                        elif event.key == pygame.K_n and episode < episodes:
                            episode += 1
                            observation, _ = env.reset(seed=seed + episode - 1)
                            frames_left = 0
                            paused = False
                    if renderer.show_help_overlay:
                        if event.key in (pygame.K_h, pygame.K_SLASH, pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                            renderer.show_help_overlay = False
                        continue
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        if renderer.show_build_panel:
                            renderer.show_build_panel = False
                        else:
                            running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_PERIOD:
                        single_step = True
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        speed_index = min(len(speed_options) - 1, speed_index + 1)
                    elif event.key == pygame.K_MINUS:
                        speed_index = max(0, speed_index - 1)
                    elif event.key == pygame.K_r:
                        observation, _ = env.reset(seed=seed + episode - 1)
                        frames_left = 0
                        paused = False
                    elif event.key == pygame.K_c:
                        renderer.cycle_theme()
                    elif event.key in (pygame.K_h, pygame.K_SLASH):
                        renderer.toggle_help()
                    elif event.key == pygame.K_v:
                        renderer.show_volume_slider = not renderer.show_volume_slider
                        if renderer.audio and renderer.show_volume_slider:
                            renderer.audio.play("click", minimum_interval_ms=50)
                    elif event.key == pygame.K_TAB:
                        renderer.show_build_panel = not renderer.show_build_panel
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if renderer.show_help_overlay:
                        renderer.show_help_overlay = False
                        continue
                    if env.done:
                        summary_action = renderer.episode_end_action_at_position(event.pos)
                        if summary_action == "replay":
                            observation, _ = env.reset(seed=seed + episode - 1)
                            frames_left = 0
                            paused = False
                        elif summary_action == "next" and episode < episodes:
                            episode += 1
                            observation, _ = env.reset(seed=seed + episode - 1)
                            frames_left = 0
                            paused = False
                        elif summary_action == "menu":
                            running = False
                    elif renderer.pause_at_position(event.pos):
                        paused = not paused

            if (
                running
                and not env.done
                and not renderer.show_build_panel
                and (not paused or single_step)
            ):
                if frames_left <= 0:
                    action, _ = model.predict(observation, deterministic=True)
                    current_action = int(np.asarray(action).item())
                    frames_left = action_repeat
                observation, _, _, _, _ = env.step(current_action)
                frames_left -= 1

            footer = (
                "SHIP BUILD  •  TAB closes this panel  •  playback paused"
                if renderer.show_build_panel
                else (
                    f"TRAINED DQN  •  deterministic  •  episode {episode}/{episodes}  •  "
                    f"seed {seed + episode - 1}  •  {speed_options[speed_index]:g}x  •  "
                    "P pause  . step  "
                    "+/- speed  TAB build  R replay  Esc exit"
                )
            )
            renderer.episode_end_has_next = episode < episodes
            renderer.render(
                process_events=False,
                footer_text=(
                    "AI PLAYBACK PAUSED  •  P or RESUME continues  •  . advances one frame"
                    if paused
                    else footer
                ),
                paused=paused,
            )

            renderer.clock.tick(
                max(1, round(env.fps * speed_options[speed_index]))
            )
    finally:
        renderer.close()
        env.close()


def build_parser(forced_control_style: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually evaluate a trained Part II DQN agent"
    )
    if forced_control_style is None:
        parser.add_argument(
            "--control-style", choices=("direct", "rotation"), required=True
        )
    else:
        parser.set_defaults(control_style=forced_control_style)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None, *, forced_control_style: str | None = None
) -> None:
    """Run evaluation, optionally with an explicit argument sequence."""

    args = build_parser(forced_control_style).parse_args(argv)
    try:
        model, metadata = load_policy(args.control_style, args.model)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    action_repeat = int(metadata.get("action_repeat", 4))
    if args.headless:
        rows, aggregate = evaluate_model(
            model,
            args.control_style,
            episodes=args.episodes,
            action_repeat=action_repeat,
            seed=args.seed,
        )
        output = ARENA_EVIDENCE_DIR / f"{args.control_style}_evaluation"
        write_benchmark(rows, aggregate, output)
        print(json.dumps(aggregate, indent=2))
    else:
        watch_policy(
            model,
            args.control_style,
            episodes=args.episodes,
            seed=args.seed,
            action_repeat=action_repeat,
        )


if __name__ == "__main__":
    main()


__all__ = ["load_policy", "main", "policy_readiness", "watch_policy"]
