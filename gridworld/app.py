"""Unified, presentation-ready Pygame application for Part I.

Run from the project root with::

    python -m gridworld

The existing command-line training and evaluation modules remain useful for
repeatable experiments.  This module is deliberately an application layer: it
does not change environment rewards, transitions, level layouts, or learned
models.  It provides one event loop for campaign play, free play, and visual
agent evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pygame

from gridworld.environment import (
    ACTION_NAMES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    GridWorldEnv,
)
from gridworld.levels import LEVELS


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "gridworld")

VIRTUAL_WIDTH = 1280
VIRTUAL_HEIGHT = 800
TARGET_FPS = 60
DEFAULT_AI_SPEED = 1.0


COLORS = {
    "ink": (229, 238, 250),
    "muted": (151, 168, 191),
    # Secondary copy still needs to remain readable after window scaling.
    "faint": (116, 134, 160),
    "night": (8, 14, 29),
    "night_2": (14, 24, 45),
    "panel": (20, 32, 56),
    "panel_2": (27, 43, 72),
    "line": (58, 78, 108),
    "blue": (77, 157, 255),
    "cyan": (77, 222, 224),
    "green": (91, 214, 151),
    "yellow": (255, 205, 87),
    "orange": (255, 133, 80),
    "red": (255, 92, 112),
    "purple": (171, 121, 255),
    "floor_a": (30, 47, 70),
    "floor_b": (34, 53, 78),
    "rock": (84, 91, 111),
    "rock_dark": (51, 58, 78),
    "fire": (244, 73, 65),
    "fire_hot": (255, 205, 74),
    "apple": (91, 214, 121),
    "apple_dark": (38, 133, 77),
    "chest": (167, 105, 67),
    "chest_dark": (101, 61, 47),
    "monster": (225, 68, 105),
    "monster_dark": (126, 38, 76),
    "player": (70, 161, 255),
    "player_dark": (28, 81, 166),
}


LEVEL_META: Dict[int, Dict[str, str]] = {
    0: {
        "title": "Orchard Run",
        "task": "Task 1 - Q-Learning",
        "objective": "Collect every apple. Rocks block movement; find the shortest complete route.",
        "mechanics": "APPLES  /  ROCKS  /  SHORTEST PATH",
        "accent": "blue",
    },
    1: {
        "title": "Cliffside Harvest",
        "task": "Task 2 - SARSA",
        "objective": "Collect every apple without touching fire. Compare the risky and conservative policies.",
        "mechanics": "FIRE  /  RISK  /  SAFE ROUTES",
        "accent": "orange",
    },
    2: {
        "title": "The Locked Cache",
        "task": "Task 3 - Planning",
        "objective": "Collect the apples and key, then use the key to open the chest.",
        "mechanics": "APPLES  /  KEY  /  CHEST",
        "accent": "yellow",
    },
    3: {
        "title": "Stone Labyrinth",
        "task": "Task 3 - Planning",
        "objective": "Plan through the rock corridors, collect the key, and claim every reward.",
        "mechanics": "MAZE  /  KEY  /  CHEST",
        "accent": "purple",
    },
    4: {
        "title": "First Contact",
        "task": "Task 4 - Stochasticity",
        "objective": "Collect every apple while a monster has a 40% chance to move after each action.",
        "mechanics": "1 MONSTER  /  40% MOVE CHANCE",
        "accent": "red",
    },
    5: {
        "title": "Hunter's Maze",
        "task": "Task 4 - Stochasticity",
        "objective": "Navigate tight corridors, avoid two moving monsters, and collect every apple.",
        "mechanics": "2 MONSTERS  /  CORRIDORS",
        "accent": "red",
    },
    6: {
        "title": "Curiosity Vault",
        "task": "Task 5 - Intrinsic Reward",
        "objective": "Explore the sparse-reward maze and compare ordinary learning with curiosity-driven learning.",
        "mechanics": "SPARSE REWARD  /  CURIOSITY",
        "accent": "cyan",
    },
}


CAMPAIGN_AI_MODELS: Dict[int, Tuple[str, bool]] = {
    0: ("qlearning", False),
    1: ("sarsa", False),
    2: ("qlearning", False),
    3: ("sarsa", False),
    4: ("qlearning", False),
    5: ("sarsa", False),
    6: ("qlearning", True),
}


# Compact labels keep the level cards scannable.  The full task names remain in
# the play sidebar and documentation, where there is enough room for them.
LEVEL_CARD_LABELS = {
    0: "TASK 1  •  Q-LEARNING",
    1: "TASK 2  •  SARSA",
    2: "TASK 3  •  KEY + CHEST",
    3: "TASK 3  •  KEY + CHEST",
    4: "TASK 4  •  MONSTERS",
    5: "TASK 4  •  MONSTERS",
    6: "TASK 5  •  INTRINSIC REWARD",
}


LEVEL_HELP = {
    0: "Rocks block movement. A blocked action leaves the agent on its current tile.",
    1: "Fire ends the episode immediately. The lower route is shorter but riskier during exploration.",
    2: "The key gives 0 reward but is required before the chest can be opened for +2.",
    3: "Plan around rock corridors: collect the key, then reach the chest and every apple.",
    4: "After each agent action, the monster has a 40% chance to make one valid random move.",
    5: "Both monsters independently have a 40% move chance. Contact in either direction is fatal.",
    6: "Sparse, distant apples make exploration difficult; this level demonstrates the count bonus.",
}


@dataclass
class UIButton:
    rect: pygame.Rect
    action: Tuple[Any, ...]
    enabled: bool = True


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    radius: float
    color: Tuple[int, int, int]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ease_out_cubic(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return 1.0 - (1.0 - value) ** 3


def _friendly_agent(agent_kind: Optional[str]) -> str:
    if agent_kind == "qlearning":
        return "Q-Learning"
    if agent_kind == "sarsa":
        return "SARSA"
    return "Manual"


class GridworldApp:
    """One-window application that orchestrates all Part I experiences."""

    def __init__(self, max_steps: Optional[int] = None) -> None:
        pygame.init()
        pygame.font.init()

        self.window = pygame.display.set_mode(
            (VIRTUAL_WIDTH, VIRTUAL_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption("Gridworld AI Lab - Part I")
        self.canvas = pygame.Surface((VIRTUAL_WIDTH, VIRTUAL_HEIGHT))
        self.clock = pygame.time.Clock()
        self.viewport = pygame.Rect(0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT)
        self.viewport_scale = 1.0

        self.fonts = {
            "hero": pygame.font.SysFont("segoeui", 54, bold=True),
            "h1": pygame.font.SysFont("segoeui", 36, bold=True),
            "h2": pygame.font.SysFont("segoeui", 25, bold=True),
            "h3": pygame.font.SysFont("segoeui", 19, bold=True),
            "card_title": pygame.font.SysFont("segoeui", 17, bold=True),
            "body": pygame.font.SysFont("segoeui", 17),
            "small": pygame.font.SysFont("segoeui", 14),
            "tiny": pygame.font.SysFont("segoeui", 12),
            "mono": pygame.font.SysFont("consolas", 15),
            "mono_small": pygame.font.SysFont("consolas", 12),
        }

        config = self._load_config()
        self.monster_move_chance = float(
            config.get("monster", {}).get("move_chance", 0.4)
        )
        configured_max = int(config.get("training", {}).get("max_steps", 500))
        self.max_steps = max(1, max_steps if max_steps is not None else configured_max)

        self.running = True
        self.scene = "menu"
        self.buttons: List[UIButton] = []
        self.mouse_virtual = (0, 0)
        self.elapsed = 0.0
        self.notice = ""
        self.notice_time = 0.0

        self.level_select_context = "free"
        self.selected_level = min(LEVELS)
        self.session_origin = "free"
        self.control_mode = "manual"
        self.campaign_mode = False
        self.current_level = min(LEVELS)
        self.current_agent_kind: Optional[str] = None
        self.current_intrinsic = False

        self.env: Optional[GridWorldEnv] = None
        self.agent: Any = None
        self.state: Any = None
        self.total_reward = 0.0
        self.steps = 0
        self.run_done = False
        self.result_kind = ""
        self.result_detail = ""
        self.info: Dict[str, Any] = {}
        self.paused = False
        self.speed_options = [0.5, 1.0, 2.0, 4.0, 8.0]
        self.speed_index = self.speed_options.index(DEFAULT_AI_SPEED)
        self.ai_accumulator = 0.0
        self.show_policy = False
        self.policy_cells: Dict[Tuple[int, int], Tuple[List[int], List[float]]] = {}
        self.last_action: Optional[int] = None
        self.trail: List[Tuple[int, int]] = []
        self.event_text = ""
        self.event_time = 0.0

        self.anim_from_agent: Optional[Tuple[float, float]] = None
        self.anim_to_agent: Optional[Tuple[float, float]] = None
        self.anim_from_monsters: List[Tuple[float, float]] = []
        self.anim_to_monsters: List[Tuple[float, float]] = []
        self.move_anim = 1.0
        self.move_anim_duration = 0.13
        self.particles: List[Particle] = []

        self.missing_level = 0
        self.missing_agent_kind = "qlearning"
        self.missing_intrinsic = False
        self.missing_reason = ""
        self.missing_origin = "showcase"

    # ------------------------------------------------------------------
    # Lifecycle and event ownership
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def run(self) -> None:
        """Run the app until the player closes the window."""
        while self.running:
            dt = min(self.clock.tick(TARGET_FPS) / 1000.0, 0.1)
            self.elapsed += dt
            self._poll_events()
            self._update(dt)
            self._draw()
            self._present()

        pygame.quit()

    def _poll_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if event.type == pygame.VIDEORESIZE:
                width = max(800, event.w)
                height = max(520, event.h)
                self.window = pygame.display.set_mode(
                    (width, height), pygame.RESIZABLE
                )
                continue

            if event.type == pygame.MOUSEMOTION:
                self.mouse_virtual = self._screen_to_virtual(event.pos)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                virtual_pos = self._screen_to_virtual(event.pos)
                for button in reversed(self.buttons):
                    if button.enabled and button.rect.collidepoint(virtual_pos):
                        self._handle_action(button.action)
                        break

            if event.type == pygame.KEYDOWN:
                self._handle_key(event)

    def _screen_to_virtual(self, pos: Sequence[int]) -> Tuple[int, int]:
        if self.viewport_scale <= 0:
            return (0, 0)
        x = int((pos[0] - self.viewport.x) / self.viewport_scale)
        y = int((pos[1] - self.viewport.y) / self.viewport_scale)
        return (x, y)

    def _handle_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_ESCAPE:
            if self.scene == "menu":
                self.running = False
            elif self.scene in ("level_select", "campaign_select", "algorithm_select", "model_missing"):
                self._go_menu()
            elif self.scene == "play":
                self._go_menu()
            return

        if self.scene == "menu":
            if event.key == pygame.K_c:
                self.scene = "campaign_select"
            elif event.key == pygame.K_f:
                self.level_select_context = "free"
                self.scene = "level_select"
            elif event.key == pygame.K_a:
                self.level_select_context = "showcase"
                self.scene = "level_select"
            elif event.key == pygame.K_q:
                self.running = False
            return

        if self.scene == "level_select":
            if event.unicode and event.unicode.isdigit():
                level = int(event.unicode)
                if level in LEVELS:
                    self._choose_level(level)
            return

        if self.scene == "algorithm_select":
            if event.key == pygame.K_q:
                self._start_level(
                    self.selected_level, "ai", "qlearning", False, "showcase"
                )
            elif event.key == pygame.K_s:
                self._start_level(
                    self.selected_level, "ai", "sarsa", False, "showcase"
                )
            return

        if self.scene != "play":
            return

        if self.run_done:
            if event.key == pygame.K_r:
                self._retry_level()
            elif event.key in (pygame.K_n, pygame.K_RETURN):
                self._next_level()
            elif event.key == pygame.K_m:
                self._go_menu()
            return

        if event.key == pygame.K_r:
            self._retry_level()
            return
        if event.key == pygame.K_m:
            self._go_menu()
            return
        if event.key == pygame.K_p:
            self.show_policy = not self.show_policy
            self._set_notice("Policy lens on" if self.show_policy else "Policy lens off")
            return

        if self.control_mode == "ai":
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
                self._set_notice("Playback paused" if self.paused else "Playback resumed")
            elif event.key in (pygame.K_PERIOD, pygame.K_RIGHT) and self.paused:
                self._perform_ai_step()
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_RIGHTBRACKET):
                self._change_speed(1)
            elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                self._change_speed(-1)
            return

        if self.paused:
            return

        key_actions = {
            pygame.K_UP: UP,
            pygame.K_w: UP,
            pygame.K_DOWN: DOWN,
            pygame.K_s: DOWN,
            pygame.K_LEFT: LEFT,
            pygame.K_a: LEFT,
            pygame.K_RIGHT: RIGHT,
            pygame.K_d: RIGHT,
        }
        action = key_actions.get(event.key)
        if action is not None:
            self._perform_step(action)

    def _handle_action(self, action: Tuple[Any, ...]) -> None:
        name = action[0]
        if name == "quit":
            self.running = False
        elif name == "menu":
            self._go_menu()
        elif name == "campaign_menu":
            self.scene = "campaign_select"
        elif name == "campaign_manual":
            self.campaign_mode = True
            self._start_level(min(LEVELS), "manual", None, False, "campaign")
        elif name == "campaign_ai":
            self.campaign_mode = True
            self.speed_index = self.speed_options.index(DEFAULT_AI_SPEED)
            kind, intrinsic = CAMPAIGN_AI_MODELS.get(min(LEVELS), ("qlearning", False))
            self._start_level(min(LEVELS), "ai", kind, intrinsic, "campaign")
        elif name == "free_select":
            self.level_select_context = "free"
            self.scene = "level_select"
        elif name == "showcase_select":
            self.level_select_context = "showcase"
            self.scene = "level_select"
        elif name == "level":
            self._choose_level(int(action[1]))
        elif name == "start_ai":
            self.speed_index = self.speed_options.index(DEFAULT_AI_SPEED)
            self._start_level(
                self.selected_level,
                "ai",
                str(action[1]),
                bool(action[2]),
                "showcase",
            )
        elif name == "retry":
            self._retry_level()
        elif name == "next":
            self._next_level()
        elif name == "toggle_pause":
            self.paused = not self.paused
        elif name == "single_step":
            if self.control_mode == "ai" and not self.run_done:
                self.paused = True
                self._perform_ai_step()
        elif name == "speed":
            self._change_speed(int(action[1]))
        elif name == "policy":
            self.show_policy = not self.show_policy
        elif name == "back_to_levels":
            self.scene = "level_select"
        elif name == "missing_back":
            self.scene = (
                "campaign_select"
                if self.missing_origin == "campaign"
                else "level_select"
            )
        elif name == "missing_manual":
            origin = "campaign" if self.missing_origin == "campaign" else "free"
            self._start_level(self.missing_level, "manual", None, False, origin)

    # ------------------------------------------------------------------
    # Scene transitions and episode control
    # ------------------------------------------------------------------

    def _go_menu(self) -> None:
        self.scene = "menu"
        self.env = None
        self.agent = None
        self.state = None
        self.buttons = []
        self.campaign_mode = False
        self.particles.clear()

    def _choose_level(self, level: int) -> None:
        self.selected_level = level
        if self.level_select_context == "free":
            self.campaign_mode = False
            self._start_level(level, "manual", None, False, "free")
        else:
            self.scene = "algorithm_select"

    def _model_path(self, level: int, agent_kind: str, intrinsic: bool) -> str:
        suffix = "_intrinsic" if intrinsic else ""
        return os.path.join(
            MODELS_DIR, f"level{level}_{agent_kind}{suffix}.pkl"
        )

    def _load_agent(self, agent_kind: str, path: str) -> Any:
        if agent_kind == "qlearning":
            from gridworld.agents.q_learning import QLearningAgent

            agent = QLearningAgent()
        elif agent_kind == "sarsa":
            from gridworld.agents.sarsa import SARSAAgent

            agent = SARSAAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_kind}")

        agent.load(path)
        agent.epsilon = 0.0
        return agent

    @staticmethod
    def _unwrap_reset(result: Any) -> Any:
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            return result[0]
        return result

    def _start_level(
        self,
        level: int,
        control_mode: str,
        agent_kind: Optional[str],
        intrinsic: bool,
        origin: str,
    ) -> None:
        model_path = ""
        if control_mode == "ai":
            assert agent_kind is not None
            model_path = self._model_path(level, agent_kind, intrinsic)
            if not os.path.exists(model_path):
                self._show_missing_model(
                    level,
                    agent_kind,
                    intrinsic,
                    f"No trained model was found at {model_path}",
                    origin,
                )
                return

        try:
            env = GridWorldEnv(
                level, monster_move_chance=self.monster_move_chance
            )
            state = self._unwrap_reset(env.reset())
            agent = (
                self._load_agent(agent_kind, model_path)
                if control_mode == "ai" and agent_kind is not None
                else None
            )
            if agent is not None:
                q_table = getattr(agent, "q_table", {})
                if state not in q_table:
                    raise ValueError(
                        "The model does not contain the level's initial state. "
                        "Its state schema may be outdated; retrain it with the current environment."
                    )
        except Exception as exc:  # Present model/schema failures in-window.
            if control_mode == "ai" and agent_kind is not None:
                self._show_missing_model(
                    level, agent_kind, intrinsic, str(exc), origin
                )
                return
            raise

        self.env = env
        self.agent = agent
        self.state = state
        self.scene = "play"
        self.current_level = level
        self.selected_level = level
        self.control_mode = control_mode
        self.current_agent_kind = agent_kind
        self.current_intrinsic = intrinsic
        self.session_origin = origin
        self.campaign_mode = origin == "campaign"

        self.total_reward = 0.0
        self.steps = 0
        self.run_done = False
        self.result_kind = ""
        self.result_detail = ""
        self.info = {}
        self.paused = False
        self.ai_accumulator = 0.0
        self.last_action = None
        self.trail = [tuple(self.env.agent_pos)]
        self.event_text = "Manual control ready" if control_mode == "manual" else "Greedy policy loaded - epsilon = 0"
        self.event_time = 2.5
        self.anim_from_agent = tuple(self.env.agent_pos)
        self.anim_to_agent = tuple(self.env.agent_pos)
        self.anim_from_monsters = [tuple(p) for p in self.env.monster_positions]
        self.anim_to_monsters = list(self.anim_from_monsters)
        self.move_anim = 1.0
        self.particles.clear()
        self._refresh_policy_cache()

    def _show_missing_model(
        self,
        level: int,
        agent_kind: str,
        intrinsic: bool,
        reason: str,
        origin: str = "showcase",
    ) -> None:
        self.missing_level = level
        self.missing_agent_kind = agent_kind
        self.missing_intrinsic = intrinsic
        self.missing_reason = reason
        self.missing_origin = origin
        self.scene = "model_missing"

    def _retry_level(self) -> None:
        if self.env is None:
            return
        self._start_level(
            self.current_level,
            self.control_mode,
            self.current_agent_kind,
            self.current_intrinsic,
            self.session_origin,
        )

    def _next_level(self) -> None:
        next_level = self.current_level + 1
        if next_level not in LEVELS:
            self._go_menu()
            self._set_notice("Campaign complete - all seven levels cleared!")
            return

        if self.campaign_mode and self.result_kind != "victory":
            self._set_notice("Clear this level to unlock the next campaign stage")
            return

        if self.campaign_mode and self.control_mode == "ai":
            kind, intrinsic = CAMPAIGN_AI_MODELS.get(
                next_level, ("qlearning", False)
            )
            self._start_level(next_level, "ai", kind, intrinsic, "campaign")
        else:
            self._start_level(
                next_level,
                self.control_mode,
                self.current_agent_kind,
                self.current_intrinsic,
                self.session_origin,
            )

    def _perform_ai_step(self) -> None:
        if self.agent is None or self.run_done:
            return
        try:
            action = int(self.agent.choose_action(self.state))
        except Exception as exc:
            self._finish_run("error", f"Agent could not select an action: {exc}")
            return
        self._perform_step(action)

    def _perform_step(self, action: int) -> None:
        if self.env is None or self.run_done:
            return

        before_agent = tuple(self.env.agent_pos)
        before_monsters = [tuple(p) for p in self.env.monster_positions]

        try:
            result = self.env.step(action)
        except Exception as exc:
            self._finish_run("error", f"Environment step failed: {exc}")
            return

        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            next_state, reward, done, info = result

        self.state = next_state
        self.total_reward += float(reward)
        self.steps += 1
        self.info = dict(info or {})
        self.last_action = action
        self.trail.append(tuple(self.env.agent_pos))

        self.anim_from_agent = before_agent
        self.anim_to_agent = tuple(self.env.agent_pos)
        self.anim_from_monsters = before_monsters
        self.anim_to_monsters = [tuple(p) for p in self.env.monster_positions]
        self.move_anim = 0.0
        interval = 1.0 / (3.0 * self.speed_options[self.speed_index])
        self.move_anim_duration = max(0.035, min(0.14, interval * 0.72))

        picked = self.info.get("picked_up")
        if picked:
            reward_text = "" if picked == "key" else f"  +{float(reward):.0f} reward"
            self.event_text = f"Collected {picked}!{reward_text}"
            self.event_time = 2.0
            particle_color = COLORS["yellow"] if picked in ("key", "chest") else COLORS["green"]
            self._spawn_particles(tuple(self.env.agent_pos), particle_color, 18)

        self._refresh_policy_cache()

        if done:
            if self.info.get("victory"):
                self._finish_run("victory", "Every collectible reward was obtained.")
            elif self.info.get("death"):
                cause = str(self.info.get("death", "hazard"))
                self._finish_run("death", f"The agent was defeated by {cause}.")
            else:
                self._finish_run("complete", "The environment ended the episode.")
        elif self.steps >= self.max_steps:
            self._finish_run(
                "timeout",
                f"Evaluation stopped safely at the {self.max_steps}-step limit.",
            )

    def _finish_run(self, result_kind: str, detail: str) -> None:
        if self.run_done:
            return
        self.run_done = True
        self.result_kind = result_kind
        self.result_detail = detail
        self.paused = True
        color = COLORS["green"] if result_kind == "victory" else COLORS["red"]
        if self.env is not None:
            self._spawn_particles(tuple(self.env.agent_pos), color, 32)

    def _change_speed(self, direction: int) -> None:
        self.speed_index = int(
            _clamp(self.speed_index + direction, 0, len(self.speed_options) - 1)
        )
        self.ai_accumulator = 0.0
        self._set_notice(f"Playback speed: {self.speed_options[self.speed_index]:g}x")

    def _set_notice(self, text: str, seconds: float = 2.0) -> None:
        self.notice = text
        self.notice_time = seconds

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update(self, dt: float) -> None:
        self.notice_time = max(0.0, self.notice_time - dt)
        self.event_time = max(0.0, self.event_time - dt)
        self.move_anim = min(
            1.0, self.move_anim + dt / max(self.move_anim_duration, 0.001)
        )

        alive_particles: List[Particle] = []
        for particle in self.particles:
            particle.life -= dt
            if particle.life <= 0:
                continue
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vy += 70.0 * dt
            alive_particles.append(particle)
        self.particles = alive_particles

        if (
            self.scene == "play"
            and self.control_mode == "ai"
            and not self.paused
            and not self.run_done
        ):
            interval = 1.0 / (3.0 * self.speed_options[self.speed_index])
            self.ai_accumulator += dt
            if self.ai_accumulator >= interval:
                self.ai_accumulator %= interval
                self._perform_ai_step()

    # ------------------------------------------------------------------
    # Drawing foundation
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self.buttons = []
        self._draw_background()

        if self.scene == "menu":
            self._draw_menu()
        elif self.scene == "campaign_select":
            self._draw_campaign_select()
        elif self.scene == "level_select":
            self._draw_level_select()
        elif self.scene == "algorithm_select":
            self._draw_algorithm_select()
        elif self.scene == "model_missing":
            self._draw_model_missing()
        elif self.scene == "play":
            self._draw_play()

        if self.notice_time > 0 and self.notice:
            self._draw_toast(self.notice)

    def _draw_background(self) -> None:
        self.canvas.fill(COLORS["night"])
        for y in range(VIRTUAL_HEIGHT):
            factor = y / VIRTUAL_HEIGHT
            color = (
                int(8 + 8 * factor),
                int(14 + 12 * factor),
                int(29 + 20 * factor),
            )
            pygame.draw.line(self.canvas, color, (0, y), (VIRTUAL_WIDTH, y))

        # A restrained animated node network gives the menu identity without assets.
        for index in range(24):
            x = int((index * 193 + 70) % VIRTUAL_WIDTH)
            base_y = (index * 89 + 40) % VIRTUAL_HEIGHT
            y = int(base_y + math.sin(self.elapsed * 0.55 + index) * 9)
            radius = 2 + index % 3
            pygame.draw.circle(self.canvas, (33, 61, 96), (x, y), radius)
            if index % 3 == 0:
                nx = int(((index + 3) * 193 + 70) % VIRTUAL_WIDTH)
                ny_base = ((index + 3) * 89 + 40) % VIRTUAL_HEIGHT
                ny = int(ny_base + math.sin(self.elapsed * 0.55 + index + 3) * 9)
                pygame.draw.aaline(self.canvas, (24, 45, 74), (x, y), (nx, ny))

    def _present(self) -> None:
        window_width, window_height = self.window.get_size()
        scale = min(window_width / VIRTUAL_WIDTH, window_height / VIRTUAL_HEIGHT)
        draw_width = max(1, int(VIRTUAL_WIDTH * scale))
        draw_height = max(1, int(VIRTUAL_HEIGHT * scale))
        x = (window_width - draw_width) // 2
        y = (window_height - draw_height) // 2
        self.viewport = pygame.Rect(x, y, draw_width, draw_height)
        self.viewport_scale = scale

        self.window.fill((3, 7, 16))
        if draw_width == VIRTUAL_WIDTH and draw_height == VIRTUAL_HEIGHT:
            scaled = self.canvas
        else:
            scaled = pygame.transform.smoothscale(
                self.canvas, (draw_width, draw_height)
            )
        self.window.blit(scaled, (x, y))
        pygame.display.flip()

    def _panel(
        self,
        rect: pygame.Rect,
        fill: Tuple[int, int, int] = COLORS["panel"],
        border: Tuple[int, int, int] = COLORS["line"],
        radius: int = 18,
        shadow: bool = True,
    ) -> None:
        if shadow:
            shadow_rect = rect.move(0, 7)
            shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                shadow_surface,
                (0, 0, 0, 105),
                shadow_surface.get_rect(),
                border_radius=radius,
            )
            self.canvas.blit(shadow_surface, shadow_rect)
        pygame.draw.rect(self.canvas, fill, rect, border_radius=radius)
        pygame.draw.rect(self.canvas, border, rect, 1, border_radius=radius)

    def _text(
        self,
        text: str,
        pos: Tuple[int, int],
        font: str = "body",
        color: Tuple[int, int, int] = COLORS["ink"],
        anchor: str = "topleft",
        max_width: Optional[int] = None,
    ) -> pygame.Rect:
        value = str(text)
        if max_width is not None:
            value = self._ellipsize(value, font, max_width)
        surface = self.fonts[font].render(value, True, color)
        rect = surface.get_rect()
        setattr(rect, anchor, pos)
        self.canvas.blit(surface, rect)
        return rect

    def _ellipsize(self, text: str, font: str, max_width: int) -> str:
        """Fit one line inside ``max_width`` without leaking into neighbours."""
        value = str(text)
        font_obj = self.fonts[font]
        if max_width <= 0:
            return ""
        if font_obj.size(value)[0] <= max_width:
            return value
        suffix = "…"
        if font_obj.size(suffix)[0] > max_width:
            return ""
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if font_obj.size(value[:middle].rstrip() + suffix)[0] <= max_width:
                low = middle
            else:
                high = middle - 1
        return value[:low].rstrip() + suffix

    def _wrapped_text(
        self,
        text: str,
        rect: pygame.Rect,
        font: str = "body",
        color: Tuple[int, int, int] = COLORS["muted"],
        line_gap: int = 4,
        max_lines: Optional[int] = None,
    ) -> int:
        words = str(text).split()
        all_lines: List[str] = []
        current = ""
        font_obj = self.fonts[font]
        for word in words:
            if font_obj.size(word)[0] > rect.width:
                word = self._ellipsize(word, font, rect.width)
            candidate = word if not current else f"{current} {word}"
            if font_obj.size(candidate)[0] <= rect.width:
                current = candidate
            else:
                if current:
                    all_lines.append(current)
                current = word
        if current:
            all_lines.append(current)

        line_height = font_obj.get_linesize() + line_gap
        height_limit = max(0, (rect.height + line_gap) // max(1, line_height))
        line_limit = height_limit if max_lines is None else min(height_limit, max_lines)
        lines = all_lines[:line_limit]
        if lines and len(all_lines) > len(lines):
            lines[-1] = self._ellipsize(lines[-1] + "…", font, rect.width)

        y = rect.y
        previous_clip = self.canvas.get_clip()
        self.canvas.set_clip(rect)
        for line in lines:
            surface = font_obj.render(line, True, color)
            self.canvas.blit(surface, (rect.x, y))
            y += line_height
        self.canvas.set_clip(previous_clip)
        return y

    def _draw_chips(
        self,
        labels: Iterable[str],
        rect: pygame.Rect,
        accent: Tuple[int, int, int],
        max_rows: int = 2,
    ) -> None:
        """Draw compact wrapping mechanic tags inside a bounded card region."""
        font_obj = self.fonts["tiny"]
        x, y = rect.x, rect.y
        row = 0
        height = 21
        gap = 6
        for raw_label in labels:
            label = str(raw_label).upper()
            width = min(rect.width, font_obj.size(label)[0] + 18)
            if x + width > rect.right and x > rect.x:
                row += 1
                if row >= max_rows:
                    break
                x = rect.x
                y += height + 5
            chip = pygame.Rect(x, y, width, height)
            pygame.draw.rect(self.canvas, COLORS["night_2"], chip, border_radius=10)
            pygame.draw.rect(self.canvas, accent, chip, 1, border_radius=10)
            self._text(
                label,
                chip.center,
                "tiny",
                accent,
                "center",
                max_width=chip.width - 12,
            )
            x += width + gap

    def _button(
        self,
        rect: pygame.Rect,
        label: str,
        action: Tuple[Any, ...],
        subtitle: str = "",
        kind: str = "secondary",
        enabled: bool = True,
        badge: str = "",
    ) -> None:
        hover = rect.collidepoint(self.mouse_virtual) and enabled
        palette = {
            "primary": (COLORS["blue"], (101, 177, 255), (10, 33, 65)),
            "secondary": (COLORS["panel_2"], (37, 57, 91), COLORS["ink"]),
            "success": (COLORS["green"], (114, 230, 172), (8, 46, 38)),
            "warning": (COLORS["yellow"], (255, 221, 120), (55, 38, 8)),
            "danger": (COLORS["red"], (255, 119, 135), (56, 13, 26)),
        }
        normal, hovered, text_color = palette.get(kind, palette["secondary"])
        if not enabled:
            normal = (38, 47, 64)
            hovered = normal
            text_color = COLORS["faint"]

        shadow = rect.move(0, 5)
        pygame.draw.rect(self.canvas, (3, 8, 18), shadow, border_radius=13)
        pygame.draw.rect(
            self.canvas, hovered if hover else normal, rect, border_radius=13
        )
        outline = (255, 255, 255) if hover else COLORS["line"]
        pygame.draw.rect(self.canvas, outline, rect, 1, border_radius=13)

        label_y = rect.centery if not subtitle else rect.y + 17
        badge_width = self.fonts["tiny"].size(badge)[0] + 12 if badge else 0
        text_width = max(1, rect.width - 36 - badge_width)
        self._text(
            label,
            (rect.x + 18, label_y),
            "h3",
            text_color,
            "midleft" if not subtitle else "topleft",
            max_width=text_width,
        )
        if subtitle:
            self._text(
                subtitle,
                (rect.x + 18, rect.y + 48),
                "small",
                text_color if kind in ("primary", "success", "warning", "danger") else COLORS["muted"],
                max_width=text_width,
            )
        if badge:
            badge_surface = self.fonts["tiny"].render(badge, True, text_color)
            badge_rect = badge_surface.get_rect()
            badge_rect.midright = (rect.right - 15, rect.centery)
            self.canvas.blit(badge_surface, badge_rect)

        self.buttons.append(UIButton(rect.copy(), action, enabled))

    def _back_button(self, action: Tuple[Any, ...] = ("menu",)) -> None:
        self._button(
            pygame.Rect(36, 28, 120, 45),
            "<  Back",
            action,
            kind="secondary",
        )

    def _section_heading(self, eyebrow: str, title: str, subtitle: str = "") -> None:
        self._text(eyebrow.upper(), (VIRTUAL_WIDTH // 2, 45), "small", COLORS["cyan"], "midtop")
        self._text(
            title,
            (VIRTUAL_WIDTH // 2, 72),
            "h1",
            COLORS["ink"],
            "midtop",
            max_width=860,
        )
        if subtitle:
            self._text(
                subtitle,
                (VIRTUAL_WIDTH // 2, 120),
                "body",
                COLORS["muted"],
                "midtop",
                max_width=920,
            )

    # ------------------------------------------------------------------
    # Menu scenes
    # ------------------------------------------------------------------

    def _draw_menu(self) -> None:
        self._draw_logo(pygame.Rect(80, 120, 390, 390))

        self._text("RMIT  /  GAMES & AI", (585, 105), "small", COLORS["cyan"])
        self._text("GRIDWORLD", (580, 139), "hero", COLORS["ink"])
        self._text("AI LAB", (583, 198), "hero", COLORS["blue"])
        self._wrapped_text(
            "Seven connected levels. Two classical reinforcement-learning agents. One visual laboratory for playing, inspecting, and presenting learned behaviour.",
            pygame.Rect(586, 276, 585, 90),
            "body",
            COLORS["muted"],
            line_gap=5,
        )

        self._button(
            pygame.Rect(580, 380, 286, 83),
            "Campaign",
            ("campaign_menu",),
            "Play Levels 0 through 6",
            "primary",
            badge="C",
        )
        self._button(
            pygame.Rect(884, 380, 286, 83),
            "Free Play",
            ("free_select",),
            "Choose any level manually",
            "secondary",
            badge="F",
        )
        self._button(
            pygame.Rect(580, 483, 590, 83),
            "AI Showcase",
            ("showcase_select",),
            "Replay a trained Q-Learning, SARSA, or intrinsic-reward policy",
            "secondary",
            badge="A",
        )
        self._button(
            pygame.Rect(580, 586, 590, 57),
            "Exit",
            ("quit",),
            kind="secondary",
            badge="ESC",
        )

        chips = [
            ("4 ACTIONS", COLORS["blue"]),
            ("ENV REWARDS UNCHANGED", COLORS["green"]),
            ("PROCEDURAL ART", COLORS["purple"]),
        ]
        x = 580
        for label, color in chips:
            width = self.fonts["tiny"].size(label)[0] + 24
            rect = pygame.Rect(x, 681, width, 28)
            pygame.draw.rect(self.canvas, COLORS["panel"], rect, border_radius=14)
            pygame.draw.rect(self.canvas, color, rect, 1, border_radius=14)
            self._text(label, rect.center, "tiny", color, "center")
            x += width + 10

        self._text(
            "Mouse or keyboard: C Campaign  /  F Free Play  /  A AI Showcase  /  ESC Exit",
            (VIRTUAL_WIDTH // 2, 764),
            "small",
            COLORS["faint"],
            "center",
        )

    def _draw_logo(self, rect: pygame.Rect) -> None:
        self._panel(rect, (15, 28, 51), (45, 77, 117), radius=28)
        cell = 46
        ox = rect.x + 34
        oy = rect.y + 34
        for row in range(7):
            for col in range(7):
                tile = pygame.Rect(ox + col * cell, oy + row * cell, cell - 5, cell - 5)
                fill = COLORS["floor_a"] if (row + col) % 2 == 0 else COLORS["floor_b"]
                pygame.draw.rect(self.canvas, fill, tile, border_radius=8)
        # A miniature path, agent, goal, and danger make the purpose readable.
        points = [
            (ox + 22, oy + 6 * cell + 20),
            (ox + 22, oy + 3 * cell + 20),
            (ox + 3 * cell + 20, oy + 3 * cell + 20),
            (ox + 3 * cell + 20, oy + cell + 20),
            (ox + 6 * cell + 20, oy + cell + 20),
        ]
        pygame.draw.lines(self.canvas, COLORS["blue"], False, points, 5)
        for point in points[1:-1]:
            pygame.draw.circle(self.canvas, COLORS["cyan"], point, 5)
        pygame.draw.circle(self.canvas, COLORS["player_dark"], points[0], 18)
        pygame.draw.circle(self.canvas, COLORS["player"], points[0], 13)
        pygame.draw.circle(self.canvas, COLORS["apple"], points[-1], 14)
        pygame.draw.line(
            self.canvas,
            COLORS["apple_dark"],
            (points[-1][0], points[-1][1] - 13),
            (points[-1][0] + 5, points[-1][1] - 20),
            4,
        )
        pulse = 26 + int(4 * math.sin(self.elapsed * 2.0))
        pygame.draw.circle(self.canvas, COLORS["blue"], points[0], pulse, 2)

    def _draw_campaign_select(self) -> None:
        self._back_button(("menu",))
        self._section_heading(
            "Campaign",
            "Choose how to cross the seven levels",
            "Progression changes only presentation; every environment mechanic and reward stays identical.",
        )

        cards = [
            (
                pygame.Rect(120, 205, 490, 420),
                "Manual Journey",
                "YOU CONTROL THE AGENT",
                "Use arrows or WASD. Learn each mechanic in order, retry safely, and unlock the next level after victory.",
                COLORS["blue"],
                ("campaign_manual",),
                "Start manual campaign",
            ),
            (
                pygame.Rect(670, 205, 490, 420),
                "Guided AI Tour",
                "GREEDY TRAINED POLICIES",
                "Watch rubric-relevant models progress in order. Pause, single-step, change speed, and inspect policy values.",
                COLORS["purple"],
                ("campaign_ai",),
                "Start AI campaign",
            ),
        ]
        for rect, title, eyebrow, body, accent, action, button_label in cards:
            self._panel(rect, COLORS["panel"], accent, radius=23)
            pygame.draw.rect(
                self.canvas,
                accent,
                pygame.Rect(rect.x, rect.y, rect.width, 8),
                border_top_left_radius=23,
                border_top_right_radius=23,
            )
            self._text(eyebrow, (rect.x + 32, rect.y + 42), "tiny", accent)
            self._text(title, (rect.x + 32, rect.y + 73), "h1", COLORS["ink"])
            self._wrapped_text(
                body,
                pygame.Rect(rect.x + 32, rect.y + 132, rect.width - 64, 110),
                "body",
                COLORS["muted"],
                line_gap=6,
            )
            self._draw_campaign_track(
                pygame.Rect(rect.x + 32, rect.y + 250, rect.width - 64, 55), accent
            )
            self._button(
                pygame.Rect(rect.x + 32, rect.bottom - 88, rect.width - 64, 58),
                button_label,
                action,
                kind="primary" if action[0] == "campaign_manual" else "secondary",
            )

        self._text(
            "AI Tour model order: Q-Learning -> SARSA comparisons -> intrinsic Q-Learning on Level 6",
            (VIRTUAL_WIDTH // 2, 686),
            "small",
            COLORS["faint"],
            "center",
        )

    def _draw_campaign_track(self, rect: pygame.Rect, accent: Tuple[int, int, int]) -> None:
        y = rect.centery
        left = rect.x + 17
        right = rect.right - 17
        pygame.draw.line(self.canvas, COLORS["line"], (left, y), (right, y), 3)
        for index, level in enumerate(sorted(LEVELS)):
            ratio = index / max(1, len(LEVELS) - 1)
            x = int(left + (right - left) * ratio)
            pygame.draw.circle(self.canvas, COLORS["panel_2"], (x, y), 14)
            pygame.draw.circle(self.canvas, accent, (x, y), 14, 2)
            self._text(str(level), (x, y), "tiny", COLORS["ink"], "center")

    def _draw_level_select(self) -> None:
        self._back_button(("menu",))
        title = "Select a level to play" if self.level_select_context == "free" else "Select a level to inspect"
        subtitle = (
            "Manual play - every level is available."
            if self.level_select_context == "free"
            else "Next, choose Q-Learning, SARSA, or an intrinsic-reward model."
        )
        self._section_heading("Level Select", title, subtitle)

        levels = sorted(LEVELS)
        card_width = 274
        card_height = 244
        gap = 18
        top_y = 176
        for index, level in enumerate(levels):
            if index < 4:
                x = 56 + index * (card_width + gap)
                y = top_y
            else:
                row_count = len(levels) - 4
                row_width = row_count * card_width + (row_count - 1) * gap
                x = (VIRTUAL_WIDTH - row_width) // 2 + (index - 4) * (card_width + gap)
                y = top_y + card_height + 20
            self._draw_level_card(level, pygame.Rect(x, y, card_width, card_height))

        self._text(
            "Click a card or press 0–6 to open a level  •  Esc returns to the main menu",
            (VIRTUAL_WIDTH // 2, 756),
            "small",
            COLORS["faint"],
            "center",
        )

    def _draw_level_card(self, level: int, rect: pygame.Rect) -> None:
        meta = self._level_meta(level)
        accent = COLORS.get(meta.get("accent", "blue"), COLORS["blue"])
        hover = rect.collidepoint(self.mouse_virtual)
        fill = (30, 47, 77) if hover else COLORS["panel"]
        border = accent if hover else COLORS["line"]
        self._panel(rect, fill, border, radius=18)

        # Clip every card's content as a final containment guarantee.  Individual
        # labels also wrap or ellipsize, so clipping should only ever be a guard.
        previous_clip = self.canvas.get_clip()
        self.canvas.set_clip(rect.inflate(-2, -2))
        try:
            badge = pygame.Rect(rect.x + 17, rect.y + 16, 44, 44)
            pygame.draw.rect(self.canvas, accent, badge, border_radius=12)
            self._text(str(level), badge.center, "h2", COLORS["night"], "center")
            self._text(
                LEVEL_CARD_LABELS.get(level, meta.get("task", "GRIDWORLD")),
                (rect.x + 72, rect.y + 18),
                "tiny",
                accent,
                max_width=rect.width - 89,
            )
            self._text(
                meta.get("title", f"Level {level}"),
                (rect.x + 72, rect.y + 39),
                "card_title",
                COLORS["ink"],
                max_width=rect.width - 89,
            )
            self._wrapped_text(
                meta.get("objective", LEVELS[level].get("description", "")),
                pygame.Rect(rect.x + 17, rect.y + 75, rect.width - 34, 70),
                "small",
                COLORS["muted"],
                line_gap=2,
                max_lines=4,
            )

            mechanics = LEVELS[level].get("mechanics", ())
            self._draw_chips(
                mechanics,
                pygame.Rect(rect.x + 17, rect.y + 153, rect.width - 34, 47),
                accent,
                max_rows=2,
            )

            footer = pygame.Rect(rect.x + 11, rect.bottom - 34, rect.width - 22, 25)
            pygame.draw.rect(self.canvas, COLORS["night_2"], footer, border_radius=9)
            if self.level_select_context == "showcase":
                ready = self._available_model_count(level)
                noun = "policy" if ready == 1 else "policies"
                status = f"{ready} trained {noun}  •  Choose AI"
            else:
                status = "Play manually  •  Open level"
            self._text(
                status,
                (footer.x + 10, footer.centery),
                "tiny",
                COLORS["ink"],
                "midleft",
                max_width=footer.width - 34,
            )
            self._text("›", (footer.right - 13, footer.centery - 1), "h3", accent, "center")
        finally:
            self.canvas.set_clip(previous_clip)

        self.buttons.append(UIButton(rect.copy(), ("level", level), True))

    def _available_model_count(self, level: int) -> int:
        return sum(
            os.path.exists(self._model_path(level, kind, intrinsic))
            for kind in ("qlearning", "sarsa")
            for intrinsic in (False, True)
        )

    @staticmethod
    def _level_meta(level: int) -> Dict[str, str]:
        """Merge presentation fallbacks with canonical level metadata.

        Older level dictionaries only contained ``description`` and ``grid``;
        newer dictionaries expose titles, task names, objectives, and mechanic
        tags.  Supporting both keeps the application layer independent from a
        particular environment revision.
        """
        meta = dict(LEVEL_META.get(level, {}))
        spec = LEVELS.get(level, {})
        if spec.get("title"):
            meta["title"] = str(spec["title"])
        task_number = spec.get("task")
        task_name = spec.get("task_name")
        if task_number is not None or task_name:
            task_prefix = f"Task {task_number}" if task_number is not None else "Part I"
            meta["task"] = (
                f"{task_prefix} - {task_name}" if task_name else task_prefix
            )
        objectives = spec.get("objectives")
        if objectives:
            meta["objective"] = " ".join(str(item) for item in objectives)
        elif spec.get("description"):
            meta["objective"] = str(spec["description"])
        mechanics = spec.get("mechanics")
        if mechanics:
            meta["mechanics"] = "  /  ".join(
                str(item).upper() for item in mechanics
            )
        return meta

    def _draw_algorithm_select(self) -> None:
        self._back_button(("back_to_levels",))
        meta = self._level_meta(self.selected_level)
        self._section_heading(
            f"Level {self.selected_level} / AI Showcase",
            "Choose a learned policy",
            meta.get("objective", LEVELS[self.selected_level].get("description", "")),
        )

        choices = [
            (
                "qlearning",
                False,
                "Q-Learning",
                "Off-policy: learns from the best next-state action.",
                COLORS["blue"],
            ),
            (
                "sarsa",
                False,
                "SARSA",
                "On-policy: learns from the next action it actually chooses.",
                COLORS["purple"],
            ),
        ]
        if self.selected_level == 6:
            choices.extend(
                [
                    (
                        "qlearning",
                        True,
                        "Q + Intrinsic",
                        "Adds the required per-episode count-based exploration bonus.",
                        COLORS["cyan"],
                    ),
                    (
                        "sarsa",
                        True,
                        "SARSA + Intrinsic",
                        "Optional on-policy curiosity variant for experimentation.",
                        COLORS["green"],
                    ),
                ]
            )

        two_card_layout = len(choices) == 2
        for index, (kind, intrinsic, title, subtitle, accent) in enumerate(choices):
            col = index % 2
            row = index // 2
            rect = (
                pygame.Rect(150 + col * 500, 255, 480, 210)
                if two_card_layout
                else pygame.Rect(150 + col * 500, 205 + row * 205, 480, 175)
            )
            path = self._model_path(self.selected_level, kind, intrinsic)
            ready = os.path.exists(path)
            recommended = CAMPAIGN_AI_MODELS.get(self.selected_level) == (kind, intrinsic)
            hover = rect.collidepoint(self.mouse_virtual)
            self._panel(
                rect,
                (30, 47, 77) if hover else COLORS["panel"],
                accent if hover or ready else COLORS["line"],
                radius=19,
            )
            pygame.draw.circle(self.canvas, accent, (rect.x + 48, rect.y + 52), 22, 3)
            letter = "Q" if kind == "qlearning" else "S"
            self._text(letter, (rect.x + 48, rect.y + 52), "h3", accent, "center")
            self._text(
                title,
                (rect.x + 85, rect.y + 27),
                "h2",
                COLORS["ink"],
                max_width=rect.width - 115,
            )
            self._wrapped_text(
                subtitle,
                pygame.Rect(rect.x + 85, rect.y + 65, rect.width - 115, 46),
                "small",
                COLORS["muted"],
                line_gap=2,
                max_lines=2,
            )
            if recommended:
                tag = pygame.Rect(rect.right - 132, rect.y + 17, 108, 24)
                pygame.draw.rect(self.canvas, COLORS["night_2"], tag, border_radius=12)
                pygame.draw.rect(self.canvas, accent, tag, 1, border_radius=12)
                self._text("CAMPAIGN PICK", tag.center, "tiny", accent, "center")
            status_color = COLORS["green"] if ready else COLORS["yellow"]
            status = (
                "Ready  •  Click to watch the trained policy"
                if ready
                else "Not trained  •  Click to see the training command"
            )
            footer = pygame.Rect(rect.x + 22, rect.bottom - 47, rect.width - 44, 29)
            pygame.draw.rect(self.canvas, COLORS["night_2"], footer, border_radius=10)
            self._text(
                status,
                (footer.x + 11, footer.centery),
                "tiny",
                status_color,
                "midleft",
                max_width=footer.width - 38,
            )
            self._text("›", (footer.right - 14, footer.centery - 1), "h3", accent, "center")
            self.buttons.append(
                UIButton(rect.copy(), ("start_ai", kind, intrinsic), True)
            )

        scope_note = (
            "Level 6 includes baseline and intrinsic variants for the required exploration comparison."
            if self.selected_level == 6
            else "Intrinsic reward is a Level 6 requirement, so curiosity variants are shown there."
        )
        self._text(
            scope_note,
            (VIRTUAL_WIDTH // 2, 645),
            "small",
            COLORS["muted"],
            "center",
            max_width=950,
        )
        self._text(
            "Playback uses ε = 0 (no exploration). Exact ties and random monster movement can still vary.",
            (VIRTUAL_WIDTH // 2, 685),
            "small",
            COLORS["faint"],
            "center",
            max_width=950,
        )

    def _draw_model_missing(self) -> None:
        self._back_button(("missing_back",))
        card = pygame.Rect(190, 135, 900, 555)
        self._panel(card, COLORS["panel"], COLORS["yellow"], radius=25)
        pygame.draw.circle(self.canvas, COLORS["yellow"], (VIRTUAL_WIDTH // 2, 214), 34, 3)
        self._text("!", (VIRTUAL_WIDTH // 2, 214), "h1", COLORS["yellow"], "center")
        self._text("This policy is not ready yet", (VIRTUAL_WIDTH // 2, 274), "h1", COLORS["ink"], "center")
        label = f"Level {self.missing_level}  /  {_friendly_agent(self.missing_agent_kind)}"
        if self.missing_intrinsic:
            label += "  /  Intrinsic reward"
        self._text(label, (VIRTUAL_WIDTH // 2, 324), "body", COLORS["cyan"], "center")

        self._wrapped_text(
            self.missing_reason,
            pygame.Rect(270, 360, 740, 72),
            "small",
            COLORS["muted"],
            line_gap=4,
            max_lines=3,
        )

        intrinsic_flag = " --intrinsic" if self.missing_intrinsic else ""
        command = (
            f"python -m gridworld.train --level {self.missing_level} "
            f"--agent {self.missing_agent_kind}{intrinsic_flag}"
        )
        command_rect = pygame.Rect(270, 440, 740, 60)
        pygame.draw.rect(self.canvas, COLORS["night"], command_rect, border_radius=12)
        pygame.draw.rect(self.canvas, COLORS["line"], command_rect, 1, border_radius=12)
        self._text(command, command_rect.center, "mono", COLORS["green"], "center")

        self._button(
            pygame.Rect(270, 545, 230, 62),
            "Choose another",
            ("missing_back",),
            kind="secondary",
        )
        self._button(
            pygame.Rect(520, 545, 230, 62),
            "Play manually",
            ("missing_manual",),
            kind="primary",
        )
        self._button(
            pygame.Rect(770, 545, 240, 62),
            "Main menu",
            ("menu",),
            kind="secondary",
        )

    # ------------------------------------------------------------------
    # Game rendering
    # ------------------------------------------------------------------

    def _draw_play(self) -> None:
        if self.env is None:
            self._go_menu()
            return

        self._draw_game_topbar()
        board_rect, cell_size, origin = self._grid_geometry()
        self._draw_board(board_rect, cell_size, origin)
        self._draw_sidebar()

        for particle in self.particles:
            alpha = int(255 * particle.life / particle.max_life)
            radius = max(1, int(particle.radius * particle.life / particle.max_life))
            surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(
                surface,
                (*particle.color, alpha),
                (radius * 2, radius * 2),
                radius,
            )
            self.canvas.blit(surface, (particle.x - radius * 2, particle.y - radius * 2))

        if self.event_time > 0 and self.event_text:
            alpha = int(220 * min(1.0, self.event_time))
            banner = pygame.Surface((430, 40), pygame.SRCALPHA)
            pygame.draw.rect(banner, (10, 20, 38, alpha), banner.get_rect(), border_radius=20)
            text_surface = self.fonts["small"].render(self.event_text, True, COLORS["ink"])
            banner.blit(text_surface, text_surface.get_rect(center=banner.get_rect().center))
            self.canvas.blit(banner, (board_rect.centerx - 215, board_rect.bottom - 53))

        if self.run_done:
            # Only result controls should be clickable through the overlay.
            self.buttons = []
            self._draw_result_overlay()

    def _draw_game_topbar(self) -> None:
        meta = self._level_meta(self.current_level)
        accent = COLORS.get(meta.get("accent", "blue"), COLORS["blue"])
        self._text(f"LEVEL {self.current_level}", (34, 26), "small", accent)
        self._text(meta.get("title", "Gridworld"), (34, 47), "h2", COLORS["ink"])

        mode = "MANUAL PLAY" if self.control_mode == "manual" else f"AI / {_friendly_agent(self.current_agent_kind).upper()}"
        if self.current_intrinsic:
            mode += " + INTRINSIC"
        mode_rect = pygame.Rect(480, 28, 285, 39)
        pygame.draw.rect(self.canvas, COLORS["panel"], mode_rect, border_radius=19)
        pygame.draw.rect(self.canvas, accent, mode_rect, 1, border_radius=19)
        self._text(mode, mode_rect.center, "tiny", accent, "center")

        if self.campaign_mode:
            start_x = 800
            for index, level in enumerate(sorted(LEVELS)):
                x = start_x + index * 42
                complete = level < self.current_level
                current = level == self.current_level
                color = COLORS["green"] if complete else accent if current else COLORS["line"]
                pygame.draw.line(self.canvas, COLORS["line"], (x, 48), (x + 42, 48), 2)
                pygame.draw.circle(self.canvas, color, (x, 48), 11)
                if not (complete or current):
                    pygame.draw.circle(self.canvas, COLORS["night"], (x, 48), 7)
                self._text(str(level), (x, 48), "tiny", COLORS["night"] if complete or current else COLORS["muted"], "center")
        else:
            self._text("FREE SESSION", (1025, 49), "tiny", COLORS["faint"], "midleft")

        self._button(
            pygame.Rect(1155, 25, 91, 45),
            "Menu",
            ("menu",),
            kind="secondary",
        )

    def _grid_geometry(self) -> Tuple[pygame.Rect, int, Tuple[int, int]]:
        area = pygame.Rect(34, 92, 716, 676)
        cell_size = max(18, min(area.width // self.env.cols, area.height // self.env.rows))
        width = cell_size * self.env.cols
        height = cell_size * self.env.rows
        origin = (
            area.x + (area.width - width) // 2,
            area.y + (area.height - height) // 2,
        )
        return pygame.Rect(origin[0], origin[1], width, height), cell_size, origin

    def _tile_rect(
        self, row: float, col: float, cell_size: int, origin: Tuple[int, int]
    ) -> pygame.Rect:
        return pygame.Rect(
            int(origin[0] + col * cell_size),
            int(origin[1] + row * cell_size),
            cell_size,
            cell_size,
        )

    def _cell_center(
        self, pos: Tuple[float, float], cell_size: int, origin: Tuple[int, int]
    ) -> Tuple[int, int]:
        row, col = pos
        return (
            int(origin[0] + (col + 0.5) * cell_size),
            int(origin[1] + (row + 0.5) * cell_size),
        )

    def _draw_board(
        self, board_rect: pygame.Rect, cell_size: int, origin: Tuple[int, int]
    ) -> None:
        shadow = board_rect.inflate(18, 18).move(0, 7)
        pygame.draw.rect(self.canvas, (2, 6, 14), shadow, border_radius=20)
        pygame.draw.rect(self.canvas, COLORS["line"], board_rect.inflate(8, 8), border_radius=17)

        # Terrain layer.
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                rect = self._tile_rect(row, col, cell_size, origin)
                tile = self.env.static_grid[row][col]
                floor = COLORS["floor_a"] if (row + col) % 2 == 0 else COLORS["floor_b"]
                pygame.draw.rect(self.canvas, floor, rect)
                if tile == "R":
                    self._draw_rock(rect)
                elif tile == "F":
                    self._draw_fire(rect, row, col)
                pygame.draw.rect(self.canvas, (45, 65, 90), rect, 1)

        # Start marker and travelled path are informational only.
        if getattr(self.env, "start_pos", None) is not None:
            center = self._cell_center(tuple(self.env.start_pos), cell_size, origin)
            pygame.draw.circle(self.canvas, COLORS["cyan"], center, max(7, cell_size // 5), 1)
            self._text("S", center, "tiny", COLORS["cyan"], "center")

        if len(self.trail) > 1:
            centers = [self._cell_center(pos, cell_size, origin) for pos in self.trail[-80:]]
            trail_surface = pygame.Surface((VIRTUAL_WIDTH, VIRTUAL_HEIGHT), pygame.SRCALPHA)
            pygame.draw.lines(trail_surface, (*COLORS["blue"], 75), False, centers, max(2, cell_size // 12))
            for center in centers[:-1: max(1, len(centers) // 15)]:
                pygame.draw.circle(trail_surface, (*COLORS["cyan"], 90), center, 2)
            self.canvas.blit(trail_surface, (0, 0))

        if self.show_policy and self.agent is not None:
            self._draw_policy_overlay(cell_size, origin)

        # Collectible layer remains visible even when an entity shares a tile.
        for item in list(getattr(self.env, "collectibles", [])):
            try:
                row, col, item_type = item
            except (TypeError, ValueError):
                continue
            rect = self._tile_rect(row, col, cell_size, origin)
            if item_type == "apple":
                self._draw_apple(rect, row, col)
            elif item_type == "chest":
                self._draw_chest(rect)

        key_pos = getattr(self.env, "key_pos", None)
        if key_pos is not None:
            self._draw_key(self._tile_rect(key_pos[0], key_pos[1], cell_size, origin))

        # Entity layer with interpolation.
        t = _ease_out_cubic(self.move_anim)
        for index, destination in enumerate(self.anim_to_monsters):
            source = self.anim_from_monsters[index] if index < len(self.anim_from_monsters) else destination
            pos = (
                source[0] + (destination[0] - source[0]) * t,
                source[1] + (destination[1] - source[1]) * t,
            )
            self._draw_monster(pos, cell_size, origin, index)

        if self.anim_to_agent is not None:
            source = self.anim_from_agent or self.anim_to_agent
            destination = self.anim_to_agent
            pos = (
                source[0] + (destination[0] - source[0]) * t,
                source[1] + (destination[1] - source[1]) * t,
            )
            self._draw_player(pos, cell_size, origin)

    def _draw_rock(self, rect: pygame.Rect) -> None:
        inset = max(3, rect.width // 14)
        body = rect.inflate(-inset * 2, -inset * 2)
        pygame.draw.rect(self.canvas, COLORS["rock_dark"], body.move(0, 3), border_radius=8)
        pygame.draw.rect(self.canvas, COLORS["rock"], body, border_radius=8)
        pygame.draw.line(
            self.canvas,
            (116, 123, 144),
            (body.x + body.width // 4, body.y + 5),
            (body.centerx, body.centery),
            2,
        )
        pygame.draw.line(
            self.canvas,
            COLORS["rock_dark"],
            (body.centerx, body.centery),
            (body.right - 8, body.bottom - 6),
            2,
        )

    def _draw_fire(self, rect: pygame.Rect, row: int, col: int) -> None:
        pygame.draw.rect(self.canvas, (94, 39, 48), rect)
        phase = self.elapsed * 5.0 + row * 0.7 + col * 1.3
        center_x = rect.centerx
        base_y = rect.bottom - max(5, rect.height // 10)
        for index, offset in enumerate((-rect.width // 5, 0, rect.width // 5)):
            wave = math.sin(phase + index * 1.8)
            height = int(rect.height * (0.47 + 0.11 * wave))
            width = max(5, rect.width // 5)
            points = [
                (center_x + offset, base_y - height),
                (center_x + offset - width, base_y),
                (center_x + offset + width, base_y),
            ]
            pygame.draw.polygon(self.canvas, COLORS["fire"], points)
            inner = [
                (center_x + offset, base_y - height // 2),
                (center_x + offset - width // 2, base_y),
                (center_x + offset + width // 2, base_y),
            ]
            pygame.draw.polygon(self.canvas, COLORS["fire_hot"], inner)

    def _draw_apple(self, rect: pygame.Rect, row: int, col: int) -> None:
        bob = int(math.sin(self.elapsed * 3.2 + row + col) * 2)
        center = (rect.centerx, rect.centery + bob + 2)
        radius = max(7, rect.width // 5)
        pygame.draw.ellipse(
            self.canvas,
            (9, 18, 28),
            pygame.Rect(center[0] - radius, center[1] + radius - 1, radius * 2, max(3, radius // 2)),
        )
        pygame.draw.circle(self.canvas, COLORS["apple_dark"], (center[0], center[1] + 2), radius + 2)
        pygame.draw.circle(self.canvas, COLORS["apple"], center, radius)
        pygame.draw.circle(self.canvas, (193, 255, 202), (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 5))
        pygame.draw.line(self.canvas, COLORS["apple_dark"], (center[0], center[1] - radius), (center[0] + 3, center[1] - radius - 7), 3)
        pygame.draw.ellipse(self.canvas, COLORS["green"], pygame.Rect(center[0] + 2, center[1] - radius - 8, radius, max(4, radius // 2)))

    def _draw_key(self, rect: pygame.Rect) -> None:
        glow = pygame.Surface(rect.size, pygame.SRCALPHA)
        pulse = int(30 + 20 * (math.sin(self.elapsed * 4.0) + 1) / 2)
        pygame.draw.circle(glow, (*COLORS["yellow"], pulse), (rect.width // 2, rect.height // 2), max(10, rect.width // 3))
        self.canvas.blit(glow, rect)
        center = (rect.centerx - rect.width // 10, rect.centery - rect.height // 12)
        ring = max(6, rect.width // 8)
        pygame.draw.circle(self.canvas, COLORS["yellow"], center, ring, 4)
        pygame.draw.line(self.canvas, COLORS["yellow"], (center[0] + ring - 1, center[1] + ring - 1), (rect.centerx + rect.width // 4, rect.centery + rect.height // 4), 5)
        pygame.draw.line(self.canvas, COLORS["yellow"], (rect.centerx + rect.width // 7, rect.centery + rect.height // 7), (rect.centerx + rect.width // 4, rect.centery), 4)

    def _draw_chest(self, rect: pygame.Rect) -> None:
        margin = max(6, rect.width // 7)
        body = pygame.Rect(rect.x + margin, rect.centery - 2, rect.width - margin * 2, rect.height // 3)
        lid = pygame.Rect(body.x, body.y - rect.height // 5, body.width, rect.height // 4)
        pygame.draw.rect(self.canvas, COLORS["chest_dark"], body.move(0, 4), border_radius=5)
        pygame.draw.rect(self.canvas, COLORS["chest"], body, border_radius=5)
        pygame.draw.rect(self.canvas, (194, 130, 78), lid, border_radius=8)
        pygame.draw.line(self.canvas, COLORS["yellow"], (body.x, body.y + 3), (body.right, body.y + 3), 3)
        lock_rect = pygame.Rect(body.centerx - 5, body.y - 2, 10, 14)
        pygame.draw.rect(self.canvas, COLORS["yellow"], lock_rect, border_radius=2)

    def _draw_monster(
        self,
        pos: Tuple[float, float],
        cell_size: int,
        origin: Tuple[int, int],
        index: int,
    ) -> None:
        center = self._cell_center(pos, cell_size, origin)
        wobble = math.sin(self.elapsed * 5.0 + index * 2.2)
        radius = max(10, cell_size // 3)
        shadow = pygame.Rect(center[0] - radius, center[1] + radius // 2, radius * 2, max(5, radius // 2))
        pygame.draw.ellipse(self.canvas, (8, 13, 25), shadow)
        body_center = (center[0], center[1] + int(wobble * 2))
        pygame.draw.circle(self.canvas, COLORS["monster_dark"], body_center, radius + 3)
        pygame.draw.circle(self.canvas, COLORS["monster"], body_center, radius)
        horn_y = body_center[1] - radius + 3
        pygame.draw.polygon(self.canvas, COLORS["monster"], [(body_center[0] - radius // 2, horn_y + 4), (body_center[0] - radius, horn_y - radius // 2), (body_center[0] - 2, horn_y + radius // 2)])
        pygame.draw.polygon(self.canvas, COLORS["monster"], [(body_center[0] + radius // 2, horn_y + 4), (body_center[0] + radius, horn_y - radius // 2), (body_center[0] + 2, horn_y + radius // 2)])
        eye_y = body_center[1] - radius // 5
        for direction in (-1, 1):
            eye_x = body_center[0] + direction * radius // 3
            pygame.draw.circle(self.canvas, COLORS["ink"], (eye_x, eye_y), max(3, radius // 5))
            pygame.draw.circle(self.canvas, COLORS["night"], (eye_x + direction, eye_y + 1), max(1, radius // 10))

    def _draw_player(
        self, pos: Tuple[float, float], cell_size: int, origin: Tuple[int, int]
    ) -> None:
        center = self._cell_center(pos, cell_size, origin)
        radius = max(11, cell_size // 3)
        pygame.draw.ellipse(
            self.canvas,
            (6, 12, 24),
            pygame.Rect(center[0] - radius, center[1] + radius // 2, radius * 2, max(6, radius // 2)),
        )
        pulse = radius + 6 + int(2 * math.sin(self.elapsed * 3.5))
        pygame.draw.circle(self.canvas, COLORS["blue"], center, pulse, 2)
        pygame.draw.circle(self.canvas, COLORS["player_dark"], center, radius + 3)
        pygame.draw.circle(self.canvas, COLORS["player"], center, radius)
        pygame.draw.circle(self.canvas, (198, 231, 255), (center[0] - radius // 3, center[1] - radius // 3), max(3, radius // 5))

        if self.last_action is not None:
            directions = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
            dx, dy = directions[self.last_action]
            tip = (center[0] + dx * (radius + 9), center[1] + dy * (radius + 9))
            side_a = (center[0] - dy * 5 + dx * radius // 2, center[1] + dx * 5 + dy * radius // 2)
            side_b = (center[0] + dy * 5 + dx * radius // 2, center[1] - dx * 5 + dy * radius // 2)
            pygame.draw.polygon(self.canvas, COLORS["ink"], [tip, side_a, side_b])

    def _refresh_policy_cache(self) -> None:
        self.policy_cells = {}
        if self.agent is None or self.env is None:
            return
        q_table = getattr(self.agent, "q_table", None)
        if not q_table:
            return

        current = self.state
        has_key = current[2] if isinstance(current, tuple) and len(current) > 2 else None
        collectibles = current[3] if isinstance(current, tuple) and len(current) > 3 else None
        confidence_by_cell: Dict[Tuple[int, int], float] = {}

        for table_state, raw_values in q_table.items():
            if not isinstance(table_state, tuple) or len(table_state) < 2:
                continue
            try:
                row, col = int(table_state[0]), int(table_state[1])
                values = [float(value) for value in raw_values]
            except (TypeError, ValueError):
                continue
            if not (0 <= row < self.env.rows and 0 <= col < self.env.cols):
                continue
            if len(table_state) > 3 and has_key is not None:
                if table_state[2] != has_key or table_state[3] != collectibles:
                    continue
            if not values:
                continue
            best_value = max(values)
            best_actions = [
                index for index, value in enumerate(values) if abs(value - best_value) <= 1e-8
            ]
            confidence = best_value - min(values)
            pos = (row, col)
            if pos not in self.policy_cells or confidence > confidence_by_cell[pos]:
                self.policy_cells[pos] = (best_actions, values)
                confidence_by_cell[pos] = confidence

    def _draw_policy_overlay(self, cell_size: int, origin: Tuple[int, int]) -> None:
        overlay = pygame.Surface((VIRTUAL_WIDTH, VIRTUAL_HEIGHT), pygame.SRCALPHA)
        for (row, col), (actions, values) in self.policy_cells.items():
            if self.env.static_grid[row][col] in ("R", "F"):
                continue
            center = self._cell_center((row, col), cell_size, origin)
            confidence = max(values) - min(values) if values else 0.0
            alpha = 75 if len(actions) > 1 else int(_clamp(105 + confidence * 40, 105, 205))
            for action in actions[:2]:
                self._draw_arrow(overlay, center, action, cell_size, (*COLORS["cyan"], alpha))
        self.canvas.blit(overlay, (0, 0))

    @staticmethod
    def _draw_arrow(
        surface: pygame.Surface,
        center: Tuple[int, int],
        action: int,
        cell_size: int,
        color: Tuple[int, int, int, int],
    ) -> None:
        directions = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
        dx, dy = directions.get(action, (0, 0))
        length = max(8, cell_size // 5)
        start = (center[0] - dx * length // 2, center[1] - dy * length // 2)
        tip = (center[0] + dx * length, center[1] + dy * length)
        pygame.draw.line(surface, color, start, tip, max(2, cell_size // 22))
        side = max(3, cell_size // 12)
        left = (tip[0] - dx * side - dy * side, tip[1] - dy * side + dx * side)
        right = (tip[0] - dx * side + dy * side, tip[1] - dy * side - dx * side)
        pygame.draw.polygon(surface, color, [tip, left, right])

    def _draw_sidebar(self) -> None:
        panel = pygame.Rect(778, 92, 468, 676)
        self._panel(panel, COLORS["panel"], COLORS["line"], radius=20)
        meta = self._level_meta(self.current_level)
        accent = COLORS.get(meta.get("accent", "blue"), COLORS["blue"])

        self._text(meta.get("task", "Part I"), (804, 116), "tiny", accent)
        self._text("OBJECTIVE", (804, 143), "small", COLORS["ink"])
        objective_bottom = self._wrapped_text(
            meta.get("objective", "Collect all rewards."),
            pygame.Rect(804, 169, 416, 72),
            "small",
            COLORS["muted"],
            line_gap=3,
            max_lines=4,
        )

        stats_y = max(239, objective_bottom + 12)
        self._stat_card(pygame.Rect(804, stats_y, 124, 65), "ENV REWARD", f"{self.total_reward:.1f}", COLORS["green"])
        self._stat_card(pygame.Rect(938, stats_y, 124, 65), "STEPS", f"{self.steps}/{self.max_steps}", COLORS["blue"])
        remaining = len(getattr(self.env, "collectibles", []))
        self._stat_card(pygame.Rect(1072, stats_y, 148, 65), "ITEMS LEFT", str(remaining), COLORS["yellow"])

        status_y = stats_y + 84
        grid_rows = LEVELS.get(self.current_level, {}).get("grid", ())
        key_required = any("K" in str(row) or "C" in str(row) for row in grid_rows)
        if key_required:
            key_status = (
                "KEY ACQUIRED"
                if getattr(self.env, "has_key", False)
                else "KEY NOT COLLECTED"
            )
        else:
            key_status = "NO KEY REQUIRED"
        monster_count = len(getattr(self.env, "monster_positions", []))
        status = f"{key_status}   /   {monster_count} MONSTER{'S' if monster_count != 1 else ''}"
        self._text(status, (804, status_y), "tiny", accent)

        if self.control_mode == "ai":
            self._draw_ai_controls(status_y + 28)
            inspector_y = status_y + 157
            self._draw_q_inspector(inspector_y)
            legend_y = inspector_y + 170
        else:
            self._draw_manual_controls(status_y + 28)
            legend_y = status_y + 155

        self._draw_legend(legend_y)
        if self.control_mode == "manual":
            self._draw_level_help(legend_y + 122, accent)

    def _stat_card(
        self, rect: pygame.Rect, label: str, value: str, color: Tuple[int, int, int]
    ) -> None:
        pygame.draw.rect(self.canvas, COLORS["night_2"], rect, border_radius=11)
        pygame.draw.rect(self.canvas, COLORS["line"], rect, 1, border_radius=11)
        self._text(label, (rect.x + 10, rect.y + 9), "tiny", COLORS["faint"])
        self._text(value, (rect.x + 10, rect.y + 29), "h3", color)

    def _draw_manual_controls(self, y: int) -> None:
        self._text("MANUAL CONTROLS", (804, y), "small", COLORS["ink"])
        controls = [
            ("ARROWS / WASD", "Move one tile"),
            ("R", "Restart this level"),
            ("M / ESC", "Return to main menu"),
        ]
        for index, (key, label) in enumerate(controls):
            row_y = y + 31 + index * 25
            self._text(key, (804, row_y), "mono_small", COLORS["blue"])
            self._text(label, (954, row_y), "small", COLORS["muted"])

        self._text(
            "Tip: collect every reward to finish the episode.",
            (804, y + 112),
            "tiny",
            COLORS["faint"],
        )

    def _draw_ai_controls(self, y: int) -> None:
        self._text("AI PLAYBACK", (804, y), "small", COLORS["ink"])
        self._text("GREEDY POLICY  •  ε = 0", (1214, y + 2), "tiny", COLORS["cyan"], "topright")
        self._button(
            pygame.Rect(804, y + 29, 118, 43),
            "Resume AI" if self.paused else "Pause AI",
            ("toggle_pause",),
            kind="secondary",
        )
        self._button(
            pygame.Rect(932, y + 29, 90, 43),
            "1 step",
            ("single_step",),
            kind="secondary",
        )
        self._button(
            pygame.Rect(1032, y + 29, 52, 43),
            "-",
            ("speed", -1),
            kind="secondary",
            enabled=self.speed_index > 0,
        )
        speed_label = f"{self.speed_options[self.speed_index]:g}x"
        speed_rect = pygame.Rect(1090, y + 29, 66, 43)
        pygame.draw.rect(self.canvas, COLORS["night_2"], speed_rect, border_radius=10)
        pygame.draw.rect(self.canvas, COLORS["line"], speed_rect, 1, border_radius=10)
        self._text(speed_label, speed_rect.center, "small", COLORS["cyan"], "center")
        self._button(
            pygame.Rect(1162, y + 29, 52, 43),
            "+",
            ("speed", 1),
            kind="secondary",
            enabled=self.speed_index < len(self.speed_options) - 1,
        )
        self._button(
            pygame.Rect(804, y + 80, 410, 39),
            "Hide learned policy arrows" if self.show_policy else "Show learned policy arrows",
            ("policy",),
            kind="secondary",
            badge="P",
        )

    def _current_q_values(self) -> Optional[List[float]]:
        if self.agent is None:
            return None
        q_table = getattr(self.agent, "q_table", None)
        if q_table is None:
            return None
        values = q_table.get(self.state)
        if values is None:
            return None
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            return None

    def _draw_q_inspector(self, y: int) -> None:
        self._text("WHY THIS MOVE?  •  CURRENT Q-VALUES", (804, y), "small", COLORS["ink"])
        values = self._current_q_values()
        if not values:
            self._text("State not present in this saved table.", (804, y + 28), "small", COLORS["yellow"])
            self._text("A greedy tie is resolved randomly.", (804, y + 51), "small", COLORS["muted"])
            return

        max_abs = max(max(abs(value) for value in values), 0.0001)
        best = max(values)
        for index, value in enumerate(values[:4]):
            row_y = y + 28 + index * 30
            name = ACTION_NAMES[index] if index < len(ACTION_NAMES) else str(index)
            best_action = abs(value - best) <= 1e-8
            color = COLORS["cyan"] if best_action else COLORS["blue"]
            self._text(name[:5], (804, row_y + 2), "mono_small", color)
            bar_rect = pygame.Rect(864, row_y + 4, 254, 14)
            pygame.draw.rect(self.canvas, COLORS["night"], bar_rect, border_radius=7)
            width = int(abs(value) / max_abs * bar_rect.width)
            pygame.draw.rect(self.canvas, color, pygame.Rect(bar_rect.x, bar_rect.y, width, bar_rect.height), border_radius=7)
            self._text(f"{value:+.3f}", (1128, row_y), "mono_small", color)
        self._text(
            "Cyan marks the highest-valued greedy action; exact ties are random.",
            (804, y + 146),
            "tiny",
            COLORS["faint"],
            max_width=410,
        )

    def _draw_legend(self, y: int) -> None:
        if y > 670:
            return
        self._text("WORLD LEGEND", (804, y), "small", COLORS["ink"])
        entries = [
            (COLORS["green"], "Apple +1"),
            (COLORS["yellow"], "Key +0"),
            (COLORS["chest"], "Chest +2"),
            (COLORS["red"], "Fire: death"),
            (COLORS["monster"], "Monster: death"),
            (COLORS["rock"], "Rock: blocked"),
        ]
        for index, (color, label) in enumerate(entries):
            col = index % 2
            row = index // 2
            x = 804 + col * 207
            row_y = y + 31 + row * 27
            pygame.draw.circle(self.canvas, color, (x + 7, row_y + 7), 6)
            self._text(label, (x + 20, row_y), "small", COLORS["muted"])

    def _draw_level_help(self, y: int, accent: Tuple[int, int, int]) -> None:
        rect = pygame.Rect(804, y, 410, 118)
        pygame.draw.rect(self.canvas, COLORS["night_2"], rect, border_radius=12)
        pygame.draw.rect(self.canvas, COLORS["line"], rect, 1, border_radius=12)
        self._text("LEVEL RULE", (rect.x + 14, rect.y + 11), "tiny", accent)
        self._wrapped_text(
            LEVEL_HELP.get(self.current_level, "Collect every item and avoid hazards."),
            pygame.Rect(rect.x + 14, rect.y + 31, rect.width - 28, 45),
            "tiny",
            COLORS["muted"],
            line_gap=2,
            max_lines=3,
        )
        pygame.draw.line(
            self.canvas,
            COLORS["line"],
            (rect.x + 14, rect.y + 82),
            (rect.right - 14, rect.y + 82),
        )
        self._text(
            "EPISODE END  •  all items collected or the agent dies",
            (rect.x + 14, rect.y + 94),
            "tiny",
            COLORS["faint"],
            max_width=rect.width - 28,
        )

    def _draw_result_overlay(self) -> None:
        shade = pygame.Surface((VIRTUAL_WIDTH, VIRTUAL_HEIGHT), pygame.SRCALPHA)
        shade.fill((3, 8, 18, 190))
        self.canvas.blit(shade, (0, 0))

        card = pygame.Rect(330, 165, 620, 470)
        if self.result_kind == "victory":
            accent = COLORS["green"]
            title = "LEVEL CLEARED"
            if self.campaign_mode and self.current_level == max(LEVELS):
                title = "CAMPAIGN COMPLETE"
        elif self.result_kind == "timeout":
            accent = COLORS["yellow"]
            title = "STEP LIMIT REACHED"
        elif self.result_kind == "error":
            accent = COLORS["orange"]
            title = "RUN STOPPED SAFELY"
        else:
            accent = COLORS["red"]
            title = "EPISODE ENDED"

        self._panel(card, COLORS["panel"], accent, radius=25)
        pulse = 43 + int(3 * math.sin(self.elapsed * 4.0))
        pygame.draw.circle(self.canvas, accent, (card.centerx, card.y + 78), pulse, 3)
        symbol = "+" if self.result_kind == "victory" else "!"
        self._text(symbol, (card.centerx, card.y + 78), "h1", accent, "center")
        self._text(title, (card.centerx, card.y + 139), "h1", COLORS["ink"], "center")
        self._wrapped_text(
            self.result_detail,
            pygame.Rect(card.x + 75, card.y + 190, card.width - 150, 65),
            "body",
            COLORS["muted"],
            line_gap=4,
            max_lines=3,
        )

        summary = f"Reward  {self.total_reward:.1f}     Steps  {self.steps}     Remaining  {len(getattr(self.env, 'collectibles', []))}"
        summary_rect = pygame.Rect(card.x + 70, card.y + 263, card.width - 140, 48)
        pygame.draw.rect(self.canvas, COLORS["night_2"], summary_rect, border_radius=12)
        self._text(summary, summary_rect.center, "mono", accent, "center")

        self._button(
            pygame.Rect(card.x + 44, card.bottom - 102, 162, 58),
            "Retry",
            ("retry",),
            kind="primary",
            badge="R",
        )
        has_next = self.current_level < max(LEVELS)
        next_enabled = has_next and (not self.campaign_mode or self.result_kind == "victory")
        next_label = "Next level" if has_next else "Finish"
        next_action = ("next",) if has_next else ("menu",)
        self._button(
            pygame.Rect(card.x + 226, card.bottom - 102, 162, 58),
            next_label,
            next_action,
            kind="success",
            enabled=next_enabled or not has_next,
            badge="N",
        )
        self._button(
            pygame.Rect(card.x + 408, card.bottom - 102, 168, 58),
            "Main menu",
            ("menu",),
            kind="secondary",
            badge="M",
        )

        if self.campaign_mode and has_next and self.result_kind != "victory":
            self._text("Win this stage to unlock Next level.", (card.centerx, card.bottom - 27), "tiny", COLORS["yellow"], "center")

    def _draw_toast(self, text: str) -> None:
        width = min(620, self.fonts["small"].size(text)[0] + 50)
        rect = pygame.Rect((VIRTUAL_WIDTH - width) // 2, 735, width, 44)
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surface, (14, 28, 49, 240), surface.get_rect(), border_radius=22)
        pygame.draw.rect(surface, (*COLORS["cyan"], 150), surface.get_rect(), 1, border_radius=22)
        label = self.fonts["small"].render(text, True, COLORS["ink"])
        surface.blit(label, label.get_rect(center=surface.get_rect().center))
        self.canvas.blit(surface, rect)

    def _spawn_particles(
        self,
        grid_pos: Tuple[int, int],
        color: Tuple[int, int, int],
        count: int,
    ) -> None:
        if self.env is None:
            return
        _, cell_size, origin = self._grid_geometry()
        center = self._cell_center(grid_pos, cell_size, origin)
        for index in range(count):
            angle = (math.tau * index / max(1, count)) + self.elapsed * 0.4
            speed = 45 + (index % 5) * 18
            life = 0.55 + (index % 4) * 0.12
            self.particles.append(
                Particle(
                    float(center[0]),
                    float(center[1]),
                    math.cos(angle) * speed,
                    math.sin(angle) * speed - 35,
                    life,
                    life,
                    3.0 + index % 3,
                    color,
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the unified Part I Gridworld experience"
    )
    parser.add_argument(
        "--mode",
        choices=["menu", "manual", "ai"],
        default="menu",
        help="Open the menu or deep-link directly into a level",
    )
    parser.add_argument(
        "--level", type=int, choices=sorted(LEVELS), default=0, help="Deep-link level"
    )
    parser.add_argument(
        "--agent",
        choices=["qlearning", "sarsa"],
        default="qlearning",
        help="Agent used with --mode ai",
    )
    parser.add_argument(
        "--intrinsic",
        action="store_true",
        help="Load the intrinsic-reward model variant",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Safe evaluation limit (default: config value or 500)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    app = GridworldApp(max_steps=args.max_steps)
    if args.mode == "manual":
        app._start_level(args.level, "manual", None, False, "free")
    elif args.mode == "ai":
        app._start_level(args.level, "ai", args.agent, args.intrinsic, "showcase")
    try:
        app.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(130)


if __name__ == "__main__":
    main()
