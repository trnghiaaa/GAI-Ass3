"""Gymnasium-compatible real-time action arena for Assignment 3 Part II.

The simulation uses continuous pixel coordinates and a fixed time step. It is
kept independent from Pygame so long training runs can execute headlessly; the
optional :class:`arena.renderer.ArenaRenderer` displays the same live state.
"""

from __future__ import annotations

from copy import deepcopy
from enum import IntEnum
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from arena.entities import (
    DangerZone,
    Enemy,
    Player,
    Projectile,
    Spawner,
    circles_overlap,
)


DIRECT_ACTIONS = {
    "NOOP": 0,
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
    "SHOOT": 5,
}

ROTATION_ACTIONS = {
    "NOOP": 0,
    "THRUST": 1,
    "ROTATE_LEFT": 2,
    "ROTATE_RIGHT": 3,
    "SHOOT": 4,
}


class ObservationIndex(IntEnum):
    """Stable indices for the agent's fixed-size feature vector."""

    PLAYER_X = 0
    PLAYER_Y = 1
    PLAYER_VELOCITY_X = 2
    PLAYER_VELOCITY_Y = 3
    PLAYER_HEADING_COS = 4
    PLAYER_HEADING_SIN = 5
    PLAYER_HEALTH = 6
    WEAPON_READY = 7
    NEAREST_ENEMY_DIRECTION_X = 8
    NEAREST_ENEMY_DIRECTION_Y = 9
    NEAREST_ENEMY_DISTANCE = 10
    NEAREST_ENEMY_HEALTH = 11
    NEAREST_SPAWNER_DIRECTION_X = 12
    NEAREST_SPAWNER_DIRECTION_Y = 13
    NEAREST_SPAWNER_DISTANCE = 14
    NEAREST_SPAWNER_HEALTH = 15
    ENEMY_COUNT = 16
    SPAWNER_COUNT = 17
    PHASE = 18
    TIME_REMAINING = 19
    NEAREST_ENEMY_AIM_ALIGNMENT = 20
    NEAREST_SPAWNER_AIM_ALIGNMENT = 21
    ACTIVE_TARGET_DIRECTION_X = 22
    ACTIVE_TARGET_DIRECTION_Y = 23
    ACTIVE_TARGET_DISTANCE = 24
    ACTIVE_TARGET_IS_SPAWNER = 25
    ACTIVE_TARGET_AIM_ALIGNMENT = 26
    ACTIVE_TARGET_TURN_DIRECTION = 27
    PLAYER_LEVEL = 28
    XP_PROGRESS = 29
    WEAPON_SHOT_COUNT = 30
    WEAPON_FIRE_RATE = 31
    WEAPON_DAMAGE = 32
    WEAPON_IS_LASER = 33
    MAX_HEALTH_BONUS = 34
    DAMAGE_RESISTANCE = 35
    PROJECTILE_RANGE = 36
    PROJECTILE_PIERCE = 37
    PROJECTILE_SPLASH = 38
    ENGINE_POWER = 39
    SUPPORT_DRONE_ACTIVE = 40
    NOVA_BOMB_ARMED = 41
    BOSS_PHASE = 42
    MINIBOSS_COUNT = 43
    NEAREST_MINIBOSS_HEALTH = 44
    HAZARD_ESCAPE_X = 45
    HAZARD_ESCAPE_Y = 46
    HAZARD_DISTANCE_TO_SAFETY = 47
    HAZARD_TIME_TO_IMPACT = 48
    HAZARD_ACTIVE = 49
    HAZARD_IS_CIRCLE = 50
    DRONE_LEVEL = 51
    DRONE_COUNT = 52
    BARRIER_CHARGES = 53
    OVERDRIVE_ACTIVE = 54
    REGENERATION = 55
    HOMING_STRENGTH = 56
    MASTERY_POWER = 57
    HAZARD_COUNT = 58
    HAZARD_COMBINED_ESCAPE_X = 59
    HAZARD_COMBINED_ESCAPE_Y = 60
    SECONDARY_HAZARD_ESCAPE_X = 61
    SECONDARY_HAZARD_ESCAPE_Y = 62
    SECONDARY_HAZARD_DISTANCE_TO_SAFETY = 63
    SECONDARY_HAZARD_TIME_TO_IMPACT = 64
    SECONDARY_HAZARD_ACTIVE = 65
    SECONDARY_HAZARD_IS_CIRCLE = 66
    CRITICAL_CHANCE = 67
    LEECH_STRENGTH = 68
    RIFTBREAKER_POWER = 69
    SECOND_ENEMY_X = 70
    SECOND_ENEMY_Y = 71
    SECOND_ENEMY_DISTANCE = 72
    SECOND_ENEMY_CLOSING = 73
    NEAREST_ENEMY_CLOSING = 74
    LOCAL_ENEMY_COUNT = 75
    CROWD_PRESSURE = 76
    CROWD_ESCAPE_X = 77
    CROWD_ESCAPE_Y = 78
    WALL_LEFT = 79
    WALL_RIGHT = 80
    WALL_TOP = 81
    WALL_BOTTOM = 82
    SPAWNER_CLEARANCE = 83
    BOSS_SHIELD = 84
    BOSS_SUMMONS_REMAINING = 85
    HAZARD_ESCAPE_ALIGNMENT = 86
    HAZARD_ESCAPE_TURN = 87
    BOSS_SUMMON_COOLDOWN = 88
    BOSS_HEALTH = 89
    BOSS_VULNERABLE = 90
    BOSS_DEFENDER_COUNT = 91
    NEAREST_DEFENDER_DIRECTION_X = 92
    NEAREST_DEFENDER_DIRECTION_Y = 93
    NEAREST_DEFENDER_DISTANCE = 94
    NEAREST_DEFENDER_HEALTH = 95
    NEAREST_MISSILE_DIRECTION_X = 96
    NEAREST_MISSILE_DIRECTION_Y = 97
    NEAREST_MISSILE_DISTANCE = 98
    NEAREST_MISSILE_TIME_TO_IMPACT = 99
    BOSS_VELOCITY_X = 100
    BOSS_VELOCITY_Y = 101
    NEAREST_MISSILE_VELOCITY_X = 102
    NEAREST_MISSILE_VELOCITY_Y = 103
    MISSILE_ESCAPE_X = 104
    MISSILE_ESCAPE_Y = 105
    MISSILE_LOCK_ON = 106


OBSERVATION_NAMES = tuple(index.name.lower() for index in ObservationIndex)
ENVIRONMENT_SCHEMA_VERSION = 10


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a configuration override without mutating either input."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class ArenaEnv(gym.Env):
    """Continuous 2-D combat arena with enemies, spawners, and phases.

    Parameters
    ----------
    control_style:
        ``"direct"`` for four-direction movement or ``"rotation"`` for
        rotation and thrust controls.
    render_mode:
        ``None`` for headless simulation, ``"human"`` for a Pygame window, or
        ``"rgb_array"`` for an RGB frame.
    config_override:
        Optional nested dictionary used by tests and later experiments.
    manual_choices:
        Pause at upgrade drafts for human selection. Headless agents resolve
        the same three-card drafts with a deterministic heuristic so the
        required action sets remain unchanged.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        control_style: str = "direct",
        render_mode: str | None = None,
        config_override: dict[str, Any] | None = None,
        manual_choices: bool = False,
    ) -> None:
        super().__init__()

        if control_style not in ("direct", "rotation"):
            raise ValueError("control_style must be 'direct' or 'rotation'")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")

        config_path = Path(__file__).with_name("config.json")
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        self.config = _merge_dict(config, config_override or {})

        self.control_style = control_style
        self.render_mode = render_mode
        self.manual_choices = bool(manual_choices)
        self.sim_cfg = self.config["simulation"]
        self.player_cfg = self.config["player"]
        self.enemy_cfg = self.config["enemy"]
        self.spawner_cfg = self.config["spawner"]
        self.projectile_cfg = self.config["projectile"]
        self.phase_cfg = self.config["phase"]
        self.progression_cfg = self.config["progression"]
        self.reward_cfg = self.config["rewards"]

        self.width = int(self.sim_cfg["window_width"])
        self.height = int(self.sim_cfg["window_height"])
        self.playfield_top = float(self.sim_cfg["hud_height"])
        self.fps = int(self.sim_cfg["fps"])
        self.dt = 1.0 / self.fps
        self.max_steps = int(self.sim_cfg["max_steps"])
        self.phase_max_steps = max(
            1,
            round(float(self.sim_cfg["phase_time_limit_seconds"]) * self.fps),
        )
        self.metadata["render_fps"] = self.fps

        self.action_names = DIRECT_ACTIONS if control_style == "direct" else ROTATION_ACTIONS
        self.action_space = spaces.Discrete(len(self.action_names))

        # The layout is fixed even when there are no enemies or spawners.
        self.observation_names = OBSERVATION_NAMES
        low = np.zeros(len(self.observation_names), dtype=np.float32)
        signed_features = (
            ObservationIndex.PLAYER_X,
            ObservationIndex.PLAYER_Y,
            ObservationIndex.PLAYER_VELOCITY_X,
            ObservationIndex.PLAYER_VELOCITY_Y,
            ObservationIndex.PLAYER_HEADING_COS,
            ObservationIndex.PLAYER_HEADING_SIN,
            ObservationIndex.NEAREST_ENEMY_DIRECTION_X,
            ObservationIndex.NEAREST_ENEMY_DIRECTION_Y,
            ObservationIndex.NEAREST_SPAWNER_DIRECTION_X,
            ObservationIndex.NEAREST_SPAWNER_DIRECTION_Y,
            ObservationIndex.NEAREST_ENEMY_AIM_ALIGNMENT,
            ObservationIndex.NEAREST_SPAWNER_AIM_ALIGNMENT,
            ObservationIndex.ACTIVE_TARGET_DIRECTION_X,
            ObservationIndex.ACTIVE_TARGET_DIRECTION_Y,
            ObservationIndex.ACTIVE_TARGET_AIM_ALIGNMENT,
            ObservationIndex.ACTIVE_TARGET_TURN_DIRECTION,
            ObservationIndex.HAZARD_ESCAPE_X,
            ObservationIndex.HAZARD_ESCAPE_Y,
            ObservationIndex.HAZARD_COMBINED_ESCAPE_X,
            ObservationIndex.HAZARD_COMBINED_ESCAPE_Y,
            ObservationIndex.SECONDARY_HAZARD_ESCAPE_X,
            ObservationIndex.SECONDARY_HAZARD_ESCAPE_Y,
            ObservationIndex.SECOND_ENEMY_X,
            ObservationIndex.SECOND_ENEMY_Y,
            ObservationIndex.SECOND_ENEMY_CLOSING,
            ObservationIndex.NEAREST_ENEMY_CLOSING,
            ObservationIndex.CROWD_ESCAPE_X,
            ObservationIndex.CROWD_ESCAPE_Y,
            ObservationIndex.HAZARD_ESCAPE_ALIGNMENT,
            ObservationIndex.HAZARD_ESCAPE_TURN,
            ObservationIndex.NEAREST_DEFENDER_DIRECTION_X,
            ObservationIndex.NEAREST_DEFENDER_DIRECTION_Y,
            ObservationIndex.NEAREST_MISSILE_DIRECTION_X,
            ObservationIndex.NEAREST_MISSILE_DIRECTION_Y,
            ObservationIndex.BOSS_VELOCITY_X,
            ObservationIndex.BOSS_VELOCITY_Y,
            ObservationIndex.NEAREST_MISSILE_VELOCITY_X,
            ObservationIndex.NEAREST_MISSILE_VELOCITY_Y,
            ObservationIndex.MISSILE_ESCAPE_X,
            ObservationIndex.MISSILE_ESCAPE_Y,
        )
        low[[int(index) for index in signed_features]] = -1.0
        high = np.ones(len(self.observation_names), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.player: Player
        self.enemies: list[Enemy] = []
        self.spawners: list[Spawner] = []
        self.projectiles: list[Projectile] = []
        self.danger_zones: list[DangerZone] = []
        self.phase = 1
        self.phase_max_steps = self._phase_step_budget()
        self.step_count = 0
        self.phase_step_count = 0
        self.phase_transition_steps = 0
        self.last_phase_cleanup_count = 0
        self.done = False
        self.last_end_reason: str | None = None
        self._next_entity_id = 1
        self._renderer = None
        self.last_events: dict[str, Any] = {}
        self.episode_stats: dict[str, float | int] = {}
        self.upgrade_banner_steps = 0
        self.last_upgrade_name = "Pulse Cannon"
        self.last_upgrade_detail = "Base weapon online"
        self.upgrade_stacks: dict[str, int] = {}
        self.pending_choice_kind: str | None = None
        self.pending_choices: list[dict[str, Any]] = []
        self._choice_queue: list[str] = []
        self.nova_bomb_armed = False
        self.support_drone_phase = 0
        self.drone_cooldown_steps = 0
        self.barrier_charges = 0
        self.overdrive_until_phase = 0
        self.boss_skill_cooldown_steps = 0
        self.boss_skill_index = 0
        self.boss_attack_serial = 0
        self.phases_since_miniboss = 0
        self._manual_fire_requested = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the full episode and return its initial observation."""

        super().reset(seed=seed)
        reset_options = dict(options or {})
        start_phase = int(reset_options.get("start_phase", 1))
        if start_phase < 1:
            raise ValueError("start_phase must be at least 1")

        self.phase = start_phase
        self.phase_max_steps = self._phase_step_budget()
        self.step_count = 0
        self.phase_step_count = 0
        self.phase_transition_steps = 0
        self.last_phase_cleanup_count = 0
        self.done = False
        self.last_end_reason = None
        self._next_entity_id = 1
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self.danger_zones = []
        self.last_events = {}
        self.upgrade_banner_steps = 0
        self.last_upgrade_name = "Pulse Cannon"
        self.last_upgrade_detail = "Base weapon online"
        self.upgrade_stacks = {
            str(item["id"]): 0 for item in self.progression_cfg["upgrade_catalog"]
        }
        self.upgrade_stacks["artifact"] = 0
        self.pending_choice_kind = None
        self.pending_choices = []
        self._choice_queue = []
        self.nova_bomb_armed = False
        self.support_drone_phase = 0
        self.drone_cooldown_steps = 0
        self.barrier_charges = 0
        self.overdrive_until_phase = 0
        self.boss_skill_cooldown_steps = 0
        self.boss_skill_index = 0
        self.boss_attack_serial = 0
        self.phases_since_miniboss = 0
        self._manual_fire_requested = False
        self.episode_stats = {
            "reward": 0.0,
            "enemies_destroyed": 0,
            "spawners_destroyed": 0,
            "phases_advanced": 0,
            "damage_dealt_enemy": 0.0,
            "damage_dealt_spawner": 0.0,
            "damage_taken": 0.0,
            "shots_fired": 0,
            "volleys_fired": 0,
            "projectile_hits": 0,
            "xp_earned": 0.0,
            "levels_gained": 0,
            "upgrades_chosen": 0,
            "phase_rewards_chosen": 0,
            "bosses_destroyed": 0,
            "minibosses_destroyed": 0,
            "miniboss_caches": 0,
            "boss_skills_cast": 0,
            "boss_skills_dodged": 0,
            "boss_skill_hits": 0,
            "boss_defenders_spawned": 0,
            "boss_defenders_destroyed": 0,
            "boss_immune_hits": 0,
            "boss_health_regenerated": 0.0,
            "missiles_fired": 0,
            "missiles_evaded": 0,
            "missile_hits": 0,
            "hazard_exposure": 0.0,
            "sustain_healed": 0.0,
            "boss_rewards_chosen": 0,
        }

        radius = float(self.player_cfg["radius"])
        max_health = float(self.player_cfg["max_health"])
        self.player = Player(
            x=self.width / 2.0,
            y=self.height / 2.0,
            radius=radius,
            angle=-math.pi / 2.0,
            max_health=max_health,
            health=max_health,
        )
        self._spawn_phase_spawners()

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the arena by one fixed simulation step."""

        if self.done:
            raise RuntimeError("Episode has ended; call reset() before step().")
        if self.pending_choice_kind is not None:
            raise RuntimeError("Resolve the pending upgrade choice before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.control_style} controls")

        self.step_count += 1
        if self.phase_transition_steps == 0:
            self.phase_step_count += 1
        events: dict[str, Any] = {
            "shot_fired": False,
            "projectiles_fired": 0,
            "projectile_hits": 0,
            "enemies_spawned": 0,
            "enemies_destroyed": 0,
            "spawners_destroyed": 0,
            "damage_dealt_enemy": 0.0,
            "damage_dealt_spawner": 0.0,
            "damage_taken": 0.0,
            "phase_advanced": False,
            "boss_phase_cleared": False,
            "enemies_dispersed": 0,
            "projectiles_cleared": 0,
            "hazards_cleared": 0,
            "spawner_progress": 0.0,
            "aim_improvement": 0.0,
            "shot_alignment": 0.0,
            "impacts": [],
            "player_hit": False,
            "xp_gained": 0.0,
            "levels_gained": 0,
            "upgrade_unlocked": None,
            "choice_offered": None,
            "choice_selected": None,
            "drone_shots": 0,
            "drone_hits": 0,
            "nova_bomb_detonated": False,
            "minibosses_spawned": 0,
            "minibosses_destroyed": 0,
            "miniboss_caches": 0,
            "boss_skills_cast": 0,
            "boss_skills_dodged": 0,
            "boss_skill_hits": 0,
            "boss_defenders_spawned": 0,
            "boss_defenders_destroyed": 0,
            "boss_immune_hits": 0,
            "boss_health_regenerated": 0.0,
            "missiles_fired": 0,
            "missiles_evaded": 0,
            "missile_hits": 0,
            "missile_escape_improvement": 0.0,
            "hazard_escape_improvement": 0.0,
            "hazard_exposure": 0.0,
            "sustain_healed": 0.0,
            "barrier_blocks": 0,
            "phase_timeout": False,
        }

        shaping_before = self._shaping_snapshot()
        self._tick_cooldowns()
        events["projectiles_fired"] = self._apply_player_action(int(action))
        if self._manual_fire_requested and not events["projectiles_fired"]:
            events["projectiles_fired"] += self._fire_projectile(
                auto_aim=self.control_style == "direct"
            )
        self._manual_fire_requested = False
        events["shot_fired"] = events["projectiles_fired"] > 0
        if events["shot_fired"]:
            events["shot_alignment"] = float(
                self._shaping_snapshot()["target_alignment"]
            )
        self._update_support_drone(events)
        self._update_boss_movement_and_defenders(events)
        self._update_projectiles(events)
        # A dense, auditable warning-zone signal teaches the policy before the
        # delayed boss strike lands. It does not alter damage or game physics.
        events["hazard_exposure"] = self._danger_zone_risk()
        self._update_boss_skills(events)
        self._update_enemies(events)
        self._update_spawners(events)
        self._update_phase(events)
        self._update_passive_systems()
        self._apply_combat_progression(events)
        if events["phase_advanced"]:
            self._queue_choice(
                "boss_reward" if events["boss_phase_cleared"] else "phase_reward"
            )
        self._prepare_next_choice(events)
        self._apply_shaping_delta(events, shaping_before)

        terminated = self.player.health <= 0.0
        phase_timeout = self.phase_step_count >= self.phase_max_steps
        safety_limit = self.step_count >= self.max_steps
        truncated = (phase_timeout or safety_limit) and not terminated
        if terminated:
            self.player.health = 0.0
            self.last_end_reason = "player_destroyed"
        elif phase_timeout:
            self.last_end_reason = "phase_timeout"
        elif safety_limit:
            self.last_end_reason = "safety_limit"
        events["phase_timeout"] = phase_timeout
        self.done = terminated or truncated

        reward, reward_breakdown = self._calculate_reward(events, terminated)
        self._update_episode_stats(events, reward)
        self.last_events = dict(events)
        info = self._get_info()
        info.update(events)
        info["reward_breakdown"] = reward_breakdown
        if self.done:
            info["episode_end"] = self.last_end_reason

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Draw the current arena or return an RGB array."""

        if self.render_mode is None:
            return None
        if self._renderer is None:
            from arena.renderer import ArenaRenderer

            self._renderer = ArenaRenderer(self, mode=self.render_mode)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def window_closed(self) -> bool:
        """Whether the user closed the human-rendering window."""

        return bool(self._renderer is not None and self._renderer.close_requested)

    # ------------------------------------------------------------------
    # Player controls
    # ------------------------------------------------------------------

    def request_manual_fire(self) -> None:
        """Queue a shot alongside the next manual movement/rotation action.

        This deliberately sits outside the Gym action space.  Human players can
        hold Space while steering, while trained agents still use exactly the
        five/six discrete actions required by the assignment specification.
        """

        if not self.manual_choices:
            raise RuntimeError("Simultaneous fire is available only in manual play.")
        self._manual_fire_requested = True

    def _apply_player_action(self, action: int) -> int:
        projectiles_fired = 0
        engine_multiplier = 1.0 + 0.075 * self.upgrade_stacks.get("engine", 0)

        if self.control_style == "direct":
            speed = float(self.player_cfg["direct_speed"]) * engine_multiplier
            self.player.vx = 0.0
            self.player.vy = 0.0
            if action == DIRECT_ACTIONS["UP"]:
                self.player.vy = -speed
                self.player.angle = -math.pi / 2.0
            elif action == DIRECT_ACTIONS["DOWN"]:
                self.player.vy = speed
                self.player.angle = math.pi / 2.0
            elif action == DIRECT_ACTIONS["LEFT"]:
                self.player.vx = -speed
                self.player.angle = math.pi
            elif action == DIRECT_ACTIONS["RIGHT"]:
                self.player.vx = speed
                self.player.angle = 0.0
            elif action == DIRECT_ACTIONS["SHOOT"]:
                # Direct controls intentionally omit an aim action. Target assist
                # keeps this control set accessible while rotation mode retains
                # explicit learned aiming as its distinct challenge.
                projectiles_fired = self._fire_projectile(auto_aim=True)
        else:
            rotation_speed = math.radians(float(self.player_cfg["rotation_speed_degrees"]))
            if action == ROTATION_ACTIONS["ROTATE_LEFT"]:
                self.player.angle -= rotation_speed * self.dt
            elif action == ROTATION_ACTIONS["ROTATE_RIGHT"]:
                self.player.angle += rotation_speed * self.dt
            elif action == ROTATION_ACTIONS["THRUST"]:
                acceleration = float(self.player_cfg["thrust_acceleration"]) * engine_multiplier
                self.player.vx += math.cos(self.player.angle) * acceleration * self.dt
                self.player.vy += math.sin(self.player.angle) * acceleration * self.dt
            elif action == ROTATION_ACTIONS["SHOOT"]:
                projectiles_fired = self._fire_projectile(auto_aim=False)

            drag = float(self.player_cfg["drag"])
            self.player.vx *= drag
            self.player.vy *= drag
            max_speed = float(self.player_cfg["max_speed"]) * engine_multiplier
            speed = math.hypot(self.player.vx, self.player.vy)
            if speed > max_speed:
                scale = max_speed / speed
                self.player.vx *= scale
                self.player.vy *= scale

        self.player.angle %= 2.0 * math.pi
        self.player.x += self.player.vx * self.dt
        self.player.y += self.player.vy * self.dt
        self.player.x = float(np.clip(self.player.x, self.player.radius, self.width - self.player.radius))
        self.player.y = float(
            np.clip(
                self.player.y,
                self.playfield_top + self.player.radius,
                self.height - self.player.radius,
            )
        )
        return projectiles_fired

    def _fire_projectile(self, auto_aim: bool) -> int:
        if self.player.fire_cooldown_steps > 0:
            return 0

        angle = self.player.angle
        if auto_aim:
            target = self._nearest_target(
                max_distance=float(self.player_cfg["target_assist_range"])
                + 20.0 * self.upgrade_stacks.get("capacitor", 0)
            )
            if target is not None:
                angle = math.atan2(target.y - self.player.y, target.x - self.player.x)
                self.player.angle = angle
        elif self.control_style == "rotation":
            # Rotation remains an explicit aiming control.  A narrow shared
            # magnetism cone only corrects near-misses for human and AI pilots.
            maximum_distance = float(self.player_cfg["rotation_aim_assist_range"])
            maximum_difference = math.radians(
                float(self.player_cfg["rotation_aim_assist_degrees"])
            )
            candidates: list[tuple[float, float]] = []
            for target in self._priority_targets():
                dx = target.x - self.player.x
                dy = target.y - self.player.y
                distance = math.hypot(dx, dy)
                if distance > maximum_distance:
                    continue
                target_angle = math.atan2(dy, dx)
                difference = (target_angle - angle + math.pi) % math.tau - math.pi
                if abs(difference) <= maximum_difference:
                    candidates.append((abs(difference) + distance * 1e-6, difference))
            if candidates:
                angle += min(candidates, key=lambda candidate: candidate[0])[1]

        profile = self.weapon_profile()
        shot_count = int(profile["shot_count"])
        spread = math.radians(float(profile["spread_degrees"]))
        offsets = (
            [0.0]
            if shot_count == 1
            else [
                spread * (index / (shot_count - 1) - 0.5)
                for index in range(shot_count)
            ]
        )
        speed = float(self.projectile_cfg["speed"]) * float(
            profile["speed_multiplier"]
        )
        damage = float(self.projectile_cfg["damage"]) * float(
            profile["damage_multiplier"]
        )
        kind = str(profile["kind"])
        radius = float(self.projectile_cfg["radius"]) + 0.8 * self.upgrade_stacks.get(
            "capacitor", 0
        )
        if kind == "laser":
            radius = max(2.0, radius - 1.0)
        offset = self.player.radius + radius + 2.0

        for angle_offset in offsets:
            projectile_angle = angle + angle_offset
            is_critical = float(self.np_random.random()) < float(
                profile["critical_chance"]
            )
            self.projectiles.append(
                Projectile(
                    x=self.player.x + math.cos(projectile_angle) * offset,
                    y=self.player.y + math.sin(projectile_angle) * offset,
                    radius=radius,
                    entity_id=self._new_id(),
                    vx=math.cos(projectile_angle) * speed,
                    vy=math.sin(projectile_angle) * speed,
                    damage=damage * (2.0 if is_critical else 1.0),
                    lifetime_steps=int(
                        float(self.projectile_cfg["lifetime_seconds"]) * self.fps
                        * float(profile["lifetime_multiplier"])
                    ),
                    weapon_kind=kind,
                    owner="player",
                    pierces_remaining=int(profile["pierces"]),
                    splash_radius=float(profile["splash_radius"]),
                    is_critical=is_critical,
                )
            )

        self.player.fire_cooldown_steps = self.weapon_cooldown_steps()
        return shot_count

    @property
    def maximum_player_level(self) -> int:
        """Legacy UI alias for the observation normalization horizon.

        Combat levels themselves are intentionally uncapped.
        """

        return int(self.progression_cfg["observation_level_cap"])

    def xp_threshold_for_level(self, level: int) -> float:
        """Return cumulative XP required to enter an uncapped combat level."""

        if level <= 1:
            return 0.0
        return float(self.progression_cfg["level_xp_base"]) * float(
            level - 1
        ) ** float(self.progression_cfg["level_xp_exponent"])

    @property
    def drone_level(self) -> int:
        return self.upgrade_stacks.get("drone", 0) + self.upgrade_stacks.get(
            "drone_mastery", 0
        )

    @property
    def drone_count(self) -> int:
        if not self.support_drone_active:
            return 0
        return self._drone_count_for_level(self.drone_level)

    @property
    def overdrive_active(self) -> bool:
        return self.phase <= self.overdrive_until_phase

    def weapon_profile(self) -> dict[str, Any]:
        """Compose the live weapon from the player's selected upgrade stacks."""

        multishot = self.upgrade_stacks.get("multishot", 0)
        fire_rate = self.upgrade_stacks.get("fire_rate", 0)
        damage = self.upgrade_stacks.get("damage", 0)
        range_stacks = self.upgrade_stacks.get("range", 0)
        laser = self.upgrade_stacks.get("laser", 0) > 0
        splash = self.upgrade_stacks.get("splash", 0)
        weapon_mastery = self.upgrade_stacks.get("weapon_mastery", 0)
        artifact = self.upgrade_stacks.get("artifact", 0)
        shot_count = 1 + multishot
        if laser:
            name, kind = "Prism Laser", "laser"
        elif splash:
            name, kind = "Nova Payload", "nova"
        elif shot_count > 1:
            name, kind = "Multi-Beam Array", "twin"
        elif fire_rate:
            name, kind = "Rapid Pulse", "rapid"
        else:
            name, kind = "Pulse Cannon", "pulse"
        return {
            "name": name,
            "kind": kind,
            "shot_count": shot_count,
            "spread_degrees": 0 if shot_count == 1 else min(24, 6 + (shot_count - 2) * 4),
            "cooldown_multiplier": 0.9**fire_rate
            * (0.78 if self.overdrive_active else 1.0),
            "speed_multiplier": 1.0 + 0.06 * range_stacks + (0.4 if laser else 0.0),
            "damage_multiplier": (
                1.0
                + 0.12 * damage
                + 0.15 * int(laser)
                + 0.04 * weapon_mastery
                + 0.08 * artifact
            )
            * (1.25 if self.overdrive_active else 1.0),
            "lifetime_multiplier": 1.0 + 0.18 * range_stacks,
            "pierces": self.upgrade_stacks.get("piercing", 0),
            "splash_radius": 28.0 * splash,
            "critical_chance": min(
                0.40, 0.08 * self.upgrade_stacks.get("critical", 0)
            ),
        }

    def weapon_cooldown_steps(self) -> int:
        """Return the active weapon's cooldown in simulation frames."""

        multiplier = float(self.weapon_profile()["cooldown_multiplier"])
        return max(
            1,
            int(float(self.player_cfg["fire_cooldown_seconds"]) * multiplier * self.fps),
        )

    def xp_progress(self) -> float:
        """Return normalized progress from the current level to the next one."""

        current = self.xp_threshold_for_level(self.player.level)
        following = self.xp_threshold_for_level(self.player.level + 1)
        return float(np.clip((self.player.xp - current) / (following - current), 0.0, 1.0))

    def xp_to_next_level(self) -> float:
        """Return remaining XP to the next uncapped combat level."""

        return max(0.0, self.xp_threshold_for_level(self.player.level + 1) - self.player.xp)

    def _apply_combat_progression(self, events: dict[str, Any]) -> None:
        """Convert completed combat objectives into non-RL experience points."""

        leech_tiers = self.upgrade_stacks.get("leech", 0)
        if leech_tiers and int(events["enemies_destroyed"]) > 0:
            health_before = self.player.health
            self.player.health = min(
                self.player.max_health,
                self.player.health
                + self.player.max_health
                * 0.02
                * leech_tiers
                * int(events["enemies_destroyed"]),
            )
            events["sustain_healed"] = self.player.health - health_before

        xp_gained = (
            int(events["enemies_destroyed"])
            * float(self.progression_cfg["enemy_xp"])
            + int(events["spawners_destroyed"])
            * float(self.progression_cfg["spawner_xp"])
            + int(bool(events["phase_advanced"]))
            * float(self.progression_cfg["phase_xp"])
            + int(events.get("minibosses_destroyed", 0))
            * float(self.progression_cfg["miniboss_xp"])
            + int(bool(events.get("boss_phase_cleared", False)))
            * float(self.progression_cfg["boss_phase_xp"])
        )
        events["xp_gained"] = xp_gained
        if xp_gained <= 0.0:
            return

        self.player.xp += xp_gained
        previous_level = self.player.level
        while self.player.xp >= self.xp_threshold_for_level(self.player.level + 1):
            self.player.level += 1

        levels_gained = self.player.level - previous_level
        events["levels_gained"] = levels_gained
        if levels_gained > 0:
            for _ in range(levels_gained):
                self._queue_choice("level_up")

    @property
    def is_boss_phase(self) -> bool:
        interval = max(1, int(self.phase_cfg["boss_interval"]))
        return self.phase % interval == 0

    @property
    def boss_encounter_number(self) -> int:
        """Return the one-based boss number, or zero outside a boss phase."""

        if not self.is_boss_phase:
            return 0
        return max(1, self.phase // max(1, int(self.phase_cfg["boss_interval"])))

    def boss_summon_limit_for_phase(self) -> int:
        """Scale a finite reinforcement budget gradually across boss encounters."""

        if not self.is_boss_phase:
            return 0
        first = int(self.phase_cfg["boss_first_summon_limit"])
        growth = int(self.phase_cfg["boss_summon_growth_per_encounter"])
        maximum = int(self.phase_cfg["boss_summon_limit"])
        return min(maximum, first + (self.boss_encounter_number - 1) * growth)

    def boss_active_summon_limit_for_phase(self) -> int:
        if not self.is_boss_phase:
            return 0
        maximum = int(self.phase_cfg["boss_summon_active_limit"])
        return min(maximum, self.boss_encounter_number)

    def boss_defender_count_for_phase(self) -> int:
        """Return the finite sentry count for the low-health intermission."""

        if not self.is_boss_phase:
            return 0
        first = int(self.phase_cfg["boss_defender_first_count"])
        growth = int(self.phase_cfg["boss_defender_count_growth"])
        maximum = int(self.phase_cfg["boss_defender_max_count"])
        return min(maximum, first + (self.boss_encounter_number - 1) * growth)

    @property
    def boss_defenders(self) -> list[Enemy]:
        return [enemy for enemy in self.enemies if enemy.is_boss_defender]

    @property
    def boss_intermission_active(self) -> bool:
        return bool(self.boss_defenders)

    def boss_is_vulnerable(self, boss: Spawner | None = None) -> bool:
        """Bosses can be damaged unless their finite sentry wave is active."""

        candidate = boss or next(
            (spawner for spawner in self.spawners if spawner.is_boss), None
        )
        return bool(candidate is not None and not self.boss_intermission_active)

    def miniboss_chance_for_phase(self) -> float:
        """Increase optional Rift Hunter frequency after the fifth phase."""

        growth_steps = max(
            0,
            self.phase - int(self.phase_cfg["miniboss_chance_growth_start_phase"]),
        )
        return min(
            float(self.phase_cfg["miniboss_chance_max"]),
            float(self.phase_cfg["miniboss_chance"])
            + growth_steps * float(self.phase_cfg["miniboss_chance_growth_per_phase"]),
        )

    def maximum_active_enemies(self) -> int:
        normal_limit = min(
            int(self.phase_cfg["maximum_enemy_absolute"]),
            int(self.enemy_cfg["maximum_active"])
            + (self.phase - 1)
            * int(self.phase_cfg["maximum_enemy_growth_per_phase"]),
        )
        if self.boss_encounter_number == 1:
            return min(
                normal_limit,
                int(self.phase_cfg["first_boss_maximum_active_enemies"]),
            )
        return normal_limit

    def _phase_step_budget(self) -> int:
        seconds = float(self.sim_cfg["phase_time_limit_seconds"])
        if self.is_boss_phase:
            bonus_key = (
                "first_boss_time_bonus_seconds"
                if self.boss_encounter_number == 1
                else "boss_time_bonus_seconds"
            )
            seconds += float(self.sim_cfg[bonus_key])
        return max(1, round(seconds * self.fps))

    @property
    def support_drone_active(self) -> bool:
        return self.drone_level > 0 or self.support_drone_phase == self.phase

    def _queue_choice(self, kind: str) -> None:
        if kind not in ("level_up", "phase_reward", "boss_reward"):
            raise ValueError(f"Unknown choice kind: {kind}")
        self._choice_queue.append(kind)

    def _choice_catalog(self, kind: str) -> list[dict[str, Any]]:
        key = {
            "level_up": "upgrade_catalog",
            "phase_reward": "phase_reward_catalog",
            "boss_reward": "boss_reward_catalog",
        }[kind]
        catalog = [dict(item) for item in self.progression_cfg[key]]
        if kind == "boss_reward":
            for item in catalog:
                if item["id"] == "artifact_core" and self.upgrade_stacks.get(
                    "artifact", 0
                ) >= 12:
                    item["name"] = "Rift Resonance"
                    item["description"] = (
                        "Artifact array saturated: gain repeatable Weapon Mastery."
                    )
                elif item["id"] == "drone_squadron" and self.upgrade_stacks.get(
                    "drone", 0
                ) >= 8:
                    item["name"] = "Drone Mastery Core"
                    item["description"] = (
                        "Core squadron complete: gain repeatable Drone Mastery."
                    )
        return catalog

    @staticmethod
    def _drone_count_for_level(level: int) -> int:
        if level <= 0:
            return 0
        return min(4, 1 + max(0, level - 1) // 3)

    def _project_drone_upgrade(self, amount: int) -> tuple[int, int, int]:
        """Return projected core tier, mastery tier and drone count."""

        core = self.upgrade_stacks.get("drone", 0)
        mastery = self.upgrade_stacks.get("drone_mastery", 0)
        core_gain = min(max(0, 8 - core), max(0, amount))
        surplus = max(0, amount - core_gain)
        next_core = core + core_gain
        next_mastery = mastery + (max(1, math.ceil(surplus / 2)) if surplus else 0)
        return (
            next_core,
            next_mastery,
            self._drone_count_for_level(next_core + next_mastery),
        )

    @classmethod
    def _drone_combat_summary(cls, core: int, mastery: int) -> str:
        level = core + mastery
        count = cls._drone_count_for_level(level)
        damage_percent = round(
            100 * (0.52 + 0.10 * math.sqrt(max(0, level)) + 0.04 * math.sqrt(max(0, mastery)))
        )
        reload_seconds = max(
            0.20,
            0.48 - 0.025 * min(8, core) - 0.012 * math.sqrt(max(0, mastery)),
        )
        return f"{count} drone(s); {damage_percent}% dmg; {reload_seconds:.3f}s reload"

    def choice_card_details(self, choice: dict[str, Any]) -> dict[str, str]:
        """Build a concise, truthful current-versus-next preview for a draft card."""

        item_id = str(choice["id"])
        kind = self.pending_choice_kind or "level_up"
        stacks = self.upgrade_stacks.get(item_id, 0)
        maximum = int(choice.get("max_stacks", 1))
        repeatable = bool(choice.get("repeatable", False))
        if repeatable:
            status = f"MASTERY {stacks} -> {stacks + 1}"
        elif stacks == 0:
            status = f"NEW - TIER 1 / {maximum}"
        else:
            status = f"TIER {stacks} -> {stacks + 1} / {maximum}"
        current = "Not installed"
        after = str(choice.get("description", "Upgrade applied"))
        name = str(choice["name"])

        if kind == "level_up":
            base_cooldown = float(self.player_cfg["fire_cooldown_seconds"])
            previews: dict[str, tuple[str, str]] = {
                "hull": (
                    f"Max hull {self.player.max_health:.0f}",
                    f"Max hull {self.player.max_health + 20:.0f}; repair 20",
                ),
                "repair": (
                    f"Hull {self.player.health:.0f}/{self.player.max_health:.0f}",
                    f"Hull {min(self.player.max_health, self.player.health + self.player.max_health * 0.45):.0f}/{self.player.max_health:.0f}",
                ),
                "damage": (
                    f"Core damage bonus +{12 * stacks}%",
                    f"Core damage bonus +{12 * (stacks + 1)}%",
                ),
                "fire_rate": (
                    f"Base reload {base_cooldown * 0.9**stacks:.3f}s",
                    f"Base reload {base_cooldown * 0.9**(stacks + 1):.3f}s",
                ),
                "multishot": (
                    f"{1 + stacks} {'beam' if 1 + stacks == 1 else 'beams'} per volley",
                    f"{2 + stacks} beams per volley",
                ),
                "range": (
                    f"Range +{18 * stacks}%; speed +{6 * stacks}%",
                    f"Range +{18 * (stacks + 1)}%; speed +{6 * (stacks + 1)}%",
                ),
                "laser": (
                    f"Weapon: {self.weapon_profile()['name']}",
                    "Prism Laser; +15% damage, +40% speed",
                ),
                "piercing": (
                    f"Pass through {stacks} extra targets",
                    f"Pass through {stacks + 1} extra targets",
                ),
                "splash": (
                    f"Blast radius {28 * stacks}px",
                    f"Blast radius {28 * (stacks + 1)}px",
                ),
                "shield": (
                    f"Damage resistance {min(50, 10 * stacks)}%",
                    f"Damage resistance {min(50, 10 * (stacks + 1))}%",
                ),
                "engine": (
                    f"Movement bonus +{7.5 * stacks:g}%",
                    f"Movement bonus +{7.5 * (stacks + 1):g}%",
                ),
                "homing": (
                    f"Guidance strength tier {stacks}",
                    f"Guidance strength tier {stacks + 1}",
                ),
                "regen": (
                    f"Regeneration {float(self.progression_cfg['passive_regen_per_second']) * stacks:.1f} hull/s",
                    f"Regeneration {float(self.progression_cfg['passive_regen_per_second']) * (stacks + 1):.1f} hull/s",
                ),
                "capacitor": (
                    f"Beam +{0.8 * stacks:.1f}px; assist +{20 * stacks}px",
                    f"Beam +{0.8 * (stacks + 1):.1f}px; assist +{20 * (stacks + 1)}px",
                ),
                "critical": (
                    f"Critical chance {min(40, 8 * stacks)}%",
                    f"Critical chance {min(40, 8 * (stacks + 1))}%",
                ),
                "leech": (
                    f"Restore {2 * stacks}% hull per kill",
                    f"Restore {2 * (stacks + 1)}% hull per kill",
                ),
                "riftbreaker": (
                    f"Boss damage bonus +{12 * stacks}%",
                    f"Boss damage bonus +{12 * (stacks + 1)}%",
                ),
                "weapon_mastery": (
                    f"Mastery damage bonus +{4 * stacks}%",
                    f"Mastery damage bonus +{4 * (stacks + 1)}%",
                ),
                "hull_mastery": (
                    f"Max hull {self.player.max_health:.0f}",
                    f"Max hull {self.player.max_health + 10:.0f}; repair 10",
                ),
                "drone_mastery": (
                    f"Drone mastery tier {stacks}",
                    f"Drone mastery tier {stacks + 1}; faster + stronger",
                ),
            }
            if item_id == "drone":
                next_core, next_mastery, _ = self._project_drone_upgrade(1)
                core = self.upgrade_stacks.get("drone", 0)
                mastery = self.upgrade_stacks.get("drone_mastery", 0)
                current = f"Core T{core}; " + self._drone_combat_summary(core, mastery)
                after = f"Core T{next_core}; " + self._drone_combat_summary(
                    next_core, next_mastery
                )
                status = (
                    f"NEW - CORE T1 / 8"
                    if self.upgrade_stacks.get("drone", 0) == 0
                    else f"CORE T{self.upgrade_stacks.get('drone', 0)} -> T{next_core} / 8"
                )
            elif item_id in previews:
                current, after = previews[item_id]
            if item_id == "repair":
                status = "INSTANT REPAIR"
        elif kind == "phase_reward":
            status = "NEXT-PHASE SUPPORT"
            if item_id == "repair_cache":
                current = f"Hull {self.player.health:.0f}/{self.player.max_health:.0f}"
                after = f"Hull {min(self.player.max_health, self.player.health + self.player.max_health * 0.60):.0f}/{self.player.max_health:.0f}"
            elif item_id == "nova_bomb":
                current = "Nova bomb armed" if self.nova_bomb_armed else "No bomb armed"
                after = "Next assault opens with an arena-wide blast"
            elif item_id == "wingman":
                name = "Wingman Core"
                next_core, next_mastery, _ = self._project_drone_upgrade(1)
                core = self.upgrade_stacks.get("drone", 0)
                mastery = self.upgrade_stacks.get("drone_mastery", 0)
                current = f"Core T{core}; " + self._drone_combat_summary(core, mastery)
                after = f"Core T{next_core}; " + self._drone_combat_summary(
                    next_core, next_mastery
                )
                status = "PERMANENT DRONE UPGRADE"
            elif item_id == "overdrive":
                current = "Inactive" if not self.overdrive_active else "Already active"
                after = "+25% damage and faster fire for next assault"
            elif item_id == "aegis":
                current = f"{self.barrier_charges} Aegis charge(s)"
                after = f"{self.barrier_charges + 2} Aegis charges"
        elif kind == "boss_reward":
            status = "PERMANENT BOSS RELIC"
            if item_id == "artifact_core":
                artifact = self.upgrade_stacks.get("artifact", 0)
                if artifact < 12:
                    current = f"Artifact T{artifact}; max hull {self.player.max_health:.0f}"
                    after = f"Artifact T{artifact + 1}; +8% damage; +15 hull"
                else:
                    mastery = self.upgrade_stacks.get("weapon_mastery", 0)
                    current = f"Weapon mastery T{mastery}"
                    after = f"Weapon mastery T{mastery + 1}; +4% damage"
            elif item_id == "drone_squadron":
                next_core, next_mastery, _ = self._project_drone_upgrade(2)
                core = self.upgrade_stacks.get("drone", 0)
                mastery = self.upgrade_stacks.get("drone_mastery", 0)
                current = f"Core T{core}; " + self._drone_combat_summary(core, mastery)
                after = f"Core T{next_core}; " + self._drone_combat_summary(
                    next_core, next_mastery
                )
            elif item_id == "full_restore":
                current = f"Hull {self.player.health:.0f}/{self.player.max_health:.0f}; Aegis {self.barrier_charges}"
                after = f"Full hull; Aegis {self.barrier_charges + 3}"
            elif item_id == "temporal_overdrive":
                current = "Standard weapon output"
                after = "+25% damage and faster fire for two assaults"

        return {
            "name": name,
            "status": status,
            "current": current,
            "after": after,
        }

    def _roll_choices(self, kind: str) -> list[dict[str, Any]]:
        catalog = self._choice_catalog(kind)
        if kind == "level_up":
            catalog = [
                item
                for item in catalog
                if self.player.level >= int(item.get("minimum_level", 1))
                and (
                    item["id"] == "repair"
                    or bool(item.get("repeatable", False))
                    or self.upgrade_stacks.get(str(item["id"]), 0)
                    < int(item.get("max_stacks", 1))
                )
            ]
        count = min(int(self.progression_cfg["upgrade_choices"]), len(catalog))
        if count == len(catalog):
            order = self.np_random.permutation(len(catalog))
            return [catalog[int(index)] for index in order]
        indices = self.np_random.choice(len(catalog), size=count, replace=False)
        return [catalog[int(index)] for index in indices]

    def _auto_choice_index(self) -> int:
        """Draft a coherent survival/build choice for non-interactive agents."""

        health_ratio = self.player.health / max(1.0, self.player.max_health)
        scores: list[float] = []
        for item in self.pending_choices:
            item_id = str(item["id"])
            if self.pending_choice_kind == "phase_reward":
                score = {
                    "repair_cache": 16.0 if health_ratio < 0.52 else 5.0,
                    "nova_bomb": 11.0 if self.is_boss_phase else 7.5,
                    "wingman": (
                        10.0 if self.drone_level < 10 else 6.0
                    ),
                    "overdrive": 10.5 if self.is_boss_phase else 8.0,
                    "aegis": 13.0 if self.is_boss_phase or health_ratio < 0.62 else 6.5,
                }.get(item_id, 0.0)
            elif self.pending_choice_kind == "boss_reward":
                score = {
                    "artifact_core": 10.5
                    - 0.5 * self.upgrade_stacks.get("artifact", 0),
                    "drone_squadron": (
                        9.5 if self.drone_level < 12 else 6.5
                    ),
                    "full_restore": 18.0 if health_ratio < 0.55 else 5.5,
                    "temporal_overdrive": 10.0,
                }.get(item_id, 0.0)
            else:
                score = float(item.get("auto_priority", 0.0))
                stacks = self.upgrade_stacks.get(item_id, 0)
                if item_id in ("repair", "hull", "hull_mastery", "leech", "regen"):
                    score += max(0.0, 0.72 - health_ratio) * 18.0
                if item_id == "repair" and health_ratio > 0.9:
                    score -= 14.0
                elif item_id == "repair" and health_ratio < 0.35:
                    score += 8.0
                if self.is_boss_phase:
                    score += {
                        "riftbreaker": 5.0,
                        "shield": 4.0,
                        "engine": 3.5,
                        "homing": 2.0,
                        "critical": 2.0,
                    }.get(item_id, 0.0)
                if self.control_style == "rotation":
                    score += {
                        "homing": 4.5,
                        "engine": 3.0,
                        "range": 2.0,
                        "shield": 1.5,
                    }.get(item_id, 0.0)
                else:
                    score += {
                        "multishot": 3.5,
                        "capacitor": 2.5,
                        "splash": 2.0,
                    }.get(item_id, 0.0)
                if self.upgrade_stacks.get("laser", 0):
                    score += {
                        "range": 2.5,
                        "piercing": 2.0,
                        "critical": 3.0,
                    }.get(item_id, 0.0)
                if self.upgrade_stacks.get("multishot", 0):
                    score += {
                        "damage": 2.0,
                        "critical": 2.5,
                        "capacitor": 1.5,
                    }.get(item_id, 0.0)
                if self.drone_level:
                    score += {
                        "drone": 2.5,
                        "drone_mastery": 3.0,
                    }.get(item_id, 0.0)
                score -= stacks * (0.45 if item.get("repeatable") else 0.65)
            scores.append(score)
        return int(np.argmax(np.asarray(scores, dtype=np.float32)))

    def _prepare_next_choice(self, events: dict[str, Any] | None = None) -> None:
        if self.pending_choice_kind is not None:
            return
        while self._choice_queue and self.pending_choice_kind is None:
            self.pending_choice_kind = self._choice_queue.pop(0)
            self.pending_choices = self._roll_choices(self.pending_choice_kind)
            if events is not None:
                events["choice_offered"] = self.pending_choice_kind
            if self.manual_choices:
                return
            self.choose_pending_choice(self._auto_choice_index(), events=events)

    def choose_pending_choice(
        self, index: int, *, events: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Apply one visible draft choice and return the selected card."""

        if self.pending_choice_kind is None or not self.pending_choices:
            raise RuntimeError("There is no pending choice.")
        if not 0 <= index < len(self.pending_choices):
            raise ValueError(f"Choice index {index} is out of range")

        kind = self.pending_choice_kind
        selected = dict(self.pending_choices[index])
        selected_id = str(selected["id"])
        preview = self.choice_card_details(selected)
        if kind == "level_up":
            if selected_id == "hull":
                self.upgrade_stacks[selected_id] += 1
                self.player.max_health += 20.0
                self.player.health = min(self.player.max_health, self.player.health + 20.0)
            elif selected_id == "hull_mastery":
                self.upgrade_stacks[selected_id] += 1
                self.player.max_health += 10.0
                self.player.health = min(self.player.max_health, self.player.health + 10.0)
            elif selected_id == "repair":
                self.player.health = min(
                    self.player.max_health,
                    self.player.health + self.player.max_health * 0.45,
                )
            else:
                self.upgrade_stacks[selected_id] += 1
            self.episode_stats["upgrades_chosen"] = int(
                self.episode_stats["upgrades_chosen"]
            ) + 1
        elif selected_id == "repair_cache":
            self.player.health = min(
                self.player.max_health,
                self.player.health + self.player.max_health * 0.60,
            )
        elif selected_id == "nova_bomb":
            self.nova_bomb_armed = True
        elif selected_id == "wingman":
            self._grant_drone_tiers(1)
            self.drone_cooldown_steps = 1
        elif selected_id == "overdrive":
            self.overdrive_until_phase = max(self.overdrive_until_phase, self.phase)
        elif selected_id == "aegis":
            self.barrier_charges += 2
        elif selected_id == "artifact_core":
            if self.upgrade_stacks["artifact"] < 12:
                self.upgrade_stacks["artifact"] += 1
                self.player.max_health += 15.0
                self.player.health = min(self.player.max_health, self.player.health + 15.0)
            else:
                self.upgrade_stacks["weapon_mastery"] += 1
        elif selected_id == "drone_squadron":
            self._grant_drone_tiers(2)
            self.drone_cooldown_steps = 1
        elif selected_id == "full_restore":
            self.player.health = self.player.max_health
            self.barrier_charges += 3
        elif selected_id == "temporal_overdrive":
            self.overdrive_until_phase = max(self.overdrive_until_phase, self.phase + 1)
        if kind == "phase_reward":
            self.episode_stats["phase_rewards_chosen"] = int(
                self.episode_stats["phase_rewards_chosen"]
            ) + 1
        elif kind == "boss_reward":
            self.episode_stats["boss_rewards_chosen"] = int(
                self.episode_stats["boss_rewards_chosen"]
            ) + 1

        self.last_upgrade_name = str(selected["name"])
        self.last_upgrade_detail = preview["after"]
        self.upgrade_banner_steps = max(
            1,
            int(float(self.progression_cfg["upgrade_banner_seconds"]) * self.fps),
        )
        if events is not None:
            events["choice_selected"] = selected_id
            events["upgrade_unlocked"] = self.last_upgrade_name
            events["upgrade_detail"] = self.last_upgrade_detail
        self.pending_choice_kind = None
        self.pending_choices = []
        self._prepare_next_choice(events)
        return selected

    def _grant_drone_tiers(self, amount: int) -> None:
        """Fill the eight core tiers, then convert surplus into mastery."""

        current = self.upgrade_stacks.get("drone", 0)
        core_gain = min(max(0, 8 - current), max(0, amount))
        self.upgrade_stacks["drone"] = current + core_gain
        surplus = max(0, amount - core_gain)
        if surplus:
            self.upgrade_stacks["drone_mastery"] += max(1, math.ceil(surplus / 2))

    # ------------------------------------------------------------------
    # Simulation updates
    # ------------------------------------------------------------------

    def _tick_cooldowns(self) -> None:
        self.player.fire_cooldown_steps = max(0, self.player.fire_cooldown_steps - 1)
        self.drone_cooldown_steps = max(0, self.drone_cooldown_steps - 1)
        self.upgrade_banner_steps = max(0, self.upgrade_banner_steps - 1)
        for enemy in self.enemies:
            enemy.attack_cooldown_steps = max(0, enemy.attack_cooldown_steps - 1)
            enemy.missile_cooldown_steps = max(0, enemy.missile_cooldown_steps - 1)

    def _spawn_boss_defenders(self, boss: Spawner, events: dict[str, Any]) -> None:
        """Deploy one finite, clearly signalled low-health defender wave."""

        count = self.boss_defender_count_for_phase()
        orbit_radius = float(self.phase_cfg["boss_defender_orbit_radius"])
        health_scale = (
            1.0
            + (self.phase - 1) * float(self.phase_cfg["enemy_health_growth"])
        ) * float(self.phase_cfg["boss_defender_health_multiplier"])
        max_health = float(self.enemy_cfg["max_health"]) * health_scale
        interval = max(
            1,
            int(
                float(self.phase_cfg["boss_defender_missile_interval_seconds"])
                * self.fps
            ),
        )
        for index in range(count):
            angle = math.tau * index / max(1, count)
            radius = 17.0 if index % 2 == 0 else 14.0
            self.enemies.append(
                Enemy(
                    x=float(
                        np.clip(
                            boss.x + math.cos(angle) * orbit_radius,
                            radius,
                            self.width - radius,
                        )
                    ),
                    y=float(
                        np.clip(
                            boss.y + math.sin(angle) * orbit_radius,
                            self.playfield_top + radius,
                            self.height - radius,
                        )
                    ),
                    radius=radius,
                    entity_id=self._new_id(),
                    max_health=max_health,
                    health=max_health,
                    speed=float(self.phase_cfg["boss_move_speed"]) * 1.35,
                    is_elite=True,
                    is_boss_defender=True,
                    defender_kind="turret" if index % 2 == 0 else "interceptor",
                    orbit_angle=angle,
                    missile_cooldown_steps=max(1, interval // 2 + index * interval // count),
                )
            )
        boss.defender_wave_started = True
        boss.defender_regen_cap = min(
            boss.max_health,
            boss.health
            + boss.max_health * float(self.phase_cfg["boss_defender_regen_fraction"]),
        )
        events["boss_defenders_spawned"] += count
        events["enemies_spawned"] += count
        self.last_upgrade_name = "BOSS AEGIS: destroy the sentry wing"
        self.last_upgrade_detail = "The boss is immune and repairing while sentries remain"
        self.upgrade_banner_steps = max(
            self.upgrade_banner_steps,
            int(float(self.progression_cfg["upgrade_banner_seconds"]) * self.fps),
        )

    def _fire_enemy_missile(self, defender: Enemy, events: dict[str, Any]) -> None:
        """Fire a telegraphed missile with brief guidance and a dodgeable path."""

        angle = math.atan2(self.player.y - defender.y, self.player.x - defender.x)
        speed = float(self.phase_cfg["boss_defender_missile_speed"])
        radius = 7.0
        offset = defender.radius + radius + 3.0
        self.projectiles.append(
            Projectile(
                x=defender.x + math.cos(angle) * offset,
                y=defender.y + math.sin(angle) * offset,
                radius=radius,
                entity_id=self._new_id(),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                damage=float(self.phase_cfg["boss_defender_missile_damage"])
                * (1.0 + 0.035 * min(12, self.phase - 1)),
                lifetime_steps=max(
                    1,
                    int(
                        float(self.phase_cfg["boss_defender_missile_lifetime_seconds"])
                        * self.fps
                    ),
                ),
                weapon_kind="enemy_missile",
                owner="enemy",
                telegraph_steps=max(
                    1,
                    int(
                        float(self.phase_cfg["boss_defender_missile_telegraph_seconds"])
                        * self.fps
                    ),
                ),
                homing_turn_rate=math.radians(
                    float(self.phase_cfg["boss_defender_missile_turn_degrees"])
                ),
                guidance_steps=max(
                    0,
                    int(
                        float(
                            self.phase_cfg[
                                "boss_defender_missile_guidance_seconds"
                            ]
                        )
                        * self.fps
                    ),
                ),
            )
        )
        events["missiles_fired"] += 1

    def _update_boss_movement_and_defenders(self, events: dict[str, Any]) -> None:
        """Advance the mobile boss and its one-shot defensive intermission."""

        boss = next((spawner for spawner in self.spawners if spawner.is_boss), None)
        if boss is None:
            return

        # The shielded opening remains a readable stationary damage check.
        # Once broken, a slow orbit prevents point-blank camping without turning
        # the rift into a fast chaser.
        if boss.shield <= 0.0:
            dx = self.player.x - boss.x
            dy = self.player.y - boss.y
            distance = max(1e-6, math.hypot(dx, dy))
            desired = float(self.phase_cfg["boss_move_standoff"])
            radial = float(np.clip((distance - desired) / desired, -0.65, 0.65))
            orbit_sign = 1.0 if self.boss_encounter_number % 2 else -1.0
            direction_x = dx / distance * radial - dy / distance * 0.58 * orbit_sign
            direction_y = dy / distance * radial + dx / distance * 0.58 * orbit_sign
            length = max(1e-6, math.hypot(direction_x, direction_y))
            speed = float(self.phase_cfg["boss_move_speed"]) * min(
                1.35, 1.0 + 0.08 * (self.boss_encounter_number - 1)
            )
            boss.vx = direction_x / length * speed
            boss.vy = direction_y / length * speed
            boss.x += boss.vx * self.dt
            boss.y += boss.vy * self.dt
            boss.x = float(np.clip(boss.x, boss.radius, self.width - boss.radius))
            boss.y = float(
                np.clip(
                    boss.y,
                    self.playfield_top + boss.radius,
                    self.height - boss.radius,
                )
            )

        health_ratio = boss.health / max(1.0, boss.max_health)
        if (
            boss.shield <= 0.0
            and not boss.defender_wave_started
            and health_ratio <= float(self.phase_cfg["boss_defender_health_gate"])
        ):
            self._spawn_boss_defenders(boss, events)

        defenders = self.boss_defenders
        if not defenders:
            return

        regeneration = (
            boss.max_health
            * float(self.phase_cfg["boss_defender_regen_per_second"])
            * self.dt
        )
        before = boss.health
        boss.health = min(boss.defender_regen_cap, boss.health + regeneration)
        events["boss_health_regenerated"] += boss.health - before

        orbit_radius = float(self.phase_cfg["boss_defender_orbit_radius"])
        missile_interval = max(
            1,
            int(
                float(self.phase_cfg["boss_defender_missile_interval_seconds"])
                * self.fps
            ),
        )
        count = len(defenders)
        for index, defender in enumerate(defenders):
            defender.orbit_angle += self.dt * (
                0.55 if defender.defender_kind == "turret" else 0.85
            )
            radius = orbit_radius + (18.0 if defender.defender_kind == "interceptor" else 0.0)
            desired_x = boss.x + math.cos(defender.orbit_angle) * radius
            desired_y = boss.y + math.sin(defender.orbit_angle) * radius
            dx = desired_x - defender.x
            dy = desired_y - defender.y
            distance = max(1e-6, math.hypot(dx, dy))
            maximum_move = defender.speed * self.dt
            scale = min(1.0, maximum_move / distance)
            defender.vx = dx / self.dt * scale
            defender.vy = dy / self.dt * scale
            defender.x += dx * scale
            defender.y += dy * scale
            defender.x = float(np.clip(defender.x, defender.radius, self.width - defender.radius))
            defender.y = float(
                np.clip(
                    defender.y,
                    self.playfield_top + defender.radius,
                    self.height - defender.radius,
                )
            )
            if defender.missile_cooldown_steps <= 0:
                self._fire_enemy_missile(defender, events)
                cadence_scale = max(0.72, 1.0 - 0.05 * (self.boss_encounter_number - 1))
                defender.missile_cooldown_steps = max(
                    1,
                    int(missile_interval * cadence_scale + index * 3 / max(1, count)),
                )

    def _update_support_drone(self, events: dict[str, Any]) -> None:
        if not self.support_drone_active or self.drone_cooldown_steps > 0:
            return
        target = self._nearest_target()
        if target is None:
            return
        mastery = self.upgrade_stacks.get("drone_mastery", 0)
        for drone_index in range(self.drone_count):
            orbit = self.step_count * 0.055 + math.tau * drone_index / self.drone_count
            orbit_radius = 34.0 + 8.0 * (drone_index % 2)
            origin_x = self.player.x + math.cos(orbit) * orbit_radius
            origin_y = self.player.y + math.sin(orbit) * orbit_radius
            angle = math.atan2(target.y - origin_y, target.x - origin_x)
            speed = float(self.projectile_cfg["speed"]) * 0.9
            self.projectiles.append(
                Projectile(
                    x=origin_x,
                    y=origin_y,
                    radius=3.0 + min(2.0, self.drone_level * 0.15),
                    entity_id=self._new_id(),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    damage=float(self.projectile_cfg["damage"])
                    * (
                        0.52
                        + 0.10 * math.sqrt(max(0, self.drone_level))
                        + 0.04 * math.sqrt(max(0, mastery))
                    ),
                    lifetime_steps=int(
                        float(self.projectile_cfg["lifetime_seconds"]) * self.fps
                    ),
                    weapon_kind="drone",
                    owner="drone",
                    pierces_remaining=min(2, mastery // 3),
                )
            )
        events["drone_shots"] += self.drone_count
        core_level = min(8, self.upgrade_stacks.get("drone", 0))
        interval = max(
            0.20,
            0.48 - 0.025 * core_level - 0.012 * math.sqrt(max(0, mastery)),
        )
        self.drone_cooldown_steps = max(1, int(interval * self.fps))

    def _steer_projectile(self, projectile: Projectile) -> None:
        """Curve player projectiles toward targets when Guidance Matrix is owned."""

        homing = self.upgrade_stacks.get("homing", 0)
        if homing <= 0 or projectile.owner != "player":
            return
        candidates = self._priority_targets()
        if not candidates:
            return
        target = min(
            candidates,
            key=lambda item: (item.x - projectile.x) ** 2 + (item.y - projectile.y) ** 2,
        )
        desired = math.atan2(target.y - projectile.y, target.x - projectile.x)
        current = math.atan2(projectile.vy, projectile.vx)
        difference = (desired - current + math.pi) % math.tau - math.pi
        maximum_turn = math.radians(1.5 + 1.8 * homing)
        angle = current + float(np.clip(difference, -maximum_turn, maximum_turn))
        speed = math.hypot(projectile.vx, projectile.vy)
        projectile.vx = math.cos(angle) * speed
        projectile.vy = math.sin(angle) * speed

    def _steer_enemy_missile(self, projectile: Projectile) -> None:
        desired = math.atan2(
            self.player.y - projectile.y, self.player.x - projectile.x
        )
        current = math.atan2(projectile.vy, projectile.vx)
        difference = (desired - current + math.pi) % math.tau - math.pi
        angle = current + float(
            np.clip(difference, -projectile.homing_turn_rate, projectile.homing_turn_rate)
        )
        speed = math.hypot(projectile.vx, projectile.vy)
        projectile.vx = math.cos(angle) * speed
        projectile.vy = math.sin(angle) * speed

    def _aim_enemy_missile_at_player(self, projectile: Projectile) -> None:
        """Track during the visible lock-on without moving the missile."""

        angle = math.atan2(
            self.player.y - projectile.y, self.player.x - projectile.x
        )
        speed = max(1.0, math.hypot(projectile.vx, projectile.vy))
        projectile.vx = math.cos(angle) * speed
        projectile.vy = math.sin(angle) * speed

    def _damage_target(
        self,
        target: Enemy | Spawner,
        amount: float,
        events: dict[str, Any],
        destroyed_enemy_ids: set[int],
        destroyed_spawner_ids: set[int],
        *,
        source: str,
        count_hit: bool,
    ) -> None:
        if target.health <= 0.0:
            return
        is_spawner = isinstance(target, Spawner)
        if is_spawner and target.is_boss and self.boss_intermission_active:
            events["boss_immune_hits"] += 1
            if count_hit:
                events["projectile_hits" if source == "player" else "drone_hits"] += 1
            events["impacts"].append(
                {
                    "x": float(target.x),
                    "y": float(target.y),
                    "kind": "boss_immune",
                    "destroyed": False,
                }
            )
            return
        effective_amount = float(amount)
        riftbreaker_tiers = self.upgrade_stacks.get("riftbreaker", 0)
        if riftbreaker_tiers and (
            is_spawner or (isinstance(target, Enemy) and target.is_miniboss)
        ):
            effective_amount *= 1.0 + 0.12 * riftbreaker_tiers
        if (
            is_spawner
            and target.is_boss
            and any(zone.telegraph_steps > 0 for zone in self.danger_zones)
        ):
            effective_amount *= float(self.phase_cfg["boss_channel_damage_multiplier"])
        absorbed = 0.0
        if is_spawner and target.shield > 0.0:
            absorbed = min(effective_amount, target.shield)
            target.shield -= absorbed
            effective_amount -= absorbed
        damage = min(effective_amount, target.health)
        target.health -= effective_amount
        events["damage_dealt_spawner" if is_spawner else "damage_dealt_enemy"] += damage + absorbed
        if count_hit:
            events["projectile_hits" if source == "player" else "drone_hits"] += 1
        destroyed = target.health <= 0.0
        events["impacts"].append(
            {
                "x": float(target.x),
                "y": float(target.y),
                "kind": "spawner" if is_spawner else "enemy",
                "destroyed": destroyed,
            }
        )
        if destroyed:
            (destroyed_spawner_ids if is_spawner else destroyed_enemy_ids).add(
                target.entity_id
            )

    def _update_projectiles(self, events: dict[str, Any]) -> None:
        surviving_projectiles: list[Projectile] = []
        destroyed_enemy_ids: set[int] = set()
        destroyed_spawner_ids: set[int] = set()
        miniboss_ids = {enemy.entity_id for enemy in self.enemies if enemy.is_miniboss}
        defender_ids = {
            enemy.entity_id for enemy in self.enemies if enemy.is_boss_defender
        }

        for projectile in self.projectiles:
            if projectile.owner == "enemy":
                projectile.lifetime_steps -= 1
                if projectile.telegraph_steps > 0:
                    self._aim_enemy_missile_at_player(projectile)
                    projectile.telegraph_steps -= 1
                    surviving_projectiles.append(projectile)
                    continue
                if projectile.guidance_steps > 0:
                    self._steer_enemy_missile(projectile)
                    projectile.guidance_steps -= 1
                projectile.x += projectile.vx * self.dt
                projectile.y += projectile.vy * self.dt
                if circles_overlap(projectile, self.player):
                    if self._apply_player_damage(projectile.damage, events) > 0.0:
                        events["missile_hits"] += 1
                    events["impacts"].append(
                        {
                            "x": float(self.player.x),
                            "y": float(self.player.y),
                            "kind": "missile",
                            "destroyed": True,
                        }
                    )
                    continue
                if (
                    projectile.lifetime_steps <= 0
                    or projectile.x < -projectile.radius
                    or projectile.x > self.width + projectile.radius
                    or projectile.y < self.playfield_top - projectile.radius
                    or projectile.y > self.height + projectile.radius
                ):
                    events["missiles_evaded"] += 1
                    continue
                surviving_projectiles.append(projectile)
                continue

            self._steer_projectile(projectile)
            projectile.x += projectile.vx * self.dt
            projectile.y += projectile.vy * self.dt
            projectile.lifetime_steps -= 1
            if (
                projectile.lifetime_steps <= 0
                or projectile.x < -projectile.radius
                or projectile.x > self.width + projectile.radius
                or projectile.y < -projectile.radius
                or projectile.y > self.height + projectile.radius
            ):
                continue

            candidates: list[Enemy | Spawner] = [*self.enemies, *self.spawners]
            target = next(
                (
                    item
                    for item in candidates
                    if item.entity_id not in projectile.hit_entity_ids
                    and item.entity_id not in destroyed_enemy_ids
                    and item.entity_id not in destroyed_spawner_ids
                    and circles_overlap(projectile, item)
                ),
                None,
            )
            if target is None:
                surviving_projectiles.append(projectile)
                continue

            projectile.hit_entity_ids.add(target.entity_id)
            self._damage_target(
                target,
                projectile.damage,
                events,
                destroyed_enemy_ids,
                destroyed_spawner_ids,
                source=projectile.owner,
                count_hit=True,
            )
            if projectile.splash_radius > 0.0:
                for nearby in candidates:
                    if nearby.entity_id == target.entity_id or nearby.health <= 0.0:
                        continue
                    if math.hypot(nearby.x - target.x, nearby.y - target.y) <= projectile.splash_radius:
                        self._damage_target(
                            nearby,
                            projectile.damage * 0.6,
                            events,
                            destroyed_enemy_ids,
                            destroyed_spawner_ids,
                            source=projectile.owner,
                            count_hit=False,
                        )
            if projectile.pierces_remaining > 0:
                projectile.pierces_remaining -= 1
                surviving_projectiles.append(projectile)

        if destroyed_enemy_ids:
            self.enemies = [item for item in self.enemies if item.entity_id not in destroyed_enemy_ids]
            events["enemies_destroyed"] += len(destroyed_enemy_ids)
            events["minibosses_destroyed"] += len(
                destroyed_enemy_ids.intersection(miniboss_ids)
            )
            events["boss_defenders_destroyed"] += len(
                destroyed_enemy_ids.intersection(defender_ids)
            )
            self._grant_miniboss_caches(events)
        if destroyed_spawner_ids:
            self.episode_stats["bosses_destroyed"] = int(
                self.episode_stats["bosses_destroyed"]
            ) + sum(
                int(item.is_boss)
                for item in self.spawners
                if item.entity_id in destroyed_spawner_ids
            )
            self.spawners = [item for item in self.spawners if item.entity_id not in destroyed_spawner_ids]
            events["spawners_destroyed"] += len(destroyed_spawner_ids)
        self.projectiles = surviving_projectiles

    def _update_enemies(self, events: dict[str, Any]) -> None:
        contact_scale = min(
            float(self.phase_cfg["contact_damage_max_multiplier"]),
            1.0 + (self.phase - 1) * float(self.phase_cfg["contact_damage_growth"]),
        )
        base_contact_damage = float(self.enemy_cfg["contact_damage"]) * contact_scale
        cooldown_steps = max(1, int(float(self.enemy_cfg["attack_cooldown_seconds"]) * self.fps))

        for enemy in self.enemies:
            dx = self.player.x - enemy.x
            dy = self.player.y - enemy.y
            distance = math.hypot(dx, dy)
            if distance > 1e-8 and not enemy.is_boss_defender:
                enemy.vx = dx / distance * enemy.speed
                enemy.vy = dy / distance * enemy.speed
                enemy.x += enemy.vx * self.dt
                enemy.y += enemy.vy * self.dt

            if circles_overlap(enemy, self.player) and enemy.attack_cooldown_steps == 0:
                events['contact_events'] = int(events.get('contact_events', 0)) + 1
                contact_damage = base_contact_damage
                if enemy.is_elite:
                    contact_damage *= 1.2
                if enemy.is_miniboss:
                    contact_damage *= float(self.phase_cfg["miniboss_contact_multiplier"])
                if enemy.is_boss_defender:
                    contact_damage *= 0.8
                self._apply_player_damage(contact_damage, events)
                enemy.attack_cooldown_steps = cooldown_steps

                if distance > 1e-8:
                    knockback = float(self.enemy_cfg["contact_knockback"])
                    enemy.x -= dx / distance * knockback
                    enemy.y -= dy / distance * knockback

            enemy.x = float(np.clip(enemy.x, enemy.radius, self.width - enemy.radius))
            enemy.y = float(
                np.clip(
                    enemy.y,
                    self.playfield_top + enemy.radius,
                    self.height - enemy.radius,
                )
            )

    def _update_spawners(self, events: dict[str, Any]) -> None:
        if self.phase_transition_steps > 0:
            return

        max_enemies = self.maximum_active_enemies()
        for spawner in self.spawners:
            if spawner.is_boss and self.boss_intermission_active:
                continue
            spawner.spawn_cooldown_steps -= 1
            if spawner.spawn_cooldown_steps <= 0:
                if len(self.enemies) < max_enemies:
                    self._spawn_enemy(spawner)
                    events["enemies_spawned"] += 1
                spawner.spawn_cooldown_steps = self._spawn_interval_steps(spawner)

    def _update_boss_skills(self, events: dict[str, Any]) -> None:
        """Advance grouped telegraphs and resolve each barrage at most once."""

        self._update_boss_summons(events)
        newly_triggered: set[int] = set()
        expired_attacks: set[int] = set()
        for zone in self.danger_zones:
            if zone.telegraph_steps > 0:
                zone.telegraph_steps -= 1
                if zone.telegraph_steps == 0:
                    zone.triggered = True
                    newly_triggered.add(zone.attack_id)
                continue
            zone.active_steps -= 1
            if zone.active_steps <= 0:
                expired_attacks.add(zone.attack_id)

        for attack_id in newly_triggered:
            group = [zone for zone in self.danger_zones if zone.attack_id == attack_id]
            hit = next(
                (zone for zone in group if self._danger_zone_contains_player(zone)),
                None,
            )
            if hit is not None:
                for zone in group:
                    zone.hit_player = True
                if self._apply_player_damage(hit.damage, events) > 0.0:
                    events["boss_skill_hits"] += 1

        for attack_id in expired_attacks:
            group = [zone for zone in self.danger_zones if zone.attack_id == attack_id]
            if group and not any(zone.hit_player for zone in group):
                events["boss_skills_dodged"] += 1
        self.danger_zones = [
            zone
            for zone in self.danger_zones
            if zone.telegraph_steps > 0 or zone.active_steps > 0
        ]

        if not any(spawner.is_boss for spawner in self.spawners):
            self.boss_skill_cooldown_steps = 0
            return
        self.boss_skill_cooldown_steps -= 1
        if self.boss_skill_cooldown_steps <= 0 and not any(
            zone.telegraph_steps > 0 for zone in self.danger_zones
        ):
            self._cast_boss_skill(events)

    def _update_boss_summons(self, events: dict[str, Any]) -> None:
        """Release a finite reinforcement budget at evenly spaced health gates.

        There is no shield regeneration or unlimited summon/XP farming. Summons
        wait while the player is near the boss, a barrage is telegraphing, or
        the finite sentry intermission is active.
        """
        limit = self.boss_summon_limit_for_phase()
        for boss in self.spawners:
            if not boss.is_boss:
                continue
            boss.summon_cooldown_steps = max(0, boss.summon_cooldown_steps - 1)
            # Three summons use 75/50/25% health gates; changing the configured
            # finite budget keeps the gates evenly distributed automatically.
            threshold = 1.0 - (boss.summons_used + 1) / (limit + 1)
            if (
                boss.summons_used >= limit or boss.summon_cooldown_steps > 0
                or self.boss_intermission_active
                or boss.health / boss.max_health > threshold
                or sum(enemy.is_miniboss for enemy in self.enemies)
                >= self.boss_active_summon_limit_for_phase()
                or len(self.enemies) >= self.maximum_active_enemies()
                or math.hypot(boss.x - self.player.x, boss.y - self.player.y) < 180.0
                or any(zone.telegraph_steps > 0 for zone in self.danger_zones)
            ):
                continue
            self._spawn_miniboss(boss)
            boss.summons_used += 1
            boss.summon_cooldown_steps = int(
                self.fps * float(self.phase_cfg["boss_summon_interval_seconds"])
            )
            events["minibosses_spawned"] += 1
            events["enemies_spawned"] += 1

    def _cast_boss_skill(self, events: dict[str, Any]) -> None:
        patterns = (
            "twin_sweep",
            "cross_barrage",
            "diagonal_lattice",
            "trident_walls",
            "nova_cage",
        )
        pattern = patterns[self.boss_skill_index % len(patterns)]
        self.boss_skill_index += 1
        self.boss_attack_serial += 1
        attack_id = self.boss_attack_serial
        telegraph = max(
            1, int(float(self.phase_cfg["boss_skill_telegraph_seconds"]) * self.fps)
        )
        active = max(
            1, int(float(self.phase_cfg["boss_skill_active_seconds"]) * self.fps)
        )
        damage_growth = 1.0 + float(self.phase_cfg["boss_skill_damage_growth"]) * min(
            12, self.phase - 1
        )
        damage = float(self.phase_cfg["boss_skill_damage"]) * damage_growth
        if self.boss_encounter_number == 1:
            damage *= float(self.phase_cfg["first_boss_skill_damage_multiplier"])
        predicted_x = float(np.clip(self.player.x + self.player.vx * 0.32, 0, self.width))
        predicted_y = float(
            np.clip(
                self.player.y + self.player.vy * 0.32,
                self.playfield_top,
                self.height,
            )
        )
        names = {
            "twin_sweep": "TWIN SWEEP",
            "cross_barrage": "CROSS BARRAGE",
            "diagonal_lattice": "DIAGONAL LATTICE",
            "trident_walls": "TRIDENT WALLS",
            "nova_cage": "NOVA CAGE",
        }
        attack_name = names[pattern]

        def line_zone(x: float, y: float, angle: float) -> DangerZone:
            return DangerZone(
                kind="line",
                x=float(np.clip(x, 0.0, self.width)),
                y=float(np.clip(y, self.playfield_top, self.height)),
                angle=angle,
                half_width=float(self.phase_cfg["boss_skill_line_half_width"]),
                half_length=math.hypot(self.width, self.height),
                telegraph_steps=telegraph,
                active_steps=active,
                maximum_telegraph_steps=telegraph,
                damage=damage,
                attack_id=attack_id,
                attack_name=attack_name,
            )

        spacing = float(self.phase_cfg["boss_skill_parallel_spacing"])
        zones: list[DangerZone]
        if pattern == "twin_sweep":
            direction = 1.0 if predicted_y < (self.playfield_top + self.height) / 2 else -1.0
            zones = [
                line_zone(predicted_x, predicted_y, 0.0),
                line_zone(predicted_x, predicted_y + direction * spacing, 0.0),
            ]
        elif pattern == "cross_barrage":
            zones = [
                line_zone(predicted_x, predicted_y, 0.0),
                line_zone(predicted_x, predicted_y, math.pi / 2.0),
            ]
        elif pattern == "diagonal_lattice":
            zones = [
                line_zone(predicted_x, predicted_y, math.pi / 4.0),
                line_zone(predicted_x, predicted_y, -math.pi / 4.0),
            ]
        elif pattern == "trident_walls":
            offsets = [-spacing, 0.0, spacing]
            if self.phase < int(self.phase_cfg["boss_skill_triple_line_start_phase"]):
                offsets = [0.0, spacing if predicted_x < self.width / 2 else -spacing]
            zones = [
                line_zone(predicted_x + offset, predicted_y, math.pi / 2.0)
                for offset in offsets
            ]
        else:
            zones = [
                DangerZone(
                    kind="circle",
                    x=predicted_x,
                    y=predicted_y,
                    radius=float(self.phase_cfg["boss_skill_circle_radius"]),
                    telegraph_steps=telegraph,
                    active_steps=active,
                    maximum_telegraph_steps=telegraph,
                    damage=damage,
                    attack_id=attack_id,
                    attack_name=attack_name,
                ),
                line_zone(predicted_x, predicted_y, math.pi / 4.0),
                line_zone(predicted_x, predicted_y, -math.pi / 4.0),
            ]
        self.danger_zones.extend(zones)
        events["boss_skills_cast"] += 1
        minimum = float(self.phase_cfg["boss_skill_min_interval_seconds"])
        interval = max(
            minimum,
            float(self.phase_cfg["boss_skill_interval_seconds"])
            - 0.10 * max(0, self.phase - int(self.phase_cfg["boss_interval"])),
        )
        self.boss_skill_cooldown_steps = max(1, int(interval * self.fps))

    def _danger_zone_contains_player(self, zone: DangerZone) -> bool:
        dx = self.player.x - zone.x
        dy = self.player.y - zone.y
        if zone.kind == "circle":
            return math.hypot(dx, dy) <= zone.radius + self.player.radius
        perpendicular = abs(-math.sin(zone.angle) * dx + math.cos(zone.angle) * dy)
        parallel = abs(math.cos(zone.angle) * dx + math.sin(zone.angle) * dy)
        return (
            perpendicular <= zone.half_width + self.player.radius
            and parallel <= zone.half_length + self.player.radius
        )

    def _single_danger_zone_features(
        self, zone: DangerZone
    ) -> tuple[float, float, float, float, float, float]:
        """Describe one hazard with the movement vector needed to leave it."""

        dx = self.player.x - zone.x
        dy = self.player.y - zone.y
        if zone.kind == "circle":
            distance = math.hypot(dx, dy)
            if distance <= 1e-8:
                escape_x, escape_y = 1.0, 0.0
            else:
                escape_x, escape_y = dx / distance, dy / distance
            safety = max(0.0, zone.radius + self.player.radius - distance)
            distance_to_safety = float(np.clip(safety / max(1.0, zone.radius), 0.0, 1.0))
        else:
            signed = -math.sin(zone.angle) * dx + math.cos(zone.angle) * dy
            direction = 1.0 if signed >= 0.0 else -1.0
            escape_x = -math.sin(zone.angle) * direction
            escape_y = math.cos(zone.angle) * direction
            safety = max(0.0, zone.half_width + self.player.radius - abs(signed))
            distance_to_safety = float(
                np.clip(safety / max(1.0, zone.half_width), 0.0, 1.0)
            )
        time_to_impact = (
            float(np.clip(zone.telegraph_steps / zone.maximum_telegraph_steps, 0.0, 1.0))
            if zone.telegraph_steps > 0
            else 0.0
        )
        return (
            float(escape_x),
            float(escape_y),
            distance_to_safety,
            time_to_impact,
            float(zone.telegraph_steps == 0),
            float(zone.kind == "circle"),
        )

    def _ordered_danger_zones(self) -> list[DangerZone]:
        return sorted(
            self.danger_zones,
            key=lambda item: (
                0 if item.telegraph_steps == 0 else item.telegraph_steps,
                item.attack_id,
                item.kind,
            ),
        )

    def _danger_zone_features(self) -> tuple[float, float, float, float, float, float]:
        """Describe the most urgent individual boss hazard."""

        zones = self._ordered_danger_zones()
        if not zones:
            return 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
        return self._single_danger_zone_features(zones[0])

    def _multi_danger_zone_features(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """Expose hazard count, a combined escape direction, and a second lane."""

        zones = self._ordered_danger_zones()
        if not zones:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
        combined_x = 0.0
        combined_y = 0.0
        for zone in zones:
            escape_x, escape_y, risk, time_left, _, _ = self._single_danger_zone_features(zone)
            weight = risk * (2.0 - time_left)
            combined_x += escape_x * weight
            combined_y += escape_y * weight
        magnitude = math.hypot(combined_x, combined_y)
        if magnitude > 1e-8:
            combined_x /= magnitude
            combined_y /= magnitude
        secondary = (
            self._single_danger_zone_features(zones[1])
            if len(zones) > 1
            else (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        )
        return (
            float(np.clip(len(zones) / 4.0, 0.0, 1.0)),
            float(combined_x),
            float(combined_y),
            *secondary,
        )

    def _danger_zone_risk(self) -> float:
        """Return urgency-weighted overlap risk for potential-based shaping."""

        if not self.danger_zones:
            return 0.0
        risks = []
        for zone in self.danger_zones:
            _, _, distance_to_safety, time_left, _, _ = self._single_danger_zone_features(zone)
            risks.append(distance_to_safety * (2.0 - time_left))
        # Parallel lanes must not dilute the danger of the lane covering the
        # player. The most urgent overlap is the decision-relevant risk.
        return float(max(risks))

    def _update_phase(self, events: dict[str, Any]) -> None:
        if self.spawners:
            return

        if self.phase_transition_steps > 0:
            self.phase_transition_steps -= 1
            if self.phase_transition_steps == 0:
                self._spawn_phase_spawners(events)
            return

        # A new phase is a clean combat encounter. Remaining hostiles retreat
        # and in-flight shots are discarded; neither grants kills, XP, or reward.
        cleared_boss_phase = self.is_boss_phase
        enemies_dispersed = len(self.enemies)
        projectiles_cleared = len(self.projectiles)
        hazards_cleared = len(self.danger_zones)
        self.enemies.clear()
        self.projectiles.clear()
        self.danger_zones.clear()
        self.last_phase_cleanup_count = enemies_dispersed
        events["enemies_dispersed"] = enemies_dispersed
        events["projectiles_cleared"] = projectiles_cleared
        events["hazards_cleared"] = hazards_cleared
        events["boss_phase_cleared"] = cleared_boss_phase

        self.phase += 1
        self.phase_step_count = 0
        self.phase_max_steps = self._phase_step_budget()
        events["phase_advanced"] = True
        self.phase_transition_steps = max(
            0, int(float(self.phase_cfg["transition_seconds"]) * self.fps)
        )
        if self.phase_transition_steps == 0:
            self._spawn_phase_spawners(events)

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------

    def _spawn_phase_spawners(self, events: dict[str, Any] | None = None) -> None:
        boss_phase = self.is_boss_phase
        growth_interval = max(
            1, int(self.phase_cfg["spawner_growth_interval_phases"])
        )
        count = 1 if boss_phase else min(
            int(self.phase_cfg["maximum_spawners"]),
            int(self.phase_cfg["initial_spawners"])
            + ((self.phase - 1) // growth_interval)
            * int(self.phase_cfg["spawners_added_per_phase"]),
        )
        positions = self._choose_spawner_positions(count)
        late_phase = max(
            0, self.phase - int(self.phase_cfg["late_health_start_phase"])
        )
        health_scale = (
            1.0
            + (self.phase - 1) * float(self.phase_cfg["spawner_health_growth"])
            + float(self.phase_cfg["spawner_late_health_growth"])
            * late_phase**1.25
        )

        for x, y in positions:
            max_health = float(self.spawner_cfg["max_health"]) * health_scale
            if boss_phase:
                health_key = (
                    "first_boss_health_multiplier"
                    if self.boss_encounter_number == 1
                    else "boss_health_multiplier"
                )
                max_health *= float(self.phase_cfg[health_key])
            boss_number = max(0, self.boss_encounter_number - 1)
            shield_base = (
                float(self.phase_cfg["first_boss_shield_fraction"])
                if self.boss_encounter_number == 1
                else float(self.phase_cfg["boss_shield_fraction"])
            )
            shield_fraction = min(
                float(self.phase_cfg["boss_shield_max_fraction"]),
                shield_base
                + boss_number
                * float(self.phase_cfg["boss_shield_growth_per_encounter"]),
            )
            initial_delay = int(
                self.np_random.uniform(0.35, 1.0)
                * self._spawn_interval_steps(None, is_boss=boss_phase)
            )
            self.spawners.append(
                Spawner(
                    x=x,
                    y=y,
                    radius=(48.0 if boss_phase else float(self.spawner_cfg["radius"])),
                    entity_id=self._new_id(),
                    max_health=max_health,
                    health=max_health,
                    spawn_cooldown_steps=max(1, initial_delay),
                    is_boss=boss_phase,
                    max_shield=(max_health * shield_fraction if boss_phase else 0.0),
                    shield=(max_health * shield_fraction if boss_phase else 0.0),
                )
            )
        if boss_phase:
            self.boss_skill_index = int(self.np_random.integers(0, 5))
            self.boss_skill_cooldown_steps = max(
                1,
                int(float(self.phase_cfg["boss_skill_initial_delay_seconds"]) * self.fps),
            )
        elif self.phase >= int(self.phase_cfg["miniboss_start_phase"]):
            pity = self.phases_since_miniboss >= int(self.phase_cfg["miniboss_pity_phases"])
            if pity or float(self.np_random.random()) < self.miniboss_chance_for_phase():
                self._spawn_miniboss(self.spawners[0])
                self.phases_since_miniboss = 0
                if events is not None:
                    events["minibosses_spawned"] += 1
            else:
                self.phases_since_miniboss += 1
        if self.nova_bomb_armed and events is not None:
            self._detonate_nova_bomb(events)

    def _apply_player_damage(self, amount: float, events: dict[str, Any]) -> float:
        """Apply upgrades and consumable barriers to one incoming hit."""

        if self.barrier_charges > 0:
            self.barrier_charges -= 1
            events["barrier_blocks"] += 1
            return 0.0
        resistance = min(0.50, 0.10 * self.upgrade_stacks.get("shield", 0))
        damage = max(0.0, float(amount) * (1.0 - resistance))
        self.player.health -= damage
        events["damage_taken"] += damage
        events["player_hit"] = damage > 0.0
        return damage

    def _grant_miniboss_caches(self, events: dict[str, Any]) -> None:
        """Grant immediate, visible survivability loot for Rift Hunter kills."""

        count = int(events.get("minibosses_destroyed", 0)) - int(
            events.get("miniboss_caches", 0)
        )
        if count <= 0:
            return
        self.barrier_charges += count
        self.player.health = min(
            self.player.max_health,
            self.player.health + self.player.max_health * 0.15 * count,
        )
        events["miniboss_caches"] += count
        self.last_upgrade_name = "Rift Hunter Cache: repair + Aegis"
        self.upgrade_banner_steps = max(
            self.upgrade_banner_steps,
            int(float(self.progression_cfg["upgrade_banner_seconds"]) * self.fps),
        )

    def _update_passive_systems(self) -> None:
        regen_stacks = self.upgrade_stacks.get("regen", 0)
        if regen_stacks > 0 and self.player.health > 0.0:
            regeneration = (
                float(self.progression_cfg["passive_regen_per_second"])
                * regen_stacks
                * self.dt
            )
            self.player.health = min(self.player.max_health, self.player.health + regeneration)

    def _detonate_nova_bomb(self, events: dict[str, Any]) -> None:
        destroyed_enemy_ids: set[int] = set()
        destroyed_spawner_ids: set[int] = set()
        miniboss_ids = {enemy.entity_id for enemy in self.enemies if enemy.is_miniboss}
        blast_damage = 70.0 + 8.0 * max(0, self.player.level - 1)
        for target in [*self.enemies, *self.spawners]:
            self._damage_target(
                target,
                blast_damage,
                events,
                destroyed_enemy_ids,
                destroyed_spawner_ids,
                source="bomb",
                count_hit=False,
            )
        if destroyed_enemy_ids:
            self.enemies = [item for item in self.enemies if item.entity_id not in destroyed_enemy_ids]
            events["enemies_destroyed"] += len(destroyed_enemy_ids)
            events["minibosses_destroyed"] += len(
                destroyed_enemy_ids.intersection(miniboss_ids)
            )
            self._grant_miniboss_caches(events)
        if destroyed_spawner_ids:
            self.episode_stats["bosses_destroyed"] = int(
                self.episode_stats["bosses_destroyed"]
            ) + sum(
                int(item.is_boss)
                for item in self.spawners
                if item.entity_id in destroyed_spawner_ids
            )
            self.spawners = [item for item in self.spawners if item.entity_id not in destroyed_spawner_ids]
            events["spawners_destroyed"] += len(destroyed_spawner_ids)
        events["nova_bomb_detonated"] = True
        self.nova_bomb_armed = False

    def _choose_spawner_positions(self, count: int) -> list[tuple[float, float]]:
        margin = float(self.spawner_cfg["edge_margin"])
        top = max(
            margin,
            self.playfield_top + float(self.spawner_cfg["radius"]) + 18.0,
        )
        bottom = self.height - margin
        candidates = [
            (margin, top),
            (self.width - margin, top),
            (margin, self.height - margin),
            (self.width - margin, self.height - margin),
            (self.width / 2.0, top),
            (self.width / 2.0, bottom),
            (margin, (top + bottom) / 2.0),
            (self.width - margin, (top + bottom) / 2.0),
        ]
        order = self.np_random.permutation(len(candidates))
        return [candidates[int(index)] for index in order[:count]]

    def _spawn_enemy(self, spawner: Spawner, *, is_miniboss: bool = False) -> None:
        angle = float(self.np_random.uniform(0.0, math.tau))
        distance = spawner.radius + float(self.enemy_cfg["radius"]) + 5.0
        enemy_radius = float(self.enemy_cfg["radius"]) * (1.55 if is_miniboss else 1.0)
        x = float(
            np.clip(
                spawner.x + math.cos(angle) * distance,
                enemy_radius,
                self.width - enemy_radius,
            )
        )
        y = float(
            np.clip(
                spawner.y + math.sin(angle) * distance,
                self.playfield_top + enemy_radius,
                self.height - enemy_radius,
            )
        )
        late_phase = max(
            0, self.phase - int(self.phase_cfg["late_health_start_phase"])
        )
        health_scale = (
            1.0
            + (self.phase - 1) * float(self.phase_cfg["enemy_health_growth"])
            + float(self.phase_cfg["enemy_late_health_growth"])
            * late_phase**1.25
        )
        speed_scale = min(
            float(self.phase_cfg["enemy_speed_max_multiplier"]),
            1.0 + (self.phase - 1) * float(self.phase_cfg["enemy_speed_growth"]),
        )
        if spawner.is_boss:
            health_key = (
                "first_boss_enemy_health_multiplier"
                if self.boss_encounter_number == 1
                else "boss_enemy_health_multiplier"
            )
            speed_key = (
                "first_boss_enemy_speed_multiplier"
                if self.boss_encounter_number == 1
                else "boss_enemy_speed_multiplier"
            )
            health_scale *= float(self.phase_cfg[health_key])
            speed_scale *= float(self.phase_cfg[speed_key])
        if is_miniboss:
            health_scale *= float(self.phase_cfg["miniboss_health_multiplier"])
            speed_scale *= float(self.phase_cfg["miniboss_speed_multiplier"])
        speed_scale = min(
            speed_scale, float(self.phase_cfg["enemy_speed_max_multiplier"])
        )
        max_health = float(self.enemy_cfg["max_health"]) * health_scale
        self.enemies.append(
            Enemy(
                x=x,
                y=y,
                radius=enemy_radius,
                entity_id=self._new_id(),
                max_health=max_health,
                health=max_health,
                speed=float(self.enemy_cfg["speed"]) * speed_scale,
                is_elite=spawner.is_boss,
                is_miniboss=is_miniboss,
            )
        )

    def _spawn_miniboss(self, spawner: Spawner) -> None:
        self._spawn_enemy(spawner, is_miniboss=True)

    def _spawn_interval_steps(
        self, spawner: Spawner | None = None, *, is_boss: bool = False
    ) -> int:
        base_seconds = float(self.spawner_cfg["spawn_interval_seconds"])
        speedup = 1.0 + (self.phase - 1) * float(self.phase_cfg["spawn_rate_growth"])
        if is_boss or (spawner is not None and spawner.is_boss):
            rate_key = (
                "first_boss_spawn_rate_multiplier"
                if self.boss_encounter_number == 1
                else "boss_spawn_rate_multiplier"
            )
            base_seconds *= float(self.phase_cfg[rate_key])
        seconds = max(
            float(self.phase_cfg["minimum_spawn_interval_seconds"]),
            base_seconds / speedup,
        )
        return max(1, int(seconds * self.fps))

    def _new_id(self) -> int:
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        return entity_id

    # ------------------------------------------------------------------
    # Observations, rewards, and diagnostics
    # ------------------------------------------------------------------

    def _nearest_target(
        self, max_distance: float | None = None
    ) -> Enemy | Spawner | None:
        targets = self._priority_targets()
        if max_distance is not None:
            max_distance_squared = max_distance * max_distance
            targets = [
                target
                for target in targets
                if (target.x - self.player.x) ** 2
                + (target.y - self.player.y) ** 2
                <= max_distance_squared
            ]
        if not targets:
            return None
        return min(
            targets,
            key=lambda target: (target.x - self.player.x) ** 2
            + (target.y - self.player.y) ** 2,
        )

    def _priority_targets(self) -> list[Enemy | Spawner]:
        """Expose the current damageable objective to aim-assist and agents.

        During the finite boss intermission, sentries are the only valid
        progression targets. This keeps direct aim assist, drones, homing beams,
        the reticle, and observation targeting consistent with boss immunity.
        """

        defenders = self.boss_defenders
        if defenders:
            return list(defenders)
        return [*self.enemies, *self.spawners]

    def _target_features(
        self, targets: list[Enemy] | list[Spawner]
    ) -> tuple[float, float, float, float]:
        """Encode unit direction, normalized distance, and health of the nearest target.

        A missing target uses direction ``(0, 0)``, maximum distance ``1``,
        and zero health. Screen-space positive Y points downward.
        """

        if not targets:
            return 0.0, 0.0, 1.0, 0.0
        target = min(
            targets,
            key=lambda item: (item.x - self.player.x) ** 2
            + (item.y - self.player.y) ** 2,
        )
        dx = target.x - self.player.x
        dy = target.y - self.player.y
        pixel_distance = math.hypot(dx, dy)
        if pixel_distance > 1e-8:
            direction_x = dx / pixel_distance
            direction_y = dy / pixel_distance
        else:
            direction_x = 0.0
            direction_y = 0.0

        playfield_height = self.height - self.playfield_top
        diagonal = math.hypot(self.width, playfield_height)
        distance = float(np.clip(pixel_distance / diagonal, 0.0, 1.0))
        health = float(np.clip(target.health / target.max_health, 0.0, 1.0))
        return direction_x, direction_y, distance, health

    def _aim_alignment(self, targets: list[Enemy] | list[Spawner]) -> float:
        """Return heading/target cosine alignment, or zero when no target exists."""

        if not targets:
            return 0.0
        target = min(
            targets,
            key=lambda item: (item.x - self.player.x) ** 2
            + (item.y - self.player.y) ** 2,
        )
        dx = target.x - self.player.x
        dy = target.y - self.player.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-8:
            return 1.0
        return float(
            np.clip(
                math.cos(self.player.angle) * dx / distance
                + math.sin(self.player.angle) * dy / distance,
                -1.0,
                1.0,
            )
        )

    def _aim_turn_direction(self, targets: list[Enemy] | list[Spawner]) -> float:
        """Return signed heading-to-target cross product for left/right steering."""

        if not targets:
            return 0.0
        target = min(
            targets,
            key=lambda item: (item.x - self.player.x) ** 2
            + (item.y - self.player.y) ** 2,
        )
        dx = target.x - self.player.x
        dy = target.y - self.player.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-8:
            return 0.0
        return float(
            np.clip(
                math.cos(self.player.angle) * dy / distance
                - math.sin(self.player.angle) * dx / distance,
                -1.0,
                1.0,
            )
        )

    def _active_target_features(self) -> tuple[float, float, float, float, float, float]:
        """Encode the same nearest target highlighted by the targeting reticle."""

        target = self._nearest_target()
        if target is None:
            return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        direction_x, direction_y, distance, _ = self._target_features([target])
        is_spawner = float(isinstance(target, Spawner))
        return (
            direction_x,
            direction_y,
            distance,
            is_spawner,
            self._aim_alignment([target]),
            self._aim_turn_direction([target]),
        )

    def _crowd_metrics(self) -> tuple[float, float, float, float]:
        """Smooth local pressure and an escape vector, not an action override."""
        pressure = escape_x = escape_y = 0.0
        count = 0
        for enemy in self.enemies:
            dx, dy = enemy.x - self.player.x, enemy.y - self.player.y
            distance = max(1e-6, math.hypot(dx, dy))
            clearance = max(0.0, distance - enemy.radius - self.player.radius)
            if clearance < 200.0:
                count += 1
                weight = (1.0 - clearance / 200.0) ** 2
                pressure += weight
                escape_x -= dx / distance * weight
                escape_y -= dy / distance * weight
        length = max(1e-6, math.hypot(escape_x, escape_y))
        return min(1.0, count / 8.0), min(1.0, pressure / 3.0), escape_x / length, escape_y / length

    def _threat_observation(self) -> list[float]:
        enemies = sorted(self.enemies, key=lambda e: (e.x-self.player.x)**2 + (e.y-self.player.y)**2)
        def closing(enemy: Enemy) -> float:
            dx, dy = enemy.x-self.player.x, enemy.y-self.player.y
            distance = max(1e-6, math.hypot(dx, dy))
            return float(np.clip(((self.player.vx-enemy.vx)*dx + (self.player.vy-enemy.vy)*dy) / distance / 560.0, -1, 1))
        second = enemies[1] if len(enemies) > 1 else None
        second_features = self._target_features([second] if second else [])[:3]
        radius = self.player.radius
        walls = [self.player.x-radius, self.width-radius-self.player.x,
                 self.player.y-self.playfield_top-radius, self.height-radius-self.player.y]
        spawner_clearance = min((math.hypot(s.x-self.player.x, s.y-self.player.y)-s.radius-radius for s in self.spawners), default=300.0)
        boss = next((s for s in self.spawners if s.is_boss), None)
        hazard = self._multi_danger_zone_features()
        hx, hy = hazard[1], hazard[2]
        cos, sin = math.cos(self.player.angle), math.sin(self.player.angle)
        return [*second_features, closing(second) if second else 0.0,
                closing(enemies[0]) if enemies else 0.0, *self._crowd_metrics(),
                *[float(np.clip(w / 160.0, 0, 1)) for w in walls],
                float(np.clip(spawner_clearance / 300.0, 0, 1)),
                boss.shield / max(1.0, boss.max_shield) if boss else 0.0,
                (1.0-boss.summons_used/max(1, self.boss_summon_limit_for_phase())) if boss else 0.0,
                hx*cos+hy*sin, hy*cos-hx*sin,
                boss.summon_cooldown_steps / max(1.0,self.fps*float(self.phase_cfg['boss_summon_interval_seconds'])) if boss else 0.0]

    def _nearest_enemy_missile(self) -> Projectile | None:
        missiles = [
            projectile
            for projectile in self.projectiles
            if projectile.owner == "enemy" and projectile.weapon_kind == "enemy_missile"
        ]
        return min(
            missiles,
            key=lambda item: (item.x - self.player.x) ** 2
            + (item.y - self.player.y) ** 2,
            default=None,
        )

    def _missile_risk(self, missile: Projectile | None = None) -> float:
        candidate = missile or self._nearest_enemy_missile()
        if candidate is None:
            return 0.0
        speed = max(1.0, math.hypot(candidate.vx, candidate.vy))
        ux, uy = candidate.vx / speed, candidate.vy / speed
        to_player_x = self.player.x - candidate.x
        to_player_y = self.player.y - candidate.y
        along = to_player_x * ux + to_player_y * uy
        lateral = abs(ux * to_player_y - uy * to_player_x)
        telegraph = candidate.telegraph_steps / max(1.0, self.fps)
        if along <= 0.0 and candidate.telegraph_steps <= 0:
            return 0.0
        time_to_cross = max(0.0, along) / speed + telegraph
        collision_lane = candidate.radius + self.player.radius + 12.0
        lateral_risk = np.clip(
            1.0 - max(0.0, lateral - collision_lane) / 105.0,
            0.0,
            1.0,
        )
        time_risk = np.clip(1.0 - time_to_cross / 2.6, 0.0, 1.0)
        return float(lateral_risk * time_risk)

    def _boss_intermission_observation(self) -> list[float]:
        """Expose boss vulnerability, sentry priority, and incoming missiles."""

        boss = next((spawner for spawner in self.spawners if spawner.is_boss), None)
        defenders = self.boss_defenders
        defender_features = self._target_features(defenders)
        missile = self._nearest_enemy_missile()
        if missile is None:
            missile_features = (0.0, 0.0, 1.0, 1.0)
            missile_motion = (0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            dx = missile.x - self.player.x
            dy = missile.y - self.player.y
            distance = max(1e-6, math.hypot(dx, dy))
            diagonal = math.hypot(self.width, self.height - self.playfield_top)
            speed = max(1.0, math.hypot(missile.vx, missile.vy))
            velocity_x, velocity_y = missile.vx / speed, missile.vy / speed
            to_player_x, to_player_y = -dx, -dy
            along = to_player_x * velocity_x + to_player_y * velocity_y
            telegraph = missile.telegraph_steps / max(1.0, self.fps)
            time_to_cross = max(0.0, along) / speed + telegraph
            left_x, left_y = -velocity_y, velocity_x
            cross = velocity_x * to_player_y - velocity_y * to_player_x
            if abs(cross) < 1e-6:
                lateral_velocity = self.player.vx * left_x + self.player.vy * left_y
                side = (
                    1.0
                    if lateral_velocity > 1e-6
                    or (abs(lateral_velocity) <= 1e-6 and missile.entity_id % 2 == 0)
                    else -1.0
                )
            else:
                side = 1.0 if cross > 0.0 else -1.0
            telegraph_total = max(
                1.0,
                float(self.phase_cfg["boss_defender_missile_telegraph_seconds"])
                * self.fps,
            )
            missile_features = (
                dx / distance,
                dy / distance,
                float(np.clip(distance / diagonal, 0.0, 1.0)),
                float(np.clip(time_to_cross / 3.0, 0.0, 1.0)),
            )
            missile_motion = (
                velocity_x,
                velocity_y,
                left_x * side,
                left_y * side,
                float(np.clip(missile.telegraph_steps / telegraph_total, 0.0, 1.0)),
            )
        move_scale = max(1.0, float(self.phase_cfg["boss_move_speed"]) * 1.35)
        return [
            boss.health / max(1.0, boss.max_health) if boss else 0.0,
            float(self.boss_is_vulnerable(boss)) if boss else 0.0,
            np.clip(
                len(defenders)
                / max(1, int(self.phase_cfg["boss_defender_max_count"])),
                0.0,
                1.0,
            ),
            *defender_features,
            *missile_features,
            np.clip(boss.vx / move_scale, -1.0, 1.0) if boss else 0.0,
            np.clip(boss.vy / move_scale, -1.0, 1.0) if boss else 0.0,
            *missile_motion,
        ]

    def _shaping_snapshot(self) -> dict[str, Any]:
        """Capture potential features used for small, explainable shaping terms."""

        missile = self._nearest_enemy_missile()
        spawner = min(
            self.spawners,
            key=lambda item: (item.x - self.player.x) ** 2
            + (item.y - self.player.y) ** 2,
            default=None,
        )
        target = self._nearest_target()
        diagonal = math.hypot(self.width, self.height - self.playfield_top)
        return {
            "spawner_id": None if spawner is None else spawner.entity_id,
            "spawner_distance": (
                0.0
                if spawner is None
                else abs(
                    max(
                        0.0,
                        math.hypot(
                            spawner.x - self.player.x,
                            spawner.y - self.player.y,
                        )
                        - spawner.radius
                        - self.player.radius,
                    )
                    - float(self.reward_cfg["spawner_standoff"])
                )
                / diagonal
            ),
            "target_id": None if target is None else target.entity_id,
            "target_alignment": (
                0.0
                if target is None
                else self._aim_alignment([target])
            ),
            "hazard_attack_ids": tuple(
                sorted({zone.attack_id for zone in self.danger_zones})
            ),
            "hazard_risk": self._danger_zone_risk(),
            "missile_id": None if missile is None else missile.entity_id,
            "missile_risk": self._missile_risk(missile),
            "enemy_pressure": {
                enemy.entity_id: (
                    max(
                        0.0,
                        1.0
                        - max(
                            0.0,
                            math.hypot(
                                enemy.x - self.player.x,
                                enemy.y - self.player.y,
                            )
                            - enemy.radius
                            - self.player.radius,
                        )
                        / 200.0,
                    )
                    ** 2
                )
                for enemy in self.enemies
            },
            "crowd_pressure": self._crowd_metrics()[1],
        }

    def _apply_shaping_delta(
        self,
        events: dict[str, Any],
        before: dict[str, Any],
    ) -> None:
        """Measure progress only while the same target remains active.

        Restricting the delta to a stable entity avoids artificial reward jumps
        when a target is destroyed or a new phase appears.
        """

        after = self._shaping_snapshot()
        events["crowd_pressure"] = float(after["crowd_pressure"])
        events["crowd_escape"] = 0.0
        # Compare only threats present on both sides of the transition. This
        # continues teaching separation when another enemy spawns, but never
        # pays the agent merely because an enemy was destroyed.
        common_enemy_ids = set(before["enemy_pressure"]).intersection(
            after["enemy_pressure"]
        )
        if common_enemy_ids:
            events["crowd_escape"] = (
                sum(float(before["enemy_pressure"][key]) for key in common_enemy_ids)
                - sum(float(after["enemy_pressure"][key]) for key in common_enemy_ids)
            ) / 3.0
        if before["spawner_id"] is not None and before["spawner_id"] == after["spawner_id"]:
            events["spawner_progress"] = float(before["spawner_distance"]) - float(
                after["spawner_distance"]
            )
        if before["target_id"] is not None and before["target_id"] == after["target_id"]:
            events["aim_improvement"] = float(after["target_alignment"]) - float(
                before["target_alignment"]
            )
        if (
            before["hazard_attack_ids"]
            and before["hazard_attack_ids"] == after["hazard_attack_ids"]
        ):
            events["hazard_escape_improvement"] = float(before["hazard_risk"]) - float(
                after["hazard_risk"]
            )
        if before["missile_id"] is not None and before["missile_id"] == after["missile_id"]:
            events["missile_escape_improvement"] = float(
                before["missile_risk"]
            ) - float(after["missile_risk"])

    def _get_observation(self) -> np.ndarray:
        engine_multiplier = 1.0 + 0.075 * self.upgrade_stacks.get("engine", 0)
        max_speed = max(
            float(self.player_cfg["max_speed"]), float(self.player_cfg["direct_speed"])
        ) * engine_multiplier
        enemy_features = self._target_features(self.enemies)
        spawner_features = self._target_features(self.spawners)
        active_target_features = self._active_target_features()
        fire_cooldown = self.weapon_cooldown_steps()
        profile = self.weapon_profile()
        max_enemies = self.maximum_active_enemies()
        minibosses = [enemy for enemy in self.enemies if enemy.is_miniboss]
        miniboss_health = self._target_features(minibosses)[3]
        hazard_features = self._danger_zone_features()
        multi_hazard_features = self._multi_danger_zone_features()
        mastery_total = sum(
            self.upgrade_stacks.get(key, 0)
            for key in ("weapon_mastery", "hull_mastery", "drone_mastery")
        )

        observation = np.array(
            [
                self.player.x / self.width * 2.0 - 1.0,
                (self.player.y - self.playfield_top)
                / (self.height - self.playfield_top)
                * 2.0
                - 1.0,
                np.clip(self.player.vx / max_speed, -1.0, 1.0),
                np.clip(self.player.vy / max_speed, -1.0, 1.0),
                math.cos(self.player.angle),
                math.sin(self.player.angle),
                np.clip(self.player.health / self.player.max_health, 0.0, 1.0),
                1.0 - np.clip(self.player.fire_cooldown_steps / fire_cooldown, 0.0, 1.0),
                *enemy_features,
                *spawner_features,
                np.clip(len(self.enemies) / max(1, max_enemies), 0.0, 1.0),
                np.clip(len(self.spawners) / max(1, int(self.phase_cfg["maximum_spawners"])), 0.0, 1.0),
                np.clip(self.phase / max(1, int(self.phase_cfg["observation_phase_cap"])), 0.0, 1.0),
                np.clip(
                    1.0 - self.phase_step_count / self.phase_max_steps,
                    0.0,
                    1.0,
                ),
                self._aim_alignment(self.enemies),
                self._aim_alignment(self.spawners),
                *active_target_features,
                np.clip(
                    (self.player.level - 1) / max(1, self.maximum_player_level - 1),
                    0.0,
                    1.0,
                ),
                self.xp_progress(),
                int(profile["shot_count"]) / 7.0,
                self.upgrade_stacks.get("fire_rate", 0) / 8.0,
                np.clip(float(profile["damage_multiplier"]) / 5.0, 0.0, 1.0),
                float(str(profile["kind"]) in ("laser", "nova")),
                np.clip(
                    (self.player.max_health - float(self.player_cfg["max_health"])) / 300.0,
                    0.0,
                    1.0,
                ),
                self.upgrade_stacks.get("shield", 0) / 5.0,
                self.upgrade_stacks.get("range", 0) / 6.0,
                self.upgrade_stacks.get("piercing", 0) / 4.0,
                self.upgrade_stacks.get("splash", 0) / 5.0,
                self.upgrade_stacks.get("engine", 0) / 6.0,
                float(self.support_drone_active),
                float(self.nova_bomb_armed),
                float(self.is_boss_phase),
                float(bool(minibosses)),
                miniboss_health,
                *hazard_features,
                np.clip(self.drone_level / 16.0, 0.0, 1.0),
                self.drone_count / 4.0,
                np.clip(self.barrier_charges / 8.0, 0.0, 1.0),
                float(self.overdrive_active),
                self.upgrade_stacks.get("regen", 0) / 5.0,
                self.upgrade_stacks.get("homing", 0) / 3.0,
                np.clip(mastery_total / 20.0, 0.0, 1.0),
                *multi_hazard_features,
                float(profile["critical_chance"]) / 0.40,
                self.upgrade_stacks.get("leech", 0) / 4.0,
                self.upgrade_stacks.get("riftbreaker", 0) / 6.0,
                *self._threat_observation(),
                *self._boss_intermission_observation(),
            ],
            dtype=np.float32,
        )
        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32, copy=False)

    def observation_as_dict(
        self, observation: np.ndarray | None = None
    ) -> dict[str, float]:
        """Return a named view of an observation for reports and debugging."""

        vector = self._get_observation() if observation is None else np.asarray(observation)
        if vector.shape != self.observation_space.shape:
            raise ValueError(
                f"Expected observation shape {self.observation_space.shape}, got {vector.shape}"
            )
        return {
            name: float(value)
            for name, value in zip(self.observation_names, vector, strict=True)
        }

    def _calculate_reward(
        self, events: dict[str, Any], terminated: bool
    ) -> tuple[float, dict[str, float]]:
        """Return the total and an auditable breakdown of every reward term."""

        breakdown = {
            "survival": float(self.reward_cfg["survival"]),
            "enemy_damage": float(events["damage_dealt_enemy"])
            * float(self.reward_cfg["enemy_damage_per_hp"]),
            "spawner_damage": float(events["damage_dealt_spawner"])
            * float(self.reward_cfg["spawner_damage_per_hp"]),
            "enemy_destroyed": int(events["enemies_destroyed"])
            * float(self.reward_cfg["enemy_destroyed"]),
            "spawner_destroyed": int(events["spawners_destroyed"])
            * float(self.reward_cfg["spawner_destroyed"]),
            "phase_advanced": (
                float(self.reward_cfg["phase_advanced"])
                if events["phase_advanced"]
                else 0.0
            ),
            "miniboss_destroyed": int(events.get("minibosses_destroyed", 0))
            * float(self.reward_cfg["miniboss_destroyed"]),
            "boss_phase_cleared": (
                float(self.reward_cfg["boss_phase_cleared"])
                if events.get("boss_phase_cleared", False)
                else 0.0
            ),
            "boss_skill_dodged": int(events.get("boss_skills_dodged", 0))
            * float(self.reward_cfg["boss_skill_dodged"]),
            "boss_skill_hit": int(events.get("boss_skill_hits", 0))
            * float(self.reward_cfg["boss_skill_hit"]),
            "boss_defender_destroyed": int(
                events.get("boss_defenders_destroyed", 0)
            )
            * float(self.reward_cfg["boss_defender_destroyed"]),
            "boss_immune_hit": int(events.get("boss_immune_hits", 0))
            * float(self.reward_cfg["boss_immune_hit"]),
            "missile_evaded": int(events.get("missiles_evaded", 0))
            * float(self.reward_cfg["missile_evaded"]),
            "missile_hit": int(events.get("missile_hits", 0))
            * float(self.reward_cfg["missile_hit"]),
            "missile_escape": float(events.get("missile_escape_improvement", 0.0))
            * float(self.reward_cfg["missile_escape"]),
            "hazard_escape": float(events.get("hazard_escape_improvement", 0.0))
            * float(self.reward_cfg["hazard_escape"]),
            "hazard_exposure": float(events.get("hazard_exposure", 0.0))
            * float(self.reward_cfg["hazard_exposure"]),
            "crowd_escape": float(events.get("crowd_escape", 0.0)) * float(self.reward_cfg["crowd_escape"]),
            "crowd_contact_risk": max(0.0, float(events.get("crowd_pressure", 0.0)) - 0.5)
            * float(self.reward_cfg["crowd_contact_risk"]),
            "phase_timeout": (
                float(self.reward_cfg["phase_timeout"])
                if events.get("phase_timeout", False)
                else 0.0
            ),
            "damage_taken": float(events["damage_taken"])
            * float(self.reward_cfg["damage_taken_per_hp"]),
            "spawner_approach": float(events["spawner_progress"])
            * float(self.reward_cfg["spawner_approach"]),
            "aim_improvement": float(events["aim_improvement"])
            * float(self.reward_cfg["aim_improvement"]),
            "shot_quality": (
                0.0
                if not events["shot_fired"]
                else (
                    max(0.0, float(events["shot_alignment"])) ** 4
                    * float(self.reward_cfg["well_aimed_shot"])
                    + (1.0 - max(0.0, float(events["shot_alignment"])))
                    * float(self.reward_cfg["poorly_aimed_shot"])
                )
            ),
            "death": float(self.reward_cfg["death"]) if terminated else 0.0,
        }
        return float(sum(breakdown.values())), breakdown

    def _update_episode_stats(self, events: dict[str, Any], reward: float) -> None:
        self.episode_stats["reward"] = float(self.episode_stats["reward"]) + reward
        for key in (
            "enemies_destroyed",
            "spawners_destroyed",
            "damage_dealt_enemy",
            "damage_dealt_spawner",
            "damage_taken",
            "projectile_hits",
            "minibosses_destroyed",
            "boss_skills_cast",
            "boss_skills_dodged",
            "boss_skill_hits",
            "boss_defenders_spawned",
            "boss_defenders_destroyed",
            "boss_immune_hits",
            "boss_health_regenerated",
            "missiles_fired",
            "missiles_evaded",
            "missile_hits",
            "hazard_exposure",
            "miniboss_caches",
            "sustain_healed",
        ):
            self.episode_stats[key] = self.episode_stats[key] + events[key]
        self.episode_stats["shots_fired"] = int(self.episode_stats["shots_fired"]) + int(
            events["projectiles_fired"]
        )
        self.episode_stats["volleys_fired"] = int(
            self.episode_stats["volleys_fired"]
        ) + int(events["shot_fired"])
        self.episode_stats["xp_earned"] = float(
            self.episode_stats["xp_earned"]
        ) + float(events["xp_gained"])
        self.episode_stats["levels_gained"] = int(
            self.episode_stats["levels_gained"]
        ) + int(events["levels_gained"])
        self.episode_stats["phases_advanced"] = int(
            self.episode_stats["phases_advanced"]
        ) + int(events["phase_advanced"])

    def _get_info(self) -> dict[str, Any]:
        return {
            "control_style": self.control_style,
            "phase": self.phase,
            "step": self.step_count,
            "time_seconds": self.step_count * self.dt,
            "phase_step": self.phase_step_count,
            "phase_time_seconds": self.phase_step_count * self.dt,
            "phase_time_remaining": max(
                0.0,
                (self.phase_max_steps - self.phase_step_count) * self.dt,
            ),
            "player_health": self.player.health,
            "player_level": self.player.level,
            "player_xp": self.player.xp,
            "xp_progress": self.xp_progress(),
            "xp_to_next_level": self.xp_to_next_level(),
            "weapon_name": str(self.weapon_profile()["name"]),
            "weapon_kind": str(self.weapon_profile()["kind"]),
            "weapon_profile": dict(self.weapon_profile()),
            "upgrade_stacks": dict(self.upgrade_stacks),
            "pending_choice_kind": self.pending_choice_kind,
            "pending_choices": [dict(item) for item in self.pending_choices],
            "boss_phase": self.is_boss_phase,
            "support_drone_active": self.support_drone_active,
            "drone_level": self.drone_level,
            "drone_count": self.drone_count,
            "barrier_charges": self.barrier_charges,
            "overdrive_active": self.overdrive_active,
            "nova_bomb_armed": self.nova_bomb_armed,
            "active_minibosses": sum(enemy.is_miniboss for enemy in self.enemies),
            "active_boss_hazards": len(self.danger_zones),
            "boss_vulnerable": self.boss_is_vulnerable(),
            "active_boss_defenders": len(self.boss_defenders),
            "active_enemy_missiles": sum(
                projectile.owner == "enemy" for projectile in self.projectiles
            ),
            "active_boss_attack": (
                self.danger_zones[0].attack_name if self.danger_zones else None
            ),
            "active_boss_attack_id": (
                self.danger_zones[0].attack_id if self.danger_zones else None
            ),
            "active_enemies": len(self.enemies),
            "active_spawners": len(self.spawners),
            "active_projectiles": len(self.projectiles),
            "episode_enemies_destroyed": int(
                self.episode_stats["enemies_destroyed"]
            ),
            "episode_spawners_destroyed": int(
                self.episode_stats["spawners_destroyed"]
            ),
            "episode_phases_advanced": int(self.episode_stats["phases_advanced"]),
            "episode_damage_taken": float(self.episode_stats["damage_taken"]),
            "episode_xp_earned": float(self.episode_stats["xp_earned"]),
            "episode_max_level": self.player.level,
            "episode_stats": dict(self.episode_stats),
        }


__all__ = [
    "ArenaEnv",
    "ENVIRONMENT_SCHEMA_VERSION",
    "DIRECT_ACTIONS",
    "ROTATION_ACTIONS",
    "ObservationIndex",
    "OBSERVATION_NAMES",
]
