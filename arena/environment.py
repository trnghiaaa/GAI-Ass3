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

from arena.entities import Enemy, Player, Projectile, Spawner, circles_overlap


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


OBSERVATION_NAMES = tuple(index.name.lower() for index in ObservationIndex)


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
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        control_style: str = "direct",
        render_mode: str | None = None,
        config_override: dict[str, Any] | None = None,
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
        self.metadata["render_fps"] = self.fps

        self.action_names = DIRECT_ACTIONS if control_style == "direct" else ROTATION_ACTIONS
        self.action_space = spaces.Discrete(len(self.action_names))

        # The layout is fixed even when there are no enemies or spawners.
        self.observation_names = OBSERVATION_NAMES
        low = np.array(
            [
                -1, -1, -1, -1, -1, -1, 0, 0,
                -1, -1, 0, 0, -1, -1, 0, 0,
                0, 0, 0, 0, -1, -1,
                -1, -1, 0, 0, -1, -1,
                0, 0, 0, 0, 0, 0,
            ],
            dtype=np.float32,
        )
        high = np.ones(len(self.observation_names), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.player: Player
        self.enemies: list[Enemy] = []
        self.spawners: list[Spawner] = []
        self.projectiles: list[Projectile] = []
        self.phase = 1
        self.step_count = 0
        self.phase_transition_steps = 0
        self.done = False
        self.last_end_reason: str | None = None
        self._next_entity_id = 1
        self._renderer = None
        self.last_events: dict[str, Any] = {}
        self.episode_stats: dict[str, float | int] = {}
        self.upgrade_banner_steps = 0
        self.last_upgrade_name = "Pulse Cannon"

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
        del options

        self.phase = 1
        self.step_count = 0
        self.phase_transition_steps = 0
        self.done = False
        self.last_end_reason = None
        self._next_entity_id = 1
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self.last_events = {}
        self.upgrade_banner_steps = 0
        self.last_upgrade_name = str(self._weapon_tiers()[0]["name"])
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
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.control_style} controls")

        self.step_count += 1
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
            "spawner_progress": 0.0,
            "aim_improvement": 0.0,
            "shot_alignment": 0.0,
            "impacts": [],
            "player_hit": False,
            "xp_gained": 0.0,
            "levels_gained": 0,
            "upgrade_unlocked": None,
        }

        shaping_before = self._shaping_snapshot()
        self._tick_cooldowns()
        events["projectiles_fired"] = self._apply_player_action(int(action))
        events["shot_fired"] = events["projectiles_fired"] > 0
        if events["shot_fired"]:
            events["shot_alignment"] = float(
                self._shaping_snapshot()["target_alignment"]
            )
        self._update_projectiles(events)
        self._update_enemies(events)
        self._update_spawners(events)
        self._update_phase(events)
        self._apply_combat_progression(events)
        self._apply_shaping_delta(events, shaping_before)

        terminated = self.player.health <= 0.0
        truncated = self.step_count >= self.max_steps and not terminated
        if terminated:
            self.player.health = 0.0
            self.last_end_reason = "player_destroyed"
        elif truncated:
            self.last_end_reason = "time_limit"
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

    def _apply_player_action(self, action: int) -> int:
        projectiles_fired = 0

        if self.control_style == "direct":
            speed = float(self.player_cfg["direct_speed"])
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
                projectiles_fired = self._fire_projectile(auto_aim=True)
        else:
            rotation_speed = math.radians(float(self.player_cfg["rotation_speed_degrees"]))
            if action == ROTATION_ACTIONS["ROTATE_LEFT"]:
                self.player.angle -= rotation_speed * self.dt
            elif action == ROTATION_ACTIONS["ROTATE_RIGHT"]:
                self.player.angle += rotation_speed * self.dt
            elif action == ROTATION_ACTIONS["THRUST"]:
                acceleration = float(self.player_cfg["thrust_acceleration"])
                self.player.vx += math.cos(self.player.angle) * acceleration * self.dt
                self.player.vy += math.sin(self.player.angle) * acceleration * self.dt
            elif action == ROTATION_ACTIONS["SHOOT"]:
                projectiles_fired = self._fire_projectile(auto_aim=False)

            drag = float(self.player_cfg["drag"])
            self.player.vx *= drag
            self.player.vy *= drag
            max_speed = float(self.player_cfg["max_speed"])
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
            target = self._nearest_target()
            if target is not None:
                angle = math.atan2(target.y - self.player.y, target.x - self.player.x)
                self.player.angle = angle

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
        radius = float(self.projectile_cfg["radius"])
        if kind == "laser":
            radius = max(2.0, radius - 1.0)
        offset = self.player.radius + radius + 2.0

        for angle_offset in offsets:
            projectile_angle = angle + angle_offset
            self.projectiles.append(
                Projectile(
                    x=self.player.x + math.cos(projectile_angle) * offset,
                    y=self.player.y + math.sin(projectile_angle) * offset,
                    radius=radius,
                    entity_id=self._new_id(),
                    vx=math.cos(projectile_angle) * speed,
                    vy=math.sin(projectile_angle) * speed,
                    damage=damage,
                    lifetime_steps=int(
                        float(self.projectile_cfg["lifetime_seconds"]) * self.fps
                    ),
                    weapon_kind=kind,
                )
            )

        self.player.fire_cooldown_steps = self.weapon_cooldown_steps()
        return shot_count

    def _weapon_tiers(self) -> list[dict[str, Any]]:
        tiers = list(self.progression_cfg["weapon_tiers"])
        thresholds = list(self.progression_cfg["level_thresholds"])
        if not tiers or len(tiers) != len(thresholds):
            raise ValueError(
                "progression.weapon_tiers and level_thresholds must have equal non-zero lengths"
            )
        return tiers

    @property
    def maximum_player_level(self) -> int:
        """Highest automatic combat-progression level available this episode."""

        return len(self._weapon_tiers())

    def weapon_profile(self) -> dict[str, Any]:
        """Return the active immutable-by-convention weapon-tier configuration."""

        level = int(np.clip(self.player.level, 1, self.maximum_player_level))
        return dict(self._weapon_tiers()[level - 1])

    def weapon_cooldown_steps(self) -> int:
        """Return the active weapon's cooldown in simulation frames."""

        multiplier = float(self.weapon_profile()["cooldown_multiplier"])
        return max(
            1,
            int(float(self.player_cfg["fire_cooldown_seconds"]) * multiplier * self.fps),
        )

    def xp_progress(self) -> float:
        """Return normalized progress from the current level to the next one."""

        thresholds = [float(value) for value in self.progression_cfg["level_thresholds"]]
        level_index = int(np.clip(self.player.level - 1, 0, len(thresholds) - 1))
        if level_index >= len(thresholds) - 1:
            return 1.0
        current = thresholds[level_index]
        following = thresholds[level_index + 1]
        return float(np.clip((self.player.xp - current) / (following - current), 0.0, 1.0))

    def xp_to_next_level(self) -> float:
        """Return remaining XP, or zero once the final tier is unlocked."""

        thresholds = [float(value) for value in self.progression_cfg["level_thresholds"]]
        if self.player.level >= len(thresholds):
            return 0.0
        return max(0.0, thresholds[self.player.level] - self.player.xp)

    def _apply_combat_progression(self, events: dict[str, Any]) -> None:
        """Convert completed combat objectives into non-RL experience points."""

        xp_gained = (
            int(events["enemies_destroyed"])
            * float(self.progression_cfg["enemy_xp"])
            + int(events["spawners_destroyed"])
            * float(self.progression_cfg["spawner_xp"])
            + int(bool(events["phase_advanced"]))
            * float(self.progression_cfg["phase_xp"])
        )
        events["xp_gained"] = xp_gained
        if xp_gained <= 0.0:
            return

        self.player.xp += xp_gained
        thresholds = [float(value) for value in self.progression_cfg["level_thresholds"]]
        previous_level = self.player.level
        while (
            self.player.level < len(thresholds)
            and self.player.xp >= thresholds[self.player.level]
        ):
            self.player.level += 1

        levels_gained = self.player.level - previous_level
        events["levels_gained"] = levels_gained
        if levels_gained > 0:
            profile = self.weapon_profile()
            self.last_upgrade_name = str(profile["name"])
            events["upgrade_unlocked"] = self.last_upgrade_name
            self.upgrade_banner_steps = max(
                1,
                int(
                    float(self.progression_cfg["upgrade_banner_seconds"])
                    * self.fps
                ),
            )

    # ------------------------------------------------------------------
    # Simulation updates
    # ------------------------------------------------------------------

    def _tick_cooldowns(self) -> None:
        self.player.fire_cooldown_steps = max(0, self.player.fire_cooldown_steps - 1)
        self.upgrade_banner_steps = max(0, self.upgrade_banner_steps - 1)
        for enemy in self.enemies:
            enemy.attack_cooldown_steps = max(0, enemy.attack_cooldown_steps - 1)

    def _update_projectiles(self, events: dict[str, Any]) -> None:
        surviving_projectiles: list[Projectile] = []
        destroyed_enemy_ids: set[int] = set()
        destroyed_spawner_ids: set[int] = set()

        for projectile in self.projectiles:
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

            hit = False
            for enemy in self.enemies:
                if enemy.entity_id in destroyed_enemy_ids:
                    continue
                if circles_overlap(projectile, enemy):
                    damage = min(projectile.damage, max(0.0, enemy.health))
                    enemy.health -= projectile.damage
                    events["damage_dealt_enemy"] += damage
                    events["projectile_hits"] += 1
                    events["impacts"].append(
                        {
                            "x": float(projectile.x),
                            "y": float(projectile.y),
                            "kind": "enemy",
                            "destroyed": enemy.health <= 0.0,
                        }
                    )
                    hit = True
                    if enemy.health <= 0.0:
                        destroyed_enemy_ids.add(enemy.entity_id)
                    break

            if not hit:
                for spawner in self.spawners:
                    if spawner.entity_id in destroyed_spawner_ids:
                        continue
                    if circles_overlap(projectile, spawner):
                        damage = min(projectile.damage, max(0.0, spawner.health))
                        spawner.health -= projectile.damage
                        events["damage_dealt_spawner"] += damage
                        events["projectile_hits"] += 1
                        events["impacts"].append(
                            {
                                "x": float(projectile.x),
                                "y": float(projectile.y),
                                "kind": "spawner",
                                "destroyed": spawner.health <= 0.0,
                            }
                        )
                        hit = True
                        if spawner.health <= 0.0:
                            destroyed_spawner_ids.add(spawner.entity_id)
                        break

            if not hit:
                surviving_projectiles.append(projectile)

        if destroyed_enemy_ids:
            self.enemies = [
                enemy for enemy in self.enemies if enemy.entity_id not in destroyed_enemy_ids
            ]
            events["enemies_destroyed"] += len(destroyed_enemy_ids)
        if destroyed_spawner_ids:
            self.spawners = [
                spawner
                for spawner in self.spawners
                if spawner.entity_id not in destroyed_spawner_ids
            ]
            events["spawners_destroyed"] += len(destroyed_spawner_ids)

        self.projectiles = surviving_projectiles

    def _update_enemies(self, events: dict[str, Any]) -> None:
        contact_damage = float(self.enemy_cfg["contact_damage"])
        cooldown_steps = max(1, int(float(self.enemy_cfg["attack_cooldown_seconds"]) * self.fps))

        for enemy in self.enemies:
            dx = self.player.x - enemy.x
            dy = self.player.y - enemy.y
            distance = math.hypot(dx, dy)
            if distance > 1e-8:
                enemy.vx = dx / distance * enemy.speed
                enemy.vy = dy / distance * enemy.speed
                enemy.x += enemy.vx * self.dt
                enemy.y += enemy.vy * self.dt

            if circles_overlap(enemy, self.player) and enemy.attack_cooldown_steps == 0:
                self.player.health -= contact_damage
                events["damage_taken"] += contact_damage
                events["player_hit"] = True
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

        max_enemies = int(self.enemy_cfg["maximum_active"])
        for spawner in self.spawners:
            spawner.spawn_cooldown_steps -= 1
            if spawner.spawn_cooldown_steps <= 0:
                if len(self.enemies) < max_enemies:
                    self._spawn_enemy(spawner)
                    events["enemies_spawned"] += 1
                spawner.spawn_cooldown_steps = self._spawn_interval_steps()

    def _update_phase(self, events: dict[str, Any]) -> None:
        if self.spawners:
            return

        if self.phase_transition_steps > 0:
            self.phase_transition_steps -= 1
            if self.phase_transition_steps == 0:
                self._spawn_phase_spawners()
            return

        self.phase += 1
        events["phase_advanced"] = True
        self.phase_transition_steps = max(
            0, int(float(self.phase_cfg["transition_seconds"]) * self.fps)
        )
        if self.phase_transition_steps == 0:
            self._spawn_phase_spawners()

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------

    def _spawn_phase_spawners(self) -> None:
        count = min(
            int(self.phase_cfg["maximum_spawners"]),
            int(self.phase_cfg["initial_spawners"])
            + (self.phase - 1) * int(self.phase_cfg["spawners_added_per_phase"]),
        )
        positions = self._choose_spawner_positions(count)
        health_scale = 1.0 + (self.phase - 1) * float(self.phase_cfg["spawner_health_growth"])

        for x, y in positions:
            max_health = float(self.spawner_cfg["max_health"]) * health_scale
            initial_delay = int(
                self.np_random.uniform(0.35, 1.0) * self._spawn_interval_steps()
            )
            self.spawners.append(
                Spawner(
                    x=x,
                    y=y,
                    radius=float(self.spawner_cfg["radius"]),
                    entity_id=self._new_id(),
                    max_health=max_health,
                    health=max_health,
                    spawn_cooldown_steps=max(1, initial_delay),
                )
            )

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

    def _spawn_enemy(self, spawner: Spawner) -> None:
        angle = float(self.np_random.uniform(0.0, math.tau))
        distance = spawner.radius + float(self.enemy_cfg["radius"]) + 5.0
        enemy_radius = float(self.enemy_cfg["radius"])
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
        health_scale = 1.0 + (self.phase - 1) * float(self.phase_cfg["enemy_health_growth"])
        speed_scale = 1.0 + (self.phase - 1) * float(self.phase_cfg["enemy_speed_growth"])
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
            )
        )

    def _spawn_interval_steps(self) -> int:
        base_seconds = float(self.spawner_cfg["spawn_interval_seconds"])
        speedup = 1.0 + (self.phase - 1) * float(self.phase_cfg["spawn_rate_growth"])
        return max(1, int(base_seconds / speedup * self.fps))

    def _new_id(self) -> int:
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        return entity_id

    # ------------------------------------------------------------------
    # Observations, rewards, and diagnostics
    # ------------------------------------------------------------------

    def _nearest_target(self) -> Enemy | Spawner | None:
        targets: list[Enemy | Spawner] = [*self.enemies, *self.spawners]
        if not targets:
            return None
        return min(
            targets,
            key=lambda target: (target.x - self.player.x) ** 2
            + (target.y - self.player.y) ** 2,
        )

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
        """Encode the same nearest target used by direct auto-aim and the reticle."""

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

    def _shaping_snapshot(self) -> dict[str, float | int | None]:
        """Capture potential features used for small, explainable shaping terms."""

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
                else math.hypot(spawner.x - self.player.x, spawner.y - self.player.y)
                / diagonal
            ),
            "target_id": None if target is None else target.entity_id,
            "target_alignment": (
                0.0
                if target is None
                else self._aim_alignment([target])
            ),
        }

    def _apply_shaping_delta(
        self,
        events: dict[str, Any],
        before: dict[str, float | int | None],
    ) -> None:
        """Measure progress only while the same target remains active.

        Restricting the delta to a stable entity avoids artificial reward jumps
        when a target is destroyed or a new phase appears.
        """

        after = self._shaping_snapshot()
        if before["spawner_id"] is not None and before["spawner_id"] == after["spawner_id"]:
            events["spawner_progress"] = float(before["spawner_distance"]) - float(
                after["spawner_distance"]
            )
        if before["target_id"] is not None and before["target_id"] == after["target_id"]:
            events["aim_improvement"] = float(after["target_alignment"]) - float(
                before["target_alignment"]
            )

    def _get_observation(self) -> np.ndarray:
        max_speed = max(
            float(self.player_cfg["max_speed"]), float(self.player_cfg["direct_speed"])
        )
        enemy_features = self._target_features(self.enemies)
        spawner_features = self._target_features(self.spawners)
        active_target_features = self._active_target_features()
        fire_cooldown = self.weapon_cooldown_steps()
        profile = self.weapon_profile()
        tiers = self._weapon_tiers()
        max_shots = max(int(tier["shot_count"]) for tier in tiers)
        max_damage_multiplier = max(float(tier["damage_multiplier"]) for tier in tiers)
        cooldown_multipliers = [float(tier["cooldown_multiplier"]) for tier in tiers]
        slowest_cooldown = max(cooldown_multipliers)
        fastest_cooldown = min(cooldown_multipliers)
        fire_rate_progress = (
            1.0
            if math.isclose(slowest_cooldown, fastest_cooldown)
            else (slowest_cooldown - float(profile["cooldown_multiplier"]))
            / (slowest_cooldown - fastest_cooldown)
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
                np.clip(len(self.enemies) / max(1, int(self.enemy_cfg["maximum_active"])), 0.0, 1.0),
                np.clip(len(self.spawners) / max(1, int(self.phase_cfg["maximum_spawners"])), 0.0, 1.0),
                np.clip(self.phase / max(1, int(self.phase_cfg["observation_phase_cap"])), 0.0, 1.0),
                np.clip(1.0 - self.step_count / self.max_steps, 0.0, 1.0),
                self._aim_alignment(self.enemies),
                self._aim_alignment(self.spawners),
                *active_target_features,
                (self.player.level - 1) / max(1, self.maximum_player_level - 1),
                self.xp_progress(),
                int(profile["shot_count"]) / max(1, max_shots),
                fire_rate_progress,
                float(profile["damage_multiplier"]) / max_damage_multiplier,
                float(str(profile["kind"]) in ("laser", "nova")),
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
            "player_health": self.player.health,
            "player_level": self.player.level,
            "player_xp": self.player.xp,
            "xp_progress": self.xp_progress(),
            "xp_to_next_level": self.xp_to_next_level(),
            "weapon_name": str(self.weapon_profile()["name"]),
            "weapon_kind": str(self.weapon_profile()["kind"]),
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
    "DIRECT_ACTIONS",
    "ROTATION_ACTIONS",
    "ObservationIndex",
    "OBSERVATION_NAMES",
]
