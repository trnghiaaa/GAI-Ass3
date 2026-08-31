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
    PLAYER_ANGULAR_VELOCITY = 20
    ENEMY_FORWARD_ALIGNMENT = 21
    ENEMY_TURN_DIRECTION = 22
    SPAWNER_FORWARD_ALIGNMENT = 23
    SPAWNER_TURN_DIRECTION = 24
    ENEMY_CLOSING_SPEED = 25
    PLAYER_FORWARD_VELOCITY = 26
    PLAYER_LATERAL_VELOCITY = 27
    SECOND_ENEMY_DIRECTION_X = 28
    SECOND_ENEMY_DIRECTION_Y = 29
    SECOND_ENEMY_DISTANCE = 30
    SECOND_ENEMY_CLOSING_SPEED = 31
    CROWD_ESCAPE_DIRECTION_X = 32
    CROWD_ESCAPE_DIRECTION_Y = 33
    CROWD_PRESSURE = 34
    DANGER_ENEMY_COUNT = 35
    ESCAPE_FORWARD_ALIGNMENT = 36
    ESCAPE_TURN_DIRECTION = 37
    SAFE_CORRIDOR_ALIGNMENT = 38
    SAFE_CORRIDOR_TURN_DIRECTION = 39
    WALL_LEFT_CLEARANCE = 40
    WALL_RIGHT_CLEARANCE = 41
    WALL_TOP_CLEARANCE = 42
    WALL_BOTTOM_CLEARANCE = 43
    BOUNDARY_INWARD_ALIGNMENT = 44
    BOUNDARY_INWARD_TURN_DIRECTION = 45
    EMERGENCY_THREAT = 46


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
        self.reward_cfg = self.config["rewards"]

        self.width = int(self.sim_cfg["window_width"])
        self.height = int(self.sim_cfg["window_height"])
        self.playfield_top = float(self.sim_cfg["hud_height"])
        self.fps = int(self.sim_cfg["fps"])
        self.playback_speed = float(self.sim_cfg.get("playback_speed", 1.0))
        if self.playback_speed <= 0.0:
            raise ValueError("simulation.playback_speed must be positive")
        self.render_fps = max(1, round(self.fps * self.playback_speed))
        self.dt = 1.0 / self.fps
        self.max_steps = int(self.sim_cfg["max_steps"])
        self.metadata = {**self.metadata, "render_fps": self.render_fps}

        self.action_names = DIRECT_ACTIONS if control_style == "direct" else ROTATION_ACTIONS
        self.action_space = spaces.Discrete(len(self.action_names))

        # The layout is fixed even when there are no enemies or spawners.
        self.observation_names = OBSERVATION_NAMES
        low = np.array(
            [
                -1, -1, -1, -1, -1, -1, 0, 0, -1, -1,
                0, 0, -1, -1, 0, 0, 0, 0, 0, 0,
                -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, 0, -1, -1, -1, 0, 0, -1, -1,
                -1, -1,
                0, 0, 0, 0, -1, -1, 0,
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
        self.wall_trap_steps = 0
        self._next_entity_id = 1
        self._renderer = None

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
        options = options or {}
        start_phase = int(options.get("start_phase", 1))
        if start_phase < 1:
            raise ValueError("start_phase must be at least 1")

        self.phase = start_phase
        self.step_count = 0
        self.phase_transition_steps = 0
        self.done = False
        self.last_end_reason = None
        self.wall_trap_steps = 0
        self._next_entity_id = 1
        self.enemies = []
        self.spawners = []
        self.projectiles = []

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
            "invalid_shot": False,
            "shot_alignment": 0.0,
            "projectile_hits": 0,
            "enemy_damage_dealt": 0.0,
            "spawner_damage_dealt": 0.0,
            "spawner_progress": 0.0,
            "spawner_navigation_active": False,
            "enemy_separation": 0.0,
            "enemy_proximity": 0.0,
            "crowd_escape": 0.0,
            "crowd_pressure": 0.0,
            "escape_alignment_progress": 0.0,
            "escape_thrust": 0.0,
            "wall_trap": 0.0,
            "wall_escape_progress": 0.0,
            "wall_outward_thrust": 0.0,
            "emergency_attack": 0.0,
            "dangerous_spawner_damage": 0.0,
            "enemies_spawned": 0,
            "enemies_destroyed": 0,
            "spawners_destroyed": 0,
            "damage_taken": 0.0,
            "phase_advanced": False,
        }

        tracked_spawner = self._nearest_entity(self.spawners)
        spawner_distance_before = self._distance_to(tracked_spawner)
        tracked_enemy = self._nearest_entity(self.enemies)
        enemy_distance_before = self._distance_to(tracked_enemy)
        crowd_before = self._crowd_threat_features()
        safe_corridor_before = self._safe_corridor_features(crowd_before)
        wall_proximity_before = self._wall_proximity()
        boundary_before = self._boundary_features()
        emergency_before = self._emergency_threat(crowd_before)
        enemy_closing_before = max(
            0.0,
            self._enemy_closing_speed_for(
                tracked_enemy if isinstance(tracked_enemy, Enemy) else None
            ),
        )
        player_speed_before = math.hypot(self.player.vx, self.player.vy) / max(
            1.0,
            float(self.player_cfg["max_speed"]),
        )

        self._tick_cooldowns()
        shoot_action = (
            DIRECT_ACTIONS["SHOOT"]
            if self.control_style == "direct"
            else ROTATION_ACTIONS["SHOOT"]
        )
        events["shot_fired"] = self._apply_player_action(int(action))
        events["invalid_shot"] = bool(action == shoot_action and not events["shot_fired"])
        if events["shot_fired"]:
            events["emergency_attack"] = emergency_before
        if self.control_style == "rotation" and action == ROTATION_ACTIONS["THRUST"]:
            events["wall_outward_thrust"] = float(
                wall_proximity_before * max(0.0, -boundary_before[4])
            )
        crowd_after_action = self._crowd_threat_features()
        safe_corridor_after_action = self._safe_corridor_features(crowd_after_action)
        if crowd_before[6] > 0.0:
            events["escape_alignment_progress"] = float(
                np.clip(
                    safe_corridor_after_action[0] - safe_corridor_before[0],
                    -0.15,
                    0.15,
                )
                * crowd_before[6]
            )
            if (
                self.control_style == "rotation"
                and action == ROTATION_ACTIONS["THRUST"]
            ):
                events["escape_thrust"] = float(
                    max(0.0, safe_corridor_after_action[0]) * crowd_before[6]
                )
        if events["shot_fired"]:
            events["shot_alignment"] = max(0.0, self._priority_target_alignment())
        self._update_projectiles(events)
        wall_proximity_after = self._wall_proximity()
        events["wall_escape_progress"] = float(
            np.clip(wall_proximity_before - wall_proximity_after, 0.0, 0.05)
        )
        if events["spawner_damage_dealt"] > 0.0:
            events["dangerous_spawner_damage"] = float(
                events["spawner_damage_dealt"] * emergency_before
            )

        navigation_emergency_limit = float(
            self.reward_cfg.get("spawner_navigation_max_emergency", 0.30)
        )
        if (
            tracked_spawner is not None
            and tracked_spawner in self.spawners
            and emergency_before <= navigation_emergency_limit
        ):
            spawner_distance_after = self._distance_to(tracked_spawner)
            diagonal = math.hypot(self.width, self.height - self.playfield_top)
            events["spawner_navigation_active"] = True
            events["spawner_progress"] = float(
                np.clip(
                    (spawner_distance_before - spawner_distance_after) / diagonal,
                    -0.02,
                    0.02,
                )
            )

        self._update_enemies(events)
        if tracked_enemy is not None and tracked_enemy in self.enemies:
            enemy_distance_after = self._distance_to(tracked_enemy)
            danger_distance = float(self.reward_cfg["enemy_danger_distance"])
            if min(enemy_distance_before, enemy_distance_after) < danger_distance:
                diagonal = math.hypot(self.width, self.height - self.playfield_top)
                events["enemy_separation"] = float(
                    np.clip(
                        (enemy_distance_after - enemy_distance_before) / diagonal,
                        -0.02,
                        0.02,
                    )
                )
        stagnation_emergency = float(
            self.reward_cfg.get("danger_stagnation_emergency", 0.35)
        )
        stagnation_speed = float(
            self.reward_cfg.get("danger_stagnation_speed", 0.22)
        )
        stagnation_closing = float(
            self.reward_cfg.get("danger_stagnation_closing_speed", 0.03)
        )
        if (
            emergency_before >= stagnation_emergency
            and enemy_closing_before >= stagnation_closing
            and player_speed_before < stagnation_speed
        ):
            low_speed = 1.0 - player_speed_before / max(stagnation_speed, 1e-6)
            events["enemy_proximity"] = float(
                emergency_before * enemy_closing_before * low_speed
            )
        crowd_after = self._crowd_threat_features()
        events["crowd_escape"] = float(
            np.clip(crowd_before[6] - crowd_after[6], -0.05, 0.05)
        )
        events["crowd_pressure"] = float(crowd_after[6])
        trap_proximity = float(self.reward_cfg.get("wall_trap_proximity", 0.75))
        if (
            wall_proximity_after >= trap_proximity
            and events["wall_escape_progress"] <= 1e-6
        ):
            self.wall_trap_steps += 1
        else:
            self.wall_trap_steps = 0
        trap_grace = int(self.reward_cfg.get("wall_trap_grace_steps", 30))
        trap_full = max(
            trap_grace + 1,
            int(self.reward_cfg.get("wall_trap_full_steps", 120)),
        )
        persistence = float(
            np.clip(
                (self.wall_trap_steps - trap_grace) / (trap_full - trap_grace),
                0.0,
                1.0,
            )
        )
        wall_severity = max(
            0.0,
            (wall_proximity_after - trap_proximity) / max(1e-6, 1.0 - trap_proximity),
        )
        events["wall_trap"] = float(wall_severity * persistence)
        self._update_spawners(events)
        self._update_phase(events)

        terminated = self.player.health <= 0.0
        truncated = self.step_count >= self.max_steps and not terminated
        if terminated:
            self.player.health = 0.0
            self.last_end_reason = "player_destroyed"
        elif truncated:
            self.last_end_reason = "time_limit"
        self.done = terminated or truncated

        reward = self._calculate_reward(events, terminated)
        info = self._get_info()
        info.update(events)
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

    def _apply_player_action(self, action: int) -> bool:
        shot_fired = False

        if self.control_style == "direct":
            self.player.angular_velocity = 0.0
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
                shot_fired = self._fire_projectile(auto_aim=True)
        else:
            rotation_speed = math.radians(float(self.player_cfg["rotation_speed_degrees"]))
            target_angular_velocity = 0.0
            if action == ROTATION_ACTIONS["ROTATE_LEFT"]:
                target_angular_velocity = -rotation_speed
            elif action == ROTATION_ACTIONS["ROTATE_RIGHT"]:
                target_angular_velocity = rotation_speed
            elif action == ROTATION_ACTIONS["THRUST"]:
                acceleration = float(self.player_cfg["thrust_acceleration"])
                self.player.vx += math.cos(self.player.angle) * acceleration * self.dt
                self.player.vy += math.sin(self.player.angle) * acceleration * self.dt
            elif action == ROTATION_ACTIONS["SHOOT"]:
                shot_fired = self._fire_projectile(auto_aim=False)
                # The five-action assignment interface cannot thrust and shoot
                # simultaneously. Preserve a small combat drift so repeated
                # firing does not turn the policy into a stationary turret.
                speed = math.hypot(self.player.vx, self.player.vy)
                minimum_drift = float(self.player_cfg.get("combat_drift_speed", 90.0))
                if speed < minimum_drift and self._forward_wall_distance() > 80.0:
                    drift_acceleration = float(
                        self.player_cfg.get("combat_drift_acceleration", 220.0)
                    )
                    self.player.vx += (
                        math.cos(self.player.angle) * drift_acceleration * self.dt
                    )
                    self.player.vy += (
                        math.sin(self.player.angle) * drift_acceleration * self.dt
                    )

                crowd_features = self._crowd_threat_features()
                emergency = self._emergency_threat(crowd_features)
                if emergency >= float(
                    self.player_cfg.get("threat_steering_threshold", 0.52)
                ):
                    _, escape_turn = self._safe_corridor_features(crowd_features)
                    if abs(escape_turn) >= 0.04:
                        threat_strength = float(
                            self.player_cfg.get("threat_steering_strength", 0.65)
                        )
                        target_angular_velocity = (
                            math.copysign(rotation_speed, escape_turn)
                            * threat_strength
                            * emergency
                        )

            # Low-level boundary steering blends with NOOP, THRUST, and SHOOT.
            # Explicit left/right actions remain entirely under the controller.
            wall_steering_distance = float(
                self.player_cfg.get("wall_steering_distance", 100.0)
            )
            if (
                action
                not in (
                    ROTATION_ACTIONS["ROTATE_LEFT"],
                    ROTATION_ACTIONS["ROTATE_RIGHT"],
                )
                and self._forward_wall_distance() < wall_steering_distance
            ):
                boundary_turn = self._boundary_features()[5]
                if abs(boundary_turn) < 0.04:
                    boundary_turn = 1.0
                wall_strength = float(
                    self.player_cfg.get("wall_steering_strength", 0.9)
                )
                target_angular_velocity = (
                    math.copysign(rotation_speed, boundary_turn) * wall_strength
                )

            turn_response = float(
                self.player_cfg["turn_response"]
                if target_angular_velocity != 0.0
                else self.player_cfg["turn_release_response"]
            )
            self.player.angular_velocity += (
                target_angular_velocity - self.player.angular_velocity
            ) * turn_response
            self.player.angle += self.player.angular_velocity * self.dt

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
        minimum_x = self.player.radius
        maximum_x = self.width - self.player.radius
        minimum_y = self.playfield_top + self.player.radius
        maximum_y = self.height - self.player.radius
        if self.player.x < minimum_x and self.player.vx < 0.0:
            self.player.vx = 0.0
        elif self.player.x > maximum_x and self.player.vx > 0.0:
            self.player.vx = 0.0
        if self.player.y < minimum_y and self.player.vy < 0.0:
            self.player.vy = 0.0
        elif self.player.y > maximum_y and self.player.vy > 0.0:
            self.player.vy = 0.0
        self.player.x = float(np.clip(self.player.x, minimum_x, maximum_x))
        self.player.y = float(
            np.clip(self.player.y, minimum_y, maximum_y)
        )
        return shot_fired

    def _forward_wall_distance(self) -> float:
        """Return ray distance from the ship heading to the first arena wall."""

        heading_x = math.cos(self.player.angle)
        heading_y = math.sin(self.player.angle)
        distances: list[float] = []
        if heading_x < -1e-8:
            distances.append((self.player.x - self.player.radius) / -heading_x)
        elif heading_x > 1e-8:
            distances.append(
                (self.width - self.player.radius - self.player.x) / heading_x
            )
        if heading_y < -1e-8:
            distances.append(
                (self.player.y - self.playfield_top - self.player.radius) / -heading_y
            )
        elif heading_y > 1e-8:
            distances.append(
                (self.height - self.player.radius - self.player.y) / heading_y
            )
        return max(0.0, min(distances, default=float("inf")))

    def _fire_projectile(self, auto_aim: bool) -> bool:
        if self.player.fire_cooldown_steps > 0:
            return False

        angle = self.player.angle
        if auto_aim:
            target = self._nearest_target()
            if target is not None:
                angle = math.atan2(target.y - self.player.y, target.x - self.player.x)
                self.player.angle = angle

        speed = float(self.projectile_cfg["speed"])
        offset = self.player.radius + float(self.projectile_cfg["radius"]) + 2.0
        projectile = Projectile(
            x=self.player.x + math.cos(angle) * offset,
            y=self.player.y + math.sin(angle) * offset,
            radius=float(self.projectile_cfg["radius"]),
            entity_id=self._new_id(),
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
            damage=float(self.projectile_cfg["damage"]),
            lifetime_steps=int(self.projectile_cfg["lifetime_seconds"] * self.fps),
        )
        self.projectiles.append(projectile)
        self.player.fire_cooldown_steps = max(
            1, int(float(self.player_cfg["fire_cooldown_seconds"]) * self.fps)
        )
        return True

    # ------------------------------------------------------------------
    # Simulation updates
    # ------------------------------------------------------------------

    def _tick_cooldowns(self) -> None:
        self.player.fire_cooldown_steps = max(0, self.player.fire_cooldown_steps - 1)
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
                    events["projectile_hits"] += 1
                    events["enemy_damage_dealt"] += damage
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
                        events["projectile_hits"] += 1
                        events["spawner_damage_dealt"] += damage
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
        self.enemies = []
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
        x = float(np.clip(spawner.x + math.cos(angle) * distance, 0.0, self.width))
        y = float(
            np.clip(
                spawner.y + math.sin(angle) * distance,
                self.playfield_top + float(self.enemy_cfg["radius"]),
                self.height,
            )
        )
        health_scale = 1.0 + (self.phase - 1) * float(self.phase_cfg["enemy_health_growth"])
        speed_scale = 1.0 + (self.phase - 1) * float(self.phase_cfg["enemy_speed_growth"])
        max_health = float(self.enemy_cfg["max_health"]) * health_scale
        self.enemies.append(
            Enemy(
                x=x,
                y=y,
                radius=float(self.enemy_cfg["radius"]),
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

    def _nearest_entity(
        self, targets: list[Enemy] | list[Spawner]
    ) -> Enemy | Spawner | None:
        if not targets:
            return None
        return min(
            targets,
            key=lambda target: (target.x - self.player.x) ** 2
            + (target.y - self.player.y) ** 2,
        )

    def _distance_to(self, target: Enemy | Spawner | None) -> float:
        if target is None:
            return math.hypot(self.width, self.height - self.playfield_top)
        return math.hypot(target.x - self.player.x, target.y - self.player.y)

    def _nearest_target(self) -> Enemy | Spawner | None:
        targets: list[Enemy | Spawner] = [*self.enemies, *self.spawners]
        if not targets:
            return None
        return min(
            targets,
            key=lambda target: (target.x - self.player.x) ** 2
            + (target.y - self.player.y) ** 2,
        )

    def _target_alignment(
        self, targets: list[Enemy] | list[Spawner]
    ) -> tuple[float, float]:
        """Return forward alignment and signed turn direction to a target.

        Alignment is ``1`` directly ahead and ``-1`` directly behind. The
        signed cross product is positive when the target is to the ship's
        right in screen coordinates and negative when it is to the left.
        """

        target = self._nearest_entity(targets)
        if target is None:
            return 0.0, 0.0

        dx = target.x - self.player.x
        dy = target.y - self.player.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-8:
            return 1.0, 0.0

        direction_x = dx / distance
        direction_y = dy / distance
        heading_x = math.cos(self.player.angle)
        heading_y = math.sin(self.player.angle)
        alignment = heading_x * direction_x + heading_y * direction_y
        turn_direction = heading_x * direction_y - heading_y * direction_x
        return float(alignment), float(turn_direction)

    def _priority_target_alignment(self) -> float:
        """Aim at an immediate enemy threat, otherwise at the objective."""

        nearest_enemy = self._nearest_entity(self.enemies)
        if (
            nearest_enemy is not None
            and self._distance_to(nearest_enemy)
            < float(self.reward_cfg["enemy_danger_distance"])
        ):
            return self._target_alignment(self.enemies)[0]
        if self.spawners:
            return self._target_alignment(self.spawners)[0]
        return self._target_alignment(self.enemies)[0]

    def _enemy_closing_speed_for(self, enemy: Enemy | None) -> float:
        """Return normalized positive speed when ``enemy`` is closing."""

        if enemy is None:
            return 0.0
        dx = enemy.x - self.player.x
        dy = enemy.y - self.player.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-8:
            return 1.0

        direction_x = dx / distance
        direction_y = dy / distance
        relative_velocity_x = enemy.vx - self.player.vx
        relative_velocity_y = enemy.vy - self.player.vy
        distance_rate = (
            relative_velocity_x * direction_x + relative_velocity_y * direction_y
        )
        max_relative_speed = float(self.enemy_cfg["speed"]) + float(
            self.player_cfg["max_speed"]
        )
        return float(np.clip(-distance_rate / max_relative_speed, -1.0, 1.0))

    def _enemy_closing_speed(self) -> float:
        """Return normalized positive speed when the nearest enemy is closing."""

        enemy = self._nearest_entity(self.enemies)
        return self._enemy_closing_speed_for(
            enemy if isinstance(enemy, Enemy) else None
        )

    def _crowd_threat_features(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """Describe a second threat and the safest direction away from the crowd.

        Nearby enemies contribute quadratically more pressure. Their weighted
        repulsion vector gives the policy a stable escape corridor even when the
        nearest-enemy identity changes from frame to frame.
        """

        diagonal = math.hypot(self.width, self.height - self.playfield_top)
        ordered = sorted(
            self.enemies,
            key=lambda enemy: (enemy.x - self.player.x) ** 2
            + (enemy.y - self.player.y) ** 2,
        )
        if len(ordered) >= 2:
            second = ordered[1]
            second_dx = second.x - self.player.x
            second_dy = second.y - self.player.y
            second_distance_pixels = math.hypot(second_dx, second_dy)
            if second_distance_pixels > 1e-8:
                second_direction_x = second_dx / second_distance_pixels
                second_direction_y = second_dy / second_distance_pixels
            else:
                second_direction_x = 0.0
                second_direction_y = 0.0
            second_distance = float(
                np.clip(second_distance_pixels / diagonal, 0.0, 1.0)
            )
            second_closing_speed = self._enemy_closing_speed_for(second)
        else:
            second_direction_x = 0.0
            second_direction_y = 0.0
            second_distance = 1.0
            second_closing_speed = 0.0

        danger_distance = float(self.reward_cfg["crowd_danger_distance"])
        pressure_cap = max(1e-8, float(self.reward_cfg["crowd_pressure_cap"]))
        repulsion_x = 0.0
        repulsion_y = 0.0
        raw_pressure = 0.0
        danger_count = 0
        for enemy in ordered:
            away_x = self.player.x - enemy.x
            away_y = self.player.y - enemy.y
            distance = math.hypot(away_x, away_y)
            closeness = max(0.0, 1.0 - distance / danger_distance)
            if closeness <= 0.0:
                continue
            danger_count += 1
            pressure = closeness * closeness
            raw_pressure += pressure
            if distance > 1e-8:
                closing_boost = 1.0 + max(
                    0.0, self._enemy_closing_speed_for(enemy)
                )
                repulsion_x += away_x / distance * pressure * closing_boost
                repulsion_y += away_y / distance * pressure * closing_boost

        repulsion_length = math.hypot(repulsion_x, repulsion_y)
        if repulsion_length <= 1e-8 and danger_count and ordered:
            nearest = ordered[0]
            repulsion_x = self.player.x - nearest.x
            repulsion_y = self.player.y - nearest.y
            repulsion_length = math.hypot(repulsion_x, repulsion_y)

        if repulsion_length > 1e-8:
            escape_x = repulsion_x / repulsion_length
            escape_y = repulsion_y / repulsion_length
            heading_x = math.cos(self.player.angle)
            heading_y = math.sin(self.player.angle)
            escape_alignment = heading_x * escape_x + heading_y * escape_y
            escape_turn = heading_x * escape_y - heading_y * escape_x
        else:
            escape_x = 0.0
            escape_y = 0.0
            escape_alignment = 0.0
            escape_turn = 0.0

        crowd_pressure = float(np.clip(raw_pressure / pressure_cap, 0.0, 1.0))
        danger_count_normalized = float(
            np.clip(
                danger_count / max(1, int(self.enemy_cfg["maximum_active"])),
                0.0,
                1.0,
            )
        )
        return (
            float(second_direction_x),
            float(second_direction_y),
            second_distance,
            second_closing_speed,
            float(escape_x),
            float(escape_y),
            crowd_pressure,
            danger_count_normalized,
            float(escape_alignment),
            float(escape_turn),
        )

    def _safe_corridor_features(
        self,
        crowd_features: tuple[
            float, float, float, float, float, float, float, float, float, float
        ]
        | None = None,
    ) -> tuple[float, float]:
        """Return body-relative alignment to a wall-safe crowd escape route."""

        crowd = self._crowd_threat_features() if crowd_features is None else crowd_features
        if crowd[6] <= 0.0:
            return 0.0, 0.0
        ordered = sorted(
            self.enemies,
            key=lambda enemy: (enemy.x - self.player.x) ** 2
            + (enemy.y - self.player.y) ** 2,
        )
        safe_x, safe_y = self._safe_escape_direction(
            ordered,
            float(self.reward_cfg["crowd_danger_distance"]),
            crowd[4],
            crowd[5],
        )
        heading_x = math.cos(self.player.angle)
        heading_y = math.sin(self.player.angle)
        return (
            float(heading_x * safe_x + heading_y * safe_y),
            float(heading_x * safe_y - heading_y * safe_x),
        )

    def _safe_escape_direction(
        self,
        enemies: list[Enemy],
        danger_distance: float,
        fallback_x: float,
        fallback_y: float,
    ) -> tuple[float, float]:
        """Choose a nearby open corridor that does not terminate at a wall."""

        lookahead = min(160.0, danger_distance * 0.6)
        player_margin = self.player.radius + 4.0
        left = player_margin
        right = self.width - player_margin
        top = self.playfield_top + player_margin
        bottom = self.height - player_margin
        best_score = -math.inf
        best_direction = (fallback_x, fallback_y)

        for index in range(24):
            angle = math.tau * index / 24.0
            direction_x = math.cos(angle)
            direction_y = math.sin(angle)
            candidate_x = self.player.x + direction_x * lookahead
            candidate_y = self.player.y + direction_y * lookahead
            if not (left <= candidate_x <= right and top <= candidate_y <= bottom):
                continue

            wall_clearance = min(
                candidate_x - left,
                right - candidate_x,
                candidate_y - top,
                bottom - candidate_y,
            )
            future_clearances: list[float] = []
            prediction_seconds = lookahead / max(
                1.0, float(self.player_cfg["max_speed"])
            )
            for enemy in enemies:
                predicted_x = enemy.x + enemy.vx * prediction_seconds
                predicted_y = enemy.y + enemy.vy * prediction_seconds
                future_clearances.append(
                    math.hypot(candidate_x - predicted_x, candidate_y - predicted_y)
                )
            future_clearances.sort()
            nearest_clearance = future_clearances[0] if future_clearances else danger_distance
            local_clearance = float(np.mean(future_clearances[:3]))
            fallback_alignment = direction_x * fallback_x + direction_y * fallback_y
            score = (
                1.4 * min(nearest_clearance / danger_distance, 2.0)
                + 0.35 * min(local_clearance / danger_distance, 2.0)
                + 0.45 * min(wall_clearance / lookahead, 1.0)
                + 0.15 * fallback_alignment
            )
            if score > best_score:
                best_score = score
                best_direction = (direction_x, direction_y)
        return best_direction

    def _wall_proximity(self) -> float:
        """Return one near a wall and zero when there is ample turning room."""

        clearance = min(
            self.player.x - self.player.radius,
            self.width - self.player.radius - self.player.x,
            self.player.y - self.playfield_top - self.player.radius,
            self.height - self.player.radius - self.player.y,
        )
        safe_clearance = 120.0
        return float(np.clip(1.0 - clearance / safe_clearance, 0.0, 1.0))

    def _boundary_features(self) -> tuple[float, float, float, float, float, float]:
        """Return wall clearances and the body-relative direction back inside."""

        safe_clearance = 160.0
        clearances = (
            self.player.x - self.player.radius,
            self.width - self.player.radius - self.player.x,
            self.player.y - self.playfield_top - self.player.radius,
            self.height - self.player.radius - self.player.y,
        )
        normalized = tuple(
            float(np.clip(clearance / safe_clearance, 0.0, 1.0))
            for clearance in clearances
        )
        left_pressure = max(0.0, 1.0 - clearances[0] / safe_clearance)
        right_pressure = max(0.0, 1.0 - clearances[1] / safe_clearance)
        top_pressure = max(0.0, 1.0 - clearances[2] / safe_clearance)
        bottom_pressure = max(0.0, 1.0 - clearances[3] / safe_clearance)
        inward_x = left_pressure - right_pressure
        inward_y = top_pressure - bottom_pressure
        inward_length = math.hypot(inward_x, inward_y)
        if inward_length <= 1e-8:
            return (*normalized, 0.0, 0.0)
        inward_x /= inward_length
        inward_y /= inward_length
        heading_x = math.cos(self.player.angle)
        heading_y = math.sin(self.player.angle)
        return (
            *normalized,
            float(heading_x * inward_x + heading_y * inward_y),
            float(heading_x * inward_y - heading_y * inward_x),
        )

    def _emergency_threat(
        self,
        crowd_features: tuple[
            float, float, float, float, float, float, float, float, float, float
        ]
        | None = None,
    ) -> float:
        """Return immediate collision urgency, distinct from objective value."""

        crowd = self._crowd_threat_features() if crowd_features is None else crowd_features
        nearest = self._nearest_entity(self.enemies)
        if not isinstance(nearest, Enemy):
            return 0.0
        emergency_distance = float(self.reward_cfg.get("emergency_distance", 180.0))
        closeness = max(0.0, 1.0 - self._distance_to(nearest) / emergency_distance)
        closing = max(0.0, self._enemy_closing_speed_for(nearest))
        immediate = closeness * (0.65 + 0.35 * closing)
        return float(np.clip(max(immediate, crowd[6]), 0.0, 1.0))

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

    def _get_observation(self) -> np.ndarray:
        max_speed = max(
            float(self.player_cfg["max_speed"]), float(self.player_cfg["direct_speed"])
        )
        enemy_features = self._target_features(self.enemies)
        spawner_features = self._target_features(self.spawners)
        enemy_alignment = self._target_alignment(self.enemies)
        spawner_alignment = self._target_alignment(self.spawners)
        fire_cooldown = max(1, int(float(self.player_cfg["fire_cooldown_seconds"]) * self.fps))
        rotation_speed = math.radians(float(self.player_cfg["rotation_speed_degrees"]))
        heading_x = math.cos(self.player.angle)
        heading_y = math.sin(self.player.angle)
        forward_velocity = (
            self.player.vx * heading_x + self.player.vy * heading_y
        ) / max_speed
        lateral_velocity = (
            self.player.vx * -heading_y + self.player.vy * heading_x
        ) / max_speed
        crowd_features = self._crowd_threat_features()
        safe_corridor_features = self._safe_corridor_features(crowd_features)
        boundary_features = self._boundary_features()

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
                np.clip(self.player.angular_velocity / rotation_speed, -1.0, 1.0),
                *enemy_alignment,
                *spawner_alignment,
                self._enemy_closing_speed(),
                np.clip(forward_velocity, -1.0, 1.0),
                np.clip(lateral_velocity, -1.0, 1.0),
                *crowd_features,
                *safe_corridor_features,
                *boundary_features,
                self._emergency_threat(crowd_features),
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

    def _calculate_reward(self, events: dict[str, Any], terminated: bool) -> float:
        """Combine sparse task rewards with dense combat/navigation feedback."""

        reward = float(self.reward_cfg["survival"])
        reward += float(events["enemy_damage_dealt"]) * float(
            self.reward_cfg["enemy_damage_per_hp"]
        )
        reward += float(events["spawner_damage_dealt"]) * float(
            self.reward_cfg["spawner_damage_per_hp"]
        )
        reward += events["enemies_destroyed"] * float(self.reward_cfg["enemy_destroyed"])
        reward += events["spawners_destroyed"] * float(self.reward_cfg["spawner_destroyed"])
        if events["shot_fired"]:
            reward += float(self.reward_cfg["shot_cost"])
            reward += float(events["shot_alignment"]) * float(
                self.reward_cfg["aimed_shot"]
            )
        if events["invalid_shot"]:
            reward += float(self.reward_cfg["invalid_shot"])
        reward += float(events["spawner_progress"]) * float(
            self.reward_cfg["spawner_progress"]
        )
        reward += float(events["enemy_separation"]) * float(
            self.reward_cfg["enemy_separation"]
        )
        reward += float(events["enemy_proximity"]) * float(
            self.reward_cfg["enemy_proximity"]
        )
        reward += float(events["crowd_escape"]) * float(
            self.reward_cfg["crowd_escape"]
        )
        reward += float(events["crowd_pressure"]) * float(
            self.reward_cfg["crowd_proximity"]
        )
        reward += float(events["escape_alignment_progress"]) * float(
            self.reward_cfg["escape_alignment_progress"]
        )
        reward += float(events["escape_thrust"]) * float(
            self.reward_cfg["escape_thrust"]
        )
        reward += float(events["wall_trap"]) * float(
            self.reward_cfg["wall_trap"]
        )
        reward += float(events["wall_escape_progress"]) * float(
            self.reward_cfg["wall_escape_progress"]
        )
        reward += float(events["wall_outward_thrust"]) * float(
            self.reward_cfg["wall_outward_thrust"]
        )
        reward += float(events["emergency_attack"]) * float(
            self.reward_cfg["emergency_attack"]
        )
        reward += float(events["dangerous_spawner_damage"]) * float(
            self.reward_cfg["dangerous_spawner_damage_per_hp"]
        )
        reward += float(events["damage_taken"]) * float(self.reward_cfg["damage_taken_per_hp"])
        if events["phase_advanced"]:
            reward += float(self.reward_cfg["phase_advanced"])
        if terminated:
            reward += float(self.reward_cfg["death"])
        return reward

    def _get_info(self) -> dict[str, Any]:
        return {
            "control_style": self.control_style,
            "phase": self.phase,
            "step": self.step_count,
            "time_seconds": self.step_count * self.dt,
            "player_health": self.player.health,
            "active_enemies": len(self.enemies),
            "active_spawners": len(self.spawners),
            "active_projectiles": len(self.projectiles),
            "nearest_enemy_distance": self._distance_to(
                self._nearest_entity(self.enemies)
            ),
            "crowd_pressure": self._crowd_threat_features()[6],
            "wall_proximity": self._wall_proximity(),
            "emergency_threat": self._emergency_threat(),
        }


__all__ = [
    "ArenaEnv",
    "DIRECT_ACTIONS",
    "ROTATION_ACTIONS",
    "ObservationIndex",
    "OBSERVATION_NAMES",
]
