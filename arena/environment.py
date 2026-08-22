"""Gymnasium-compatible real-time action arena for Assignment 3 Part II.

The simulation uses continuous pixel coordinates and a fixed time step. It is
kept independent from Pygame so long training runs can execute headlessly; the
optional :class:`arena.renderer.ArenaRenderer` displays the same live state.
"""

from __future__ import annotations

from copy import deepcopy
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
        self.dt = 1.0 / self.fps
        self.max_steps = int(self.sim_cfg["max_steps"])
        self.metadata["render_fps"] = self.fps

        self.action_names = DIRECT_ACTIONS if control_style == "direct" else ROTATION_ACTIONS
        self.action_space = spaces.Discrete(len(self.action_names))

        # The layout is fixed even when there are no enemies or spawners.
        self.observation_names = (
            "player_x",
            "player_y",
            "player_velocity_x",
            "player_velocity_y",
            "player_heading_cos",
            "player_heading_sin",
            "player_health",
            "weapon_ready",
            "nearest_enemy_dx",
            "nearest_enemy_dy",
            "nearest_enemy_distance",
            "nearest_enemy_health",
            "nearest_spawner_dx",
            "nearest_spawner_dy",
            "nearest_spawner_distance",
            "nearest_spawner_health",
            "enemy_count",
            "spawner_count",
            "phase",
            "time_remaining",
        )
        low = np.array(
            [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
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
            "enemies_spawned": 0,
            "enemies_destroyed": 0,
            "spawners_destroyed": 0,
            "damage_taken": 0.0,
            "phase_advanced": False,
        }

        self._tick_cooldowns()
        events["shot_fired"] = self._apply_player_action(int(action))
        self._update_projectiles(events)
        self._update_enemies(events)
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

    # ------------------------------------------------------------------
    # Player controls
    # ------------------------------------------------------------------

    def _apply_player_action(self, action: int) -> bool:
        shot_fired = False

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
                shot_fired = self._fire_projectile(auto_aim=True)
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
                shot_fired = self._fire_projectile(auto_aim=False)

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
        return shot_fired

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
                    enemy.health -= projectile.damage
                    hit = True
                    if enemy.health <= 0.0:
                        destroyed_enemy_ids.add(enemy.entity_id)
                    break

            if not hit:
                for spawner in self.spawners:
                    if spawner.entity_id in destroyed_spawner_ids:
                        continue
                    if circles_overlap(projectile, spawner):
                        spawner.health -= projectile.damage
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
        if not targets:
            return 0.0, 0.0, 1.0, 0.0
        target = min(
            targets,
            key=lambda item: (item.x - self.player.x) ** 2
            + (item.y - self.player.y) ** 2,
        )
        dx = (target.x - self.player.x) / self.width
        playfield_height = self.height - self.playfield_top
        dy = (target.y - self.player.y) / playfield_height
        diagonal = math.hypot(self.width, self.height)
        distance = math.hypot(target.x - self.player.x, target.y - self.player.y) / diagonal
        health = max(0.0, target.health / target.max_health)
        return dx, dy, distance, health

    def _get_observation(self) -> np.ndarray:
        max_speed = max(
            float(self.player_cfg["max_speed"]), float(self.player_cfg["direct_speed"])
        )
        enemy_features = self._target_features(self.enemies)
        spawner_features = self._target_features(self.spawners)
        fire_cooldown = max(1, int(float(self.player_cfg["fire_cooldown_seconds"]) * self.fps))

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
            ],
            dtype=np.float32,
        )
        return observation

    def _calculate_reward(self, events: dict[str, Any], terminated: bool) -> float:
        """Provide configurable event rewards; detailed tuning is a later task."""

        reward = float(self.reward_cfg["survival"])
        reward += events["enemies_destroyed"] * float(self.reward_cfg["enemy_destroyed"])
        reward += events["spawners_destroyed"] * float(self.reward_cfg["spawner_destroyed"])
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
        }


__all__ = ["ArenaEnv", "DIRECT_ACTIONS", "ROTATION_ACTIONS"]
