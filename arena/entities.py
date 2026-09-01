"""Lightweight data objects used by the action-arena simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Body:
    """A circular body represented in continuous pixel coordinates."""

    x: float
    y: float
    radius: float


@dataclass
class Player(Body):
    angle: float
    max_health: float
    health: float
    vx: float = 0.0
    vy: float = 0.0
    fire_cooldown_steps: int = 0
    level: int = 1
    xp: float = 0.0


@dataclass
class Enemy(Body):
    entity_id: int
    max_health: float
    health: float
    speed: float
    vx: float = 0.0
    vy: float = 0.0
    attack_cooldown_steps: int = 0
    is_elite: bool = False
    is_miniboss: bool = False
    is_boss_defender: bool = False
    defender_kind: str = ""
    orbit_angle: float = 0.0
    missile_cooldown_steps: int = 0


@dataclass
class Spawner(Body):
    entity_id: int
    max_health: float
    health: float
    spawn_cooldown_steps: int
    is_boss: bool = False
    max_shield: float = 0.0
    shield: float = 0.0
    summons_used: int = 0
    summon_cooldown_steps: int = 0
    vx: float = 0.0
    vy: float = 0.0
    defender_wave_started: bool = False
    defender_regen_cap: float = 0.0


@dataclass
class Projectile(Body):
    entity_id: int
    vx: float
    vy: float
    damage: float
    lifetime_steps: int
    weapon_kind: str = "pulse"
    owner: str = "player"
    pierces_remaining: int = 0
    splash_radius: float = 0.0
    is_critical: bool = False
    telegraph_steps: int = 0
    homing_turn_rate: float = 0.0
    guidance_steps: int = 0
    hit_entity_ids: set[int] = field(default_factory=set)


@dataclass
class DangerZone:
    """Telegraphed boss attack resolved in continuous arena coordinates."""

    kind: str
    x: float
    y: float
    angle: float = 0.0
    radius: float = 0.0
    half_width: float = 0.0
    half_length: float = 0.0
    telegraph_steps: int = 1
    active_steps: int = 1
    maximum_telegraph_steps: int = 1
    damage: float = 0.0
    attack_id: int = 0
    attack_name: str = "BOSS ATTACK"
    triggered: bool = False
    hit_player: bool = False


def circles_overlap(first: Body, second: Body) -> bool:
    """Return whether two circular bodies intersect."""

    dx = first.x - second.x
    dy = first.y - second.y
    radii = first.radius + second.radius
    return dx * dx + dy * dy <= radii * radii


__all__ = [
    "Body",
    "Player",
    "Enemy",
    "Spawner",
    "Projectile",
    "DangerZone",
    "circles_overlap",
]
