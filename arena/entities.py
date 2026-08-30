"""Lightweight data objects used by the action-arena simulation."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class Spawner(Body):
    entity_id: int
    max_health: float
    health: float
    spawn_cooldown_steps: int


@dataclass
class Projectile(Body):
    entity_id: int
    vx: float
    vy: float
    damage: float
    lifetime_steps: int
    weapon_kind: str = "pulse"


def circles_overlap(first: Body, second: Body) -> bool:
    """Return whether two circular bodies intersect."""

    dx = first.x - second.x
    dy = first.y - second.y
    radii = first.radius + second.radius
    return dx * dx + dy * dy <= radii * radii
