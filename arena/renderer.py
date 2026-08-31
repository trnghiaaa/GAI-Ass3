"""Pygame renderer for the continuous Part II action arena."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import pygame

if TYPE_CHECKING:
    from arena.environment import ArenaEnv


COLORS = {
    "space": (7, 10, 25),
    "nebula": (18, 25, 52),
    "star": (150, 185, 225),
    "player": (65, 210, 255),
    "player_core": (224, 250, 255),
    "thrust": (255, 154, 54),
    "enemy": (241, 73, 91),
    "enemy_dark": (104, 22, 44),
    "spawner": (181, 82, 255),
    "spawner_core": (255, 116, 226),
    "projectile": (255, 235, 99),
    "health": (63, 220, 135),
    "health_low": (244, 71, 88),
    "bar_bg": (38, 45, 68),
    "hud": (13, 18, 38),
    "text": (232, 241, 255),
    "muted": (150, 167, 196),
    "accent": (106, 173, 255),
}


class ArenaRenderer:
    """Draw the live environment in a window or an off-screen RGB surface."""

    def __init__(self, env: "ArenaEnv", mode: str = "human") -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError("mode must be 'human' or 'rgb_array'")

        self.env = env
        self.mode = mode
        self.close_requested = False
        pygame.init()
        pygame.font.init()

        size = (env.width, env.height)
        if mode == "human":
            self.surface = pygame.display.set_mode(size)
            pygame.display.set_caption(f"Neon Rift Arena — {env.control_style.title()} Controls")
        else:
            self.surface = pygame.Surface(size)

        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas", 15)
        self.font_medium = pygame.font.SysFont("consolas", 21, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 38, bold=True)

        star_rng = random.Random(8071)
        self.stars = [
            (
                star_rng.randrange(0, env.width),
                star_rng.randrange(0, env.height),
                star_rng.choice((1, 1, 1, 2)),
                star_rng.randrange(110, 220),
            )
            for _ in range(95)
        ]

    def render(
        self,
        *,
        process_events: bool = True,
        footer_text: str | None = None,
    ) -> np.ndarray | None:
        if process_events and self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close_requested = True

        self._draw_background()
        self._draw_spawners()
        self._draw_projectiles()
        self._draw_enemies()
        self._draw_player()
        self._draw_hud(footer_text)

        if self.env.phase_transition_steps > 0 and not self.env.done:
            self._draw_phase_banner()
        if self.env.done:
            self._draw_episode_end()

        if self.mode == "human":
            pygame.display.flip()
            return None

        frame = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame, (1, 0, 2)).copy()

    def _draw_background(self) -> None:
        self.surface.fill(COLORS["space"])

        # Soft geometric nebula bands give a spatial feel without resembling a
        # tile grid.
        nebula = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        pygame.draw.ellipse(
            nebula,
            (*COLORS["nebula"], 100),
            (-self.env.width // 5, self.env.height // 4, self.env.width, self.env.height),
        )
        pygame.draw.ellipse(
            nebula,
            (44, 20, 72, 55),
            (self.env.width // 2, -self.env.height // 3, self.env.width, self.env.height),
        )
        self.surface.blit(nebula, (0, 0))

        shimmer = self.env.step_count % 90
        for index, (x, y, radius, brightness) in enumerate(self.stars):
            pulse = 20 if (index * 13 + shimmer) % 90 < 4 else 0
            color = tuple(min(255, component + pulse) for component in COLORS["star"])
            scaled = tuple(component * brightness // 220 for component in color)
            pygame.draw.circle(self.surface, scaled, (x, y), radius)

    def _draw_player(self) -> None:
        player = self.env.player
        angle = player.angle
        nose = (player.x + math.cos(angle) * 24, player.y + math.sin(angle) * 24)
        left = (
            player.x + math.cos(angle + 2.45) * 18,
            player.y + math.sin(angle + 2.45) * 18,
        )
        right = (
            player.x + math.cos(angle - 2.45) * 18,
            player.y + math.sin(angle - 2.45) * 18,
        )

        glow = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*COLORS["player"], 35), (32, 32), 29)
        self.surface.blit(glow, (int(player.x - 32), int(player.y - 32)))

        speed = math.hypot(player.vx, player.vy)
        if speed > 15:
            tail = (
                player.x - math.cos(angle) * (23 + min(speed / 18, 12)),
                player.y - math.sin(angle) * (23 + min(speed / 18, 12)),
            )
            pygame.draw.line(self.surface, COLORS["thrust"], left, tail, 4)
            pygame.draw.line(self.surface, COLORS["thrust"], right, tail, 4)

        pygame.draw.polygon(self.surface, COLORS["player"], (nose, left, right))
        pygame.draw.polygon(self.surface, COLORS["player_core"], (nose, left, right), 2)
        pygame.draw.circle(
            self.surface, COLORS["player_core"], (round(player.x), round(player.y)), 4
        )

    def _draw_enemies(self) -> None:
        for enemy in self.env.enemies:
            center = (round(enemy.x), round(enemy.y))
            glow = pygame.Surface((48, 48), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*COLORS["enemy"], 35), (24, 24), 22)
            self.surface.blit(glow, (center[0] - 24, center[1] - 24))

            pygame.draw.circle(self.surface, COLORS["enemy_dark"], center, round(enemy.radius))
            pygame.draw.circle(self.surface, COLORS["enemy"], center, round(enemy.radius), 3)
            heading = math.atan2(enemy.vy, enemy.vx)
            eye = (
                round(enemy.x + math.cos(heading) * 7),
                round(enemy.y + math.sin(heading) * 7),
            )
            pygame.draw.circle(self.surface, COLORS["player_core"], eye, 3)
            self._draw_health_bar(
                enemy.x - 18,
                enemy.y - enemy.radius - 10,
                36,
                enemy.health / enemy.max_health,
                height=4,
            )

    def _draw_spawners(self) -> None:
        for spawner in self.env.spawners:
            center = (round(spawner.x), round(spawner.y))
            pulse = 2.0 + math.sin(self.env.step_count * 0.12 + spawner.entity_id) * 2.0
            pygame.draw.circle(
                self.surface,
                COLORS["spawner"],
                center,
                round(spawner.radius + 8 + pulse),
                2,
            )

            rotation = self.env.step_count * 0.02 + spawner.entity_id
            points = [
                (
                    spawner.x + math.cos(rotation + index * math.tau / 6) * spawner.radius,
                    spawner.y + math.sin(rotation + index * math.tau / 6) * spawner.radius,
                )
                for index in range(6)
            ]
            pygame.draw.polygon(self.surface, (45, 25, 75), points)
            pygame.draw.polygon(self.surface, COLORS["spawner"], points, 3)
            pygame.draw.circle(self.surface, COLORS["spawner_core"], center, 8)
            pygame.draw.circle(self.surface, COLORS["player_core"], center, 3)
            self._draw_health_bar(
                spawner.x - 28,
                spawner.y - spawner.radius - 15,
                56,
                spawner.health / spawner.max_health,
                height=6,
            )

    def _draw_projectiles(self) -> None:
        for projectile in self.env.projectiles:
            center = (round(projectile.x), round(projectile.y))
            speed = math.hypot(projectile.vx, projectile.vy)
            if speed > 1e-8:
                tail = (
                    round(projectile.x - projectile.vx / speed * 18),
                    round(projectile.y - projectile.vy / speed * 18),
                )
                pygame.draw.line(self.surface, (255, 154, 54), tail, center, 7)
                pygame.draw.line(self.surface, COLORS["projectile"], tail, center, 3)
            pygame.draw.circle(self.surface, (255, 184, 54), center, 7)
            pygame.draw.circle(self.surface, COLORS["projectile"], center, 4)

    def _draw_hud(self, footer_text: str | None) -> None:
        panel = pygame.Surface((self.env.width, 74), pygame.SRCALPHA)
        panel.fill((*COLORS["hud"], 225))
        self.surface.blit(panel, (0, 0))

        self._text(f"PHASE {self.env.phase}", 18, 12, self.font_medium, COLORS["accent"])
        time_remaining = max(0.0, (self.env.max_steps - self.env.step_count) / self.env.fps)
        self._text(
            f"TIME {time_remaining:05.1f}s", 18, 42, self.font_small, COLORS["muted"]
        )

        self._text("HULL", 155, 15, self.font_small, COLORS["muted"])
        health_ratio = self.env.player.health / self.env.player.max_health
        self._draw_health_bar(155, 40, 205, health_ratio, height=13)
        self._text(
            f"{max(0, math.ceil(self.env.player.health)):3d}",
            368,
            35,
            self.font_small,
            COLORS["text"],
        )

        stats = f"SPAWNERS {len(self.env.spawners)}   HOSTILES {len(self.env.enemies)}"
        stats_surface = self.font_medium.render(stats, True, COLORS["text"])
        self.surface.blit(stats_surface, (self.env.width - stats_surface.get_width() - 18, 14))
        mode = f"{self.env.control_style.upper()} CONTROL"
        mode_surface = self.font_small.render(mode, True, COLORS["muted"])
        self.surface.blit(mode_surface, (self.env.width - mode_surface.get_width() - 18, 45))

        if footer_text:
            footer = pygame.Surface((self.env.width, 28), pygame.SRCALPHA)
            footer.fill((*COLORS["hud"], 205))
            self.surface.blit(footer, (0, self.env.height - 28))
            text_surface = self.font_small.render(footer_text, True, COLORS["muted"])
            self.surface.blit(
                text_surface,
                ((self.env.width - text_surface.get_width()) // 2, self.env.height - 22),
            )

    def _draw_health_bar(
        self, x: float, y: float, width: int, ratio: float, *, height: int
    ) -> None:
        ratio = max(0.0, min(1.0, ratio))
        rect = pygame.Rect(round(x), round(y), width, height)
        pygame.draw.rect(self.surface, COLORS["bar_bg"], rect, border_radius=height // 2)
        fill_width = round(width * ratio)
        if fill_width > 0:
            fill_color = COLORS["health"] if ratio > 0.3 else COLORS["health_low"]
            fill = pygame.Rect(rect.x, rect.y, fill_width, height)
            pygame.draw.rect(self.surface, fill_color, fill, border_radius=height // 2)

    def _draw_phase_banner(self) -> None:
        remaining = self.env.phase_transition_steps / self.env.fps
        banner = pygame.Surface((390, 92), pygame.SRCALPHA)
        banner.fill((10, 13, 31, 220))
        pygame.draw.rect(banner, COLORS["spawner"], banner.get_rect(), 2, border_radius=12)
        title = self.font_large.render(f"PHASE {self.env.phase}", True, COLORS["text"])
        subtitle = self.font_small.render(
            f"Rift signatures arriving in {remaining:0.1f}s", True, COLORS["muted"]
        )
        banner.blit(title, ((banner.get_width() - title.get_width()) // 2, 8))
        banner.blit(subtitle, ((banner.get_width() - subtitle.get_width()) // 2, 61))
        self.surface.blit(
            banner,
            (
                (self.env.width - banner.get_width()) // 2,
                (self.env.height - banner.get_height()) // 2,
            ),
        )

    def _draw_episode_end(self) -> None:
        overlay = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        overlay.fill((4, 6, 16, 185))
        self.surface.blit(overlay, (0, 0))
        title_text = "SHIP DESTROYED" if self.env.last_end_reason == "player_destroyed" else "TIME LIMIT"
        title = self.font_large.render(title_text, True, COLORS["text"])
        subtitle = self.font_medium.render(
            f"Reached phase {self.env.phase}  •  Press R to restart",
            True,
            COLORS["accent"],
        )
        self.surface.blit(
            title,
            ((self.env.width - title.get_width()) // 2, self.env.height // 2 - 45),
        )
        self.surface.blit(
            subtitle,
            ((self.env.width - subtitle.get_width()) // 2, self.env.height // 2 + 10),
        )

    def _text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        self.surface.blit(font.render(text, True, color), (x, y))

    def close(self) -> None:
        pygame.quit()


__all__ = ["ArenaRenderer"]
