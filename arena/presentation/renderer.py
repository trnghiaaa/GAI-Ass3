"""Pygame renderer for the continuous Part II action arena."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import pygame

from arena.presentation.audio import ArenaAudio

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
    "boss": (255, 84, 176),
    "miniboss": (255, 154, 54),
    "defender": (255, 196, 72),
    "missile": (255, 62, 94),
    "missile_dark": (116, 20, 49),
    "allied_fire": (76, 220, 255),
    "hazard": (255, 67, 112),
    "hazard_safe": (255, 195, 67),
    "drone": (78, 229, 181),
    "projectile": (255, 235, 99),
    "laser": (82, 242, 255),
    "nova": (141, 112, 255),
    "xp": (78, 229, 181),
    "health": (63, 220, 135),
    "health_low": (244, 71, 88),
    "bar_bg": (38, 45, 68),
    "hud": (13, 18, 38),
    "text": (232, 241, 255),
    "muted": (184, 199, 222),
    "accent": (106, 173, 255),
}


class ArenaRenderer:
    """Draw the live environment in a window or an off-screen RGB surface."""

    def __init__(
        self,
        env: "ArenaEnv",
        mode: str = "human",
        audio: ArenaAudio | None = None,
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError("mode must be 'human' or 'rgb_array'")

        self.env = env
        self.mode = mode
        self.audio = audio if audio is not None else (ArenaAudio() if mode == "human" else None)
        if self.audio and self.audio.available:
            self.audio.start_music()
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
        # Bahnschrift has clean, wide counters at game-HUD sizes. Pygame falls
        # back to its default font on platforms where it is unavailable.
        self.font_tiny = pygame.font.SysFont("bahnschrift", 13, bold=True)
        self.font_small = pygame.font.SysFont("bahnschrift", 15)
        self.font_medium = pygame.font.SysFont("bahnschrift", 21, bold=True)
        self.font_large = pygame.font.SysFont("bahnschrift", 40, bold=True)
        self.effect_rng = random.Random(4207)
        self.particles: list[dict[str, object]] = []
        self.last_effect_step = -1
        self.show_build_panel = False
        self.pause_button_rect = pygame.Rect(env.width - 108, env.height - 29, 92, 24)
        self.episode_end_has_next = False

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
        paused: bool = False,
    ) -> np.ndarray | None:
        if process_events and self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close_requested = True

        self._sync_effects()
        self._draw_background()
        self._draw_danger_zones()
        self._draw_spawners()
        self._draw_projectiles()
        self._draw_enemies()
        self._draw_targeting_reticle()
        self._draw_player()
        self._draw_support_drone()
        self._draw_particles()
        self._draw_progression_fx()
        self._draw_vignette()
        self._draw_hud(footer_text)
        self._draw_projectile_legend()
        self._draw_pause_button(paused)

        if (
            self.env.phase_transition_steps > 0
            and self.env.pending_choice_kind is None
            and not self.env.done
        ):
            self._draw_phase_banner()
        if (
            self.env.upgrade_banner_steps > 0
            and self.env.pending_choice_kind is None
            and not self.env.done
        ):
            self._draw_upgrade_banner()
        if self.show_build_panel and self.env.pending_choice_kind is None and not self.env.done:
            self._draw_build_panel()
        if self.env.pending_choice_kind is not None and not self.env.done:
            self._draw_choice_overlay()
        elif paused and not self.env.done and not self.show_build_panel:
            self._draw_pause_overlay()
        if self.env.done:
            self._draw_episode_end()

        if self.mode == "human":
            pygame.display.flip()
            return None

        frame = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame, (1, 0, 2)).copy()

    def pause_at_position(self, position: tuple[int, int]) -> bool:
        """Return whether a click targets the always-visible pause control."""

        return self.pause_button_rect.collidepoint(position)

    def episode_end_button_rects(self) -> dict[str, pygame.Rect]:
        """Return stable click targets for the mission-summary actions."""

        if self.episode_end_has_next:
            return {
                "replay": pygame.Rect(151, 402, 150, 46),
                "next": pygame.Rect(325, 402, 150, 46),
                "menu": pygame.Rect(499, 402, 150, 46),
            }
        return {
            "replay": pygame.Rect(238, 402, 150, 46),
            "menu": pygame.Rect(412, 402, 150, 46),
        }

    def episode_end_action_at_position(
        self, position: tuple[int, int]
    ) -> str | None:
        """Resolve a mission-summary mouse click without consuming events."""

        return next(
            (
                action
                for action, rect in self.episode_end_button_rects().items()
                if rect.collidepoint(position)
            ),
            None,
        )

    def _draw_pause_button(self, paused: bool) -> None:
        """Draw a compact mouse-accessible PAUSE / RESUME control."""

        rect = self.pause_button_rect
        accent = COLORS["xp"] if paused else COLORS["accent"]
        shadow = rect.move(0, 3)
        pygame.draw.rect(self.surface, (4, 7, 18), shadow, border_radius=7)
        pygame.draw.rect(self.surface, COLORS["hud"], rect, border_radius=7)
        pygame.draw.rect(self.surface, accent, rect, width=1, border_radius=7)
        icon_x = rect.x + 14
        if paused:
            pygame.draw.polygon(
                self.surface,
                accent,
                ((icon_x - 2, rect.centery - 5), (icon_x + 6, rect.centery), (icon_x - 2, rect.centery + 5)),
            )
        else:
            pygame.draw.rect(self.surface, accent, (icon_x - 3, rect.centery - 5, 3, 10))
            pygame.draw.rect(self.surface, accent, (icon_x + 3, rect.centery - 5, 3, 10))
        label = self.font_tiny.render("RESUME" if paused else "PAUSE", True, COLORS["text"])
        self.surface.blit(label, (rect.x + 29, rect.centery - label.get_height() // 2))

    def _draw_pause_overlay(self) -> None:
        veil = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        veil.fill((3, 7, 20, 182))
        self.surface.blit(veil, (0, 0))
        panel = pygame.Rect(210, 205, 380, 170)
        pygame.draw.rect(self.surface, (14, 22, 47), panel, border_radius=18)
        pygame.draw.rect(self.surface, COLORS["xp"], panel, width=2, border_radius=18)
        title = self.font_large.render("PAUSED", True, COLORS["text"])
        self.surface.blit(title, title.get_rect(center=(panel.centerx, panel.y + 55)))
        subtitle = self.font_small.render(
            "Press P or click RESUME when you are ready.", True, COLORS["muted"]
        )
        self.surface.blit(subtitle, subtitle.get_rect(center=(panel.centerx, panel.y + 108)))
        hint = self.font_tiny.render(
            "The phase timer and every enemy are frozen.", True, COLORS["xp"]
        )
        self.surface.blit(hint, hint.get_rect(center=(panel.centerx, panel.y + 137)))
        # The veil covers the ordinary HUD, so redraw the active exit control
        # on top and keep the mouse path back to play visually obvious.
        self._draw_pause_button(True)

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
        parallax_x = int((self.env.player.x - self.env.width / 2) * 0.012)
        parallax_y = int((self.env.player.y - self.env.height / 2) * 0.012)
        for index, (x, y, radius, brightness) in enumerate(self.stars):
            pulse = 20 if (index * 13 + shimmer) % 90 < 4 else 0
            color = tuple(min(255, component + pulse) for component in COLORS["star"])
            scaled = tuple(component * brightness // 220 for component in color)
            pygame.draw.circle(
                self.surface,
                scaled,
                ((x - parallax_x) % self.env.width, (y - parallax_y) % self.env.height),
                radius,
            )

    def _draw_danger_zones(self) -> None:
        """Render boss attacks as strong telegraphs before their damage frame."""

        labeled_attacks: set[int] = set()
        for zone in self.env.danger_zones:
            layer = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
            telegraphing = zone.telegraph_steps > 0
            progress = (
                1.0 - zone.telegraph_steps / max(1, zone.maximum_telegraph_steps)
                if telegraphing
                else 1.0
            )
            pulse = 0.5 + 0.5 * math.sin(self.env.step_count * 0.34)
            color = COLORS["hazard_safe"] if telegraphing else COLORS["hazard"]
            alpha = int(28 + 58 * progress + (22 * pulse if telegraphing else 90))
            outline = 2 + int(3 * progress)
            if zone.kind == "circle":
                center = (round(zone.x), round(zone.y))
                radius = max(1, round(zone.radius))
                pygame.draw.circle(layer, (*color, alpha), center, radius)
                pygame.draw.circle(layer, (*color, 245), center, radius, outline)
                if telegraphing:
                    inner = max(4, round(radius * progress))
                    pygame.draw.circle(layer, (*color, 210), center, inner, 2)
            else:
                along_x = math.cos(zone.angle) * zone.half_length
                along_y = math.sin(zone.angle) * zone.half_length
                across_x = -math.sin(zone.angle) * zone.half_width
                across_y = math.cos(zone.angle) * zone.half_width
                points = [
                    (zone.x - along_x - across_x, zone.y - along_y - across_y),
                    (zone.x + along_x - across_x, zone.y + along_y - across_y),
                    (zone.x + along_x + across_x, zone.y + along_y + across_y),
                    (zone.x - along_x + across_x, zone.y - along_y + across_y),
                ]
                pygame.draw.polygon(layer, (*color, alpha), points)
                pygame.draw.polygon(layer, (*color, 235), points, outline)
                if telegraphing:
                    pygame.draw.line(
                        layer,
                        (*color, 245),
                        (zone.x - along_x, zone.y - along_y),
                        (zone.x + along_x, zone.y + along_y),
                        2,
                    )
            self.surface.blit(layer, (0, 0))

            if telegraphing and zone.attack_id not in labeled_attacks:
                labeled_attacks.add(zone.attack_id)
                seconds = zone.telegraph_steps / self.env.fps
                message = f"{zone.attack_name}  •  MOVE  •  {seconds:.1f}s"
                label = self._fit_text(
                    message, self.font_small, 310, COLORS["text"]
                )
                label_box = pygame.Rect(
                    (self.env.width - min(330, label.get_width() + 28)) // 2,
                    round(self.env.playfield_top + 14),
                    min(330, label.get_width() + 28),
                    34,
                )
                pill = pygame.Surface(label_box.size, pygame.SRCALPHA)
                pill.fill((18, 10, 28, 220))
                pygame.draw.rect(
                    pill,
                    COLORS["hazard_safe"],
                    pill.get_rect(),
                    2,
                    border_radius=12,
                )
                pill.blit(label, label.get_rect(center=pill.get_rect().center))
                self.surface.blit(pill, label_box.topleft)

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
        shield_alpha = 55 + int(20 * math.sin(self.env.step_count * 0.08))
        shield = pygame.Surface((52, 52), pygame.SRCALPHA)
        pygame.draw.circle(shield, (*COLORS["player"], shield_alpha), (26, 26), 24, 1)
        self.surface.blit(shield, (round(player.x) - 26, round(player.y) - 26))
        for charge in range(min(3, self.env.barrier_charges)):
            pygame.draw.circle(
                self.surface,
                COLORS["xp"],
                (round(player.x), round(player.y)),
                27 + charge * 3,
                1,
            )

    def _draw_support_drone(self) -> None:
        if not self.env.support_drone_active:
            return
        for drone_index in range(self.env.drone_count):
            orbit = (
                self.env.step_count * 0.055
                + math.tau * drone_index / self.env.drone_count
            )
            orbit_radius = 34.0 + 8.0 * (drone_index % 2)
            x = self.env.player.x + math.cos(orbit) * orbit_radius
            y = self.env.player.y + math.sin(orbit) * orbit_radius
            center = (round(x), round(y))
            pygame.draw.circle(self.surface, COLORS["drone"], center, 8)
            pygame.draw.circle(self.surface, COLORS["player_core"], center, 3)
            pygame.draw.arc(
                self.surface,
                COLORS["drone"],
                pygame.Rect(center[0] - 13, center[1] - 13, 26, 26),
                self.env.step_count * 0.08,
                self.env.step_count * 0.08 + math.pi,
                2,
            )

    def _draw_enemies(self) -> None:
        for enemy in self.env.enemies:
            center = (round(enemy.x), round(enemy.y))
            enemy_color = (
                COLORS["defender"]
                if enemy.is_boss_defender
                else (
                    COLORS["miniboss"]
                    if enemy.is_miniboss
                    else (COLORS["boss"] if enemy.is_elite else COLORS["enemy"])
                )
            )
            glow_size = 76 if enemy.is_miniboss else (58 if enemy.is_boss_defender else 48)
            glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(
                glow, (*enemy_color, 42), (glow_size // 2, glow_size // 2), glow_size // 2 - 2
            )
            self.surface.blit(
                glow, (center[0] - glow_size // 2, center[1] - glow_size // 2)
            )

            if enemy.is_boss_defender and enemy.defender_kind == "turret":
                body = pygame.Rect(0, 0, round(enemy.radius * 1.7), round(enemy.radius * 1.7))
                body.center = center
                pygame.draw.rect(self.surface, COLORS["enemy_dark"], body, border_radius=4)
                pygame.draw.rect(self.surface, enemy_color, body, 3, border_radius=4)
                pygame.draw.circle(self.surface, enemy_color, center, 5, 2)
            else:
                pygame.draw.circle(self.surface, COLORS["enemy_dark"], center, round(enemy.radius))
                pygame.draw.circle(self.surface, enemy_color, center, round(enemy.radius), 3)
            if enemy.is_boss_defender:
                # A compact in-body glyph stays readable at arena edges and
                # avoids overlapping labels when several sentries orbit close.
                glyph = self.font_tiny.render(
                    "T" if enemy.defender_kind == "turret" else "I",
                    True,
                    COLORS["player_core"],
                )
                self.surface.blit(glyph, glyph.get_rect(center=center))
            else:
                heading = math.atan2(enemy.vy, enemy.vx)
                eye = (
                    round(enemy.x + math.cos(heading) * 7),
                    round(enemy.y + math.sin(heading) * 7),
                )
                pygame.draw.circle(self.surface, COLORS["player_core"], eye, 3)
            if enemy.is_miniboss:
                label = self.font_tiny.render("RIFT HUNTER", True, COLORS["miniboss"])
                self.surface.blit(
                    label,
                    label.get_rect(center=(center[0], center[1] + enemy.radius + 18)),
                )
            self._draw_health_bar(
                enemy.x - (38 if enemy.is_miniboss else 18),
                enemy.y - enemy.radius - 10,
                76 if enemy.is_miniboss else 36,
                enemy.health / enemy.max_health,
                height=4,
            )

    def _draw_spawners(self) -> None:
        for spawner in self.env.spawners:
            center = (round(spawner.x), round(spawner.y))
            spawner_color = COLORS["boss"] if spawner.is_boss else COLORS["spawner"]
            pulse = 2.0 + math.sin(self.env.step_count * 0.12 + spawner.entity_id) * 2.0
            pygame.draw.circle(
                self.surface,
                spawner_color,
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
            pygame.draw.polygon(self.surface, spawner_color, points, 4 if spawner.is_boss else 3)
            pygame.draw.circle(self.surface, COLORS["spawner_core"], center, 8)
            pygame.draw.circle(self.surface, COLORS["player_core"], center, 3)
            if spawner.is_boss:
                channeling = any(
                    zone.telegraph_steps > 0 for zone in self.env.danger_zones
                )
                if channeling:
                    pygame.draw.circle(
                        self.surface,
                        COLORS["hazard_safe"],
                        center,
                        round(spawner.radius + 14 + pulse),
                        3,
                    )
                if self.env.boss_intermission_active:
                    pygame.draw.circle(
                        self.surface,
                        COLORS["defender"],
                        center,
                        round(spawner.radius + 18 + pulse),
                        4,
                    )
            bar_width = 100 if spawner.is_boss else 56
            if spawner.is_boss and spawner.max_shield > 0.0:
                shield_rect = pygame.Rect(round(spawner.x - 50), max(self.env.playfield_top + 5, round(spawner.y - spawner.radius - 25)), 100, 5)
                pygame.draw.rect(self.surface, (25, 45, 65), shield_rect)
                fill = shield_rect.copy()
                fill.width = round(100 * max(0.0, spawner.shield / spawner.max_shield))
                pygame.draw.rect(self.surface, (80, 200, 255), fill)
                if self.env.boss_intermission_active:
                    boss_status = f"AEGIS ACTIVE  •  SENTRIES {len(self.env.boss_defenders)}"
                    status_color = COLORS["defender"]
                else:
                    boss_status = (
                        f"SHIELD {max(0, round(spawner.shield))}  •  "
                        f"SUMMONS {spawner.summons_used}/{self.env.boss_summon_limit_for_phase()}"
                    )
                    status_color = (120, 215, 255)
                shield_label = self.font_tiny.render(
                    boss_status,
                    True,
                    status_color,
                )
                label_rect = shield_label.get_rect(
                    center=(center[0], center[1] + spawner.radius + 61)
                )
                label_rect.clamp_ip(self.surface.get_rect().inflate(-12, -12))
                # A small backing plate keeps the dense boss telemetry legible
                # over beams, wingmen, and the animated arena background.
                backing = label_rect.inflate(10, 6)
                pygame.draw.rect(self.surface, (8, 20, 40), backing, border_radius=4)
                pygame.draw.rect(
                    self.surface, (45, 104, 150), backing, width=1, border_radius=4
                )
                self.surface.blit(shield_label, label_rect)
            self._draw_health_bar(
                spawner.x - bar_width / 2,
                max(self.env.playfield_top + 15, spawner.y - spawner.radius - 15),
                bar_width,
                spawner.health / spawner.max_health,
                height=6,
            )

    def _draw_projectiles(self) -> None:
        for projectile in self.env.projectiles:
            center = (round(projectile.x), round(projectile.y))
            if projectile.owner == "enemy" and projectile.weapon_kind == "enemy_missile":
                if projectile.telegraph_steps > 0:
                    pulse = 9 + int(3 * math.sin(self.env.step_count * 0.45))
                    pygame.draw.line(
                        self.surface,
                        COLORS["missile"],
                        center,
                        (round(self.env.player.x), round(self.env.player.y)),
                        2,
                    )
                    pygame.draw.circle(self.surface, COLORS["missile"], center, pulse, 2)
                    pygame.draw.circle(self.surface, COLORS["missile"], center, 4)
                else:
                    angle = math.atan2(projectile.vy, projectile.vx)
                    rear = (
                        round(projectile.x - math.cos(angle) * 15),
                        round(projectile.y - math.sin(angle) * 15),
                    )
                    pygame.draw.line(self.surface, COLORS["missile_dark"], rear, center, 7)
                    pygame.draw.line(self.surface, COLORS["missile"], rear, center, 3)
                    nose = (
                        round(projectile.x + math.cos(angle) * 10),
                        round(projectile.y + math.sin(angle) * 10),
                    )
                    left = (
                        round(projectile.x + math.cos(angle + 2.45) * 8),
                        round(projectile.y + math.sin(angle + 2.45) * 8),
                    )
                    right = (
                        round(projectile.x + math.cos(angle - 2.45) * 8),
                        round(projectile.y + math.sin(angle - 2.45) * 8),
                    )
                    pygame.draw.polygon(self.surface, COLORS["missile"], (nose, left, right))
                    pygame.draw.circle(self.surface, (255, 238, 244), center, 2)
                continue
            speed = max(1.0, math.hypot(projectile.vx, projectile.vy))
            is_beam = projectile.weapon_kind in ("laser", "nova")
            tail_length = 30 if is_beam else 15
            tail = (
                round(projectile.x - projectile.vx / speed * tail_length),
                round(projectile.y - projectile.vy / speed * tail_length),
            )
            if projectile.weapon_kind == "drone":
                outer, core = COLORS["drone"], COLORS["player_core"]
            elif projectile.weapon_kind == "nova":
                outer, core = COLORS["nova"], COLORS["player_core"]
            elif projectile.weapon_kind == "laser":
                outer, core = COLORS["laser"], COLORS["player_core"]
            elif projectile.weapon_kind == "rapid":
                outer, core = (80, 170, 255), (235, 250, 255)
            elif projectile.weapon_kind == "twin":
                outer, core = (80, 235, 224), (235, 250, 255)
            else:
                outer, core = COLORS["allied_fire"], (235, 250, 255)
            if projectile.is_critical:
                outer, core = (255, 198, 62), (255, 255, 255)
                pulse = 10 + int(2 * math.sin(self.env.step_count * 0.4))
                pygame.draw.circle(self.surface, outer, center, pulse, 2)
            pygame.draw.line(self.surface, outer, tail, center, 5 if is_beam else 3)
            pygame.draw.line(self.surface, core, tail, center, 2)
            pygame.draw.circle(self.surface, outer, center, 7 if is_beam else 8)
            pygame.draw.circle(self.surface, core, center, 3 if is_beam else 4)

    def _draw_projectile_legend(self) -> None:
        """Show an unobtrusive combat legend while hostile missiles exist."""

        if not self.env.boss_intermission_active and not any(
            projectile.owner == "enemy" for projectile in self.env.projectiles
        ):
            return
        panel = pygame.Surface((222, 26), pygame.SRCALPHA)
        panel.fill((8, 13, 31, 218))
        pygame.draw.rect(panel, (55, 75, 112), panel.get_rect(), 1, border_radius=7)
        pygame.draw.circle(panel, COLORS["allied_fire"], (13, 13), 4)
        panel.blit(self.font_tiny.render("CYAN: ALLY", True, COLORS["muted"]), (23, 5))
        pygame.draw.polygon(
            panel,
            COLORS["missile"],
            ((119, 8), (127, 13), (119, 18)),
        )
        panel.blit(self.font_tiny.render("RED: HOSTILE", True, COLORS["text"]), (135, 5))
        self.surface.blit(panel, (14, self.env.height - 64))

    def _draw_targeting_reticle(self) -> None:
        target = self.env._nearest_target()
        if target is None:
            return
        target_distance = math.hypot(
            target.x - self.env.player.x, target.y - self.env.player.y
        )
        assist_locked = (
            self.env.control_style == "direct"
            and target_distance
            <= float(self.env.player_cfg["target_assist_range"])
            + 20.0 * self.env.upgrade_stacks.get("capacitor", 0)
        )
        color = (
            COLORS["xp"]
            if assist_locked
            else (
                COLORS["spawner_core"]
                if target in self.env.spawners
                else COLORS["enemy"]
            )
        )
        center = (round(target.x), round(target.y))
        radius = round(target.radius + 11 + 2 * math.sin(self.env.step_count * 0.1))
        for start, end in ((-45, 35), (45, 125), (135, 215), (225, 305)):
            pygame.draw.arc(
                self.surface,
                color,
                pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2),
                math.radians(start),
                math.radians(end),
                2,
            )
        heading_x = self.env.player.x + math.cos(self.env.player.angle) * 44
        heading_y = self.env.player.y + math.sin(self.env.player.angle) * 44
        pygame.draw.circle(
            self.surface, COLORS["player"], (round(heading_x), round(heading_y)), 3, 1
        )

    def _draw_hud(self, footer_text: str | None) -> None:
        hud_height = round(self.env.playfield_top)
        panel = pygame.Surface((self.env.width, hud_height), pygame.SRCALPHA)
        panel.fill((*COLORS["hud"], 225))
        self.surface.blit(panel, (0, 0))

        phase_label = (
            f"BOSS T{self.env.boss_threat_tier} / P{self.env.phase}"
            if self.env.is_boss_phase
            else f"PHASE {self.env.phase}"
        )
        phase_color = COLORS["boss"] if self.env.is_boss_phase else COLORS["accent"]
        self._text(phase_label, 16, 10, self.font_medium, phase_color)
        time_remaining = max(
            0.0,
            (self.env.phase_max_steps - self.env.phase_step_count) / self.env.fps,
        )
        self._text(
            f"PHASE TIME  {time_remaining:04.1f}s",
            16,
            49,
            self.font_tiny,
            COLORS["muted"],
        )

        self._text("HULL", 150, 12, self.font_tiny, COLORS["muted"])
        health_ratio = self.env.player.health / self.env.player.max_health
        self._draw_health_bar(150, 47, 112, health_ratio, height=13)
        self._text(
            f"{max(0, math.ceil(self.env.player.health))}/{math.ceil(self.env.player.max_health)}",
            270,
            43,
            self.font_tiny,
            COLORS["text"],
        )

        weapon_name = str(self.env.weapon_profile()["name"]).upper()
        level_label = self._fit_text(
            f"LVL {self.env.player.level}  •  {weapon_name}",
            self.font_small,
            205,
            COLORS["xp"],
        )
        self.surface.blit(level_label, (330, 11))
        self._text("XP", 330, 49, self.font_tiny, COLORS["muted"])
        xp_rect = pygame.Rect(356, 52, 106, 9)
        pygame.draw.rect(self.surface, COLORS["bar_bg"], xp_rect, border_radius=4)
        xp_fill = round(xp_rect.width * self.env.xp_progress())
        if xp_fill > 0:
            pygame.draw.rect(
                self.surface,
                COLORS["xp"],
                pygame.Rect(xp_rect.x, xp_rect.y, xp_fill, xp_rect.height),
                border_radius=4,
            )
        xp_text = f"{math.ceil(self.env.xp_to_next_level())} TO NEXT"
        self._text(xp_text, 470, 47, self.font_tiny, COLORS["muted"])

        miniboss_count = sum(enemy.is_miniboss for enemy in self.env.enemies)
        stats = f"RIFTS {len(self.env.spawners)}   HOSTILES {len(self.env.enemies)}"
        pressure = self.env.adaptive_pressure()
        if miniboss_count:
            stats += f"   HUNTER {miniboss_count}"
        if pressure >= 0.15:
            stats += f"   SURGE {round(pressure * 100)}%"
        stats_surface = self._fit_text(
            stats,
            self.font_tiny if miniboss_count or pressure >= 0.15 else self.font_small,
            244,
            COLORS["xp"] if pressure >= 0.15 else COLORS["text"],
        )
        self.surface.blit(stats_surface, (self.env.width - stats_surface.get_width() - 16, 11))
        destroyed = int(self.env.episode_stats.get("enemies_destroyed", 0))
        rifts = int(self.env.episode_stats.get("spawners_destroyed", 0))
        mode = f"KILLS {destroyed}  •  RIFTS {rifts}"
        if self.env.drone_count:
            mode += f"  •  DRONES {self.env.drone_count}"
        if self.env.barrier_charges:
            mode += f"  •  AEGIS {self.env.barrier_charges}"
        mode_surface = self._fit_text(mode, self.font_tiny, 244, COLORS["muted"])
        self.surface.blit(mode_surface, (self.env.width - mode_surface.get_width() - 16, 49))

        if footer_text:
            footer_height = 32
            footer = pygame.Surface((self.env.width, footer_height), pygame.SRCALPHA)
            footer.fill((*COLORS["hud"], 205))
            self.surface.blit(footer, (0, self.env.height - footer_height))
            text_surface = self._fit_text(
                footer_text,
                self.font_small,
                self.env.width - self.pause_button_rect.width - 48,
                COLORS["muted"],
            )
            self.surface.blit(
                text_surface,
                (16, self.env.height - 26),
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
        banner = pygame.Surface((486, 126), pygame.SRCALPHA)
        banner.fill((10, 13, 31, 232))
        accent = COLORS["boss"] if self.env.is_boss_phase else COLORS["spawner"]
        pygame.draw.rect(banner, accent, banner.get_rect(), 2, border_radius=14)
        eyebrow_text = (
            f"BOSS TIER {self.env.boss_threat_tier}  •  HIGH THREAT"
            if self.env.is_boss_phase
            else "RIFT NETWORK RECONFIGURING"
        )
        eyebrow = self.font_tiny.render(eyebrow_text, True, accent)
        title_text = (
            f"BOSS PHASE {self.env.phase}"
            if self.env.is_boss_phase
            else f"PHASE {self.env.phase} INBOUND"
        )
        title = self._fit_text(title_text, self.font_large, 452, COLORS["text"])
        cleanup = self.env.last_phase_cleanup_count
        subtitle_text = (
            f"{cleanup} hostile{'s' if cleanup != 1 else ''} withdrew  •  deployment in {remaining:0.1f}s"
            if cleanup
            else f"Combat deployment in {remaining:0.1f}s"
        )
        subtitle = self._fit_text(
            subtitle_text, self.font_small, banner.get_width() - 28, COLORS["muted"]
        )
        banner.blit(eyebrow, ((banner.get_width() - eyebrow.get_width()) // 2, 9))
        banner.blit(title, ((banner.get_width() - title.get_width()) // 2, 29))
        banner.blit(subtitle, ((banner.get_width() - subtitle.get_width()) // 2, 83))
        duration = max(1.0, float(self.env.phase_cfg["transition_seconds"]))
        progress = float(np.clip(1.0 - remaining / duration, 0.0, 1.0))
        track = pygame.Rect(28, 111, banner.get_width() - 56, 5)
        pygame.draw.rect(banner, COLORS["bar_bg"], track, border_radius=3)
        if progress > 0:
            pygame.draw.rect(
                banner,
                accent,
                pygame.Rect(track.x, track.y, round(track.width * progress), track.height),
                border_radius=3,
            )
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
        panel = pygame.Rect(70, 168, 660, 306)
        pygame.draw.rect(self.surface, (15, 24, 48), panel, border_radius=20)
        pygame.draw.rect(self.surface, COLORS["accent"], panel, 2, border_radius=20)
        title_text = {
            "player_destroyed": "SHIP DESTROYED",
            "phase_timeout": "PHASE TIME EXPIRED",
            "safety_limit": "MISSION LIMIT REACHED",
        }.get(self.env.last_end_reason, "MISSION COMPLETE")
        title = self.font_large.render(title_text, True, COLORS["text"])
        subtitle = self.font_medium.render(
            f"Reached Phase {self.env.phase}  •  Ship Level {self.env.player.level}",
            True,
            COLORS["accent"],
        )
        self.surface.blit(
            title,
            (panel.centerx - title.get_width() // 2, panel.y + 28),
        )
        self.surface.blit(
            subtitle,
            (panel.centerx - subtitle.get_width() // 2, panel.y + 82),
        )
        stats = self.env.episode_stats
        detail = self._fit_text(
            f"Enemies {int(stats.get('enemies_destroyed', 0))}  •  "
            f"Rifts {int(stats.get('spawners_destroyed', 0))}  •  "
            f"Bosses {int(stats.get('bosses_destroyed', 0))}  •  "
            f"Reward {float(stats.get('reward', 0.0)):.1f}",
            self.font_small,
            panel.width - 56,
            COLORS["muted"],
        )
        self.surface.blit(
            detail,
            (panel.centerx - detail.get_width() // 2, panel.y + 126),
        )
        hint_text = (
            "Review this run, replay it, continue to the next seed, or return to the launcher."
            if self.episode_end_has_next
            else "Review this run, then replay it or return to the launcher."
        )
        hint = self._fit_text(
            hint_text, self.font_small, panel.width - 56, COLORS["muted"]
        )
        self.surface.blit(hint, (panel.centerx - hint.get_width() // 2, panel.y + 164))

        mouse = pygame.mouse.get_pos()
        labels = {
            "replay": ("REPLAY", "R"),
            "next": ("NEXT RUN", "N"),
            "menu": ("MAIN MENU", "ESC"),
        }
        for action, rect in self.episode_end_button_rects().items():
            hovered = self.mode == "human" and rect.collidepoint(mouse)
            accent = COLORS["xp"] if action != "menu" else COLORS["accent"]
            pygame.draw.rect(
                self.surface,
                (28, 43, 75) if hovered else COLORS["hud"],
                rect,
                border_radius=10,
            )
            pygame.draw.rect(self.surface, accent, rect, 2, border_radius=10)
            label, shortcut = labels[action]
            label_surface = self.font_small.render(label, True, COLORS["text"])
            shortcut_surface = self.font_tiny.render(shortcut, True, accent)
            label_area = pygame.Rect(
                rect.x + 8, rect.y, rect.width - shortcut_surface.get_width() - 30, rect.height
            )
            self.surface.blit(
                label_surface,
                label_surface.get_rect(center=label_area.center),
            )
            self.surface.blit(
                shortcut_surface,
                (rect.right - shortcut_surface.get_width() - 9, rect.centery - 7),
            )

    def _draw_upgrade_banner(self) -> None:
        """Show a readable unlock notification without pausing the simulation."""

        duration = max(
            1,
            int(float(self.env.progression_cfg["upgrade_banner_seconds"]) * self.env.fps),
        )
        ratio = self.env.upgrade_banner_steps / duration
        alpha = min(230, round(255 * min(1.0, ratio * 3.0)))
        banner = pygame.Surface((500, 96), pygame.SRCALPHA)
        banner.fill((8, 19, 36, alpha))
        pygame.draw.rect(banner, (*COLORS["xp"], alpha), banner.get_rect(), 2, border_radius=14)
        eyebrow = self.font_tiny.render(
            f"SHIP LEVEL {self.env.player.level}  •  BUILD UPDATED",
            True,
            COLORS["xp"],
        )
        title = self._fit_text(
            self.env.last_upgrade_name.upper(), self.font_medium, 438, COLORS["text"]
        )
        detail = self._fit_text(
            self.env.last_upgrade_detail,
            self.font_small,
            446,
            COLORS["muted"],
        )
        banner.blit(eyebrow, ((banner.get_width() - eyebrow.get_width()) // 2, 8))
        banner.blit(title, ((banner.get_width() - title.get_width()) // 2, 31))
        banner.blit(detail, ((banner.get_width() - detail.get_width()) // 2, 65))
        chevron_y = banner.get_height() // 2 + 1
        pygame.draw.polygon(banner, COLORS["xp"], ((16, chevron_y), (25, chevron_y - 7), (25, chevron_y + 7)))
        pygame.draw.polygon(
            banner,
            COLORS["xp"],
            ((banner.get_width() - 16, chevron_y), (banner.get_width() - 25, chevron_y - 7), (banner.get_width() - 25, chevron_y + 7)),
        )
        self.surface.blit(banner, ((self.env.width - banner.get_width()) // 2, 86))

    def _draw_progression_fx(self) -> None:
        """Animate phase deployment and ship upgrades in the live arena."""

        layer = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        center = (round(self.env.player.x), round(self.env.player.y))
        if self.env.phase_transition_steps > 0:
            total = max(
                1,
                int(float(self.env.phase_cfg["transition_seconds"]) * self.env.fps),
            )
            progress = float(
                np.clip(1.0 - self.env.phase_transition_steps / total, 0.0, 1.0)
            )
            accent = COLORS["boss"] if self.env.is_boss_phase else COLORS["spawner"]
            for index in range(4):
                radius = int(34 + ((progress + index * 0.22) % 1.0) * 245)
                alpha = max(0, int(150 * (1.0 - radius / 290.0)))
                pygame.draw.circle(layer, (*accent, alpha), center, radius, 2)
            for index in range(12 if self.env.is_boss_phase else 8):
                angle = index * math.tau / (12 if self.env.is_boss_phase else 8) + progress
                inner = 52 + 75 * progress
                outer = inner + (72 if self.env.is_boss_phase else 48)
                pygame.draw.line(
                    layer,
                    (*accent, 105),
                    (center[0] + math.cos(angle) * inner, center[1] + math.sin(angle) * inner),
                    (center[0] + math.cos(angle) * outer, center[1] + math.sin(angle) * outer),
                    2,
                )
            if self.env.is_boss_phase:
                edge_alpha = 42 + int(20 * math.sin(self.env.step_count * 0.28))
                pygame.draw.rect(
                    layer,
                    (*COLORS["boss"], edge_alpha),
                    layer.get_rect(),
                    width=12,
                )
        if self.env.upgrade_banner_steps > 0:
            duration = max(
                1,
                int(float(self.env.progression_cfg["upgrade_banner_seconds"]) * self.env.fps),
            )
            elapsed = 1.0 - self.env.upgrade_banner_steps / duration
            rotation = self.env.step_count * 0.11
            for index in range(3):
                radius = int(30 + ((elapsed + index * 0.3) % 1.0) * 95)
                alpha = max(0, 155 - radius)
                pygame.draw.circle(layer, (*COLORS["xp"], alpha), center, radius, 2)
            arc_rect = pygame.Rect(center[0] - 47, center[1] - 47, 94, 94)
            for index in range(3):
                pygame.draw.arc(
                    layer,
                    (*COLORS["xp"], 185),
                    arc_rect.inflate(index * 14, index * 14),
                    rotation + index * 1.7,
                    rotation + index * 1.7 + 1.0,
                    3,
                )
        self.surface.blit(layer, (0, 0))

    def choice_rects(self) -> list[pygame.Rect]:
        count = max(1, len(self.env.pending_choices))
        width, gap = 224, 18
        total = count * width + (count - 1) * gap
        start_x = (self.env.width - total) // 2
        return [pygame.Rect(start_x + index * (width + gap), 238, width, 250) for index in range(count)]

    def choice_at_position(self, position: tuple[int, int]) -> int | None:
        for index, rect in enumerate(self.choice_rects()):
            if rect.collidepoint(position):
                return index
        return None

    def _wrapped_lines(
        self, text: str, font: pygame.font.Font, max_width: int, max_lines: int
    ) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
                if len(lines) == max_lines - 1:
                    break
        if current and len(lines) < max_lines:
            lines.append(current)
        consumed = " ".join(lines)
        if len(consumed) < len(text) and lines:
            while lines[-1] and font.size(lines[-1] + "…")[0] > max_width:
                lines[-1] = lines[-1][:-1]
            lines[-1] = lines[-1].rstrip() + "…"
        return lines

    def _draw_choice_overlay(self) -> None:
        overlay = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        overlay.fill((3, 6, 18, 224))
        self.surface.blit(overlay, (0, 0))
        phase_reward = self.env.pending_choice_kind == "phase_reward"
        boss_reward = self.env.pending_choice_kind == "boss_reward"
        accent = (
            COLORS["boss"]
            if boss_reward
            else (COLORS["spawner_core"] if phase_reward else COLORS["xp"])
        )
        if boss_reward:
            eyebrow_text = "BOSS RIFT COLLAPSED  •  CLAIM ONE RELIC"
            title_text = "A worthy victory deserves lasting power"
        elif phase_reward:
            eyebrow_text = "PHASE CLEARED  •  CHOOSE ONE SUPPORT DROP"
            title_text = "Prepare for the next assault"
        else:
            eyebrow_text = f"SHIP LEVEL {self.env.player.level}  •  CHOOSE ONE UPGRADE"
            title_text = "Evolve your build"
        eyebrow = self.font_small.render(eyebrow_text, True, accent)
        title = self.font_large.render(title_text, True, COLORS["text"])
        hint = self.font_small.render("Click a card or press 1, 2, or 3. The battle timer is paused.", True, COLORS["muted"])
        self.surface.blit(eyebrow, eyebrow.get_rect(center=(self.env.width // 2, 112)))
        self.surface.blit(title, title.get_rect(center=(self.env.width // 2, 158)))
        self.surface.blit(hint, hint.get_rect(center=(self.env.width // 2, 203)))

        mouse = pygame.mouse.get_pos() if self.mode == "human" else (-1, -1)
        for index, (rect, choice) in enumerate(zip(self.choice_rects(), self.env.pending_choices)):
            details = self.env.choice_card_details(choice)
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.surface, (27, 40, 72) if hover else (17, 27, 52), rect, border_radius=16)
            pygame.draw.rect(self.surface, accent if hover else (66, 84, 122), rect, 2, border_radius=16)
            badge = pygame.Rect(rect.x + 15, rect.y + 15, 34, 34)
            pygame.draw.rect(self.surface, accent, badge, border_radius=9)
            number = self.font_medium.render(str(index + 1), True, COLORS["space"])
            self.surface.blit(number, number.get_rect(center=badge.center))
            name_lines = self._wrapped_lines(
                details["name"], self.font_medium, rect.width - 76, 2
            )
            for line_index, line in enumerate(name_lines):
                self._text(
                    line,
                    rect.x + 60,
                    rect.y + 14 + line_index * 23,
                    self.font_medium,
                    COLORS["text"],
                )

            status_y = rect.y + (62 if len(name_lines) > 1 else 55)
            status = self._fit_text(
                details["status"], self.font_tiny, rect.width - 30, accent
            )
            self.surface.blit(status, (rect.x + 15, status_y))
            divider_y = status_y + 23
            pygame.draw.line(
                self.surface,
                (58, 77, 114),
                (rect.x + 15, divider_y),
                (rect.right - 15, divider_y),
            )

            self._text("CURRENT", rect.x + 15, divider_y + 10, self.font_tiny, COLORS["muted"])
            current_lines = self._wrapped_lines(
                details["current"], self.font_small, rect.width - 30, 2
            )
            for line_index, line in enumerate(current_lines):
                self._text(
                    line,
                    rect.x + 15,
                    divider_y + 29 + line_index * 18,
                    self.font_small,
                    COLORS["text"],
                )

            after_y = divider_y + 70
            self._text("AFTER PICKING", rect.x + 15, after_y, self.font_tiny, accent)
            after_lines = self._wrapped_lines(
                details["after"], self.font_small, rect.width - 30, 3
            )
            for line_index, line in enumerate(after_lines):
                self._text(
                    line,
                    rect.x + 15,
                    after_y + 19 + line_index * 18,
                    self.font_small,
                    COLORS["text"],
                )
        self._draw_modal_footer(
            "BATTLE PAUSED  •  SELECT WITH MOUSE OR KEYS 1–3",
            accent,
        )

    def _draw_build_panel(self) -> None:
        overlay = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        overlay.fill((3, 6, 18, 205))
        self.surface.blit(overlay, (0, 0))
        panel = pygame.Rect(40, 70, 720, 462)
        pygame.draw.rect(self.surface, (17, 27, 52), panel, border_radius=18)
        pygame.draw.rect(self.surface, COLORS["xp"], panel, 2, border_radius=18)
        title = self.font_large.render("SHIP BUILD", True, COLORS["text"])
        self.surface.blit(title, (panel.x + 28, panel.y + 22))
        self._text("TAB to close  •  battle timer paused", panel.x + 30, panel.y + 68, self.font_small, COLORS["muted"])

        profile = self.env.weapon_profile()
        damage = float(self.env.projectile_cfg["damage"]) * float(profile["damage_multiplier"])
        cooldown = self.env.weapon_cooldown_steps() / self.env.fps
        lifetime = float(self.env.projectile_cfg["lifetime_seconds"]) * float(profile["lifetime_multiplier"])
        self._text(str(profile["name"]).upper(), panel.x + 30, panel.y + 112, self.font_medium, COLORS["xp"])
        weapon_lines = (
            f"Beams        {int(profile['shot_count'])}",
            f"Damage       {damage:.1f} each",
            f"Cooldown     {cooldown:.2f}s",
            f"Beam range   {lifetime:.1f}s flight",
            f"Piercing     +{int(profile['pierces'])} targets",
            f"Blast radius {float(profile['splash_radius']):.0f}px",
            f"Critical     {float(profile['critical_chance']):.0%} ×2 damage",
        )
        for index, line in enumerate(weapon_lines):
            self._text(line, panel.x + 30, panel.y + 150 + index * 25, self.font_small, COLORS["text"])

        upgrade_x = panel.x + 285
        self._text("SELECTED UPGRADES", upgrade_x, panel.y + 112, self.font_medium, COLORS["accent"])
        names = {str(item["id"]): str(item["name"]) for item in self.env.progression_cfg["upgrade_catalog"]}
        selected = [
            (names.get(key, "Rift Artifact" if key == "artifact" else key.replace("_", " ").title()), value)
            for key, value in self.env.upgrade_stacks.items()
            if value > 0 and key != "repair"
        ]
        if not selected:
            self._text("No permanent upgrades yet.", upgrade_x, panel.y + 151, self.font_small, COLORS["muted"])
        for index, (name, stacks) in enumerate(selected[:20]):
            column = index // 10
            row = index % 10
            label = self._fit_text(
                f"{name}  T{stacks}", self.font_small, 188, COLORS["text"]
            )
            self.surface.blit(
                label,
                (upgrade_x + column * 198, panel.y + 151 + row * 22),
            )
        if len(selected) > 20:
            self._text(
                f"+{len(selected) - 20} additional mastery path(s)",
                upgrade_x,
                panel.y + 375,
                self.font_tiny,
                COLORS["muted"],
            )
        support = []
        if self.env.support_drone_active:
            support.append(
                f"DRONES {self.env.drone_count}  •  CORE TIER {self.env.drone_level}"
            )
        if self.env.nova_bomb_armed:
            support.append("NOVA BOMB ARMED")
        if self.env.barrier_charges:
            support.append(f"AEGIS {self.env.barrier_charges}")
        if self.env.overdrive_active:
            support.append("OVERDRIVE ACTIVE")
        support_text = "  •  ".join(support) if support else "No active support systems"
        support_surface = self._fit_text(
            support_text, self.font_tiny, panel.width - 60, COLORS["drone"] if support else COLORS["muted"]
        )
        self.surface.blit(support_surface, (panel.x + 30, panel.bottom - 42))
        self._draw_modal_footer("BATTLE PAUSED  •  PRESS TAB TO RETURN", COLORS["xp"])

    def _draw_modal_footer(
        self, text: str, accent: tuple[int, int, int]
    ) -> None:
        footer_height = 32
        footer = pygame.Surface((self.env.width, footer_height), pygame.SRCALPHA)
        footer.fill((7, 12, 29, 248))
        pygame.draw.line(footer, accent, (0, 0), (self.env.width, 0), 1)
        label = self.font_tiny.render(text, True, COLORS["text"])
        footer.blit(label, label.get_rect(center=(self.env.width // 2, 17)))
        self.surface.blit(footer, (0, self.env.height - footer_height))

    def _sync_effects(self) -> None:
        if self.env.step_count == self.last_effect_step:
            return
        self.last_effect_step = self.env.step_count
        events = self.env.last_events
        for impact in events.get("impacts", []):
            color = (
                COLORS["spawner_core"]
                if impact.get("kind") == "spawner"
                else COLORS["enemy"]
            )
            count = 24 if impact.get("destroyed") else 10
            self._burst(float(impact["x"]), float(impact["y"]), color, count)
        if events.get("player_hit"):
            self._burst(
                self.env.player.x,
                self.env.player.y,
                COLORS["player"],
                18,
            )
        if events.get("levels_gained"):
            self._burst(
                self.env.player.x,
                self.env.player.y,
                COLORS["xp"],
                42,
            )
        if events.get("nova_bomb_detonated"):
            self._burst(
                self.env.width / 2,
                self.env.height / 2,
                COLORS["nova"],
                110,
            )
        if self.audio:
            self.audio.sync_events(events, self.env.done)

    def _burst(
        self, x: float, y: float, color: tuple[int, int, int], count: int
    ) -> None:
        for index in range(count):
            angle = self.effect_rng.uniform(0.0, math.tau)
            speed = self.effect_rng.uniform(55.0, 180.0)
            life = self.effect_rng.uniform(0.28, 0.65)
            self.particles.append(
                {
                    "x": x,
                    "y": y,
                    "vx": math.cos(angle) * speed,
                    "vy": math.sin(angle) * speed,
                    "life": life,
                    "maximum": life,
                    "size": 2 + index % 3,
                    "color": color,
                }
            )

    def _draw_particles(self) -> None:
        alive: list[dict[str, object]] = []
        for particle in self.particles:
            life = float(particle["life"]) - 1.0 / self.env.fps
            if life <= 0:
                continue
            particle["life"] = life
            particle["x"] = float(particle["x"]) + float(particle["vx"]) / self.env.fps
            particle["y"] = float(particle["y"]) + float(particle["vy"]) / self.env.fps
            ratio = life / float(particle["maximum"])
            color = tuple(int(component * ratio) for component in particle["color"])
            pygame.draw.circle(
                self.surface,
                color,
                (round(float(particle["x"])), round(float(particle["y"]))),
                int(particle["size"]),
            )
            alive.append(particle)
        self.particles = alive

    def _draw_vignette(self) -> None:
        health_ratio = max(0.0, self.env.player.health / self.env.player.max_health)
        if health_ratio > 0.45 and not self.env.last_events.get("player_hit"):
            return
        alpha = int((1.0 - health_ratio) * 75)
        if self.env.last_events.get("player_hit"):
            alpha = max(alpha, 55)
        overlay = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA)
        pygame.draw.rect(
            overlay,
            (*COLORS["health_low"], alpha),
            overlay.get_rect(),
            width=14,
        )
        self.surface.blit(overlay, (0, 0))

    @staticmethod
    def _fit_text(
        text: str,
        font: pygame.font.Font,
        max_width: int,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        if font.size(text)[0] <= max_width:
            return font.render(text, True, color)
        suffix = "…"
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if font.size(text[:middle] + suffix)[0] <= max_width:
                low = middle
            else:
                high = middle - 1
        return font.render(text[:low].rstrip() + suffix, True, color)

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
        if self.audio:
            self.audio.close()
        pygame.quit()


__all__ = ["ArenaRenderer"]
