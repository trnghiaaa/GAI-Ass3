"""Friendly visual launcher for manual play and both trained DQN agents."""

from __future__ import annotations

from dataclasses import dataclass
import random

import pygame

from arena.evaluation.evaluate import load_policy, policy_readiness, watch_policy
from arena.presentation.audio import ArenaAudio
from arena.presentation.play import play_manual


WIDTH, HEIGHT = 800, 600
DEMO_SEEDS = {"direct": 22003, "rotation": 53006}
BACKGROUND = (7, 10, 25)
PANEL = (19, 29, 55)
PANEL_HOVER = (28, 43, 75)
LINE = (51, 72, 112)
TEXT = (235, 243, 255)
MUTED = (184, 199, 222)
CYAN = (65, 210, 255)
PURPLE = (181, 82, 255)
GREEN = (63, 220, 135)
YELLOW = (255, 205, 87)


@dataclass(frozen=True)
class LaunchCard:
    rect: pygame.Rect
    title: str
    subtitle: str
    mode: str
    control_style: str
    accent: tuple[int, int, int]


def _fit_text(
    text: str, font: pygame.font.Font, max_width: int, color: tuple[int, int, int]
) -> pygame.Surface:
    if font.size(text)[0] <= max_width:
        return font.render(text, True, color)
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    return font.render(text.rstrip() + "…", True, color)


def _draw_menu(
    screen: pygame.Surface,
    cards: list[LaunchCard],
    readiness: dict[str, tuple[bool, str]],
    mouse: tuple[int, int],
    elapsed: float,
    notice: str,
    fonts: dict[str, pygame.font.Font],
) -> None:
    screen.fill(BACKGROUND)
    nebula = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.ellipse(nebula, (30, 46, 92, 100), (-180, 140, 720, 540))
    pygame.draw.ellipse(nebula, (81, 30, 104, 65), (420, -180, 620, 540))
    screen.blit(nebula, (0, 0))

    star_rng = random.Random(9042)
    for index in range(70):
        x = star_rng.randrange(WIDTH)
        y = star_rng.randrange(HEIGHT)
        pulse = 35 if (index * 11 + int(elapsed * 20)) % 90 < 5 else 0
        shade = 115 + star_rng.randrange(80) + pulse
        pygame.draw.circle(screen, (shade // 2, shade * 3 // 4, min(255, shade)), (x, y), 1)

    eyebrow = fonts["eyebrow"].render("ASSIGNMENT 3  •  PART II", True, CYAN)
    screen.blit(eyebrow, eyebrow.get_rect(center=(WIDTH // 2, 42)))
    hero = fonts["hero"].render("NEON RIFT ARENA", True, TEXT)
    screen.blit(hero, hero.get_rect(center=(WIDTH // 2, 85)))
    subtitle = fonts["subtitle"].render(
        "Choose a control scheme, then play it yourself or inspect its learned DQN policy.",
        True,
        MUTED,
    )
    screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, 126)))

    for card in cards:
        hover = card.rect.collidepoint(mouse)
        shadow = card.rect.move(0, 6)
        pygame.draw.rect(screen, (2, 5, 15), shadow, border_radius=18)
        pygame.draw.rect(
            screen, PANEL_HOVER if hover else PANEL, card.rect, border_radius=18
        )
        pygame.draw.rect(
            screen, card.accent if hover else LINE, card.rect, 2, border_radius=18
        )
        icon_center = (card.rect.x + 44, card.rect.y + 45)
        pygame.draw.circle(screen, (*card.accent,), icon_center, 24, 2)
        icon = "AI" if card.mode == "ai" else "M"
        icon_surface = fonts["title"].render(icon, True, card.accent)
        screen.blit(icon_surface, icon_surface.get_rect(center=icon_center))
        title = _fit_text(card.title, fonts["title"], card.rect.width - 100, TEXT)
        screen.blit(title, (card.rect.x + 82, card.rect.y + 23))
        body = _fit_text(card.subtitle, fonts["body"], card.rect.width - 34, MUTED)
        screen.blit(body, (card.rect.x + 17, card.rect.y + 84))

        if card.mode == "ai":
            ready, status_text = readiness[card.control_style]
            status_color = GREEN if ready else YELLOW
        else:
            status_text = "PLAY NOW"
            status_color = GREEN
        status = fonts["tiny"].render(status_text, True, status_color)
        screen.blit(status, (card.rect.x + 17, card.rect.bottom - 28))
        arrow = fonts["title"].render("›", True, card.accent)
        screen.blit(arrow, (card.rect.right - 28, card.rect.bottom - 36))

    tags = "21 BUILD PATHS   •   RIFT HUNTERS   •   BOSS BARRAGES   •   DEEP-RL AGENTS"
    tag_surface = fonts["tiny"].render(tags, True, MUTED)
    screen.blit(tag_surface, tag_surface.get_rect(center=(WIDTH // 2, 535)))
    footer = notice or "Click a card to launch  •  V audio  •  Esc exits"
    footer_color = YELLOW if notice else MUTED
    footer_surface = _fit_text(footer, fonts["body"], WIDTH - 70, footer_color)
    screen.blit(footer_surface, footer_surface.get_rect(center=(WIDTH // 2, 570)))


def _menu_selection(notice: str = "") -> tuple[str, str] | None:
    pygame.init()
    pygame.font.init()
    audio = ArenaAudio()
    if audio.available:
        audio.start_music()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neon Rift Arena — Part II")
    clock = pygame.time.Clock()
    fonts = {
        "eyebrow": pygame.font.SysFont("bahnschrift", 14, bold=True),
        "hero": pygame.font.SysFont("bahnschrift", 46, bold=True),
        "subtitle": pygame.font.SysFont("bahnschrift", 17),
        "title": pygame.font.SysFont("bahnschrift", 23, bold=True),
        "body": pygame.font.SysFont("bahnschrift", 15),
        "tiny": pygame.font.SysFont("bahnschrift", 13, bold=True),
    }
    cards = [
        LaunchCard(pygame.Rect(55, 170, 330, 145), "Manual: Direct", "Move and fire together; nearby targets receive assist.", "manual", "direct", CYAN),
        LaunchCard(pygame.Rect(415, 170, 330, 145), "Manual: Rotation", "Rotate or thrust while firing along your heading.", "manual", "rotation", PURPLE),
        LaunchCard(pygame.Rect(55, 340, 330, 145), "Watch Direct DQN", "Watch a learned policy evolve an uncapped build.", "ai", "direct", CYAN),
        LaunchCard(pygame.Rect(415, 340, 330, 145), "Watch Rotation DQN", "Inspect learned aiming and boss-hazard avoidance.", "ai", "rotation", PURPLE),
    ]
    readiness = {
        style: policy_readiness(style) for style in ("direct", "rotation")
    }
    elapsed = 0.0
    last_hovered_card: LaunchCard | None = None
    while True:
        elapsed += clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()
        hovered_card = next((c for c in cards if c.rect.collidepoint(mouse)), None)
        if hovered_card is not None and hovered_card != last_hovered_card:
            audio.play("click", minimum_interval_ms=100)
        last_hovered_card = hovered_card

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                audio.close()
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    audio.close()
                    pygame.quit()
                    return None
                if event.key == pygame.K_v:
                    audio.toggle()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for card in cards:
                    if card.rect.collidepoint(event.pos):
                        audio.play("menu_select")
                        pygame.time.delay(120)
                        audio.close()
                        pygame.quit()
                        return card.mode, card.control_style
        _draw_menu(screen, cards, readiness, mouse, elapsed, notice, fonts)
        pygame.display.flip()


def main() -> None:
    """Keep returning to the launcher after each manual or AI session."""

    notice = ""
    while True:
        selection = _menu_selection(notice)
        notice = ""
        if selection is None:
            return
        mode, control_style = selection
        if mode == "manual":
            play_manual(control_style)
            continue
        try:
            model, metadata = load_policy(control_style)
        except (FileNotFoundError, ValueError) as exc:
            notice = str(exc)
            continue
        watch_policy(
            model,
            control_style,
            episodes=1,
            seed=DEMO_SEEDS[control_style],
            action_repeat=int(metadata.get("action_repeat", 4)),
        )


__all__ = ["main"]
