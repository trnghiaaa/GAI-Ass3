"""Friendly visual launcher for manual play and both trained DQN agents."""

from __future__ import annotations

from dataclasses import dataclass
import random

import pygame

from arena.evaluation.evaluate import policy_readiness
from arena.presentation.audio import ArenaAudio
from arena.presentation.guidebook import open_guidebook
from arena.presentation.play import play_manual


WIDTH, HEIGHT = 800, 600
DEMO_SEEDS = {"direct": 22003, "rotation": 590003}
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
GUIDE_RECT = pygame.Rect(622, 22, 154, 38)


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

    guide_hover = GUIDE_RECT.collidepoint(mouse)
    pygame.draw.rect(screen, (2, 5, 15), GUIDE_RECT.move(0, 3), border_radius=10)
    pygame.draw.rect(
        screen,
        PANEL_HOVER if guide_hover else PANEL,
        GUIDE_RECT,
        border_radius=10,
    )
    pygame.draw.rect(
        screen, CYAN if guide_hover else LINE, GUIDE_RECT, 1, border_radius=10
    )
    # Tiny open-book mark keeps the entry recognizable without external art.
    pygame.draw.line(screen, CYAN, (637, 34), (637, 50), 2)
    pygame.draw.line(screen, CYAN, (637, 34), (629, 31), 2)
    pygame.draw.line(screen, CYAN, (637, 34), (645, 31), 2)
    pygame.draw.line(screen, CYAN, (629, 31), (629, 47), 2)
    pygame.draw.line(screen, CYAN, (645, 31), (645, 47), 2)
    guide = fonts["tiny"].render("PILOT GUIDE  G", True, TEXT)
    screen.blit(guide, (654, 34))

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
    footer = notice or "Click a card to launch  •  G pilot guide  •  H quick help  •  V audio"
    footer_color = YELLOW if notice else MUTED
    footer_surface = _fit_text(footer, fonts["body"], WIDTH - 70, footer_color)
    screen.blit(footer_surface, footer_surface.get_rect(center=(WIDTH // 2, 570)))


def _draw_menu_help_overlay(screen: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((3, 6, 18, 225))
    screen.blit(overlay, (0, 0))

    panel = pygame.Rect(40, 50, 720, 500)
    pygame.draw.rect(screen, (17, 27, 52), panel, border_radius=18)
    pygame.draw.rect(screen, CYAN, panel, 2, border_radius=18)

    title = fonts["hero"].render("NEON RIFT ARENA — GUIDE", True, TEXT)
    screen.blit(title, (panel.x + 28, panel.y + 20))
    sub = fonts["subtitle"].render("Click anywhere or press [ H ] / [ ESC ] to return", True, MUTED)
    screen.blit(sub, (panel.x + 30, panel.y + 64))

    # Col 1: Modes & Controls
    col1_x = panel.x + 30
    head1 = fonts["title"].render("FLIGHT & MODES", True, CYAN)
    screen.blit(head1, (col1_x, panel.y + 98))

    controls = (
        ("Direct Control Scheme", "WASD to move, left-click / space to shoot"),
        ("Rotation Control Scheme", "A / D to rotate, W to thrust, space to fire"),
        ("Nova Bomb (Phase 6+)", "Vaporizes all regular hostile spawns"),
        ("21 Weapon Builds", "Draft upgrade cards on level up (1 / 2 / 3)"),
        ("Aegis & Drones", "Shield blocks damage; drones auto-fire"),
    )
    for i, (heading, desc) in enumerate(controls):
        h_s = fonts["tiny"].render(heading, True, YELLOW)
        d_s = fonts["body"].render(desc, True, TEXT)
        screen.blit(h_s, (col1_x, panel.y + 130 + i * 50))
        screen.blit(d_s, (col1_x, panel.y + 148 + i * 50))

    # Col 2: In-Game Hotkeys
    col2_x = panel.x + 380
    head2 = fonts["title"].render("IN-GAME HOTKEYS", True, YELLOW)
    screen.blit(head2, (col2_x, panel.y + 98))

    hotkeys = (
        ("[ C ]", "Cycle 5 Neon Ship Skins & Laser SFX"),
        ("[ TAB ]", "Open 21-Tier Ship Build Panel"),
        ("[ V ]", "Audio Volume Slider HUD"),
        ("[ M ]", "Mute / Unmute Audio"),
        ("[ P ]", "Pause / Resume Battle"),
        ("[ R ]", "Instant Replay / Restart Current Seed"),
        ("[ G ]", "Open the Illustrated Pilot Guide"),
        ("[ ESC / Q ]", "Return to Menu / Exit Mission"),
    )
    for i, (key, desc) in enumerate(hotkeys):
        k_s = fonts["tiny"].render(key, True, CYAN)
        d_s = fonts["body"].render(desc, True, TEXT)
        screen.blit(k_s, (col2_x, panel.y + 130 + i * 40))
        screen.blit(d_s, (col2_x, panel.y + 146 + i * 40))

    hint_text = "CLICK ANYWHERE OR PRESS [ H ] / [ ESC ] TO CLOSE"
    h_surf = fonts["tiny"].render(hint_text, True, CYAN)
    h_rect = pygame.Rect(panel.centerx - h_surf.get_width() // 2 - 16, panel.bottom - 38, h_surf.get_width() + 32, 26)
    pygame.draw.rect(screen, (28, 43, 75), h_rect, border_radius=13)
    pygame.draw.rect(screen, CYAN, h_rect, 1, border_radius=13)
    screen.blit(h_surf, (h_rect.centerx - h_surf.get_width() // 2, h_rect.y + 5))


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
    show_volume_slider = False
    show_help_overlay = False
    dragging_volume = False
    vol_panel = pygame.Rect(WIDTH - 300, 15, 280, 130)
    vol_track = pygame.Rect(WIDTH - 280, 68, 240, 10)

    while True:
        elapsed += clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()
        hovered_card = next((c for c in cards if c.rect.collidepoint(mouse)), None)
        if hovered_card is not None and hovered_card != last_hovered_card and not show_volume_slider and not show_help_overlay:
            audio.play("click", minimum_interval_ms=100)
        last_hovered_card = hovered_card

        if show_volume_slider and dragging_volume:
            rel_x = max(0, min(vol_track.width, mouse[0] - vol_track.x))
            audio.set_volume(rel_x / vol_track.width)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                audio.close()
                pygame.quit()
                return None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if show_help_overlay:
                    show_help_overlay = False
                    if audio.available:
                        audio.play("click", minimum_interval_ms=50)
                    continue
                if show_volume_slider:
                    if vol_panel.collidepoint(event.pos):
                        if vol_track.inflate(0, 16).collidepoint(event.pos):
                            dragging_volume = True
                            rel_x = max(0, min(vol_track.width, event.pos[0] - vol_track.x))
                            audio.set_volume(rel_x / vol_track.width)
                        elif event.pos[1] >= vol_panel.y + 88:
                            btn_minus = pygame.Rect(vol_panel.x + 16, vol_panel.y + 88, 46, 26)
                            btn_mute = pygame.Rect(vol_panel.x + 70, vol_panel.y + 88, 140, 26)
                            btn_plus = pygame.Rect(vol_panel.x + 218, vol_panel.y + 88, 46, 26)
                            if btn_minus.collidepoint(event.pos):
                                audio.set_volume(audio.get_volume() - 0.05)
                                audio.play("click", minimum_interval_ms=50)
                            elif btn_plus.collidepoint(event.pos):
                                audio.set_volume(audio.get_volume() + 0.05)
                                audio.play("click", minimum_interval_ms=50)
                            elif btn_mute.collidepoint(event.pos):
                                audio.toggle()
                        continue
                    else:
                        show_volume_slider = False
                        dragging_volume = False
                if GUIDE_RECT.collidepoint(event.pos):
                    audio.play("menu_select")
                    open_guidebook(screen, clock, audio)
                    continue
                for card in cards:
                    if card.rect.collidepoint(event.pos):
                        audio.play("menu_select")
                        pygame.time.delay(120)
                        audio.close()
                        pygame.quit()
                        return card.mode, card.control_style

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_volume = False

            if event.type == pygame.KEYDOWN:
                if show_help_overlay:
                    if event.key in (pygame.K_h, pygame.K_SLASH, pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                        show_help_overlay = False
                        if audio.available:
                            audio.play("click", minimum_interval_ms=50)
                    continue
                if event.key == pygame.K_g:
                    audio.play("menu_select")
                    open_guidebook(screen, clock, audio)
                    continue
                if event.key in (pygame.K_h, pygame.K_SLASH):
                    show_help_overlay = not show_help_overlay
                    if audio.available:
                        audio.play("click", minimum_interval_ms=50)
                    continue
                if show_volume_slider:
                    if event.key in (pygame.K_LEFT, pygame.K_DOWN, pygame.K_MINUS):
                        audio.set_volume(audio.get_volume() - 0.05)
                        audio.play("click", minimum_interval_ms=50)
                        continue
                    if event.key in (pygame.K_RIGHT, pygame.K_UP, pygame.K_PLUS, pygame.K_EQUALS):
                        audio.set_volume(audio.get_volume() + 0.05)
                        audio.play("click", minimum_interval_ms=50)
                        continue
                    if event.key == pygame.K_m:
                        audio.toggle()
                        continue
                    if event.key in (pygame.K_ESCAPE, pygame.K_v):
                        show_volume_slider = False
                        continue
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    audio.close()
                    pygame.quit()
                    return None
                if event.key == pygame.K_v:
                    show_volume_slider = not show_volume_slider
                    if show_volume_slider and audio.available:
                        audio.play("click", minimum_interval_ms=50)

        _draw_menu(screen, cards, readiness, mouse, elapsed, notice, fonts)

        if show_volume_slider and audio.available:
            shadow = vol_panel.move(0, 4)
            pygame.draw.rect(screen, (2, 5, 15), shadow, border_radius=12)
            panel = pygame.Surface(vol_panel.size, pygame.SRCALPHA)
            panel.fill((14, 22, 44, 245))
            screen.blit(panel, vol_panel.topleft)
            pygame.draw.rect(screen, CYAN, vol_panel, 2, border_radius=12)

            vol_pct = int(audio.get_volume() * 100)
            status_text = f"VOLUME: {vol_pct}%" if audio.enabled and vol_pct > 0 else "VOLUME: MUTED"
            status_color = CYAN if audio.enabled and vol_pct > 0 else YELLOW
            title_surf = fonts["body"].render(status_text, True, status_color)
            screen.blit(title_surf, (vol_panel.x + 16, vol_panel.y + 16))

            hint_surf = fonts["tiny"].render("V to close", True, MUTED)
            screen.blit(hint_surf, (vol_panel.right - 16 - hint_surf.get_width(), vol_panel.y + 18))

            fill_w = int(vol_track.width * audio.get_volume())
            fill_rect = pygame.Rect(vol_track.x, vol_track.y, fill_w, vol_track.height)
            pygame.draw.rect(screen, BACKGROUND, vol_track, border_radius=5)
            if fill_w > 0:
                pygame.draw.rect(screen, CYAN, fill_rect, border_radius=5)

            knob_x = vol_track.x + fill_w
            knob_y = vol_track.centery
            pygame.draw.circle(screen, TEXT, (knob_x, knob_y), 7)
            pygame.draw.circle(screen, CYAN, (knob_x, knob_y), 4)

            btn_minus = pygame.Rect(vol_panel.x + 16, vol_panel.y + 88, 46, 26)
            btn_mute = pygame.Rect(vol_panel.x + 70, vol_panel.y + 88, 140, 26)
            btn_plus = pygame.Rect(vol_panel.x + 218, vol_panel.y + 88, 46, 26)

            for b_rect, b_text in ((btn_minus, "-5%"), (btn_mute, "UNMUTE" if not audio.enabled else "MUTE"), (btn_plus, "+5%")):
                hover = b_rect.collidepoint(mouse)
                bg = PANEL_HOVER if hover else PANEL
                border = CYAN if hover else LINE
                pygame.draw.rect(screen, bg, b_rect, border_radius=4)
                pygame.draw.rect(screen, border, b_rect, 1, border_radius=4)
                t_surf = fonts["tiny"].render(b_text, True, TEXT if hover else MUTED)
                screen.blit(t_surf, t_surf.get_rect(center=b_rect.center))

        if show_help_overlay:
            _draw_menu_help_overlay(screen, fonts)

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
        from arena.evaluation.evaluate import load_policy, watch_policy

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
