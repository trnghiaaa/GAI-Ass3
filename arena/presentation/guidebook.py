"""Interactive, illustrated pilot guide for the Part II launcher."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import pygame

from arena.settings import load_config


WIDTH, HEIGHT = 800, 600
BACKGROUND = (7, 10, 25)
PANEL = (17, 27, 52)
PANEL_DARK = (10, 17, 36)
PANEL_HOVER = (27, 42, 74)
LINE = (50, 72, 112)
TEXT = (235, 243, 255)
MUTED = (181, 199, 224)
CYAN = (65, 210, 255)
PURPLE = (181, 82, 255)
GREEN = (67, 222, 145)
YELLOW = (255, 205, 87)
ORANGE = (255, 143, 73)
RED = (255, 79, 120)

BACK_RECT = pygame.Rect(24, 22, 104, 38)
SIDEBAR_RECT = pygame.Rect(24, 88, 174, 466)
CONTENT_RECT = pygame.Rect(210, 88, 566, 466)
PREVIOUS_RECT = pygame.Rect(226, 518, 118, 28)
NEXT_RECT = pygame.Rect(642, 518, 118, 28)


@dataclass(frozen=True)
class GuidePage:
    key: str
    tab: str
    eyebrow: str
    title: str
    subtitle: str
    accent: tuple[int, int, int]


PAGES = (
    GuidePage(
        "flight",
        "FLIGHT SCHOOL",
        "01 / QUICK START",
        "Fly, aim, survive",
        "Both control schemes use the same arena, rewards, enemies, and phase rules.",
        CYAN,
    ),
    GuidePage(
        "battlefield",
        "BATTLEFIELD",
        "02 / ARENA INTEL",
        "Read the battlefield",
        "Know the objective, HUD, hostile silhouettes, and phase clock at a glance.",
        GREEN,
    ),
    GuidePage(
        "boss",
        "BOSS PROTOCOL",
        "03 / PRIORITY TARGETS",
        "Break the Aegis lock",
        "Boss warnings are dodgeable. Sentries change the target priority completely.",
        RED,
    ),
    GuidePage(
        "drafts",
        "REWARD DRAFTS",
        "04 / BUILD FLOW",
        "Three ways to grow",
        "Level up, clear a phase, or defeat a boss to choose one of three rewards.",
        YELLOW,
    ),
    GuidePage(
        "upgrades_a",
        "UPGRADES 1–7",
        "05 / SHIP SYSTEMS",
        "Weapons and foundations",
        "Upgrade cards stack into a build and immediately show their before/after effect.",
        CYAN,
    ),
    GuidePage(
        "upgrades_b",
        "UPGRADES 8–14",
        "06 / SHIP SYSTEMS",
        "Defense and guidance",
        "Piercing, blast damage, protection, mobility, homing, regeneration, and drones.",
        PURPLE,
    ),
    GuidePage(
        "upgrades_c",
        "UPGRADES 15–21",
        "07 / SHIP SYSTEMS",
        "Advanced and mastery systems",
        "Critical hits, sustain, objective damage, and repeatable late-game scaling.",
        PURPLE,
    ),
    GuidePage(
        "phase_rewards",
        "PHASE REWARDS",
        "08 / SUPPORT CACHE",
        "Prepare the next assault",
        "Choose one support item after every cleared phase.",
        GREEN,
    ),
    GuidePage(
        "boss_rewards",
        "BOSS RELICS",
        "09 / BOSS REWARD",
        "Claim a powerful relic",
        "Boss victories offer stronger permanent or multi-phase rewards.",
        ORANGE,
    ),
    GuidePage(
        "build_style",
        "BUILD & STYLE",
        "10 / SHIP CUSTOMIZATION",
        "Inspect and personalize",
        "Pause to inspect your complete build, then showcase five audiovisual ship skins.",
        CYAN,
    ),
    GuidePage(
        "other_features",
        "OTHER FEATURES",
        "11 / PLAYER TOOLS",
        "Control the experience",
        "Pause the action, replay a run, tune the audio, or study trained AI at your pace.",
        YELLOW,
    ),
)


def guide_page_count() -> int:
    """Return the stable number of guidebook chapters."""

    return len(PAGES)


def guide_catalog_entries() -> dict[str, tuple[dict[str, Any], ...]]:
    """Load the displayed catalogues from the authoritative game config."""

    progression = load_config()["progression"]
    return {
        "upgrades": tuple(dict(item) for item in progression["upgrade_catalog"]),
        "phase_rewards": tuple(
            dict(item) for item in progression["phase_reward_catalog"]
        ),
        "boss_rewards": tuple(
            dict(item) for item in progression["boss_reward_catalog"]
        ),
    }


def _fonts() -> dict[str, pygame.font.Font]:
    return {
        "eyebrow": pygame.font.SysFont("bahnschrift", 12, bold=True),
        "hero": pygame.font.SysFont("bahnschrift", 32, bold=True),
        "title": pygame.font.SysFont("bahnschrift", 20, bold=True),
        "body": pygame.font.SysFont("bahnschrift", 13),
        "body_bold": pygame.font.SysFont("bahnschrift", 13, bold=True),
        "small": pygame.font.SysFont("bahnschrift", 11),
        "small_bold": pygame.font.SysFont("bahnschrift", 11, bold=True),
        "micro": pygame.font.SysFont("bahnschrift", 10),
    }


def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and font.size(candidate)[0] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _wrapped_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    *,
    line_height: int | None = None,
    max_lines: int | None = None,
) -> int:
    lines = _wrap(text, font, rect.width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        final = lines[-1]
        while final and font.size(final + "…")[0] > rect.width:
            final = final[:-1]
        lines[-1] = final.rstrip() + "…"
    spacing = line_height or font.get_linesize()
    for index, line in enumerate(lines):
        surface.blit(font.render(line, True, color), (rect.x, rect.y + index * spacing))
    return rect.y + len(lines) * spacing


def _panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    fill: tuple[int, int, int] = PANEL,
    border: tuple[int, int, int] = LINE,
    radius: int = 12,
) -> None:
    pygame.draw.rect(surface, (2, 5, 15), rect.move(0, 4), border_radius=radius)
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 1, border_radius=radius)


def _draw_background(surface: pygame.Surface, elapsed: float) -> None:
    surface.fill(BACKGROUND)
    haze = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.ellipse(haze, (28, 61, 104, 70), (-220, 70, 650, 520))
    pygame.draw.ellipse(haze, (92, 28, 112, 55), (480, -170, 540, 520))
    surface.blit(haze, (0, 0))
    rng = random.Random(7319)
    pulse_step = int(elapsed * 3.0)
    for index in range(58):
        x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        bright = 45 if (index + pulse_step) % 17 == 0 else 0
        shade = min(235, 105 + rng.randrange(75) + bright)
        pygame.draw.circle(surface, (shade // 2, shade * 3 // 4, shade), (x, y), 1)


def _draw_ship(
    surface: pygame.Surface,
    center: tuple[int, int],
    angle: float,
    color: tuple[int, int, int] = CYAN,
    scale: float = 1.0,
) -> None:
    cx, cy = center
    forward = (math.cos(angle), math.sin(angle))
    side = (-forward[1], forward[0])
    points = [
        (cx + forward[0] * 18 * scale, cy + forward[1] * 18 * scale),
        (cx - forward[0] * 12 * scale + side[0] * 10 * scale,
         cy - forward[1] * 12 * scale + side[1] * 10 * scale),
        (cx - forward[0] * 7 * scale, cy - forward[1] * 7 * scale),
        (cx - forward[0] * 12 * scale - side[0] * 10 * scale,
         cy - forward[1] * 12 * scale - side[1] * 10 * scale),
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, TEXT, points, 1)
    pygame.draw.circle(surface, BACKGROUND, center, max(2, int(3 * scale)))


def _draw_key(
    surface: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    rect: pygame.Rect,
    label: str,
    color: tuple[int, int, int] = CYAN,
) -> None:
    pygame.draw.rect(surface, PANEL_HOVER, rect, border_radius=5)
    pygame.draw.rect(surface, color, rect, 1, border_radius=5)
    rendered = fonts["small_bold"].render(label, True, TEXT)
    surface.blit(rendered, rendered.get_rect(center=rect.center))


def _draw_chapter_frame(
    surface: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    page_index: int,
    mouse: tuple[int, int],
) -> dict[str, pygame.Rect]:
    page = PAGES[page_index]
    back_fill = PANEL_HOVER if BACK_RECT.collidepoint(mouse) else PANEL
    _panel(surface, BACK_RECT, fill=back_fill, border=page.accent, radius=10)
    back = fonts["body_bold"].render("‹  MENU", True, TEXT)
    surface.blit(back, back.get_rect(center=BACK_RECT.center))

    heading = fonts["eyebrow"].render("NEON RIFT ARENA  •  PILOT GUIDE", True, CYAN)
    surface.blit(heading, (150, 20))
    title = fonts["title"].render("TACTICAL FIELD MANUAL", True, TEXT)
    surface.blit(title, (150, 38))
    page_label = fonts["body_bold"].render(
        f"{page_index + 1:02d}  /  {len(PAGES):02d}", True, page.accent
    )
    surface.blit(page_label, (WIDTH - 24 - page_label.get_width(), 34))

    _panel(surface, SIDEBAR_RECT, fill=PANEL_DARK, radius=14)
    tab_rects: dict[str, pygame.Rect] = {}
    for index, chapter in enumerate(PAGES):
        # Eleven chapters remain individually clickable without crowding the
        # navigation hint at the foot of the handbook.
        tab = pygame.Rect(34, 94 + index * 38, 154, 33)
        tab_rects[f"page_{index}"] = tab
        active = index == page_index
        hover = tab.collidepoint(mouse)
        if active or hover:
            pygame.draw.rect(surface, (25, 40, 70), tab, border_radius=9)
        if active:
            pygame.draw.rect(surface, chapter.accent, (tab.x, tab.y, 3, tab.height), border_radius=2)
        number = fonts["eyebrow"].render(f"{index + 1:02d}", True, chapter.accent)
        surface.blit(number, (tab.x + 10, tab.y + 3))
        label = fonts["small_bold"].render(chapter.tab, True, TEXT if active else MUTED)
        surface.blit(label, (tab.x + 35, tab.y + 10))

    hint = fonts["micro"].render("← →  PAGES   •   ESC  MENU", True, MUTED)
    surface.blit(hint, hint.get_rect(center=(SIDEBAR_RECT.centerx, SIDEBAR_RECT.bottom - 20)))

    _panel(surface, CONTENT_RECT, border=page.accent, radius=14)
    surface.blit(fonts["eyebrow"].render(page.eyebrow, True, page.accent), (228, 106))
    surface.blit(fonts["hero"].render(page.title, True, TEXT), (226, 124))
    _wrapped_text(
        surface,
        page.subtitle,
        fonts["body"],
        MUTED,
        pygame.Rect(228, 159, 524, 30),
        max_lines=2,
    )
    pygame.draw.line(surface, LINE, (228, 181), (758, 181))

    if page_index > 0:
        fill = PANEL_HOVER if PREVIOUS_RECT.collidepoint(mouse) else PANEL_DARK
        pygame.draw.rect(surface, fill, PREVIOUS_RECT, border_radius=7)
        pygame.draw.rect(surface, LINE, PREVIOUS_RECT, 1, border_radius=7)
        previous = fonts["small_bold"].render("‹  PREVIOUS", True, MUTED)
        surface.blit(previous, previous.get_rect(center=PREVIOUS_RECT.center))
    if page_index < len(PAGES) - 1:
        fill = PANEL_HOVER if NEXT_RECT.collidepoint(mouse) else PANEL_DARK
        pygame.draw.rect(surface, fill, NEXT_RECT, border_radius=7)
        pygame.draw.rect(surface, page.accent, NEXT_RECT, 1, border_radius=7)
        following = fonts["small_bold"].render("NEXT  ›", True, TEXT)
        surface.blit(following, following.get_rect(center=NEXT_RECT.center))

    return {"back": BACK_RECT, "previous": PREVIOUS_RECT, "next": NEXT_RECT, **tab_rects}


def _draw_flight_page(surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    cards = (
        (pygame.Rect(226, 194, 256, 150), "DIRECT MOVEMENT", CYAN),
        (pygame.Rect(492, 194, 268, 150), "ROTATION + THRUST", PURPLE),
    )
    for rect, title, color in cards:
        pygame.draw.rect(surface, PANEL_DARK, rect, border_radius=11)
        pygame.draw.rect(surface, color, rect, 1, border_radius=11)
        surface.blit(fonts["small_bold"].render(title, True, color), (rect.x + 14, rect.y + 12))

    # Direct-control illustration.
    center = (286, 265)
    _draw_key(surface, fonts, pygame.Rect(center[0] - 14, center[1] - 46, 28, 24), "W")
    _draw_key(surface, fonts, pygame.Rect(center[0] - 46, center[1] - 16, 28, 24), "A")
    _draw_key(surface, fonts, pygame.Rect(center[0] - 14, center[1] - 16, 28, 24), "S")
    _draw_key(surface, fonts, pygame.Rect(center[0] + 18, center[1] - 16, 28, 24), "D")
    surface.blit(fonts["small_bold"].render("MOVE", True, TEXT), (257, 302))
    direct_lines = (
        "WASD / arrows move freely",
        "Space or click shoots",
        "Move and fire together",
    )
    for index, line in enumerate(direct_lines):
        surface.blit(fonts["small"].render(line, True, MUTED), (336, 237 + index * 24))

    # Rotation-control illustration.
    _draw_ship(surface, (548, 263), -0.35, PURPLE, 1.2)
    pygame.draw.arc(surface, CYAN, pygame.Rect(515, 230, 66, 66), 0.35, 2.65, 2)
    pygame.draw.arc(surface, CYAN, pygame.Rect(515, 230, 66, 66), 3.5, 5.8, 2)
    surface.blit(fonts["small_bold"].render("A / D", True, CYAN), (526, 303))
    rotation_lines = (
        "A / D rotate the nose",
        "W applies forward thrust",
        "Hold Space while steering",
        "Momentum needs early turns",
    )
    for index, line in enumerate(rotation_lines):
        surface.blit(fonts["small"].render(line, True, MUTED), (592, 225 + index * 23))

    shoot = pygame.Rect(226, 355, 534, 63)
    pygame.draw.rect(surface, (12, 24, 46), shoot, border_radius=10)
    pygame.draw.rect(surface, YELLOW, shoot, 1, border_radius=10)
    _draw_ship(surface, (258, 386), 0.0, CYAN, 0.7)
    pygame.draw.line(surface, YELLOW, (277, 386), (327, 386), 3)
    pygame.draw.circle(surface, RED, (339, 386), 8)
    surface.blit(fonts["body_bold"].render("SHOOTING", True, YELLOW), (359, 367))
    surface.blit(fonts["small"].render("Cyan/your-color fire is friendly; red and orange fire is hostile.", True, MUTED), (359, 383))
    surface.blit(fonts["micro"].render("P pause  •  Tab build  •  R restart  •  C skin  •  V volume  •  M mute", True, CYAN), (359, 401))

    surface.blit(fonts["eyebrow"].render("MISSION LOOP", True, GREEN), (228, 434))
    labels = ("DESTROY RIFTS", "HOSTILES RETREAT", "CHOOSE REWARD", "NEXT PHASE")
    x = 226
    for index, label in enumerate(labels):
        pill = pygame.Rect(x, 455, 116, 31)
        pygame.draw.rect(surface, PANEL_DARK, pill, border_radius=15)
        pygame.draw.rect(surface, GREEN if index == 2 else LINE, pill, 1, border_radius=15)
        rendered = fonts["micro"].render(label, True, TEXT)
        surface.blit(rendered, rendered.get_rect(center=pill.center))
        if index < len(labels) - 1:
            pygame.draw.line(surface, MUTED, (pill.right + 3, pill.centery), (pill.right + 15, pill.centery), 1)
            pygame.draw.polygon(surface, MUTED, [(pill.right + 15, pill.centery), (pill.right + 10, pill.centery - 3), (pill.right + 10, pill.centery + 3)])
        x += 134


def _draw_battlefield_page(surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    arena = pygame.Rect(226, 194, 330, 279)
    pygame.draw.rect(surface, (9, 18, 38), arena, border_radius=12)
    pygame.draw.rect(surface, GREEN, arena, 1, border_radius=12)
    for x in range(arena.x + 24, arena.right, 44):
        pygame.draw.line(surface, (21, 43, 67), (x, arena.y + 42), (x, arena.bottom - 8))
    for y in range(arena.y + 58, arena.bottom, 44):
        pygame.draw.line(surface, (21, 43, 67), (arena.x + 8, y), (arena.right - 8, y))
    pygame.draw.rect(surface, PANEL, (arena.x + 10, arena.y + 10, 132, 23), border_radius=6)
    pygame.draw.rect(surface, RED, (arena.x + 17, arena.y + 17, 88, 8), border_radius=4)
    surface.blit(fonts["micro"].render("HULL", True, TEXT), (arena.x + 111, arena.y + 16))
    pygame.draw.rect(surface, PANEL, (arena.right - 117, arena.y + 10, 107, 23), border_radius=6)
    surface.blit(fonts["micro"].render("PHASE 2  42s", True, YELLOW), (arena.right - 108, arena.y + 17))

    _draw_ship(surface, (327, 369), -0.55, CYAN, 1.0)
    pygame.draw.circle(surface, PURPLE, (470, 278), 27, 3)
    pygame.draw.circle(surface, RED, (416, 384), 10)
    pygame.draw.circle(surface, ORANGE, (487, 422), 15, 2)
    pygame.draw.line(surface, CYAN, (344, 358), (397, 326), 2)
    pygame.draw.circle(surface, RED, (402, 323), 7)
    surface.blit(fonts["micro"].render("PLAYER", True, CYAN), (291, 394))
    surface.blit(fonts["micro"].render("RIFT", True, PURPLE), (454, 311))
    surface.blit(fonts["micro"].render("ENEMY", True, RED), (397, 401))
    surface.blit(fonts["micro"].render("HUNTER", True, ORANGE), (464, 443))

    surface.blit(fonts["eyebrow"].render("WHAT TO WATCH", True, GREEN), (574, 198))
    intel = (
        (CYAN, "Hull + Aegis", "Aegis blocks entire hits before hull."),
        (YELLOW, "Phase clock", "Clear every active rift before time expires."),
        (PURPLE, "Rifts", "Spawners create enemies; destroy all to advance."),
        (RED, "Hostiles", "Enemies chase, collide, and pressure your position."),
        (ORANGE, "Rift Hunter", "Optional miniboss with an XP and repair cache."),
    )
    for index, (color, title, description) in enumerate(intel):
        y = 223 + index * 49
        pygame.draw.circle(surface, color, (581, y + 7), 5)
        surface.blit(fonts["small_bold"].render(title, True, TEXT), (593, y))
        _wrapped_text(surface, description, fonts["micro"], MUTED, pygame.Rect(593, y + 15, 158, 27), max_lines=2)

    note = pygame.Rect(226, 483, 534, 25)
    pygame.draw.rect(surface, PANEL_DARK, note, border_radius=6)
    surface.blit(fonts["micro"].render("CLOCKS RESET EACH PHASE  •  ordinary 60s  •  onboarding boss 90s  •  later bosses 70s", True, MUTED), (236, 491))


def _draw_boss_page(surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    diagram = pygame.Rect(226, 194, 332, 294)
    pygame.draw.rect(surface, (12, 16, 37), diagram, border_radius=12)
    pygame.draw.rect(surface, RED, diagram, 1, border_radius=12)
    overlay = pygame.Surface(diagram.size, pygame.SRCALPHA)
    horizontal_lane = pygame.Rect(0, 62, diagram.width, 38)
    vertical_lane = pygame.Rect(143, 0, 40, diagram.height)
    pygame.draw.rect(overlay, (255, 57, 100, 92), horizontal_lane)
    pygame.draw.rect(overlay, (255, 57, 100, 92), vertical_lane)
    pygame.draw.rect(overlay, (255, 115, 145, 205), horizontal_lane, 2)
    pygame.draw.rect(overlay, (255, 115, 145, 205), vertical_lane, 2)
    # Warning stripes remain visible even on a dim projector or recording.
    for x in range(-28, diagram.width + 30, 28):
        pygame.draw.line(overlay, (255, 170, 188, 125), (x, 62), (x + 20, 100), 2)
    for y in range(-28, diagram.height + 30, 28):
        pygame.draw.line(overlay, (255, 170, 188, 125), (143, y + 18), (183, y), 2)
    surface.blit(overlay, diagram.topleft)

    boss_center = (392, 319)
    pygame.draw.circle(surface, RED, boss_center, 43, 3)
    pygame.draw.circle(surface, PURPLE, boss_center, 54, 2)
    surface.blit(fonts["small_bold"].render("BOSS", True, TEXT), fonts["small_bold"].render("BOSS", True, TEXT).get_rect(center=boss_center))
    sentries = ((286, 250), (492, 399))
    for sentry in sentries:
        pygame.draw.circle(surface, ORANGE, sentry, 18, 3)
        pygame.draw.rect(surface, ORANGE, pygame.Rect(sentry[0] - 5, sentry[1] - 5, 10, 10), 1)
        pygame.draw.line(surface, GREEN, sentry, boss_center, 3)
    _draw_ship(surface, (284, 425), -0.55, CYAN, 0.9)
    pygame.draw.arc(surface, ORANGE, pygame.Rect(267, 339, 110, 90), 4.1, 5.9, 2)
    pygame.draw.polygon(surface, ORANGE, [(365, 348), (357, 349), (362, 355)])
    surface.blit(fonts["micro"].render("HEAL + IMMUNITY", True, GREEN), (345, 376))
    surface.blit(fonts["micro"].render("AEGIS SENTRY", True, ORANGE), (244, 219))
    surface.blit(fonts["micro"].render("GUIDED MISSILE", True, ORANGE), (243, 459))
    danger_label = pygame.Rect(337, 200, 178, 22)
    pygame.draw.rect(surface, (70, 20, 42), danger_label, border_radius=6)
    pygame.draw.rect(surface, RED, danger_label, 1, border_radius=6)
    danger_text = fonts["small_bold"].render("TELEGRAPHED BOSS SKILL", True, TEXT)
    surface.blit(danger_text, danger_text.get_rect(center=danger_label.center))
    for center in ((251, 275), (534, 275), (392, 232), (392, 469)):
        pygame.draw.circle(surface, RED, center, 10)
        mark = fonts["small_bold"].render("!", True, TEXT)
        surface.blit(mark, mark.get_rect(center=center))

    surface.blit(fonts["eyebrow"].render("TARGET PRIORITY", True, RED), (576, 197))
    steps = (
        ("1", "DODGE THE WARNING", "Leave red lanes/circles before impact."),
        ("2", "BREAK SHIELD", "Damage the boss until its shield collapses."),
        ("3", "KILL SENTRIES", "At low health, Aegis towers make the boss immune and heal it."),
        ("4", "FINISH BOSS", "Destroy every sentry, then resume boss damage."),
    )
    for index, (number, title, detail) in enumerate(steps):
        y = 222 + index * 64
        pygame.draw.circle(surface, RED if index in (0, 2) else PURPLE, (587, y + 9), 11)
        numeral = fonts["small_bold"].render(number, True, TEXT)
        surface.blit(numeral, numeral.get_rect(center=(587, y + 9)))
        surface.blit(fonts["small_bold"].render(title, True, TEXT), (605, y + 1))
        _wrapped_text(
            surface,
            detail,
            fonts["small"],
            MUTED,
            pygame.Rect(605, y + 17, 145, 40),
            line_height=12,
            max_lines=3,
        )

    warning = pygame.Rect(574, 470, 178, 39)
    pygame.draw.rect(surface, (54, 24, 39), warning, border_radius=7)
    pygame.draw.rect(surface, YELLOW, warning, 1, border_radius=7)
    heading = fonts["small_bold"].render("BOSS IMMUNE?", True, YELLOW)
    detail = fonts["micro"].render("TARGET ORANGE SENTRIES", True, TEXT)
    surface.blit(heading, heading.get_rect(center=(warning.centerx, warning.y + 12)))
    surface.blit(detail, detail.get_rect(center=(warning.centerx, warning.y + 28)))


def _draw_drafts_page(surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    drafts = (
        ("LEVEL UP", "EARN XP", "Choose one permanent ship upgrade.", CYAN, "XP BAR FULL"),
        ("PHASE CLEAR", "DESTROY ALL RIFTS", "Choose support for the next assault.", GREEN, "EVERY PHASE"),
        ("BOSS CLEAR", "DEFEAT THE BOSS", "Choose a stronger relic reward.", ORANGE, "PHASE 3, 6, 9…"),
    )
    x_positions = (226, 405, 584)
    for index, ((title, trigger, detail, color, cadence), x) in enumerate(zip(drafts, x_positions)):
        rect = pygame.Rect(x, 198, 166, 248)
        pygame.draw.rect(surface, PANEL_DARK, rect, border_radius=12)
        pygame.draw.rect(surface, color, rect, 2, border_radius=12)
        pygame.draw.circle(surface, color, (rect.centerx, rect.y + 42), 25, 2)
        symbol = ("+", "✓", "★")[index]
        symbol_surface = fonts["title"].render(symbol, True, color)
        surface.blit(symbol_surface, symbol_surface.get_rect(center=(rect.centerx, rect.y + 42)))
        heading = fonts["body_bold"].render(title, True, TEXT)
        surface.blit(heading, heading.get_rect(center=(rect.centerx, rect.y + 83)))
        trigger_surface = fonts["micro"].render(trigger, True, color)
        surface.blit(trigger_surface, trigger_surface.get_rect(center=(rect.centerx, rect.y + 105)))
        _wrapped_text(surface, detail, fonts["small"], MUTED, pygame.Rect(rect.x + 14, rect.y + 130, rect.width - 28, 45), max_lines=3)
        pygame.draw.line(surface, LINE, (rect.x + 14, rect.y + 183), (rect.right - 14, rect.y + 183))
        choice = fonts["small_bold"].render("CHOOSE 1 OF 3", True, YELLOW)
        surface.blit(choice, choice.get_rect(center=(rect.centerx, rect.y + 202)))
        cadence_surface = fonts["micro"].render(cadence, True, MUTED)
        surface.blit(cadence_surface, cadence_surface.get_rect(center=(rect.centerx, rect.y + 225)))

    callout = pygame.Rect(226, 459, 524, 43)
    pygame.draw.rect(surface, (14, 31, 51), callout, border_radius=9)
    pygame.draw.rect(surface, YELLOW, callout, 1, border_radius=9)
    surface.blit(fonts["small_bold"].render("THE BATTLE PAUSES WHILE YOU CHOOSE", True, YELLOW), (240, 469))
    surface.blit(fonts["micro"].render("Click a card or press 1 / 2 / 3. AI pilots choose from the same seeded offers.", True, MUTED), (240, 486))


def _upgrade_color(identifier: str) -> tuple[int, int, int]:
    if identifier in {"hull", "repair", "shield", "regen", "hull_mastery", "leech"}:
        return GREEN
    if identifier in {"drone", "drone_mastery", "homing", "engine"}:
        return PURPLE
    if identifier in {"damage", "critical", "riftbreaker", "weapon_mastery"}:
        return ORANGE
    return CYAN


def _draw_upgrade_entry(
    surface: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    item: dict[str, Any],
    rect: pygame.Rect,
    number: int,
) -> None:
    identifier = str(item["id"])
    color = _upgrade_color(identifier)
    pygame.draw.rect(surface, PANEL_DARK, rect, border_radius=8)
    pygame.draw.rect(surface, LINE, rect, 1, border_radius=8)
    pygame.draw.circle(surface, color, (rect.x + 18, rect.y + 19), 11, 2)
    number_surface = fonts["micro"].render(str(number), True, TEXT)
    surface.blit(number_surface, number_surface.get_rect(center=(rect.x + 18, rect.y + 19)))
    name = fonts["small_bold"].render(str(item["name"]), True, color)
    surface.blit(name, (rect.x + 35, rect.y + 7))
    _wrapped_text(
        surface,
        str(item["description"]),
        fonts["small"],
        MUTED,
        pygame.Rect(rect.x + 35, rect.y + 22, rect.width - 43, rect.height - 25),
        line_height=12,
        max_lines=3,
    )


def _draw_upgrade_page(
    surface: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    start: int,
    stop: int,
) -> None:
    upgrades = guide_catalog_entries()["upgrades"][start:stop]
    per_column = 4
    for local_index, item in enumerate(upgrades):
        column = local_index // per_column
        row = local_index % per_column
        rect = pygame.Rect(226 + column * 267, 193 + row * 73, 256, 66)
        _draw_upgrade_entry(surface, fonts, item, rect, start + local_index + 1)

    note = pygame.Rect(226, 491, 524, 20)
    if start == 0:
        message = "BUILD TIP  •  offense clears clocks; mobility and defense keep the damage window open."
    elif start < 14:
        message = "SURVIVAL TIP  •  combine mobility or shields with damage instead of choosing only offense."
    else:
        message = "MASTERY CARDS 19–21 unlock after ship level 8 and remain repeatable for endless scaling."
    label = fonts["micro"].render(message, True, YELLOW)
    surface.blit(label, label.get_rect(center=note.center))


def _draw_reward_entry(
    surface: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    item: dict[str, Any],
    rect: pygame.Rect,
    color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(surface, PANEL_DARK, rect, border_radius=8)
    pygame.draw.rect(surface, LINE, rect, 1, border_radius=8)
    pygame.draw.circle(surface, color, (rect.x + 17, rect.y + 18), 8, 2)
    surface.blit(fonts["small_bold"].render(str(item["name"]), True, color), (rect.x + 33, rect.y + 7))
    _wrapped_text(
        surface,
        str(item["description"]),
        fonts["small"],
        MUTED,
        pygame.Rect(rect.x + 12, rect.y + 30, rect.width - 24, rect.height - 35),
        line_height=13,
        max_lines=4,
    )


def _draw_phase_rewards_page(
    surface: pygame.Surface, fonts: dict[str, pygame.font.Font]
) -> None:
    items = guide_catalog_entries()["phase_rewards"]
    for index, item in enumerate(items):
        column, row = divmod(index, 3)
        rect = pygame.Rect(226 + column * 267, 198 + row * 96, 256, 86)
        _draw_reward_entry(surface, fonts, item, rect, GREEN)
    note = pygame.Rect(226, 491, 524, 20)
    label = fonts["micro"].render(
        "PHASE SUPPORT  •  consumed or activated around the next assault",
        True,
        YELLOW,
    )
    surface.blit(label, label.get_rect(center=note.center))


def _draw_boss_rewards_page(
    surface: pygame.Surface, fonts: dict[str, pygame.font.Font]
) -> None:
    items = guide_catalog_entries()["boss_rewards"]
    for index, item in enumerate(items):
        column, row = divmod(index, 2)
        rect = pygame.Rect(226 + column * 267, 198 + row * 140, 256, 128)
        _draw_reward_entry(surface, fonts, item, rect, ORANGE)
    note = pygame.Rect(226, 491, 524, 20)
    label = fonts["micro"].render(
        "BOSS RELICS  •  stronger rewards match the risk of every third phase",
        True,
        YELLOW,
    )
    surface.blit(label, label.get_rect(center=note.center))


def _draw_build_style_page(
    surface: pygame.Surface, fonts: dict[str, pygame.font.Font]
) -> None:
    # A readable miniature of the real TAB build panel.
    build = pygame.Rect(226, 194, 326, 306)
    pygame.draw.rect(surface, PANEL_DARK, build, border_radius=12)
    pygame.draw.rect(surface, CYAN, build, 1, border_radius=12)
    surface.blit(fonts["title"].render("SHIP BUILD", True, TEXT), (242, 208))
    tab = pygame.Rect(451, 207, 83, 24)
    pygame.draw.rect(surface, PANEL_HOVER, tab, border_radius=6)
    pygame.draw.rect(surface, CYAN, tab, 1, border_radius=6)
    tab_label = fonts["small_bold"].render("PRESS TAB", True, CYAN)
    surface.blit(tab_label, tab_label.get_rect(center=tab.center))
    surface.blit(
        fonts["small"].render("Battle and phase clock pause while open.", True, MUTED),
        (242, 239),
    )
    pygame.draw.line(surface, LINE, (242, 260), (536, 260))

    surface.blit(fonts["eyebrow"].render("PRISM LASER", True, GREEN), (242, 273))
    weapon_rows = (
        ("BEAMS", "3"),
        ("DAMAGE", "27.6 each"),
        ("COOLDOWN", "0.22s"),
        ("PIERCING", "+2 targets"),
        ("CRITICAL", "16% ×2"),
    )
    for index, (label, value) in enumerate(weapon_rows):
        y = 299 + index * 26
        surface.blit(fonts["small_bold"].render(label, True, MUTED), (242, y))
        rendered = fonts["small"].render(value, True, TEXT)
        surface.blit(rendered, (376, y))

    surface.blit(fonts["eyebrow"].render("SELECTED UPGRADES", True, PURPLE), (432, 273))
    selected = (
        "Multi-Beam Array  T2",
        "Prism Laser  T1",
        "Reactive Shield  T2",
        "Wingman Core  T3",
        "Vector Thrusters  T1",
    )
    for index, label in enumerate(selected):
        chip = pygame.Rect(432, 295 + index * 31, 104, 24)
        pygame.draw.rect(surface, (20, 32, 60), chip, border_radius=6)
        pygame.draw.rect(surface, LINE, chip, 1, border_radius=6)
        fitted = label
        while fitted and fonts["micro"].size(fitted + "…")[0] > chip.width - 10:
            fitted = fitted[:-1]
        if fitted != label:
            fitted = fitted.rstrip() + "…"
        rendered = fonts["micro"].render(fitted, True, TEXT)
        surface.blit(rendered, (chip.x + 6, chip.y + 7))

    support = pygame.Rect(242, 450, 294, 34)
    pygame.draw.rect(surface, (12, 38, 47), support, border_radius=7)
    pygame.draw.rect(surface, GREEN, support, 1, border_radius=7)
    support_text = fonts["micro"].render(
        "DRONES 2  •  AEGIS 3  •  OVERDRIVE ACTIVE", True, GREEN
    )
    surface.blit(support_text, support_text.get_rect(center=support.center))

    # Skin carousel mirrors the five renderer palettes and their matching SFX.
    skin_panel = pygame.Rect(564, 194, 188, 306)
    pygame.draw.rect(surface, PANEL_DARK, skin_panel, border_radius=11)
    pygame.draw.rect(surface, PURPLE, skin_panel, 1, border_radius=11)
    surface.blit(fonts["eyebrow"].render("SHIP SKINS", True, PURPLE), (578, 207))
    surface.blit(fonts["small_bold"].render("PRESS C TO CYCLE", True, TEXT), (578, 226))
    pygame.draw.circle(surface, (18, 35, 62), (658, 276), 31)
    pygame.draw.circle(surface, CYAN, (658, 276), 31, 1)
    pygame.draw.circle(surface, CYAN, (658, 276), 23, 1)
    _draw_ship(surface, (658, 276), -0.35, CYAN, 0.78)

    skin_rows = (
        ("NEON CYBER", CYAN),
        ("SOLAR FLARE", ORANGE),
        ("VOID PHANTOM", PURPLE),
        ("EMERALD AEGIS", GREEN),
        ("SYNTHWAVE PINK", (255, 85, 195)),
    )
    for index, (label, color) in enumerate(skin_rows):
        row = pygame.Rect(578, 318 + index * 27, 160, 22)
        pygame.draw.rect(surface, (17, 29, 54), row, border_radius=5)
        pygame.draw.circle(surface, color, (row.x + 10, row.centery), 4)
        surface.blit(fonts["micro"].render(label, True, TEXT), (row.x + 20, row.y + 6))

    pygame.draw.line(surface, LINE, (578, 459), (738, 459))
    skin_audio = fonts["micro"].render("MATCHED THRUST + LASER SFX", True, MUTED)
    surface.blit(skin_audio, skin_audio.get_rect(center=(658, 473)))
    cosmetic = fonts["micro"].render("COSMETIC  •  RULES UNCHANGED", True, GREEN)
    surface.blit(cosmetic, cosmetic.get_rect(center=(658, 488)))


def _draw_other_features_page(
    surface: pygame.Surface, fonts: dict[str, pygame.font.Font]
) -> None:
    """Illustrate the quality-of-life and presentation systems players can use."""

    pause_card = pygame.Rect(226, 194, 256, 136)
    replay_card = pygame.Rect(492, 194, 268, 136)
    audio_card = pygame.Rect(226, 340, 326, 160)
    tools_card = pygame.Rect(562, 340, 198, 160)
    for rect, accent in (
        (pause_card, CYAN),
        (replay_card, GREEN),
        (audio_card, PURPLE),
        (tools_card, YELLOW),
    ):
        pygame.draw.rect(surface, PANEL_DARK, rect, border_radius=11)
        pygame.draw.rect(surface, accent, rect, 1, border_radius=11)

    # Pause: the same keyboard command and visible HUD button work in both modes.
    icon = pygame.Rect(241, 208, 42, 42)
    pygame.draw.circle(surface, (17, 45, 67), icon.center, 21)
    pygame.draw.circle(surface, CYAN, icon.center, 21, 2)
    pygame.draw.rect(surface, CYAN, (icon.x + 13, icon.y + 10, 5, 22), border_radius=2)
    pygame.draw.rect(surface, CYAN, (icon.x + 24, icon.y + 10, 5, 22), border_radius=2)
    surface.blit(fonts["title"].render("Pause & resume", True, TEXT), (294, 208))
    _draw_key(surface, fonts, pygame.Rect(294, 235, 30, 24), "P")
    surface.blit(fonts["small"].render("or click the HUD button", True, MUTED), (332, 241))
    _wrapped_text(
        surface,
        "Simulation and the phase clock freeze. Resume when ready—no mission time is lost.",
        fonts["small"],
        MUTED,
        pygame.Rect(241, 272, 226, 48),
        line_height=18,
        max_lines=3,
    )

    # Replay: distinguish an immediate seeded restart from the persistent summary.
    replay_center = (518, 229)
    pygame.draw.arc(surface, GREEN, (497, 208, 42, 42), 0.25, math.tau - 0.7, 3)
    pygame.draw.polygon(surface, GREEN, ((535, 207), (540, 220), (526, 218)))
    surface.blit(fonts["title"].render("Replay the run", True, TEXT), (550, 208))
    _draw_key(surface, fonts, pygame.Rect(550, 235, 30, 24), "R", GREEN)
    surface.blit(fonts["small"].render("restart the current seed", True, MUTED), (588, 241))
    _wrapped_text(
        surface,
        "After game over, the mission summary stays open until you choose Replay or Main Menu.",
        fonts["small"],
        MUTED,
        pygame.Rect(507, 272, 238, 48),
        line_height=18,
        max_lines=3,
    )

    # Audio: all music and effects are synthesized, so the repository remains portable.
    surface.blit(fonts["eyebrow"].render("PROCEDURAL MUSIC + SOUND", True, PURPLE), (241, 353))
    for index, height in enumerate((12, 25, 18, 33, 22, 29, 15, 35, 20)):
        x = 243 + index * 9
        pygame.draw.rect(surface, PURPLE, (x, 403 - height // 2, 4, height), border_radius=2)
    pygame.draw.circle(surface, (29, 23, 59), (342, 393), 25)
    pygame.draw.circle(surface, PURPLE, (342, 393), 25, 2)
    pygame.draw.circle(surface, TEXT, (342, 393), 5)
    pygame.draw.line(surface, PURPLE, (342, 388), (342, 374), 3)
    pygame.draw.line(surface, PURPLE, (342, 374), (358, 370), 3)
    pygame.draw.circle(surface, PURPLE, (358, 379), 5)
    audio_lines = (
        "Original synthwave loop",
        "Combat, boss + reward cues",
        "Skin-specific laser voices",
    )
    for index, line in enumerate(audio_lines):
        pygame.draw.circle(surface, PURPLE, (379, 378 + index * 21), 3)
        surface.blit(fonts["small"].render(line, True, MUTED), (388, 372 + index * 21))
    _draw_key(surface, fonts, pygame.Rect(241, 454, 30, 24), "V", PURPLE)
    surface.blit(fonts["small"].render("opens mixer", True, TEXT), (279, 460))
    _draw_key(surface, fonts, pygame.Rect(366, 454, 30, 24), "M", PURPLE)
    surface.blit(fonts["small"].render("mute while mixer is open", True, TEXT), (404, 460))
    surface.blit(fonts["micro"].render("Audio is presentation-only; gameplay and learning stay unchanged.", True, GREEN), (241, 484))

    # Evaluation playback and quick-reference overlays.
    surface.blit(fonts["eyebrow"].render("AI PLAYBACK", True, YELLOW), (577, 353))
    _draw_key(surface, fonts, pygame.Rect(577, 374, 30, 24), ".", YELLOW)
    surface.blit(fonts["small"].render("single step", True, MUTED), (615, 380))
    _draw_key(surface, fonts, pygame.Rect(577, 405, 30, 24), "±", YELLOW)
    surface.blit(fonts["small"].render("0.5× to 4×", True, MUTED), (615, 411))
    pygame.draw.line(surface, LINE, (577, 440), (745, 440))
    surface.blit(fonts["eyebrow"].render("QUICK ACCESS", True, CYAN), (577, 449))
    surface.blit(fonts["small"].render("H or /   help overlay", True, MUTED), (577, 468))
    surface.blit(fonts["small"].render("Tab / C   build + skin", True, MUTED), (577, 486))


def draw_guidebook_page(
    surface: pygame.Surface,
    page_index: int,
    *,
    mouse: tuple[int, int] = (-1, -1),
    elapsed: float = 0.0,
    fonts: dict[str, pygame.font.Font] | None = None,
) -> dict[str, pygame.Rect]:
    """Render one complete guide page and return its navigation hitboxes."""

    if not 0 <= page_index < len(PAGES):
        raise ValueError(f"page_index must be between 0 and {len(PAGES) - 1}")
    active_fonts = fonts or _fonts()
    _draw_background(surface, elapsed)
    hitboxes = _draw_chapter_frame(surface, active_fonts, page_index, mouse)
    key = PAGES[page_index].key
    if key == "flight":
        _draw_flight_page(surface, active_fonts)
    elif key == "battlefield":
        _draw_battlefield_page(surface, active_fonts)
    elif key == "boss":
        _draw_boss_page(surface, active_fonts)
    elif key == "drafts":
        _draw_drafts_page(surface, active_fonts)
    elif key == "upgrades_a":
        _draw_upgrade_page(surface, active_fonts, 0, 7)
    elif key == "upgrades_b":
        _draw_upgrade_page(surface, active_fonts, 7, 14)
    elif key == "upgrades_c":
        _draw_upgrade_page(surface, active_fonts, 14, 21)
    elif key == "phase_rewards":
        _draw_phase_rewards_page(surface, active_fonts)
    elif key == "boss_rewards":
        _draw_boss_rewards_page(surface, active_fonts)
    elif key == "build_style":
        _draw_build_style_page(surface, active_fonts)
    else:
        _draw_other_features_page(surface, active_fonts)
    return hitboxes


def open_guidebook(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    audio: Any | None = None,
    *,
    start_page: int = 0,
) -> None:
    """Run the menu-owned guidebook until the player returns to the launcher."""

    page_index = max(0, min(len(PAGES) - 1, int(start_page)))
    fonts = _fonts()
    elapsed = 0.0

    def play_click() -> None:
        if audio is not None and getattr(audio, "available", False):
            audio.play("click", minimum_interval_ms=40)

    while True:
        elapsed += clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()
        hitboxes = draw_guidebook_page(
            screen,
            page_index,
            mouse=mouse,
            elapsed=elapsed,
            fonts=fonts,
        )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_g, pygame.K_q):
                    play_click()
                    return
                if event.key in (pygame.K_RIGHT, pygame.K_PAGEDOWN, pygame.K_d):
                    next_page = min(len(PAGES) - 1, page_index + 1)
                    if next_page != page_index:
                        page_index = next_page
                        play_click()
                elif event.key in (pygame.K_LEFT, pygame.K_PAGEUP, pygame.K_a):
                    next_page = max(0, page_index - 1)
                    if next_page != page_index:
                        page_index = next_page
                        play_click()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    page_index = min(len(PAGES) - 1, event.key - pygame.K_1)
                    play_click()
                elif event.key == pygame.K_0:
                    page_index = 9
                    play_click()
                elif event.key == pygame.K_o:
                    page_index = 10
                    play_click()
            elif event.type == pygame.MOUSEWHEEL:
                next_page = max(0, min(len(PAGES) - 1, page_index - event.y))
                if next_page != page_index:
                    page_index = next_page
                    play_click()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hitboxes["back"].collidepoint(event.pos):
                    play_click()
                    return
                if page_index > 0 and hitboxes["previous"].collidepoint(event.pos):
                    page_index -= 1
                    play_click()
                    continue
                if page_index < len(PAGES) - 1 and hitboxes["next"].collidepoint(event.pos):
                    page_index += 1
                    play_click()
                    continue
                for index in range(len(PAGES)):
                    if hitboxes[f"page_{index}"].collidepoint(event.pos):
                        page_index = index
                        play_click()
                        break
        pygame.display.flip()


__all__ = [
    "PAGES",
    "draw_guidebook_page",
    "guide_catalog_entries",
    "guide_page_count",
    "open_guidebook",
]
