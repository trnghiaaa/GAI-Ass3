"""
Pygame renderer for the GridWorld environment.

Draws the grid, agent, items, monsters, fire, and a HUD bar showing
episode info.
"""

import pygame
import sys

# ── Colour palette ─────────────────────────────────────────────────────
COLORS = {
    "bg":           (245, 245, 240),   # warm off-white
    "grid_line":    (200, 200, 195),
    "empty":        (235, 235, 230),
    "rock":         ( 90,  80,  70),
    "fire":         (220,  60,  30),
    "fire_accent":  (255, 160,  40),
    "apple":        ( 76, 175,  80),
    "apple_leaf":   ( 46, 125,  50),
    "key":          (255, 193,   7),
    "key_dark":     (200, 150,   0),
    "chest":        (141, 110,  99),
    "chest_lock":   (255, 193,   7),
    "monster":      (183,  28,  28),
    "monster_eye":  (255, 255, 255),
    "player":       ( 33, 150, 243),
    "player_dark":  ( 21, 101, 192),
    "hud_bg":       ( 50,  50,  50),
    "hud_text":     (240, 240, 240),
}

HUD_HEIGHT = 40  # pixels


class GridWorldRenderer:
    """Render the gridworld with Pygame."""

    def __init__(self, env, cell_size=60, fps=10, title="Gridworld RL"):
        self.env = env
        self.cell_size = cell_size
        self.fps = fps

        self.width = env.cols * cell_size
        self.height = env.rows * cell_size + HUD_HEIGHT

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self, episode=None, step=None, total_reward=None):
        """Draw the current state of the environment."""
        self._handle_events()

        self.screen.fill(COLORS["bg"])
        grid = self.env.get_grid_for_render()

        cs = self.cell_size
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x = c * cs
                y = r * cs
                rect = pygame.Rect(x, y, cs, cs)
                tile = grid[r][c]

                # Base fill
                if tile == "R":
                    self._draw_rock(rect)
                elif tile == "F":
                    self._draw_fire(rect)
                else:
                    pygame.draw.rect(self.screen, COLORS["empty"], rect)

                # Grid line
                pygame.draw.rect(self.screen, COLORS["grid_line"], rect, 1)

                # Entities on top of empty tiles
                if tile == "A":
                    self._draw_apple(rect)
                elif tile == "K":
                    self._draw_key(rect)
                elif tile == "C":
                    self._draw_chest(rect)
                elif tile == "M":
                    self._draw_monster(rect)
                elif tile == "P":
                    self._draw_player(rect)

        # HUD
        self._draw_hud(episode, step, total_reward)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        """Shut down Pygame."""
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_rock(self, rect):
        pygame.draw.rect(self.screen, COLORS["rock"], rect)
        # Subtle texture lines
        for offset in range(4, rect.width, 8):
            pygame.draw.line(self.screen, (100, 90, 80),
                             (rect.x + offset, rect.y),
                             (rect.x + offset, rect.y + rect.height), 1)

    def _draw_fire(self, rect):
        pygame.draw.rect(self.screen, COLORS["fire"], rect)
        # Flame accent triangle
        cx = rect.centerx
        points = [
            (cx, rect.y + 6),
            (cx - rect.width // 4, rect.y + rect.height - 6),
            (cx + rect.width // 4, rect.y + rect.height - 6),
        ]
        pygame.draw.polygon(self.screen, COLORS["fire_accent"], points)

    def _draw_apple(self, rect):
        cx, cy = rect.centerx, rect.centery
        radius = self.cell_size // 4
        pygame.draw.circle(self.screen, COLORS["apple"], (cx, cy + 2), radius)
        # Stem
        pygame.draw.line(self.screen, COLORS["apple_leaf"],
                         (cx, cy - radius + 2), (cx + 3, cy - radius - 4), 2)

    def _draw_key(self, rect):
        cx, cy = rect.centerx, rect.centery
        r = self.cell_size // 6
        pygame.draw.circle(self.screen, COLORS["key"], (cx, cy - 4), r, 3)
        pygame.draw.line(self.screen, COLORS["key"],
                         (cx, cy), (cx, cy + r + 6), 3)
        pygame.draw.line(self.screen, COLORS["key"],
                         (cx, cy + r + 2), (cx + 5, cy + r + 2), 3)

    def _draw_chest(self, rect):
        margin = self.cell_size // 5
        body = pygame.Rect(rect.x + margin, rect.y + margin + 4,
                           rect.width - 2 * margin, rect.height - 2 * margin - 4)
        pygame.draw.rect(self.screen, COLORS["chest"], body, border_radius=3)
        # Lock
        lx = body.centerx
        ly = body.centery
        pygame.draw.circle(self.screen, COLORS["chest_lock"], (lx, ly), 4)

    def _draw_monster(self, rect):
        cx, cy = rect.centerx, rect.centery
        r = self.cell_size // 3
        # Body
        pygame.draw.circle(self.screen, COLORS["monster"], (cx, cy), r)
        # Eyes
        eye_r = max(2, r // 4)
        pygame.draw.circle(self.screen, COLORS["monster_eye"],
                           (cx - r // 3, cy - r // 4), eye_r)
        pygame.draw.circle(self.screen, COLORS["monster_eye"],
                           (cx + r // 3, cy - r // 4), eye_r)

    def _draw_player(self, rect):
        cx, cy = rect.centerx, rect.centery
        r = self.cell_size // 3
        pygame.draw.circle(self.screen, COLORS["player"], (cx, cy), r)
        pygame.draw.circle(self.screen, COLORS["player_dark"], (cx, cy), r, 3)
        # Inner dot
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, r // 4))

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, episode, step, total_reward):
        hud_y = self.env.rows * self.cell_size
        hud_rect = pygame.Rect(0, hud_y, self.width, HUD_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["hud_bg"], hud_rect)

        parts = []
        if episode is not None:
            parts.append(f"Episode: {episode}")
        if step is not None:
            parts.append(f"Step: {step}")
        if total_reward is not None:
            parts.append(f"Reward: {total_reward:.1f}")
        if self.env.has_key:
            parts.append("KEY [Y]")

        text = "   |   ".join(parts) if parts else ""
        surf = self.font.render(text, True, COLORS["hud_text"])
        self.screen.blit(surf, (12, hud_y + 10))

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _handle_events(self):
        """Process Pygame events so the window stays responsive."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
