"""Unified master launcher and visual portal for Assignment 3 RL projects.

Examples::

    python main.py                  # Open the interactive Visual Master Hub
    python main.py --part 1         # Launch Part I (Gridworld AI Lab) directly
    python main.py --part 2         # Launch Part II (Neon Rift Arena) directly
    python main.py --menu           # Explicitly launch Visual Master Hub
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Optional, Sequence

import numpy as np
import pygame


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 22_050


class MasterAudio:
    """Original minimal ambient soundtrack and UI sound mixer for the Master Portal."""

    def __init__(self) -> None:
        self.available = False
        self.enabled = True
        self.volume: float = 0.8
        self.music_channel: pygame.mixer.Channel | None = None
        self.music: pygame.mixer.Sound | None = None
        self.effects: dict[str, pygame.mixer.Sound] = {}

        try:
            current = pygame.mixer.get_init()
            if current != (SAMPLE_RATE, -16, 2):
                if current is not None:
                    pygame.mixer.quit()
                pygame.mixer.init(
                    frequency=SAMPLE_RATE,
                    size=-16,
                    channels=2,
                    buffer=512,
                )
            pygame.mixer.set_num_channels(max(12, pygame.mixer.get_num_channels()))
            # Reserve Channel 0 strictly for ambient music
            pygame.mixer.set_reserved(1)
            self.music_channel = pygame.mixer.Channel(0)
            self.music = self._make_sound(self._build_ambient_portal_music())
            self.music.set_volume(self.volume * 0.22)
            self.effects = self._build_effects()
            for sound in self.effects.values():
                sound.set_volume(self.volume)
            self.available = True
        except (pygame.error, ValueError, TypeError):
            self.available = False
            self.music_channel = None
            self.music = None
            self.effects = {}

    def set_volume(self, volume: float) -> float:
        self.volume = max(0.0, min(1.0, float(volume)))
        if not self.available:
            return self.volume
        if self.volume <= 0.001:
            self.enabled = False
            if self.music_channel:
                self.music_channel.set_volume(0.0)
            for sound in self.effects.values():
                sound.set_volume(0.0)
        else:
            self.enabled = True
            if self.music_channel:
                self.music_channel.set_volume(self.volume * 0.22)
            for sound in self.effects.values():
                sound.set_volume(self.volume)
        return self.volume

    def get_volume(self) -> float:
        return self.volume

    def toggle(self) -> bool:
        if not self.available:
            return False
        self.enabled = not self.enabled
        if not self.enabled:
            if self.music_channel:
                self.music_channel.set_volume(0.0)
            for sound in self.effects.values():
                sound.set_volume(0.0)
        else:
            if self.volume <= 0.01:
                self.volume = 0.8
            if self.music_channel:
                self.music_channel.set_volume(self.volume * 0.22)
            for sound in self.effects.values():
                sound.set_volume(self.volume)
        return self.enabled

    def start_music(self) -> None:
        if self.available and self.music and self.music_channel and not self.music_channel.get_busy():
            self.music_channel.play(self.music, loops=-1)

    def play(self, name: str) -> None:
        if not self.available or not self.enabled:
            return
        sound = self.effects.get(name)
        if sound:
            sound.play()

    def close(self) -> None:
        if self.music_channel:
            self.music_channel.stop()
        self.music_channel = None
        self.music = None
        self.effects = {}

    @staticmethod
    def _pad_note(frequency: float, duration: float, volume: float = 0.2) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        if frequency <= 0:
            return np.zeros(count, dtype=np.float64)
        t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        lfo = 0.6 * np.sin(math.tau * 3.5 * t)
        fundamental = np.sin(math.tau * (frequency + lfo) * t)
        warmth = 0.22 * np.sin(math.tau * (frequency * 2.0) * t)
        sub = 0.15 * np.sin(math.tau * (frequency * 0.5) * t)
        wave = (fundamental + warmth + sub) * volume

        attack = int(count * 0.25)
        release = int(count * 0.35)
        envelope = np.ones(count, dtype=np.float64)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        return wave * envelope

    @staticmethod
    def _bell_note(frequency: float, duration: float, volume: float = 0.25) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        if frequency <= 0:
            return np.zeros(count, dtype=np.float64)
        t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        fundamental = np.sin(math.tau * frequency * t)
        crystal = 0.32 * np.sin(math.tau * frequency * 2.75 * t)
        wave = (fundamental + crystal) * volume
        decay = np.exp(-t * 3.2)
        return wave * decay

    @classmethod
    def _build_ambient_portal_music(cls) -> np.ndarray:
        """Synthesize a serene, balanced minimal ambient loop uniting Tabular and Deep RL."""
        chord_dur = 2.4
        chord_samples = int(SAMPLE_RATE * chord_dur)
        total_samples = chord_samples * 4
        output = np.zeros(total_samples, dtype=np.float64)

        chords = (
            (130.81, 196.00, 293.66, 329.63, 493.88),  # Cmaj9
            (110.00, 164.81, 261.63, 329.63, 392.00),  # Amin9
            (87.31, 130.81, 220.00, 293.66, 329.63),   # Fmaj9
            (98.00, 146.83, 196.00, 261.63, 293.66),   # Gsus4/add9
        )

        arps = (
            (659.25, 987.77, 783.99, 587.33),  # E5, B5, G5, D5
            (523.25, 659.25, 783.99, 493.88),  # C5, E5, G5, B4
            (880.00, 659.25, 523.25, 783.99),  # A5, E5, C5, G5
            (587.33, 880.00, 783.99, 987.77),  # D5, A5, G5, B5
        )

        for i, (chord, arp) in enumerate(zip(chords, arps)):
            start = i * chord_samples
            pad_mix = np.zeros(chord_samples, dtype=np.float64)
            for freq in chord:
                pad_mix += cls._pad_note(freq, chord_dur, volume=0.065)

            sub_bass = cls._pad_note(chord[0] / 2.0, chord_dur, volume=0.09)

            arp_mix = np.zeros(chord_samples, dtype=np.float64)
            step_dur = chord_dur / 4.0
            for j, note_freq in enumerate(arp):
                note_start = int(j * step_dur * SAMPLE_RATE)
                bell = cls._bell_note(note_freq, step_dur * 1.5, volume=0.08)
                end = min(chord_samples, note_start + len(bell))
                arp_mix[note_start:end] += bell[: end - note_start]

            combined = pad_mix + sub_bass + arp_mix
            output[start : start + chord_samples] += combined

        return np.clip(output, -1.0, 1.0)

    @classmethod
    def _make_sound(cls, mono: np.ndarray) -> pygame.mixer.Sound:
        pcm = np.asarray(np.clip(mono, -1.0, 1.0) * 32767, dtype=np.int16)
        stereo = np.ascontiguousarray(np.column_stack((pcm, pcm)))
        return pygame.sndarray.make_sound(stereo)

    @classmethod
    def _build_effects(cls) -> dict[str, pygame.mixer.Sound]:
        hover_wave = cls._bell_note(1046.50, 0.06, 0.12)
        s1 = cls._bell_note(587.33, 0.10, 0.16)
        s2 = cls._bell_note(880.00, 0.26, 0.20)
        select_wave = np.concatenate([s1, s2])
        click_wave = cls._bell_note(440.0, 0.04, 0.14)

        return {
            "hover": cls._make_sound(hover_wave),
            "select": cls._make_sound(select_wave),
            "click": cls._make_sound(click_wave),
        }


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_python() -> Path | None:
    """Return this checkout's virtual-environment interpreter when available."""
    candidates = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _same_interpreter(first: str | Path, second: str | Path) -> bool:
    """Compare interpreter paths using the host platform's path semantics."""
    return os.path.normcase(os.path.realpath(first)) == os.path.normcase(
        os.path.realpath(second)
    )


def _ensure_project_runtime() -> None:
    """Re-launch through ``.venv`` if the selected Python lacks Pygame."""
    if importlib.util.find_spec("pygame") is not None:
        return

    project_python = _project_python()
    if project_python is not None and not _same_interpreter(
        sys.executable, project_python
    ):
        completed = subprocess.run(
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(completed.returncode)

    raise SystemExit(
        "Pygame is not installed and no usable project .venv was found.\n"
        "Create the environment and install the requirements with:\n"
        "  python -m venv .venv\n"
        "  .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    )


class MasterLauncher:
    """Presentation-ready visual hub uniting Part I and Part II."""

    WIDTH = 1020
    HEIGHT = 680

    COLORS = {
        "space": (8, 13, 27),
        "panel": (15, 23, 44),
        "panel_hover": (20, 31, 58),
        "card_bg": (12, 19, 38),
        "card_hover": (18, 28, 54),
        "border": (30, 44, 78),
        "border_subtle": (22, 32, 58),
        "text": (240, 246, 255),
        "muted": (136, 153, 184),
        "faint": (82, 98, 128),
        "p1_accent": (46, 204, 113),       # Emerald
        "p1_tag": (34, 150, 83),
        "p1_card_border": (39, 174, 96),
        "p2_accent": (65, 210, 255),       # Cyan
        "p2_tag": (30, 140, 190),
        "p2_card_border": (50, 180, 230),
        "gold": (255, 204, 60),
        "purple": (180, 115, 255),
    }

    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Reinforcement Learning Lab — Assignment 3 Master Portal")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("bahnschrift", 34, bold=True)
        self.font_subtitle = pygame.font.SysFont("bahnschrift", 15)
        self.font_card_title = pygame.font.SysFont("bahnschrift", 20, bold=True)
        self.font_card_tag = pygame.font.SysFont("bahnschrift", 12, bold=True)
        self.font_body = pygame.font.SysFont("bahnschrift", 14)
        self.font_button = pygame.font.SysFont("bahnschrift", 16, bold=True)
        self.font_footer = pygame.font.SysFont("bahnschrift", 13)

        # Ambient Starfield
        star_rng = random.Random(9102)
        self.stars = [
            (
                star_rng.randrange(0, self.WIDTH),
                star_rng.randrange(0, self.HEIGHT),
                star_rng.choice((1, 1, 2)),
                star_rng.randrange(80, 200),
            )
            for _ in range(90)
        ]

        # Card geometries
        self.card_p1 = pygame.Rect(55, 145, 435, 445)
        self.card_p2 = pygame.Rect(530, 145, 435, 445)
        self.btn_p1 = pygame.Rect(self.card_p1.x + 30, self.card_p1.bottom - 68, self.card_p1.width - 60, 48)
        self.btn_p2 = pygame.Rect(self.card_p2.x + 30, self.card_p2.bottom - 68, self.card_p2.width - 60, 48)

        # Volume Controls & Audio
        self.volume = 0.8
        self.show_volume_slider = False
        self.show_help_overlay = False
        self.dragging_volume = False
        self.volume_btn_rect = pygame.Rect(self.WIDTH - 120, self.HEIGHT - 38, 100, 28)
        self.help_btn_rect = pygame.Rect(self.WIDTH - 230, self.HEIGHT - 38, 100, 28)
        self.volume_panel_rect = pygame.Rect(self.WIDTH - 300, self.HEIGHT - 175, 280, 130)
        self.volume_track_rect = pygame.Rect(self.WIDTH - 280, self.HEIGHT - 123, 240, 10)

        self.audio = MasterAudio()
        self.audio.start_music()
        self.last_hover: str | None = None

    def run(self) -> str:
        """Run the Master Hub event loop and return selected project ('1', '2', or 'quit')."""
        while True:
            mouse_pos = pygame.mouse.get_pos()
            hover_p1 = self.card_p1.collidepoint(mouse_pos)
            hover_p2 = self.card_p2.collidepoint(mouse_pos)
            hover_vol_btn = self.volume_btn_rect.collidepoint(mouse_pos)
            hover_help_btn = self.help_btn_rect.collidepoint(mouse_pos)

            current_hover = "p1" if hover_p1 else ("p2" if hover_p2 else None)
            if current_hover != self.last_hover and current_hover is not None:
                if self.audio:
                    self.audio.play("hover")
            self.last_hover = current_hover

            if hover_p1 or hover_p2 or hover_vol_btn or hover_help_btn or self.show_volume_slider or self.show_help_overlay:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._cleanup_audio()
                    pygame.quit()
                    return "quit"

                elif event.type == pygame.KEYDOWN:
                    if self.show_help_overlay:
                        if event.key in (pygame.K_h, pygame.K_SLASH, pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                            self.show_help_overlay = False
                            if self.audio:
                                self.audio.play("click")
                        continue
                    if event.key in (pygame.K_h, pygame.K_SLASH):
                        self.show_help_overlay = not self.show_help_overlay
                        if self.audio:
                            self.audio.play("click")
                    elif event.key == pygame.K_1 or event.key == pygame.K_g:
                        self._play_select_sound()
                        self._cleanup_audio()
                        return "1"
                    elif event.key == pygame.K_2 or event.key == pygame.K_a:
                        self._play_select_sound()
                        self._cleanup_audio()
                        return "2"
                    elif event.key == pygame.K_v:
                        self.show_volume_slider = not self.show_volume_slider
                    elif event.key == pygame.K_m and self.audio:
                        self.audio.toggle()
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        if self.show_volume_slider:
                            self.show_volume_slider = False
                        else:
                            self._cleanup_audio()
                            return "quit"

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.show_help_overlay:
                        self.show_help_overlay = False
                        if self.audio:
                            self.audio.play("click")
                        continue
                    if self.show_volume_slider and self.volume_panel_rect.collidepoint(event.pos):
                        self._handle_volume_click(event.pos)
                    elif self.volume_btn_rect.collidepoint(event.pos):
                        self.show_volume_slider = not self.show_volume_slider
                    elif self.help_btn_rect.collidepoint(event.pos):
                        self.show_help_overlay = not self.show_help_overlay
                        if self.audio:
                            self.audio.play("click")
                    elif hover_p1:
                        self._play_select_sound()
                        self._cleanup_audio()
                        return "1"
                    elif hover_p2:
                        self._play_select_sound()
                        self._cleanup_audio()
                        return "2"
                    elif self.show_volume_slider:
                        self.show_volume_slider = False

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_volume = False

                elif event.type == pygame.MOUSEMOTION and self.dragging_volume:
                    self._update_volume_drag(event.pos[0])

            self._draw(hover_p1, hover_p2, hover_vol_btn, hover_help_btn)
            pygame.display.flip()
            self.clock.tick(60)

    def _cleanup_audio(self) -> None:
        if self.audio:
            self.audio.close()
            self.audio = None

    def _play_select_sound(self) -> None:
        if self.audio and self.audio.available:
            self.audio.play("select")

    def _handle_volume_click(self, pos: tuple[int, int]) -> None:
        rect = self.volume_panel_rect
        btn_m5 = pygame.Rect(rect.x + 16, rect.y + 88, 45, 26)
        btn_p5 = pygame.Rect(rect.x + 67, rect.y + 88, 45, 26)
        btn_mute = pygame.Rect(rect.right - 80, rect.y + 88, 64, 26)

        if btn_m5.collidepoint(pos) and self.audio:
            self.audio.set_volume(max(0.0, self.audio.get_volume() - 0.05))
        elif btn_p5.collidepoint(pos) and self.audio:
            self.audio.set_volume(min(1.0, self.audio.get_volume() + 0.05))
        elif btn_mute.collidepoint(pos) and self.audio:
            self.audio.toggle()
        elif self.volume_track_rect.inflate(10, 16).collidepoint(pos):
            self.dragging_volume = True
            self._update_volume_drag(pos[0])

    def _update_volume_drag(self, mouse_x: int) -> None:
        track = self.volume_track_rect
        norm = (mouse_x - track.x) / max(1, track.width)
        vol = max(0.0, min(1.0, norm))
        if self.audio:
            self.audio.set_volume(vol)

    def _draw(
        self,
        hover_p1: bool,
        hover_p2: bool,
        hover_vol_btn: bool,
        hover_help_btn: bool = False,
    ) -> None:
        self.screen.fill(self.COLORS["space"])

        # Starfield
        for star in self.stars:
            pygame.draw.circle(
                self.screen,
                (star[3], star[3], min(255, star[3] + 30)),
                (star[0], star[1]),
                star[2],
            )

        # Header Title & Subtitle
        title = self.font_title.render("REINFORCEMENT LEARNING LAB", True, self.COLORS["text"])
        self.screen.blit(title, title.get_rect(center=(self.WIDTH // 2, 52)))

        sub = self.font_subtitle.render(
            "CS Deep Reinforcement Learning • Comprehensive Tabular & Deep Agent Portfolio",
            True,
            self.COLORS["muted"],
        )
        self.screen.blit(sub, sub.get_rect(center=(self.WIDTH // 2, 92)))

        # Draw Cards
        self._draw_part1_card(hover_p1)
        self._draw_part2_card(hover_p2)

        # Footer
        footer_text = "Press [1] or [2] to Launch  •  [H] Quick-Help  •  [V] Volume  •  [ESC] Exit"
        footer_surf = self.font_footer.render(footer_text, True, self.COLORS["faint"])
        self.screen.blit(footer_surf, (55, self.HEIGHT - 32))

        # Help Button in bottom right
        help_bg = self.COLORS["panel_hover"] if hover_help_btn else self.COLORS["panel"]
        pygame.draw.rect(self.screen, help_bg, self.help_btn_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLORS["border"], self.help_btn_rect, 1, border_radius=6)
        help_surf = self.font_footer.render("[?] HELP", True, self.COLORS["text"])
        self.screen.blit(help_surf, help_surf.get_rect(center=self.help_btn_rect.center))

        # Volume Button in bottom right
        vol_bg = self.COLORS["panel_hover"] if hover_vol_btn else self.COLORS["panel"]
        pygame.draw.rect(self.screen, vol_bg, self.volume_btn_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLORS["border"], self.volume_btn_rect, 1, border_radius=6)
        vol_pct = int(self.audio.volume * 100) if self.audio and self.audio.enabled else 0
        vol_label = "MUTE" if self.audio and not self.audio.enabled else f"VOL {vol_pct}%"
        vol_surf = self.font_footer.render(vol_label, True, self.COLORS["p2_accent"] if vol_pct > 0 else self.COLORS["muted"])
        self.screen.blit(vol_surf, vol_surf.get_rect(center=self.volume_btn_rect.center))

        # Volume Slider Modal
        if self.show_volume_slider:
            self._draw_volume_slider()

        # Help Overlay Modal
        if self.show_help_overlay:
            self._draw_help_overlay()

    def _draw_part1_card(self, hover: bool) -> None:
        rect = self.card_p1
        bg = self.COLORS["card_hover"] if hover else self.COLORS["card_bg"]
        border_color = self.COLORS["p1_accent"] if hover else self.COLORS["border"]

        # Outer Card
        pygame.draw.rect(self.screen, bg, rect, border_radius=16)
        pygame.draw.rect(self.screen, border_color, rect, 2 if hover else 1, border_radius=16)

        # Top Badge Pill
        tag_rect = pygame.Rect(rect.x + 24, rect.y + 24, 130, 24)
        pygame.draw.rect(self.screen, (20, 50, 35), tag_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLORS["p1_accent"], tag_rect, 1, border_radius=6)
        tag_surf = self.font_card_tag.render("TABULAR RL", True, self.COLORS["p1_accent"])
        self.screen.blit(tag_surf, tag_surf.get_rect(center=tag_rect.center))

        # Title & Subtitle
        title = self.font_card_title.render("Part I: Gridworld AI Lab", True, self.COLORS["text"])
        self.screen.blit(title, (rect.x + 24, rect.y + 60))

        sub = self.font_body.render("Q-Learning • SARSA • Intrinsic Curiosity Bonus", True, self.COLORS["p1_accent"])
        self.screen.blit(sub, (rect.x + 24, rect.y + 90))

        # Feature Bullets
        features = [
            "• 8 Canonical spatial benchmark stages",
            "• Moving & stochastic monster hazard dynamics",
            "• Count-based intrinsic curiosity exploration",
            "• Live interactive Q-table heatmap & policy view",
            "• Full campaign mode & step-by-step inspector",
        ]
        y = rect.y + 130
        for feat in features:
            f_surf = self.font_body.render(feat, True, self.COLORS["muted"])
            self.screen.blit(f_surf, (rect.x + 24, y))
            y += 28

        # Launch Button
        btn_bg = self.COLORS["p1_accent"] if hover else (25, 60, 42)
        btn_text_color = (10, 25, 18) if hover else self.COLORS["p1_accent"]
        pygame.draw.rect(self.screen, btn_bg, self.btn_p1, border_radius=10)
        pygame.draw.rect(self.screen, self.COLORS["p1_accent"], self.btn_p1, 1, border_radius=10)
        btn_surf = self.font_button.render("LAUNCH PART I  [ 1 ]", True, btn_text_color)
        self.screen.blit(btn_surf, btn_surf.get_rect(center=self.btn_p1.center))

    def _draw_part2_card(self, hover: bool) -> None:
        rect = self.card_p2
        bg = self.COLORS["card_hover"] if hover else self.COLORS["card_bg"]
        border_color = self.COLORS["p2_accent"] if hover else self.COLORS["border"]

        # Outer Card
        pygame.draw.rect(self.screen, bg, rect, border_radius=16)
        pygame.draw.rect(self.screen, border_color, rect, 2 if hover else 1, border_radius=16)

        # Top Badge Pill
        tag_rect = pygame.Rect(rect.x + 24, rect.y + 24, 110, 24)
        pygame.draw.rect(self.screen, (18, 48, 65), tag_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLORS["p2_accent"], tag_rect, 1, border_radius=6)
        tag_surf = self.font_card_tag.render("DEEP RL", True, self.COLORS["p2_accent"])
        self.screen.blit(tag_surf, tag_surf.get_rect(center=tag_rect.center))

        # Title & Subtitle
        title = self.font_card_title.render("Part II: Neon Rift Arena", True, self.COLORS["text"])
        self.screen.blit(title, (rect.x + 24, rect.y + 60))

        sub = self.font_body.render("Deep Q-Networks (DQN) • Combat Arena", True, self.COLORS["p2_accent"])
        self.screen.blit(sub, (rect.x + 24, rect.y + 90))

        # Feature Bullets
        features = [
            "• 48-dim continuous feature vector observation",
            "• Multi-lane boss rift hazards & sentry missiles",
            "• In-game weapon evolution & support drone draft",
            "• Dynamic screen shake & procedural audio suite",
            "• Manual & autonomous neural policy evaluation",
        ]
        y = rect.y + 130
        for feat in features:
            f_surf = self.font_body.render(feat, True, self.COLORS["muted"])
            self.screen.blit(f_surf, (rect.x + 24, y))
            y += 28

        # Launch Button
        btn_bg = self.COLORS["p2_accent"] if hover else (20, 52, 72)
        btn_text_color = (10, 22, 32) if hover else self.COLORS["p2_accent"]
        pygame.draw.rect(self.screen, btn_bg, self.btn_p2, border_radius=10)
        pygame.draw.rect(self.screen, self.COLORS["p2_accent"], self.btn_p2, 1, border_radius=10)
        btn_surf = self.font_button.render("LAUNCH PART II  [ 2 ]", True, btn_text_color)
        self.screen.blit(btn_surf, btn_surf.get_rect(center=self.btn_p2.center))

    def _draw_volume_slider(self) -> None:
        rect = self.volume_panel_rect
        shadow = rect.inflate(8, 8)
        pygame.draw.rect(self.screen, (4, 8, 18), shadow, border_radius=12)

        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill((14, 22, 44, 250))
        self.screen.blit(panel, rect.topleft)
        pygame.draw.rect(self.screen, self.COLORS["p2_accent"], rect, 2, border_radius=12)

        vol = self.audio.volume if self.audio else 0.8
        is_muted = self.audio and not self.audio.enabled
        pct_text = "MUTED" if is_muted else f"{int(vol * 100)}%"

        title_surf = self.font_button.render(f"Master Volume: {pct_text}", True, self.COLORS["text"])
        self.screen.blit(title_surf, (rect.x + 16, rect.y + 16))

        track = self.volume_track_rect
        pygame.draw.rect(self.screen, (7, 10, 25), track, border_radius=5)
        if not is_muted and vol > 0.0:
            fill_rect = pygame.Rect(track.x, track.y, int(track.width * vol), track.height)
            pygame.draw.rect(self.screen, self.COLORS["p2_accent"], fill_rect, border_radius=5)

        knob_x = track.x + int(track.width * (0.0 if is_muted else vol))
        knob_y = track.centery
        pygame.draw.circle(self.screen, (235, 243, 255), (knob_x, knob_y), 7)
        pygame.draw.circle(self.screen, self.COLORS["p2_accent"], (knob_x, knob_y), 4)

        # Quick action buttons
        btn_m5 = pygame.Rect(rect.x + 16, rect.y + 88, 45, 26)
        btn_p5 = pygame.Rect(rect.x + 67, rect.y + 88, 45, 26)
        btn_mute = pygame.Rect(rect.right - 80, rect.y + 88, 64, 26)

        for b_rect, text in ((btn_m5, "-5%"), (btn_p5, "+5%"), (btn_mute, "UNMUTE" if is_muted else "MUTE")):
            pygame.draw.rect(self.screen, (20, 30, 56), b_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLORS["border"], b_rect, 1, border_radius=4)
            t_surf = self.font_footer.render(text, True, self.COLORS["text"])
            self.screen.blit(t_surf, t_surf.get_rect(center=b_rect.center))

    def _draw_help_overlay(self) -> None:
        """Draw holographic quick-help and keyboard controls modal for the Master Hub."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((4, 7, 18, 225))
        self.screen.blit(overlay, (0, 0))

        panel = pygame.Rect((self.WIDTH - 720) // 2, (self.HEIGHT - 480) // 2, 720, 480)
        pygame.draw.rect(self.screen, (14, 22, 44), panel, border_radius=18)
        pygame.draw.rect(self.screen, self.COLORS["p2_accent"], panel, 2, border_radius=18)

        # Header
        title = self.font_card_title.render("ASSIGNMENT 3 — QUICK-HELP & HOTKEYS", True, self.COLORS["text"])
        self.screen.blit(title, (panel.x + 28, panel.y + 22))
        sub = self.font_footer.render("Click anywhere or press [ H ] / [ ESC ] to return", True, self.COLORS["muted"])
        self.screen.blit(sub, (panel.x + 30, panel.y + 54))

        # Column 1: Projects & Navigation
        col1_x = panel.x + 30
        head1 = self.font_card_tag.render("PROJECT NAVIGATION", True, self.COLORS["p1_accent"])
        self.screen.blit(head1, (col1_x, panel.y + 88))

        nav_items = (
            ("[ 1 ] or [ G ]", "Launch Part I: Gridworld AI Lab"),
            ("[ 2 ] or [ A ]", "Launch Part II: Neon Rift Arena"),
            ("[ H ] or [ ? ]", "Toggle this Quick-Help & Hotkeys Guide"),
            ("[ V ]", "Master Hub Ambient Audio Volume Slider"),
            ("[ M ]", "Mute / Unmute Ambient Soundtrack"),
            ("[ ESC / Q ]", "Close Hub / Exit Application"),
        )
        for i, (key, desc) in enumerate(nav_items):
            k_surf = self.font_footer.render(key, True, self.COLORS["p1_accent"])
            d_surf = self.font_body.render(desc, True, self.COLORS["text"])
            self.screen.blit(k_surf, (col1_x, panel.y + 118 + i * 44))
            self.screen.blit(d_surf, (col1_x, panel.y + 134 + i * 44))

        # Column 2: In-Game Quick Features
        col2_x = panel.x + 380
        head2 = self.font_card_tag.render("FEATURE HIGHLIGHTS", True, self.COLORS["p2_accent"])
        self.screen.blit(head2, (col2_x, panel.y + 88))

        game_items = (
            ("Part 1: Tabular RL", "Q-Learning, SARSA, Curiosity Bonus (P key)"),
            ("Part 2: Deep RL", "48-dim state, 21 weapons, boss rifts"),
            ("Ship Customization", "Press [ C ] in Part 2 for 5 Neon skins & SFX"),
            ("Audio Synthesizer", "Zero-asset algorithmic chiptune/synthwave"),
            ("Visual Game Juice", "Screen shake, defeat slow-mo, victory confetti"),
            ("Test Coverage", "150+ unit tests across the entire codebase"),
        )
        for i, (head, desc) in enumerate(game_items):
            h_surf = self.font_footer.render(head, True, self.COLORS["p2_accent"])
            d_surf = self.font_body.render(desc, True, self.COLORS["text"])
            self.screen.blit(h_surf, (col2_x, panel.y + 118 + i * 44))
            self.screen.blit(d_surf, (col2_x, panel.y + 134 + i * 44))

        # Footer close button
        hint = self.font_footer.render("CLICK ANYWHERE OR PRESS [ H ] / [ ESC ] TO CLOSE", True, self.COLORS["p2_accent"])
        h_rect = pygame.Rect(panel.centerx - hint.get_width() // 2 - 16, panel.bottom - 42, hint.get_width() + 32, 28)
        pygame.draw.rect(self.screen, (20, 36, 68), h_rect, border_radius=14)
        pygame.draw.rect(self.screen, self.COLORS["p2_accent"], h_rect, 1, border_radius=14)
        self.screen.blit(hint, (h_rect.centerx - hint.get_width() // 2, h_rect.y + 6))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Part I Gridworld or Part II Neon Rift Arena."
    )
    parser.add_argument(
        "--part",
        choices=("1", "2", "menu"),
        default="menu",
        help="project to launch (1, 2, or menu for the visual portal; default: menu)",
    )
    return parser


def launch_master_hub() -> str:
    """Launch the interactive graphical Master Hub and return user's choice."""
    launcher = MasterLauncher()
    return launcher.run()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Launch the visual portal or selected project directly with argument forwarding."""
    args, remaining = build_parser().parse_known_args(() if argv is None else argv)
    _ensure_project_runtime()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    if args.part == "1":
        from gridworld.app import main as launch_gridworld
        launch_gridworld(remaining)
        return

    if args.part == "2":
        if remaining:
            build_parser().error(
                f"Part II does not accept launcher arguments: {' '.join(remaining)}"
            )
        from arena.app import main as launch_arena
        launch_arena()
        return

    # Visual Portal Loop (returns between projects)
    while True:
        choice = launch_master_hub()
        if choice == "1":
            from gridworld.app import main as launch_gridworld
            launch_gridworld(remaining)
        elif choice == "2":
            from arena.app import main as launch_arena
            launch_arena()
        else:
            break


if __name__ == "__main__":
    main(sys.argv[1:])

