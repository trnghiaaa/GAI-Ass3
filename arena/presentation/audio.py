"""Procedural sci-fi music and sound effects for the Part II arena.

The audio layer is presentation-only: it consumes UI and simulation events and
never alters physics, observations, rewards, actions, or random streams.
All waveforms are synthesized mathematically in memory using NumPy and played
via Pygame mixer, requiring zero external audio files.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pygame


SAMPLE_RATE = 22_050


class ArenaAudio:
    """Device-safe procedural sci-fi soundtrack and sound-effect mixer."""

    def __init__(self) -> None:
        self.available = False
        self.enabled = True
        self.volume: float = 0.8
        self.music_channel: pygame.mixer.Channel | None = None
        self.hazard_channel: pygame.mixer.Channel | None = None
        self.music: pygame.mixer.Sound | None = None
        self.effects: dict[str, pygame.mixer.Sound] = {}
        self._last_played: dict[str, int] = {}

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
            pygame.mixer.set_num_channels(max(16, pygame.mixer.get_num_channels()))
            self.music_channel = pygame.mixer.Channel(0)
            self.hazard_channel = pygame.mixer.Channel(1)
            self.music = self._make_sound(self._build_synthwave_music())
            self.music.set_volume(self.volume * 0.20)
            self.effects = self._build_effects()
            for sound in self.effects.values():
                sound.set_volume(self.volume)
            self.available = True
        except (pygame.error, ValueError, TypeError):
            # Headless training, test runners, and machines without sound devices
            # must continue executing smoothly without errors.
            self.available = False
            self.music_channel = None
            self.hazard_channel = None
            self.music = None
            self.effects = {}

    @staticmethod
    def _note(
        frequency: float, duration: float, volume: float = 0.5, overtone: float = 0.25
    ) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        if frequency <= 0:
            return np.zeros(count, dtype=np.float64)
        time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        fundamental = np.sin(math.tau * frequency * time)
        harmonic = overtone * np.sin(math.tau * frequency * 2.0 * time)
        wave = (fundamental + harmonic) * volume

        attack = max(1, min(count // 4, int(SAMPLE_RATE * 0.008)))
        release = max(1, min(count // 2, int(SAMPLE_RATE * 0.04)))
        envelope = np.ones(count, dtype=np.float64)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
        return wave * envelope

    @staticmethod
    def _sweep(
        f_start: float, f_end: float, duration: float, volume: float = 0.5
    ) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        freqs = np.geomspace(max(10.0, f_start), max(10.0, f_end), count)
        phases = np.cumsum(freqs) * (math.tau / SAMPLE_RATE)
        wave = np.sin(phases) * volume

        decay = np.exp(-time * (4.5 / max(0.01, duration)))
        return wave * decay

    @staticmethod
    def _noise(duration: float, volume: float = 0.5, decay_rate: float = 8.0) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        raw = np.random.uniform(-1.0, 1.0, count)
        envelope = np.exp(-time * decay_rate) * volume
        return raw * envelope

    @staticmethod
    def _mix(*waves: np.ndarray) -> np.ndarray:
        if not waves:
            return np.zeros(0, dtype=np.float64)
        max_len = max(len(w) for w in waves)
        out = np.zeros(max_len, dtype=np.float64)
        for w in waves:
            out[: len(w)] += w
        return out

    @classmethod
    def _sequence(
        cls, notes: Iterable[tuple[float, float, float]]
    ) -> np.ndarray:
        pieces = [cls._note(freq, dur, vol) for freq, dur, vol in notes]
        return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float64)

    @classmethod
    def _build_synthwave_music(cls) -> np.ndarray:
        """Synthesize a rhythmic 16-bar cyberpunk background loop."""
        chords = (
            (146.83, 220.00, 261.63, 293.66),  # Dm
            (116.54, 174.61, 233.08, 293.66),  # Bb
            (130.81, 196.00, 261.63, 329.63),  # C
            (110.00, 164.81, 220.00, 261.63),  # Am
        )
        beat_dur = 0.22  # ~136 BPM
        notes: list[tuple[float, float, float]] = []

        for chord in chords:
            for bar in range(2):
                pattern = (0, 2, 1, 3, 2, 1, 2, 3) if bar == 0 else (0, 1, 2, 3, 2, 3, 1, 0)
                for step in pattern:
                    freq = chord[step] * 2.0
                    notes.append((freq, beat_dur, 0.16))

        music_lead = cls._sequence(notes)
        total_len = len(music_lead)

        bass = np.zeros(total_len, dtype=np.float64)
        samples_per_chord = total_len // len(chords)
        for idx, chord in enumerate(chords):
            start = idx * samples_per_chord
            root = chord[0]
            for step in range(16):
                p_start = start + int(step * (samples_per_chord / 16))
                p_wave = cls._note(root, beat_dur * 0.45, 0.18, overtone=0.35)
                p_end = min(total_len, p_start + len(p_wave))
                bass[p_start:p_end] += p_wave[: p_end - p_start]

        return np.clip(music_lead + bass, -1.0, 1.0)

    @classmethod
    def _make_sound(cls, mono: np.ndarray) -> pygame.mixer.Sound:
        pcm = np.asarray(np.clip(mono, -1.0, 1.0) * 32767, dtype=np.int16)
        stereo = np.ascontiguousarray(np.column_stack((pcm, pcm)))
        return pygame.sndarray.make_sound(stereo)

    @classmethod
    def _build_effects(cls) -> dict[str, pygame.mixer.Sound]:
        """Build the full sci-fi combat sound palette."""
        laser = cls._sweep(1200.0, 260.0, 0.055, 0.22)
        laser_heavy = cls._sweep(700.0, 140.0, 0.090, 0.28)
        drone = cls._sweep(1550.0, 650.0, 0.038, 0.16)

        # Crisp arcade plasma plink / hit-marker (bright dual harmonic chime + short spark)
        hit = cls._mix(
            cls._note(1318.5, 0.024, 0.22),
            cls._note(1975.5, 0.016, 0.14),
            cls._noise(0.012, 0.16, decay_rate=50.0),
        )
        shield_hit = cls._mix(cls._note(1760.0, 0.08, 0.28), cls._note(2640.0, 0.05, 0.18))

        # Melodic arcade disintegration blip (descending 2-tone stinger G5 -> C5 with low-end pop)
        kill_enemy = cls._mix(
            cls._sequence([
                (783.99, 0.028, 0.30),
                (523.25, 0.045, 0.34),
            ]),
            cls._sweep(180.0, 45.0, 0.070, 0.28),
            cls._noise(0.040, 0.20, decay_rate=25.0),
        )
        # Massive spawner rift obliteration (deep sub-bass shockwave + warp implosion)
        destroy_spawner = cls._mix(
            cls._sweep(220.0, 25.0, 0.28, 0.45),
            cls._noise(0.28, 0.35, decay_rate=6.0),
            cls._sweep(420.0, 60.0, 0.12, 0.25),
        )
        player_hit = cls._mix(cls._noise(0.12, 0.36, decay_rate=9.0), cls._sweep(200.0, 50.0, 0.12, 0.28))

        missile_launch = cls._mix(cls._sweep(350.0, 920.0, 0.11, 0.22), cls._noise(0.11, 0.16, decay_rate=7.0))
        boss_telegraph = cls._mix(cls._note(110.0, 0.14, 0.28, overtone=0.5), cls._note(220.0, 0.14, 0.18))

        level_up = cls._sequence([
            (329.63, 0.055, 0.25),
            (440.00, 0.055, 0.28),
            (659.25, 0.065, 0.30),
            (880.00, 0.160, 0.34),
        ])

        phase_clear = cls._sequence([
            (392.00, 0.075, 0.28),
            (523.25, 0.085, 0.30),
            (659.25, 0.095, 0.32),
            (783.99, 0.220, 0.36),
        ])

        game_over = cls._mix(cls._sweep(330.0, 45.0, 0.38, 0.34), cls._noise(0.38, 0.20, decay_rate=3.0))

        click = cls._mix(cls._note(1100.0, 0.015, 0.18), cls._noise(0.012, 0.12, decay_rate=50.0))
        menu_select = cls._sequence([
            (587.33, 0.035, 0.24),
            (880.00, 0.055, 0.28),
        ])

        return {
            "laser": cls._make_sound(laser),
            "laser_heavy": cls._make_sound(laser_heavy),
            "drone": cls._make_sound(drone),
            "hit": cls._make_sound(hit),
            "shield_hit": cls._make_sound(shield_hit),
            "kill_enemy": cls._make_sound(kill_enemy),
            "destroy_spawner": cls._make_sound(destroy_spawner),
            "player_hit": cls._make_sound(player_hit),
            "missile_launch": cls._make_sound(missile_launch),
            "boss_telegraph": cls._make_sound(boss_telegraph),
            "level_up": cls._make_sound(level_up),
            "phase_clear": cls._make_sound(phase_clear),
            "game_over": cls._make_sound(game_over),
            "click": cls._make_sound(click),
            "menu_select": cls._make_sound(menu_select),
        }

    def start_music(self) -> None:
        if not self.available or not self.enabled or self.music_channel is None or self.music is None:
            return
        if not self.music_channel.get_busy():
            self.music_channel.play(self.music, loops=-1, fade_ms=500)

    def play(self, name: str, *, minimum_interval_ms: int = 0) -> None:
        if not self.available or not self.enabled:
            return
        sound = self.effects.get(name)
        if sound is None:
            return
        now = pygame.time.get_ticks()
        if now - self._last_played.get(name, -1_000_000) < minimum_interval_ms:
            return
        self._last_played[name] = now
        sound.play()

    def sync_events(self, events: dict[str, Any], done: bool = False) -> None:
        """Trigger corresponding SFX for simulation events with rate-limiting."""
        if not self.available or not self.enabled or not events:
            return

        if events.get("projectiles_fired"):
            self.play("laser", minimum_interval_ms=75)
        elif events.get("drone_shots"):
            self.play("drone", minimum_interval_ms=100)

        if events.get("spawners_destroyed"):
            self.play("destroy_spawner", minimum_interval_ms=120)
        elif events.get("enemies_destroyed"):
            self.play("kill_enemy", minimum_interval_ms=60)
        elif events.get("damage_dealt_spawner") or events.get("damage_dealt_enemy"):
            if events.get("boss_immune_hits"):
                self.play("shield_hit", minimum_interval_ms=80)
            else:
                self.play("hit", minimum_interval_ms=60)

        if events.get("player_hit"):
            self.play("player_hit", minimum_interval_ms=150)

        if events.get("missiles_fired"):
            self.play("missile_launch", minimum_interval_ms=200)

        if events.get("boss_skills_cast"):
            self.play("boss_telegraph", minimum_interval_ms=300)

        if events.get("levels_gained"):
            self.play("level_up", minimum_interval_ms=250)

        if events.get("phase_advanced"):
            self.play("phase_clear", minimum_interval_ms=400)

        if done and events.get("damage_taken", 0) > 0:
            self.play("game_over", minimum_interval_ms=1000)

    def set_volume(self, volume: float) -> float:
        """Set normalized master volume [0.0, 1.0] and scale music and SFX."""
        self.volume = max(0.0, min(1.0, float(volume)))
        if self.volume <= 0.001:
            self.enabled = False
            if self.available:
                pygame.mixer.stop()
            return 0.0
        self.enabled = True
        if self.available:
            if self.music is not None:
                self.music.set_volume(self.volume * 0.20)
            for sound in self.effects.values():
                sound.set_volume(self.volume)
            self.start_music()
        return self.volume

    def get_volume(self) -> float:
        return self.volume if self.enabled else 0.0

    def toggle(self) -> bool:
        """Toggle all audio on/off and return the new state."""
        if self.enabled:
            self.enabled = False
            if self.available:
                pygame.mixer.stop()
        else:
            self.enabled = True
            if self.volume <= 0.05:
                self.volume = 0.8
            self.set_volume(self.volume)
        return self.enabled

    def close(self) -> None:
        if self.available:
            pygame.mixer.stop()


__all__ = ["ArenaAudio"]
