"""Original procedural music and sound effects for the Part I application.

The audio layer is presentation-only: it consumes UI/environment events and
never changes transitions, rewards, agent actions, or random-number streams.
All waveforms are synthesized in memory, so the project has no external audio
assets or licensing requirements.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pygame


SAMPLE_RATE = 22_050


class GridworldAudio:
    """Small, device-safe procedural soundtrack and sound-effect mixer."""

    def __init__(self) -> None:
        self.available = False
        self.enabled = True
        self.music_channel: pygame.mixer.Channel | None = None
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
            pygame.mixer.set_num_channels(max(12, pygame.mixer.get_num_channels()))
            self.music_channel = pygame.mixer.Channel(0)
            self.music = self._make_sound(self._build_music())
            self.music.set_volume(0.24)
            self.effects = self._build_effects()
            self.available = True
        except (pygame.error, ValueError, TypeError):
            # Headless machines and lab PCs without an audio device must still
            # run the complete visual application and test suite.
            self.available = False
            self.music_channel = None
            self.music = None
            self.effects = {}

    @staticmethod
    def _note(frequency: float, duration: float, volume: float = 0.5) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        if frequency <= 0:
            return np.zeros(count, dtype=np.float64)
        time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        fundamental = np.sin(math.tau * frequency * time)
        overtone = 0.22 * np.sin(math.tau * frequency * 2.0 * time)
        wave = (fundamental + overtone) * volume

        attack = max(1, min(count // 3, int(SAMPLE_RATE * 0.012)))
        release = max(1, min(count // 2, int(SAMPLE_RATE * 0.06)))
        envelope = np.ones(count, dtype=np.float64)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
        return wave * envelope

    @classmethod
    def _sequence(
        cls, notes: Iterable[tuple[float, float, float]]
    ) -> np.ndarray:
        pieces = [cls._note(frequency, duration, volume) for frequency, duration, volume in notes]
        return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float64)

    @classmethod
    def _build_music(cls) -> np.ndarray:
        # A calm original arpeggio that leaves space for short gameplay cues.
        progression: Sequence[Sequence[float]] = (
            (261.63, 329.63, 392.00, 523.25),
            (220.00, 261.63, 329.63, 440.00),
            (174.61, 220.00, 261.63, 349.23),
            (196.00, 246.94, 293.66, 392.00),
        )
        notes: list[tuple[float, float, float]] = []
        for chord in progression:
            for _ in range(2):
                for frequency in chord:
                    notes.append((frequency, 0.24, 0.23))
        music = cls._sequence(notes)

        # A quiet bass pulse makes the loop feel musical without becoming busy.
        bass = np.zeros_like(music)
        chord_samples = int(SAMPLE_RATE * 0.24 * 8)
        for index, chord in enumerate(progression):
            start = index * chord_samples
            pulse = cls._note(chord[0] / 2.0, chord_samples / SAMPLE_RATE, 0.13)
            end = min(len(bass), start + len(pulse))
            bass[start:end] += pulse[: end - start]
        return np.clip(music + bass, -1.0, 1.0)

    @classmethod
    def _make_sound(cls, mono: np.ndarray) -> pygame.mixer.Sound:
        pcm = np.asarray(np.clip(mono, -1.0, 1.0) * 32767, dtype=np.int16)
        stereo = np.ascontiguousarray(np.column_stack((pcm, pcm)))
        return pygame.sndarray.make_sound(stereo)

    @classmethod
    def _build_effects(cls) -> dict[str, pygame.mixer.Sound]:
        definitions: dict[str, Sequence[tuple[float, float, float]]] = {
            "click": ((520, 0.035, 0.18),),
            "move": ((150, 0.035, 0.10),),
            "blocked": ((115, 0.07, 0.28), (82, 0.08, 0.20)),
            "monster": ((145, 0.06, 0.13),),
            "apple": ((659.25, 0.08, 0.34), (783.99, 0.11, 0.30)),
            "key": ((880.00, 0.08, 0.30), (1174.66, 0.15, 0.32)),
            "chest": ((392.00, 0.08, 0.30), (523.25, 0.09, 0.32), (783.99, 0.18, 0.34)),
            "locked": ((146.83, 0.10, 0.28), (110.00, 0.14, 0.22)),
            "start": ((261.63, 0.07, 0.23), (392.00, 0.08, 0.24), (523.25, 0.12, 0.26)),
            "victory": ((523.25, 0.10, 0.35), (659.25, 0.10, 0.35), (783.99, 0.12, 0.36), (1046.50, 0.28, 0.38)),
            "death": ((220.00, 0.12, 0.34), (164.81, 0.14, 0.31), (110.00, 0.28, 0.28)),
            "timeout": ((293.66, 0.12, 0.24), (220.00, 0.22, 0.24)),
        }
        return {
            name: cls._make_sound(cls._sequence(notes))
            for name, notes in definitions.items()
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

    def toggle(self) -> bool:
        """Toggle all audio and return the new enabled state."""

        self.enabled = not self.enabled
        if not self.available:
            return self.enabled
        if self.enabled:
            self.start_music()
        else:
            pygame.mixer.stop()
        return self.enabled

    def close(self) -> None:
        if self.available:
            pygame.mixer.stop()

