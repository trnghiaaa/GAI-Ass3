"""Tests for the optional, presentation-only Gridworld audio layer."""


def test_procedural_audio_initializes_and_toggles_with_dummy_driver(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from gridworld.audio import GridworldAudio

    pygame.quit()
    pygame.init()
    audio = GridworldAudio()
    try:
        assert audio.available
        assert {"move", "apple", "key", "chest", "locked", "death", "victory"} <= set(audio.effects)
        audio.start_music()
        assert audio.music_channel is not None
        assert audio.music_channel.get_busy()

        # Test volume control
        assert abs(audio.set_volume(0.5) - 0.5) < 1e-5
        assert abs(audio.get_volume() - 0.5) < 1e-5
        assert audio.set_volume(1.5) == 1.0  # Clamped to 1.0
        assert audio.set_volume(-0.5) == 0.0  # Clamped to 0.0
        assert audio.set_volume(0.8) == 0.8

        assert audio.toggle() is False
        assert audio.toggle() is True
    finally:
        audio.close()
        pygame.quit()


def test_audio_gracefully_disables_when_mixer_initialization_fails(monkeypatch):
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from gridworld.audio import GridworldAudio

    pygame.mixer.quit()
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)

    def fail_init(**_kwargs):
        raise pygame.error("no audio device")

    monkeypatch.setattr(pygame.mixer, "init", fail_init)
    audio = GridworldAudio()
    assert not audio.available
    audio.start_music()
    audio.play("victory")
