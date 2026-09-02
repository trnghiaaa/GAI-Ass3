"""Tests for the presentation-only Arena audio layer."""


def test_arena_procedural_audio_initializes_and_toggles_with_dummy_driver(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from arena.presentation.audio import ArenaAudio

    pygame.quit()
    pygame.init()
    audio = ArenaAudio()
    try:
        assert audio.available
        required_effects = {
            "laser",
            "laser_heavy",
            "drone",
            "hit",
            "shield_hit",
            "kill_enemy",
            "destroy_spawner",
            "player_hit",
            "missile_launch",
            "boss_telegraph",
            "level_up",
            "phase_clear",
            "game_over",
            "click",
            "menu_select",
        }
        assert required_effects <= set(audio.effects)
        audio.start_music()
        assert audio.music_channel is not None
        assert audio.music_channel.get_busy()

        # Test event dispatching
        audio.sync_events({"projectiles_fired": 1})
        audio.sync_events({"spawners_destroyed": 1, "levels_gained": 1})
        audio.sync_events({"player_hit": True, "missiles_fired": 1})
        audio.sync_events({"damage_taken": 25}, done=True)

        # Test volume control
        assert abs(audio.set_volume(0.5) - 0.5) < 1e-5
        assert abs(audio.get_volume() - 0.5) < 1e-5
        assert audio.set_volume(1.5) == 1.0
        assert audio.set_volume(-0.5) == 0.0
        assert audio.set_volume(0.8) == 0.8

        assert audio.toggle() is False
        assert audio.toggle() is True
    finally:
        audio.close()
        pygame.quit()


def test_arena_audio_gracefully_disables_when_mixer_fails(monkeypatch):
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from arena.presentation.audio import ArenaAudio

    pygame.mixer.quit()
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)

    def fail_init(**_kwargs):
        raise pygame.error("no audio device")

    monkeypatch.setattr(pygame.mixer, "init", fail_init)
    audio = ArenaAudio()
    assert not audio.available
    audio.start_music()
    audio.sync_events({"projectiles_fired": 1})
    assert audio.toggle() is False


def test_screen_shake_accumulates_and_decays():
    from arena.environment import ArenaEnv
    from arena.presentation.renderer import ArenaRenderer

    env = ArenaEnv(render_mode="rgb_array")
    renderer = ArenaRenderer(env, mode="rgb_array")

    assert renderer.shake_intensity == 0.0
    assert renderer._update_shake() == (0.0, 0.0)

    # Accumulate shake
    renderer.add_screen_shake(5.0)
    assert renderer.shake_intensity == 5.0
    offset_x, offset_y = renderer._update_shake()
    assert offset_x != 0.0 or offset_y != 0.0
    assert renderer.shake_intensity < 5.0  # Decayed

    # Ceiling clamping
    renderer.add_screen_shake(100.0)
    assert renderer.shake_intensity == 14.0

    # Exponential decay to zero
    for _ in range(50):
        renderer._update_shake()
    assert renderer.shake_intensity == 0.0
    assert renderer._update_shake() == (0.0, 0.0)


def test_renderer_composite_surface_renders_with_shake():
    from arena.environment import ArenaEnv
    from arena.presentation.renderer import ArenaRenderer

    env = ArenaEnv(render_mode="rgb_array")
    env.reset(seed=42)
    renderer = ArenaRenderer(env, mode="rgb_array")

    renderer.add_screen_shake(8.0)
    renderer.damage_flash_alpha = 80.0

    frame = renderer.render()
    assert frame is not None
    assert frame.shape == (env.height, env.width, 3)
    assert frame.dtype.name == "uint8"
