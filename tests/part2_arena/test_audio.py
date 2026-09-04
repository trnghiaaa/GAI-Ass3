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
            "laser",
            "laser_cyber_cyan",
            "laser_solar_flare",
            "laser_void_phantom",
            "laser_emerald_aegis",
            "laser_synth_pink",
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
        audio.sync_events({"projectiles_fired": 1}, theme="solar_flare")
        audio.sync_events({"projectiles_fired": 1}, theme="void_phantom")
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


def test_ship_themes_cycle_and_set():
    from arena.environment import ArenaEnv
    from arena.presentation.renderer import ArenaRenderer, SHIP_THEMES

    assert len(SHIP_THEMES) == 5
    for theme_name in ("cyber_cyan", "solar_flare", "void_phantom", "emerald_aegis", "synth_pink"):
        assert theme_name in SHIP_THEMES
        assert "player" in SHIP_THEMES[theme_name]
        assert "thrust" in SHIP_THEMES[theme_name]
        assert "laser" in SHIP_THEMES[theme_name]
        assert "name" in SHIP_THEMES[theme_name]

    env = ArenaEnv(render_mode="rgb_array")
    env.reset(seed=42)
    renderer = ArenaRenderer(env, mode="rgb_array")

    assert renderer.current_theme == "cyber_cyan"
    assert renderer.get_color("player") == (65, 210, 255)

    # Cycle through all themes
    visited = [renderer.current_theme]
    for _ in range(len(SHIP_THEMES) - 1):
        visited.append(renderer.cycle_theme())
    assert len(set(visited)) == 5

    # Wraps back to first
    assert renderer.cycle_theme() == "cyber_cyan"

    # Direct theme selection
    renderer.set_theme("solar_flare")
    assert renderer.current_theme == "solar_flare"
    assert renderer.get_color("player") == (255, 175, 40)
    assert renderer.get_color("thrust") == (255, 65, 25)

    # Render frame under custom theme
    frame = renderer.render()
    assert frame is not None
    assert frame.shape == (env.height, env.width, 3)


def test_arena_defeat_slowmo_and_vignette():
    from arena.environment import ArenaEnv
    from arena.presentation.renderer import ArenaRenderer

    env = ArenaEnv(render_mode="rgb_array")
    env.reset(seed=42)
    renderer = ArenaRenderer(env, mode="rgb_array")

    # Simulate player destruction
    env.done = True
    env.last_end_reason = "player_destroyed"
    renderer._sync_effects()

    assert renderer.defeat_slowmo_timer == 0.75
    assert renderer.defeat_shockwave_origin is not None
    assert renderer.defeat_vignette_alpha == 180.0
    assert renderer.shake_intensity >= 7.0

    # Render slow-mo frame
    frame = renderer.render()
    assert frame is not None
    assert frame.shape == (env.height, env.width, 3)
    assert renderer.defeat_slowmo_timer < 0.75
    assert renderer.defeat_shockwave_radius > 12.0



def test_arena_volume_slider_events(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from arena.environment import ArenaEnv
    from arena.presentation.audio import ArenaAudio
    from arena.presentation.renderer import ArenaRenderer

    env = ArenaEnv(render_mode="rgb_array")
    audio = ArenaAudio()
    renderer = ArenaRenderer(env, mode="rgb_array", audio=audio)

    # Event ignored when slider is closed
    assert renderer.show_volume_slider is False
    ev_key = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
    assert renderer.handle_volume_event(ev_key) is False

    # Open slider
    renderer.show_volume_slider = True

    # Adjust volume up via key
    vol_start = audio.get_volume()
    renderer.handle_volume_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    assert audio.get_volume() >= vol_start

    # Adjust volume down via key
    renderer.handle_volume_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    renderer.handle_volume_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    assert audio.get_volume() <= vol_start

    # Toggle mute
    renderer.handle_volume_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
    assert audio.get_volume() == 0.0 or not audio.enabled

    # Click inside panel track to set volume
    panel = renderer.volume_panel_rect
    track = renderer.volume_track_rect
    click_track = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(track.centerx, track.centery)
    )
    assert renderer.handle_volume_event(click_track) is True
    assert 0.0 < audio.get_volume() <= 1.0

    # Click outside panel dismisses slider
    click_outside = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(10, 10)
    )
    assert renderer.handle_volume_event(click_outside) is False
    assert renderer.show_volume_slider is False


