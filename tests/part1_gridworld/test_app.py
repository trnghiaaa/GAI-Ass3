import os


def test_unified_app_draws_menu_and_game_with_dummy_video(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import DEFAULT_AI_SPEED, GridworldApp

    app = GridworldApp(max_steps=20)
    try:
        assert app.speed_options[app.speed_index] == DEFAULT_AI_SPEED == 1.0
        app._draw()
        assert app.scene == "menu"
        assert app.buttons
        menu_path = tmp_path / "menu.png"
        pygame.image.save(app.canvas, menu_path)
        assert menu_path.stat().st_size > 10_000

        app._start_level(2, "manual", None, False, "campaign")
        app._draw()
        assert app.scene == "play"
        assert app.env.level_id == 2
        assert app.campaign_mode
        game_path = tmp_path / "level2.png"
        pygame.image.save(app.canvas, game_path)
        assert game_path.stat().st_size > 10_000

        original_position = app.env.agent_pos
        app._perform_step(3)
        assert app.steps == 1
        assert app.trail[0] == original_position
    finally:
        pygame.quit()


def test_level_cards_contain_text_and_ai_entry_resets_to_normal_speed(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import DEFAULT_AI_SPEED, GridworldApp

    app = GridworldApp(max_steps=20)
    try:
        marker = (1, 2, 3)
        app.canvas.fill(marker)
        app.mouse_virtual = (-100, -100)
        card = pygame.Rect(100, 100, 274, 244)
        app.level_select_context = "free"
        app._draw_level_card(2, card)

        # Text, chips, and the footer must never draw beyond the card edge.
        outside_x = card.right + 1
        assert all(app.canvas.get_at((outside_x, y))[:3] == marker for y in range(card.top, card.bottom))

        app.speed_index = len(app.speed_options) - 1
        app._handle_action(("campaign_ai",))
        assert app.speed_options[app.speed_index] == DEFAULT_AI_SPEED == 1.0

        clipped = app._ellipsize("A deliberately very long interface label", "small", 80)
        assert app.fonts["small"].size(clipped)[0] <= 80
        assert clipped.endswith("…")
    finally:
        pygame.quit()


def test_ai_result_popup_controls_replay_speed_and_next_level_resets(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import DEFAULT_AI_SPEED, GridworldApp

    app = GridworldApp(max_steps=500)
    try:
        app._start_level(0, "ai", "qlearning", False, "campaign")
        app.speed_index = len(app.speed_options) - 1
        app.run_done = True
        app.result_kind = "victory"
        app.result_detail = "Every collectible reward was obtained."
        app._draw()

        actions = [button.action for button in app.buttons]
        assert ("speed", -1) in actions
        assert ("speed", 1) in actions
        assert ("speed_reset",) in actions

        app._handle_action(("speed_reset",))
        assert app.speed_options[app.speed_index] == DEFAULT_AI_SPEED == 1.0

        app.speed_index = len(app.speed_options) - 1
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        assert app.speed_options[app.speed_index] == 4.0
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1))
        assert app.speed_options[app.speed_index] == 1.0

        app.speed_index = len(app.speed_options) - 1
        app._next_level()
        assert app.current_level == 1
        assert app.current_agent_kind == "sarsa"
        assert not app.current_intrinsic
        assert app.speed_options[app.speed_index] == DEFAULT_AI_SPEED == 1.0
    finally:
        pygame.quit()


def test_free_ai_next_level_keeps_supported_policy_then_uses_safe_fallback(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import DEFAULT_AI_SPEED, GridworldApp

    app = GridworldApp(max_steps=500)
    try:
        app._start_level(4, "ai", "sarsa", False, "showcase")
        assert app.scene == "play"

        # SARSA is a supported saved policy on Level 5, so the selection stays.
        app.run_done = True
        app.result_kind = "victory"
        app._next_level()
        assert app.scene == "play"
        assert app.current_level == 5
        assert (app.current_agent_kind, app.current_intrinsic) == ("sarsa", False)

        # Level 6's rubric showcase is baseline Q versus Q + Intrinsic. The app
        # must switch to its recommended trained model instead of showing the
        # missing Level 6 SARSA recovery screen.
        app.speed_index = len(app.speed_options) - 1
        app.run_done = True
        app.result_kind = "victory"
        assert app._resolve_next_ai_policy(6) == ("qlearning", True)
        app._next_level()
        assert app.scene == "play"
        assert app.current_level == 6
        assert (app.current_agent_kind, app.current_intrinsic) == ("qlearning", True)
        assert app.speed_options[app.speed_index] == DEFAULT_AI_SPEED == 1.0
        assert app.notice == "Level 6: switched to Q + Intrinsic"
    finally:
        pygame.quit()


def test_manual_blocked_input_is_presented_as_a_counted_action(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import GridworldApp
    from gridworld.environment import LEFT

    app = GridworldApp(max_steps=20)
    try:
        app._start_level(0, "manual", None, False, "free")
        start = tuple(app.env.agent_pos)
        app._perform_step(LEFT)
        assert tuple(app.env.agent_pos) == start
        assert app.steps == 1
        assert app.info["blocked_reason"] == "boundary"
        assert "action still counted" in app.event_text
    finally:
        pygame.quit()


def test_showcase_only_offers_rubric_relevant_saved_policies(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import GridworldApp, RUBRIC_POLICIES

    app = GridworldApp(max_steps=20)
    try:
        for level, expected in RUBRIC_POLICIES.items():
            app.selected_level = level
            app.buttons = []
            app._draw_algorithm_select()
            offered = [
                (button.action[1], button.action[2])
                for button in app.buttons
                if button.action[0] == "start_ai"
            ]
            assert offered == list(expected)
            assert app._available_model_count(level) == len(expected)
    finally:
        pygame.quit()


def test_legacy_renderer_does_not_consume_keyboard_events(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.environment import GridWorldEnv
    from gridworld.renderer import GridWorldRenderer

    env = GridWorldEnv(0)
    env.reset()
    renderer = GridWorldRenderer(env, cell_size=20, fps=1000)
    try:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        renderer.render(step=0, total_reward=0)
        events = pygame.event.get(pygame.KEYDOWN)
        assert [event.key for event in events] == [pygame.K_RIGHT]
    finally:
        renderer.close()


def test_gridworld_defeat_slowmo_and_vignette(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import GridworldApp

    app = GridworldApp(max_steps=20)
    try:
        app._start_level(1, "manual", None, False, "free")
        app._finish_run("death", "The agent was defeated by hazard.")
        assert app.defeat_slowmo_timer == 0.75
        assert app.defeat_flash_alpha == 140.0
        assert app.defeat_vignette_alpha == 180.0
        assert app.defeat_shockwave_origin is not None

        # Draw during slowmo
        app._draw()
        assert app.scene == "play"

        # Update decay
        app._update(0.1)
        assert app.defeat_slowmo_timer < 0.75
        assert app.defeat_flash_alpha < 140.0
    finally:
        pygame.quit()


def test_gridworld_help_overlay_toggle(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import GridworldApp

    app = GridworldApp(max_steps=20)
    try:
        assert app.show_help_overlay is False

        # Toggle help on with H key
        event_h = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h)
        app._handle_key(event_h)
        assert app.show_help_overlay is True

        # Draw frame while help is open
        app._draw()
        app._present()

        # Dismiss help with ESC key - verify scene remains 'menu'
        event_esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        app._handle_key(event_esc)
        assert app.show_help_overlay is False
        assert app.running is True
        assert app.scene == "menu"

        # Toggle help on again
        app._handle_key(event_h)
        assert app.show_help_overlay is True

        # Dismiss help with mouse click
        pygame.event.clear()
        event_mouse = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(200, 200))
        pygame.event.post(event_mouse)
        app._poll_events()
        assert app.show_help_overlay is False
        assert app.running is True
        assert app.scene == "menu"

        # Verify extraneous keys are swallowed while help overlay is active
        app._handle_key(event_h)
        assert app.show_help_overlay is True
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        # Scene should remain "menu", not change to "campaign_select"
        assert app.scene == "menu"
        assert app.show_help_overlay is True
        app._handle_key(event_esc)
        assert app.show_help_overlay is False
    finally:
        pygame.quit()


def test_gridworld_victory_celebration_effects(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import GridworldApp

    app = GridworldApp(max_steps=20)
    try:
        app._start_level(0, "manual", None, False, "free")
        app._finish_run("victory", "Goal reached!")
        assert app.victory_flash_alpha == 150.0
        assert app.run_done is True
        assert app.result_kind == "victory"
        # Verify celebratory confetti fountain spawned
        assert len(app.particles) >= 120

        # Draw frame during victory celebration
        app._draw()
        assert app.scene == "play"

        # Update decay
        app._update(0.1)
        assert app.victory_flash_alpha < 150.0
        # Ambient confetti generation
        app._update(0.2)
        assert len(app.particles) > 0
    finally:
        pygame.quit()


def test_gridworld_volume_slider_interaction(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from gridworld.app import GridworldApp

    app = GridworldApp(max_steps=20)
    try:
        # Toggle volume slider with V
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
        assert app.show_volume_slider is True

        # Keyboard volume adjustments
        vol_before = app.audio.get_volume()
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        assert app.audio.get_volume() >= vol_before

        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        assert app.audio.get_volume() <= vol_before

        # Mute toggle with M
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        assert app.audio.get_volume() == 0.0 or not app.audio.enabled

        # ESC dismisses volume slider without exiting scene
        app._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        assert app.show_volume_slider is False
        assert app.running is True
        assert app.scene == "menu"
    finally:
        pygame.quit()
