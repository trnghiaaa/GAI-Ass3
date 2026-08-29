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
