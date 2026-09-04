"""Regression tests for the illustrated Part II pilot guide."""

import pytest


def test_guidebook_covers_every_configured_reward_and_upgrade(monkeypatch):
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from arena.presentation.guidebook import guide_catalog_entries
    from arena.settings import load_config

    displayed = guide_catalog_entries()
    configured = load_config()["progression"]

    assert len(displayed["upgrades"]) == 21
    assert {item["id"] for item in displayed["upgrades"]} == {
        item["id"] for item in configured["upgrade_catalog"]
    }
    assert {item["id"] for item in displayed["phase_rewards"]} == {
        item["id"] for item in configured["phase_reward_catalog"]
    }
    assert {item["id"] for item in displayed["boss_rewards"]} == {
        item["id"] for item in configured["boss_reward_catalog"]
    }


def test_every_guidebook_page_renders_with_reachable_navigation(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from arena.presentation.guidebook import draw_guidebook_page, guide_page_count

    pygame.font.init()
    surface = pygame.Surface((800, 600))
    try:
        for page_index in range(guide_page_count()):
            hitboxes = draw_guidebook_page(surface, page_index)
            assert hitboxes["back"].colliderect(surface.get_rect())
            assert hitboxes[f"page_{page_index}"].colliderect(surface.get_rect())
            assert surface.get_bounding_rect().width == 800
            assert surface.get_bounding_rect().height == 600
    finally:
        pygame.font.quit()


def test_guidebook_has_separate_build_and_player_tools_chapters(monkeypatch):
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from arena.presentation.guidebook import PAGES, guide_page_count

    assert guide_page_count() == 11
    assert PAGES[9].key == "build_style"
    assert PAGES[10].key == "other_features"


def test_guidebook_rejects_an_unknown_page(monkeypatch):
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from arena.presentation.guidebook import draw_guidebook_page

    pygame.font.init()
    try:
        with pytest.raises(ValueError, match="page_index"):
            draw_guidebook_page(pygame.Surface((800, 600)), 99)
    finally:
        pygame.font.quit()


def test_guidebook_escape_returns_to_the_existing_launcher(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import pygame

    from arena.presentation.guidebook import open_guidebook

    pygame.quit()
    pygame.init()
    try:
        screen = pygame.display.set_mode((800, 600))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        open_guidebook(screen, pygame.time.Clock())
    finally:
        pygame.quit()
