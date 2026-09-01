"""Shared deterministic fixtures for the Part I test suite."""

import pytest

from gridworld.levels import LEVELS


@pytest.fixture
def level_factory(monkeypatch):
    """Install a small temporary level without mutating canonical maps."""
    next_id = 100

    def create(grid, **metadata):
        nonlocal next_id
        level_id = next_id
        next_id += 1
        definition = {
            "title": metadata.pop("title", "Test Level"),
            "task": metadata.pop("task", 0),
            "description": metadata.pop("description", "Mechanics fixture"),
            "objectives": metadata.pop("objectives", ("Exercise one rule",)),
            "mechanics": metadata.pop("mechanics", ("test",)),
            "grid": list(grid),
            **metadata,
        }
        monkeypatch.setitem(LEVELS, level_id, definition)
        return level_id

    return create
