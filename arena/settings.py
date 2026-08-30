"""Shared configuration and artifact paths for Part II tooling."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")
ARENA_MODEL_DIR = PROJECT_ROOT / "models" / "arena"
ARENA_LOG_DIR = PROJECT_ROOT / "logs" / "arena"
TENSORBOARD_DIR = ARENA_LOG_DIR / "tensorboard"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def training_settings(control_style: str, profile: str = "balanced") -> dict[str, Any]:
    """Resolve common, named-profile, and control-style training settings."""

    if control_style not in ("direct", "rotation"):
        raise ValueError("control_style must be 'direct' or 'rotation'")
    training = load_config()["training"]
    profiles = training.pop("profiles", {})
    style_overrides = training.pop("style_overrides", {})
    if profile not in profiles:
        choices = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown training profile {profile!r}; choose from {choices}")
    return deep_merge(
        deep_merge(training, style_overrides.get(control_style, {})), profiles[profile]
    )


def model_path(control_style: str, run_name: str | None = None) -> Path:
    stem = run_name or f"dqn_{control_style}"
    return ARENA_MODEL_DIR / f"{stem}.zip"


def metadata_path(control_style: str, run_name: str | None = None) -> Path:
    return model_path(control_style, run_name).with_suffix(".metadata.json")


def ensure_artifact_directories() -> None:
    ARENA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARENA_LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ARENA_LOG_DIR",
    "ARENA_MODEL_DIR",
    "CONFIG_PATH",
    "PROJECT_ROOT",
    "TENSORBOARD_DIR",
    "deep_merge",
    "ensure_artifact_directories",
    "load_config",
    "metadata_path",
    "model_path",
    "training_settings",
]
