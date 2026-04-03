"""Model config loader: TOML file + env var overrides."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


def load_model_config(path: Path) -> dict[str, Any]:
    """Load model configuration from a TOML file. Returns {} if missing."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
        return {}


def resolve_model_id(
    tier: str,
    toml_tiers: dict[str, str] | None = None,
    hardcoded: str | None = None,
) -> str | None:
    """Resolve model ID: env var > TOML > hardcoded default."""
    env_key = f"SAGE_MODEL_{tier.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    if toml_tiers and tier in toml_tiers:
        return toml_tiers[tier]
    return hardcoded


# Hardcoded last-resort fallbacks — must match config/models.toml.
# Updated 2026-04-03.
_TIER_DEFAULTS: dict[str, str] = {
    "fast": "gemini-3.1-flash-lite-preview",
    "mutator": "gpt-5.4-mini",
    "reasoner": "gemini-3.1-pro-preview",
    "codex": "gpt-5.4",
    "codex_max": "gpt-5.4-pro",
    "budget": "deepseek-chat",
    "fallback": "deepseek-chat",
}

# Cached TOML tiers (loaded once, on first call).
_toml_tiers_cache: dict[str, str] | None = None


def get_tier_model(tier: str) -> str:
    """Get the model ID for a tier.  Resolution: env var > models.toml > fallback.

    Safe to call from any module — no circular imports, caches TOML read.
    """
    global _toml_tiers_cache
    if _toml_tiers_cache is None:
        _toml_tiers_cache = {}
        for search_dir in [
            Path.cwd() / "config",
            Path(__file__).parent.parent.parent / "config",
            Path.home() / ".sage",
        ]:
            cfg = load_model_config(search_dir / "models.toml")
            if cfg and "tiers" in cfg:
                _toml_tiers_cache = cfg["tiers"]
                break
    return resolve_model_id(tier, _toml_tiers_cache, _TIER_DEFAULTS.get(tier, "deepseek-chat")) or "deepseek-chat"
