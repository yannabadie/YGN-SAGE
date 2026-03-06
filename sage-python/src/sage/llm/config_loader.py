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
