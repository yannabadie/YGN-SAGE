"""SOTA March 2026 Model Router for YGN-SAGE.

Model IDs loaded from: env vars > config/models.toml > hardcoded defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from sage.llm.base import LLMConfig
from sage.llm.config_loader import load_model_config, resolve_model_id

Tier = Literal[
    "fast", "mutator", "reasoner", "codex", "codex_max",
    "budget", "critical", "fallback",
]

_HARDCODED = {
    "codex": "gpt-5.3-codex",
    "codex_max": "gpt-5.2",
    "fast": "gemini-3.1-flash-lite-preview",
    "mutator": "gemini-3-flash-preview",
    "reasoner": "gemini-3.1-pro-preview",
    "budget": "gemini-2.5-flash-lite",
    "fallback": "gemini-2.5-flash",
}

_MAX_TOKENS = {
    "codex": 8192, "codex_max": 16384, "reasoner": 8192, "critical": 8192,
    "mutator": 4096, "budget": 2048, "fallback": 4096, "fast": 4096,
}

_CODEX_TIERS = {"codex", "codex_max"}


class ModelRouter:
    """Routes requests to the optimal model for each task type."""

    MODELS: dict[str, str] = {}

    @classmethod
    def _load_config(cls) -> None:
        """Load model IDs from TOML + env vars, falling back to hardcoded."""
        toml_tiers: dict[str, str] = {}
        for search_dir in [
            Path.cwd() / "config",
            Path(__file__).parent.parent.parent.parent / "config",
            Path.home() / ".sage",
        ]:
            toml_path = search_dir / "models.toml"
            cfg = load_model_config(toml_path)
            if cfg:
                toml_tiers = cfg.get("tiers", {})
                break

        for tier, hardcoded in _HARDCODED.items():
            cls.MODELS[tier] = resolve_model_id(tier, toml_tiers, hardcoded) or hardcoded

    @staticmethod
    def get_config(
        tier: Tier = "fast",
        temperature: float = 0.7,
        json_schema: type | dict | None = None,
    ) -> LLMConfig:
        if not ModelRouter.MODELS:
            ModelRouter._load_config()

        lookup_tier = "reasoner" if tier == "critical" else tier
        model = ModelRouter.MODELS.get(lookup_tier, _HARDCODED.get(lookup_tier, _HARDCODED["fast"]))
        provider = "codex" if tier in _CODEX_TIERS else "google"
        max_tokens = _MAX_TOKENS.get(tier, 4096)

        extra: dict = {}
        if tier == "codex_max":
            extra["reasoning_effort"] = "xhigh"

        return LLMConfig(
            provider=provider, model=model, max_tokens=max_tokens,
            temperature=temperature, json_schema=json_schema,
            extra=extra,
        )


ModelRouter._load_config()
