"""Model Router for YGN-SAGE.

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

# All models use API providers — no CLI-based providers.
# "codex"/"codex_max" tiers map to OpenAI API models, not Codex CLI.
_HARDCODED = {
    "codex": "gpt-5.4",
    "codex_max": "gpt-5.4-pro",
    "fast": "gemini-3.1-flash-lite-preview",
    "mutator": "gpt-5.4-mini",
    "reasoner": "gemini-3.1-pro-preview",
    "budget": "deepseek-chat",
    "fallback": "deepseek-chat",
}

_MAX_TOKENS = {
    "codex": 8192, "codex_max": 16384, "reasoner": 8192, "critical": 8192,
    "mutator": 4096, "budget": 2048, "fallback": 4096, "fast": 4096,
}


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
        tier: Tier | str = "fast",
        temperature: float = 0.7,
        json_schema: type | dict | None = None,
    ) -> LLMConfig:
        if not ModelRouter.MODELS:
            ModelRouter._load_config()

        lookup_tier = "reasoner" if tier == "critical" else tier
        model = ModelRouter.MODELS.get(lookup_tier, _HARDCODED.get(lookup_tier, _HARDCODED["fast"]))
        max_tokens = _MAX_TOKENS.get(tier, 4096)

        # Infer provider from model_id (all API-based, no CLI providers)
        if "deepseek" in model:
            provider = "deepseek"
        elif "gpt-" in model:
            # cards.toml only wires gpt-5.x. No o1/o3/o4 — see
            # docs/patterns/knowledge-cutoff-checks.md.
            provider = "openai"
        elif "grok" in model:
            provider = "xai"
        elif "gemini" in model:
            provider = "google"
        elif "minimax" in model.lower():
            provider = "minimax"
        elif "moonshot" in model or "kimi" in model:
            provider = "kimi"
        elif "qwen" in model:
            provider = "openrouter"
        else:
            provider = "google"  # fallback

        extra: dict = {}
        if tier == "codex_max" or (tier == "critical" and "gpt-5" in model):
            extra["reasoning_effort"] = "xhigh"

        return LLMConfig(
            provider=provider, model=model, max_tokens=max_tokens,
            temperature=temperature, json_schema=json_schema,
            extra=extra,
        )


ModelRouter._load_config()
