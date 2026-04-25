"""Boot sub-module: LLM provider initialization."""
from __future__ import annotations

import logging
import os
from typing import Any

from sage.llm.base import LLMConfig
from sage.llm.mock import MockProvider
from sage.llm.router import ModelRouter

__all__ = ["init_llm_provider"]

_log = logging.getLogger("sage.boot")


def init_llm_provider(
    use_mock_llm: bool,
    llm_tier: str,
) -> tuple[Any, LLMConfig]:
    """Detect and instantiate the best available LLM provider.

    Returns:
        (provider, llm_config) tuple.
    """
    if use_mock_llm:
        provider = MockProvider(responses=["<think>Processing</think>\nDone."])
        llm_config = LLMConfig(provider="mock", model="mock")
        return provider, llm_config

    # Auto-detect best available API provider (no CLI-based providers).
    if llm_tier == "auto":
        from sage.providers.connector import get_available_providers, PROVIDER_CONFIGS
        available = get_available_providers()
        if available:
            cfg = available[0]
            _log.info("Auto-detected provider: %s (%s)",
                      cfg["provider"], cfg.get("default_model", ""))
            llm_tier = "budget"
        else:
            raise RuntimeError(
                "No LLM provider available. Set at least one API key: "
                + ", ".join(c["api_key_env"] for c in PROVIDER_CONFIGS)
            )

    llm_config = ModelRouter.get_config(llm_tier)

    # Route to correct provider via Pydantic AI (replaced LiteLLM on
    # 2026-04-18, migration docs at docs/plans/2026-04-18-pydantic-ai-migration.md).
    # PydanticAIProvider.for_sage_provider preserves the same signature —
    # one-line swap, no caller changes.
    from sage.providers.connector import (
        get_provider_for_model, get_available_providers, PROVIDER_CONFIGS,
    )
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    model_id = llm_config.model or ""
    matched = False

    # Match by model_id -> provider via connector registry
    prov_name = get_provider_for_model(model_id)
    if prov_name:
        api_key = os.environ.get(
            next((c["api_key_env"] for c in PROVIDER_CONFIGS if c["provider"] == prov_name), ""), ""
        )
        if api_key or prov_name == "google":  # Google uses ADC
            provider = PydanticAIProvider.for_sage_provider(prov_name, model_id, api_key or None)
            matched = True

    # Fallback: try available providers in connector config order
    if not matched:
        for cfg in get_available_providers():
            api_key = os.environ.get(cfg["api_key_env"], "")
            provider = PydanticAIProvider.for_sage_provider(
                cfg["provider"], cfg.get("default_model", ""), api_key or None,
            )
            matched = True
            _log.info("Provider fallback: %s (%s)",
                      cfg["provider"], cfg.get("default_model", "native"))
            break

        if not matched:
            raise RuntimeError(
                "No LLM provider available. Set at least one API key: "
                + ", ".join(c["api_key_env"] for c in PROVIDER_CONFIGS)
            )

    return provider, llm_config
