"""Boot sub-module: LLM provider initialization."""
from __future__ import annotations

import logging
import os
import shutil
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

    # Auto-detect best available provider
    if llm_tier == "auto":
        if shutil.which("codex"):
            llm_tier = "codex"
        elif os.environ.get("GOOGLE_API_KEY"):
            llm_tier = "fast"
        else:
            raise RuntimeError(
                "No LLM provider available. Install Codex CLI or set GOOGLE_API_KEY."
            )

    llm_config = ModelRouter.get_config(llm_tier)
    if llm_config.provider == "codex":
        from sage.llm.codex import CodexProvider
        provider = CodexProvider()
    else:
        # Route to correct provider based on model_id
        # All URLs from connector.py (single source of truth)
        from sage.providers.connector import (
            get_provider_for_model, get_provider_config,
            get_available_providers, PROVIDER_CONFIGS,
        )
        from sage.providers.openai_compat import OpenAICompatProvider

        model_id = llm_config.model or ""
        matched = False

        # Match by model_id -> provider via connector registry
        prov_name = get_provider_for_model(model_id)
        if prov_name:
            cfg = get_provider_config(prov_name)
            if cfg and cfg.get("sdk") == "google-genai":
                from sage.llm.google import GoogleProvider
                provider = GoogleProvider()
                matched = True
            elif cfg:
                api_key = os.environ.get(cfg["api_key_env"], "")
                if api_key:
                    provider = OpenAICompatProvider(
                        api_key=api_key,
                        base_url=cfg["base_url"],
                        model_id=model_id,
                        provider_name=prov_name,
                    )
                    matched = True

        # Fallback: try available providers in connector config order
        if not matched:
            for cfg in get_available_providers():
                api_key = os.environ.get(cfg["api_key_env"], "")
                if cfg.get("sdk") == "google-genai":
                    from sage.llm.google import GoogleProvider
                    provider = GoogleProvider()
                else:
                    provider = OpenAICompatProvider(
                        api_key=api_key,
                        base_url=cfg["base_url"],
                        model_id=cfg.get("default_model", ""),
                        provider_name=cfg["provider"],
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
