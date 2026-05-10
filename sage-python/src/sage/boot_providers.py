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


def _runtime_policy_violation_reason(provider_name: str) -> str | None:
    try:
        from sage.pipeline_v2.provider_policy import provider_policy_from_env

        policy = provider_policy_from_env()
    except Exception:  # noqa: BLE001 - boot should preserve legacy no-policy behavior.
        return None
    return policy.violation_reason(provider_name)


def _runtime_policy_active() -> bool:
    try:
        from sage.pipeline_v2.provider_policy import provider_policy_from_env

        return provider_policy_from_env().active
    except Exception:  # noqa: BLE001 - boot should preserve legacy no-policy behavior.
        return False


def _api_key_for_config(cfg: dict[str, Any]) -> str:
    api_key = os.environ.get(str(cfg["api_key_env"]), "")
    if not api_key and cfg["provider"] == "deepseek":
        api_key = os.environ.get("DEEP_SEEK_API_KEY", "")
    return api_key


def _fallback_model_for_config(cfg: dict[str, Any]) -> str:
    default_model = str(cfg.get("default_model", "") or "")
    if default_model:
        return default_model
    hardcoded = cfg.get("hardcoded_models")
    if isinstance(hardcoded, list) and hardcoded:
        candidate = hardcoded[0]
        if isinstance(candidate, str):
            return candidate
    return ""


def _available_allowed_providers(
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        cfg for cfg in configs
        if _api_key_for_config(cfg)
        and _fallback_model_for_config(cfg)
        and _runtime_policy_violation_reason(str(cfg["provider"])) is None
    ]


def _provider_config(
    configs: list[dict[str, Any]],
    provider_name: str,
) -> dict[str, Any] | None:
    return next(
        (cfg for cfg in configs if cfg["provider"] == provider_name),
        None,
    )


def _fallback_config_for_provider(
    cfg: dict[str, Any],
    requested_config: LLMConfig,
) -> LLMConfig:
    extra: dict[str, Any] = {}
    if cfg["provider"] == "deepseek" and _fallback_model_for_config(cfg) == "deepseek-v4-flash":
        extra["thinking"] = "disabled"
    return LLMConfig(
        provider=str(cfg["provider"]),
        model=_fallback_model_for_config(cfg),
        max_tokens=requested_config.max_tokens,
        context_window=requested_config.context_window,
        temperature=requested_config.temperature,
        top_p=requested_config.top_p,
        json_schema=requested_config.json_schema,
        extra=extra,
    )


def _no_available_provider_error(
    configs: list[dict[str, Any]],
) -> RuntimeError:
    if not _runtime_policy_active():
        return RuntimeError(
            "No LLM provider available. Set at least one API key: "
            + ", ".join(c["api_key_env"] for c in configs)
        )
    from sage.pipeline_v2.provider_policy import ProviderPolicyViolation

    return ProviderPolicyViolation(
        "No LLM provider available under runtime provider policy. "
        "Set an API key for an allowed provider: "
        + ", ".join(c["api_key_env"] for c in configs)
    )


def init_llm_provider(
    use_mock_llm: bool,
    llm_tier: str,
) -> tuple[Any, LLMConfig]:
    """Detect and instantiate the best available LLM provider.

    Returns:
        (provider, llm_config) tuple.
    """
    provider: Any
    if use_mock_llm:
        provider = MockProvider(responses=["<think>Processing</think>\nDone."])
        llm_config = LLMConfig(provider="mock", model="mock")
        return provider, llm_config

    # Auto-detect best available API provider (no CLI-based providers).
    if llm_tier == "auto":
        from sage.providers.connector import get_available_providers, PROVIDER_CONFIGS
        available = [
            cfg for cfg in get_available_providers()
            if _runtime_policy_violation_reason(str(cfg["provider"])) is None
        ]
        if available:
            cfg = available[0]
            _log.info("Auto-detected provider: %s (%s)",
                      cfg["provider"], cfg.get("default_model", ""))
            llm_tier = "budget"
        else:
            raise _no_available_provider_error(PROVIDER_CONFIGS)

    llm_config = ModelRouter.get_config(llm_tier)

    # Route to correct provider via Pydantic AI (replaced LiteLLM on
    # 2026-04-18, migration docs at docs/plans/2026-04-18-pydantic-ai-migration.md).
    # PydanticAIProvider.for_sage_provider preserves the same signature —
    # one-line swap, no caller changes.
    from sage.providers.connector import (
        get_provider_for_model, PROVIDER_CONFIGS,
    )
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    model_id = llm_config.model or ""
    matched = False

    # Match by model_id -> provider via connector registry
    prov_name = get_provider_for_model(model_id)
    if prov_name:
        reason = _runtime_policy_violation_reason(prov_name)
        if reason is None:
            provider_cfg = _provider_config(PROVIDER_CONFIGS, prov_name)
            api_key = _api_key_for_config(provider_cfg) if provider_cfg is not None else ""
            if api_key or prov_name == "google":  # Google uses ADC
                provider = PydanticAIProvider.for_sage_provider(
                    prov_name,
                    model_id,
                    api_key or None,
                )
                matched = True
        else:
            allowed = _available_allowed_providers(PROVIDER_CONFIGS)
            if not allowed:
                raise _no_available_provider_error(PROVIDER_CONFIGS)
            cfg = allowed[0]
            api_key = _api_key_for_config(cfg)
            fallback_model = _fallback_model_for_config(cfg)
            provider = PydanticAIProvider.for_sage_provider(
                str(cfg["provider"]),
                fallback_model,
                api_key or None,
            )
            llm_config = _fallback_config_for_provider(cfg, llm_config)
            matched = True
            _log.info(
                "Provider policy fallback: blocked %s (%s: %s), using %s (%s)",
                prov_name,
                model_id,
                reason,
                cfg["provider"],
                fallback_model,
            )

    # Fallback: try available providers in connector config order
    if not matched:
        for cfg in _available_allowed_providers(PROVIDER_CONFIGS):
            api_key = _api_key_for_config(cfg)
            fallback_model = _fallback_model_for_config(cfg)
            provider = PydanticAIProvider.for_sage_provider(
                cfg["provider"], fallback_model, api_key or None,
            )
            llm_config = _fallback_config_for_provider(cfg, llm_config)
            matched = True
            _log.info("Provider fallback: %s (%s)",
                      cfg["provider"], cfg.get("default_model", "native"))
            break

        if not matched:
            raise _no_available_provider_error(PROVIDER_CONFIGS)

    return provider, llm_config
