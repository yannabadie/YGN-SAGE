"""Provider connectors for auto-discovering available LLM models.

Each connector queries a provider API to enumerate available models.
Failures are isolated per-provider -- one provider going down never crashes boot.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Discovery cache (24h TTL, ~/.sage/discovery_cache/) ──────────────────────
import json as _json
import time as _time
from pathlib import Path as _Path

_CACHE_DIR = _Path.home() / ".sage" / "discovery_cache"
_CACHE_TTL_SECONDS = 24 * 3600  # 24 hours


def _cache_path(provider: str, cache_dir: _Path | None = None) -> _Path:
    return (cache_dir or _CACHE_DIR) / f"{provider}_models.json"


def _read_cache(
    provider: str, cache_dir: _Path | None = None, ttl: int = _CACHE_TTL_SECONDS
) -> list[dict[str, Any]] | None:
    path = _cache_path(provider, cache_dir)
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        if _time.time() - data.get("timestamp", 0) > ttl:
            return None
        return data.get("models", [])
    except (ValueError, OSError):
        return None


def _write_cache(
    provider: str, models: list[dict[str, Any]], cache_dir: _Path | None = None
) -> None:
    d = cache_dir or _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _time.time(), "models": models}
    _cache_path(provider, d).write_text(
        _json.dumps(payload, indent=2), encoding="utf-8"
    )


# ── Provider configuration table ──────────────────────────────────────────────
# sdk: "google-genai" uses the google.genai client; "openai" uses openai.OpenAI.
# hardcoded_models: skip API discovery and use these model IDs instead.

# ── SINGLE SOURCE OF TRUTH for all provider configurations ──────────────────
# Every file that needs provider URLs, API key env vars, or default models
# MUST import from here. Do NOT hardcode URLs elsewhere.
# Updated: March 31, 2026

PROVIDER_CONFIGS: list[dict[str, Any]] = [
    {
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "sdk": "openai",
        "default_model": "deepseek-chat",
    },
    {
        "provider": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "sdk": "openai",  # OpenAI-compatible endpoint (avoids aiohttp SSL issues with google-genai)
        "default_model": "gemini-3.1-flash-lite-preview",
    },
    {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "sdk": "openai",
        "default_model": "gpt-5.4",
    },
    {
        "provider": "xai",
        "api_key_env": "GROK_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "sdk": "openai",
        "default_model": "grok-4-1-fast-reasoning",
    },
    {
        "provider": "kimi",
        "api_key_env": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "sdk": "openai",
        "default_model": "kimi-k2.5",
    },
    {
        "provider": "minimax",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "sdk": "openai",
        "default_model": "minimax-m2.7",
        "hardcoded_models": ["minimax-m2.7"],
    },
    {
        "provider": "openrouter",
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "sdk": "openai",
        "hardcoded_models": ["qwen/qwen3.5-plus-02-15"],
    },
]

# ── Helper functions — USE THESE instead of hardcoding URLs ────────────────

_CONFIG_BY_PROVIDER: dict[str, dict[str, Any]] = {
    cfg["provider"]: cfg for cfg in PROVIDER_CONFIGS
}

# Model prefix → provider name (for routing by model_id)
_MODEL_PREFIX_TO_PROVIDER: dict[str, str] = {
    "deepseek": "deepseek",
    "gpt-": "openai",
    "grok": "xai",
    "gemini": "google",
    "kimi": "kimi",
    "minimax": "minimax",
    "MiniMax": "minimax",
    "qwen": "openrouter",
}


def get_provider_config(provider: str) -> dict[str, Any] | None:
    """Get config for a provider by name. Returns None if unknown."""
    return _CONFIG_BY_PROVIDER.get(provider)


def get_provider_for_model(model_id: str) -> str | None:
    """Infer provider name from model_id prefix. Returns None if unknown."""
    for prefix, provider in _MODEL_PREFIX_TO_PROVIDER.items():
        if prefix in model_id:
            return provider
    return None


def get_base_url(provider: str) -> str | None:
    """Get the API base URL for a provider. None for Google (native SDK)."""
    cfg = _CONFIG_BY_PROVIDER.get(provider)
    return cfg["base_url"] if cfg else None


def get_api_key_env(provider: str) -> str | None:
    """Get the env var name for a provider's API key."""
    cfg = _CONFIG_BY_PROVIDER.get(provider)
    return cfg["api_key_env"] if cfg else None


def get_default_model(provider: str) -> str | None:
    """Get the default model for a provider."""
    cfg = _CONFIG_BY_PROVIDER.get(provider)
    return cfg.get("default_model") if cfg else None


def get_available_providers() -> list[dict[str, Any]]:
    """Return configs for all providers that have an API key set."""
    return [
        cfg for cfg in PROVIDER_CONFIGS
        if os.environ.get(cfg["api_key_env"])
    ]


@dataclass
class DiscoveredModel:
    """A model discovered from a provider API."""

    id: str
    provider: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_thinking: bool = False
    raw_meta: dict[str, Any] = field(default_factory=dict)


class ProviderConnector:
    """Auto-discovers models from all configured providers.

    Usage::

        connector = ProviderConnector()
        models = await connector.discover_all()
        # models: list[DiscoveredModel]
    """

    def __init__(self, configs: list[dict[str, Any]] | None = None):
        self.configs = configs if configs is not None else PROVIDER_CONFIGS

    async def discover_all(self) -> list[DiscoveredModel]:
        """Query every configured provider and return all discovered models.

        Provider failures are logged and silently skipped.
        """
        all_models: list[DiscoveredModel] = []

        for cfg in self.configs:
            provider = cfg["provider"]
            api_key = os.environ.get(cfg["api_key_env"], "")
            # Fallback: also check legacy DEEP_SEEK_API_KEY spelling
            if not api_key and cfg["provider"] == "deepseek":
                api_key = os.environ.get("DEEP_SEEK_API_KEY", "")

            if not api_key:
                logger.debug("Skipping %s: %s not set", provider, cfg["api_key_env"])
                continue

            try:
                if "hardcoded_models" in cfg:
                    models = self._hardcoded(cfg, api_key)
                elif cfg["sdk"] == "google-genai":
                    models = await self._discover_google(api_key)
                else:
                    # Wrap sync discovery in thread with timeout to prevent hanging
                    import asyncio
                    import concurrent.futures
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        try:
                            models = await asyncio.wait_for(
                                loop.run_in_executor(
                                    executor,
                                    self._discover_openai_compat, cfg, api_key,
                                ),
                                timeout=15.0,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("Provider %s discovery timed out (15s)", provider)
                            continue

                all_models.extend(models)
                logger.info("Discovered %d models from %s", len(models), provider)

            except Exception as exc:
                logger.warning("Provider %s discovery failed: %s", provider, exc)

        # Codex CLI removed — all models use API providers only

        return all_models

    # ── Per-SDK discovery methods ─────────────────────────────────────────

    # Non-text model prefixes to skip during discovery
    _SKIP_PREFIXES = (
        "dall-e", "tts-", "whisper-", "text-embedding", "sora-",
        "omni-moderation", "gpt-image", "chatgpt-image",
        "imagen-", "veo-", "gemma-", "aqa", "nano-banana",
        "grok-imagine", "embedding",
    )

    async def _discover_google(self, api_key: str) -> list[DiscoveredModel]:
        """Discover models via google-genai SDK."""
        cached = _read_cache("google")
        if cached is not None:
            logger.info("Using cached Google model list (%d models)", len(cached))
            return [DiscoveredModel(**m) for m in cached]

        try:
            from google import genai
        except ImportError:
            logger.debug("google-genai not installed, skipping Google discovery")
            return []

        import httpx
        from sage.llm._ssl import ssl_verify
        client = genai.Client(api_key=api_key)
        client._api_client._httpx_client = httpx.Client(verify=ssl_verify(), timeout=30)
        models: list[DiscoveredModel] = []

        for m in client.models.list():
            name: str = m.name or ""
            # API returns "models/gemini-..." -- strip the prefix
            model_id = name.removeprefix("models/")
            if not model_id:
                continue
            # Filter non-text models
            if any(model_id.lower().startswith(p) for p in self._SKIP_PREFIXES):
                continue
            # Skip audio/tts/image specialized models
            if any(x in model_id.lower() for x in ("-tts", "-audio", "-image", "native-audio")):
                continue

            ctx = getattr(m, "input_token_limit", None)
            out = getattr(m, "output_token_limit", None)
            thinking = False
            # Check for thinking support in supported_actions or description
            supported_actions = getattr(m, "supported_actions", None)
            if supported_actions and "thinking" in str(supported_actions).lower():
                thinking = True

            models.append(DiscoveredModel(
                id=model_id,
                provider="google",
                context_window=ctx,
                max_output_tokens=out,
                supports_thinking=thinking,
                raw_meta={"input_token_limit": ctx, "output_token_limit": out},
            ))

        _write_cache("google", [
            {"id": m.id, "provider": m.provider,
             "context_window": m.context_window,
             "max_output_tokens": m.max_output_tokens,
             "supports_thinking": m.supports_thinking}
            for m in models
        ])
        return models

    def _discover_openai_compat(
        self, cfg: dict[str, Any], api_key: str
    ) -> list[DiscoveredModel]:
        """Discover models via OpenAI-compatible models.list() endpoint."""
        provider_name = cfg["provider"]
        cached = _read_cache(provider_name)
        if cached is not None:
            logger.info("Using cached %s model list (%d models)", provider_name, len(cached))
            return [DiscoveredModel(**m) for m in cached]

        try:
            import openai
        except ImportError:
            logger.debug("openai package not installed, skipping %s", cfg["provider"])
            return []

        import httpx
        from sage.llm._ssl import ssl_verify
        http_client = httpx.Client(verify=ssl_verify(), timeout=15)
        client = openai.OpenAI(
            api_key=api_key,
            base_url=cfg.get("base_url"),
            http_client=http_client,
        )
        response = client.models.list()
        models: list[DiscoveredModel] = []

        for m in response:
            model_id = m.id
            if not model_id:
                continue
            # Filter non-text models
            if any(model_id.lower().startswith(p) for p in self._SKIP_PREFIXES):
                continue
            if any(x in model_id.lower() for x in ("-tts", "-audio", "-image", "-transcribe", "realtime", "search-api", "search-preview")):
                continue
            models.append(DiscoveredModel(
                id=model_id,
                provider=cfg["provider"],
            ))

        _write_cache(provider_name, [
            {"id": m.id, "provider": m.provider} for m in models
        ])
        return models

    def _hardcoded(
        self, cfg: dict[str, Any], api_key: str
    ) -> list[DiscoveredModel]:
        """Return hardcoded model list (for providers without discovery endpoints)."""
        _ = api_key  # Key is valid if we got here
        return [
            DiscoveredModel(id=mid, provider=cfg["provider"])
            for mid in cfg["hardcoded_models"]
        ]

    def _discover_codex(self) -> list[DiscoveredModel]:
        """Check if Codex CLI is available on PATH."""
        if shutil.which("codex"):
            logger.info("Codex CLI detected on PATH")
            return [DiscoveredModel(
                id="codex-cli",
                provider="codex",
                supports_thinking=True,
            )]
        return []
