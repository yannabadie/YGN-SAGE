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

# ── Provider configuration table ──────────────────────────────────────────────
# sdk: "google-genai" uses the google.genai client; "openai" uses openai.OpenAI.
# hardcoded_models: skip API discovery and use these model IDs instead.

PROVIDER_CONFIGS: list[dict[str, Any]] = [
    {
        "provider": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": None,
        "sdk": "google-genai",
    },
    {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "sdk": "openai",
    },
    {
        "provider": "xai",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "sdk": "openai",
    },
    {
        "provider": "deepseek",
        "api_key_env": "DEEP_SEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "sdk": "openai",
    },
    {
        "provider": "minimax",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "sdk": "openai",
        "hardcoded_models": ["MiniMax-Text-01"],
    },
    {
        "provider": "kimi",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "sdk": "openai",
    },
    {
        "provider": "glm",
        "api_key_env": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "sdk": "openai",
    },
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

            if not api_key:
                logger.debug("Skipping %s: %s not set", provider, cfg["api_key_env"])
                continue

            try:
                if "hardcoded_models" in cfg:
                    models = self._hardcoded(cfg, api_key)
                elif cfg["sdk"] == "google-genai":
                    models = await self._discover_google(api_key)
                else:
                    models = self._discover_openai_compat(cfg, api_key)

                all_models.extend(models)
                logger.info("Discovered %d models from %s", len(models), provider)

            except Exception as exc:
                logger.warning("Provider %s discovery failed: %s", provider, exc)

        # Codex CLI (subprocess-based, not API-based)
        codex_models = self._discover_codex()
        all_models.extend(codex_models)

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
        try:
            from google import genai
        except ImportError:
            logger.debug("google-genai not installed, skipping Google discovery")
            return []

        client = genai.Client(api_key=api_key)
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

        return models

    def _discover_openai_compat(
        self, cfg: dict[str, Any], api_key: str
    ) -> list[DiscoveredModel]:
        """Discover models via OpenAI-compatible models.list() endpoint."""
        try:
            import openai
        except ImportError:
            logger.debug("openai package not installed, skipping %s", cfg["provider"])
            return []

        client = openai.OpenAI(api_key=api_key, base_url=cfg.get("base_url"))
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
