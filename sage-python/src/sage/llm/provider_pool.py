"""ProviderPool — resolve model_id to live LLMProvider at execution time.

Used by TopologyRunner to get per-node providers based on model_id
assigned by ModelAssigner.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider

log = logging.getLogger(__name__)


class ProviderPool:
    """Resolve model_id → (LLMProvider, LLMConfig) with caching + fallback.

    Parameters
    ----------
    default_provider : LLMProvider
        Fallback when model_id is unknown or provider unavailable.
    registry : sage.providers.registry.ModelRegistry
        Runtime discovery registry with available models and connectors.
    default_config : LLMConfig, optional
        Config to use with default_provider.
    providers : dict[str, LLMProvider], optional
        Pre-built provider instances keyed by provider name (e.g. "google",
        "openai"). When a model's provider name matches a key here, that
        instance is returned instead of the default.
    """

    def __init__(
        self,
        default_provider: LLMProvider,
        registry: Any,
        default_config: LLMConfig | None = None,
        providers: dict[str, LLMProvider] | None = None,
    ) -> None:
        self._default = default_provider
        self._default_config = default_config
        self._registry = registry
        self._providers: dict[str, LLMProvider] = providers or {}
        self._cache: dict[str, tuple[LLMProvider, LLMConfig]] = {}

    def resolve(self, model_id: str) -> tuple[LLMProvider, LLMConfig]:
        """Resolve model_id to (provider, config). Falls back to default.

        Resolution order:
        1. Return cached result if already resolved.
        2. Look up model profile in registry via ``registry.get(model_id)``.
        3. Match profile's provider name against injected ``providers`` dict.
        4. Fall back to default_provider on any miss or error.

        Parameters
        ----------
        model_id:
            Fully-qualified model identifier (e.g. "gemini-2.5-flash").

        Returns
        -------
        tuple[LLMProvider, LLMConfig]
            Always returns a valid pair — never raises.
        """
        if not model_id:
            return (
                self._default,
                self._default_config or LLMConfig(provider="default", model="default"),
            )

        if model_id in self._cache:
            return self._cache[model_id]

        try:
            profile = (
                self._registry.get(model_id) if self._registry is not None else None
            )

            if profile is None:
                log.debug(
                    "ProviderPool: model_id=%s not found in registry, using default",
                    model_id,
                )
                return (
                    self._default,
                    self._default_config or LLMConfig(provider="default", model=model_id),
                )

            provider_name: str = getattr(profile, "provider", "")
            config = LLMConfig(provider=provider_name, model=model_id)

            provider = self._providers.get(provider_name)
            if provider is None:
                log.debug(
                    "ProviderPool: no live provider for provider_name=%s, using default",
                    provider_name,
                )
                result: tuple[LLMProvider, LLMConfig] = (self._default, config)
            else:
                result = (provider, config)

            self._cache[model_id] = result
            return result

        except Exception as exc:
            log.warning(
                "ProviderPool: resolve(%s) failed: %s, using default", model_id, exc
            )
            return (
                self._default,
                self._default_config or LLMConfig(provider="default", model=model_id),
            )

    def warm(self, model_ids: list[str]) -> None:
        """Pre-resolve a list of model IDs into the cache.

        Useful at topology load time to surface registry misses early.
        """
        for mid in model_ids:
            self.resolve(mid)
