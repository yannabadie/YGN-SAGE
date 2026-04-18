"""ProviderPool — resolve model_id to live LLMProvider at execution time.

Used by TopologyRunner to get per-node providers based on model_id
assigned by ModelAssigner.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider
from sage.resilience import CircuitBreaker

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
        self._breakers: dict[str, CircuitBreaker] = {}

    # -- Boot-time health check -------------------------------------------

    async def health_check(self, timeout: float = 10.0) -> dict[str, bool]:
        """Probe every provider with a minimal request. Open circuit for dead ones.

        Classification (2026-04-18 refined after Codex review):
          - Connection errors (DNS/SSL/timeout/refused) → DEAD (provider unreachable)
          - Quota exhaustion (429 insufficient_quota / rate_limit_exceeded with
            "quota" in message) → DEAD (reachable but unusable; quota typically
            resets on a time boundary). ModelAssigner should route elsewhere.
          - Transient rate-limit 429 without quota wording → ALIVE (probe noise,
            don't kill a perfectly good provider because the probe got rate-limited)
          - Other API errors (400 bad params, 401 auth, 403) → ALIVE (provider
            reachable, probe params just mismatched the adapter default model)
        """
        import asyncio
        from sage.llm.base import Message, Role, LLMConfig as _Cfg

        results: dict[str, bool] = {}
        probe_msg = [Message(role=Role.USER, content="hi")]

        for name, provider in list(self._providers.items()):
            try:
                model_id = getattr(provider, "model_id", "") or getattr(provider, "model_string", "")
                cfg = _Cfg(provider=name, model=model_id, max_tokens=10)
                await asyncio.wait_for(
                    provider.generate(messages=probe_msg, config=cfg),
                    timeout=timeout,
                )
                results[name] = True
                self.record_success(name)
            except Exception as exc:
                exc_name = type(exc).__name__
                exc_str = str(exc).lower()
                # Connection/DNS/SSL/timeout = provider unreachable
                is_connection_error = any(s in exc_str for s in [
                    "connection", "dns", "ssl", "timeout", "getaddrinfo",
                    "refused", "unreachable", "network",
                ])
                # Quota exhaustion: 429 with quota/billing/credits wording.
                # Pure rate-limit without quota signal is probe noise, not a
                # dead provider — handled in the else branch.
                is_quota_exhaustion = (
                    "429" in exc_str or "ratelimit" in exc_name.lower()
                ) and any(s in exc_str for s in [
                    "insufficient_quota", "quota", "billing", "credit",
                    "exceeded your current quota", "payment",
                ])
                if is_connection_error or is_quota_exhaustion:
                    results[name] = False
                    for _ in range(3):
                        self.record_failure(name, exc)
                    reason = "unreachable" if is_connection_error else "quota exhausted"
                    log.warning(
                        "Health check DEAD for %s (%s): %s — circuit opened",
                        name, reason, exc_name,
                    )
                else:
                    # API error (400/401/403/transient 429) — provider reachable
                    results[name] = True
                    self.record_success(name)
                    log.info(
                        "Health check OK for %s (API error on probe, but reachable: %s)",
                        name, exc_name,
                    )

        alive = sum(1 for v in results.values() if v)
        log.info("Provider health check: %d/%d alive %s", alive, len(results), results)
        return results

    # -- Provider inference ------------------------------------------------

    def infer_provider(self, model_id: str) -> str:
        """Infer provider name from model_id. Registry first, string fallback.

        String fallback matches actual model_id patterns in cards.toml (April 2026):
        google: gemini-*  |  openai: gpt-*  |  xai: grok-*
        deepseek: deepseek-*  |  minimax: minimax-*, MiniMax-*
        kimi: kimi-*  |  openrouter: qwen/*
        """
        if self._registry:
            profile = self._registry.get(model_id) if hasattr(self._registry, 'get') else None
            pname = getattr(profile, 'provider', '') if profile else ''
            if pname:
                return pname
        mid = model_id.lower()
        if mid.startswith("gemini"): return "google"
        if mid.startswith("gpt-"): return "openai"
        if mid.startswith("grok"): return "xai"
        if mid.startswith("deepseek"): return "deepseek"
        if "minimax" in mid: return "minimax"
        if mid.startswith("kimi"): return "kimi"
        if "qwen" in mid: return "openrouter"
        return ""

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model's provider is in the pool AND circuit is closed."""
        pname = self.infer_provider(model_id)
        if not pname:
            return True  # Unknown provider — let it try
        if pname not in self._providers:
            return False
        return self.is_available(pname)

    # -- Per-provider circuit breaker API --------------------------------

    def _get_breaker(self, provider_name: str) -> CircuitBreaker:
        if provider_name not in self._breakers:
            self._breakers[provider_name] = CircuitBreaker(
                name=f"provider_{provider_name}",
                max_failures=3,
            )
        return self._breakers[provider_name]

    def record_failure(self, provider_name: str, error: Exception) -> None:
        """Record a provider failure. After 3 failures, circuit opens."""
        self._get_breaker(provider_name).record_failure(error)

    def record_success(self, provider_name: str) -> None:
        """Record success, reset failure counter."""
        self._get_breaker(provider_name).record_success()

    def is_available(self, provider_name: str) -> bool:
        """Check if provider circuit is closed (available)."""
        return not self._get_breaker(provider_name).should_skip()

    # -- Resolution --------------------------------------------------------

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
                # Try to infer provider from model_id prefix before falling back
                _PROVIDER_HINTS = {
                    "deepseek": "deepseek",
                    "gpt-": "openai",
                    "gemini": "google",
                    "grok": "xai",
                    "minimax": "minimax",
                    "moonshot": "kimi",
                    "qwen": "openrouter",
                }
                inferred = None
                for hint, pname in _PROVIDER_HINTS.items():
                    if hint in model_id.lower():
                        inferred = self._providers.get(pname)
                        if inferred is not None:
                            log.debug(
                                "ProviderPool: inferred provider=%s for model_id=%s",
                                pname, model_id,
                            )
                            result = (inferred, LLMConfig(provider=pname, model=model_id))
                            self._cache[model_id] = result
                            return result

                log.debug(
                    "ProviderPool: model_id=%s not found in registry, using default",
                    model_id,
                )
                return (
                    self._default,
                    self._default_config or LLMConfig(provider="default", model=model_id),
                )

            provider_name: str = getattr(profile, "provider", "")
            cw = getattr(profile, "context_window", 128000) or 128000
            config = LLMConfig(provider=provider_name, model=model_id, context_window=cw)

            provider = self._providers.get(provider_name)
            circuit_open = False
            if provider is not None and not self.is_available(provider_name):
                log.info(
                    "ProviderPool: %s circuit open, falling back to default",
                    provider_name,
                )
                provider = None  # fall through to default
                circuit_open = True
            if provider is None:
                log.debug(
                    "ProviderPool: no live provider for provider_name=%s, using default",
                    provider_name,
                )
                result: tuple[LLMProvider, LLMConfig] = (self._default, config)
            else:
                result = (provider, config)

            # Don't cache when circuit is open — allow recovery after cooldown
            if not circuit_open:
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
