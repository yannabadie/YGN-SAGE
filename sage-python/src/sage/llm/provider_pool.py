"""ProviderPool — resolve model_id to live LLMProvider at execution time.

Used by TopologyRunner to get per-node providers based on model_id
assigned by ModelAssigner.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider
from sage.resilience import CircuitBreaker

log = logging.getLogger(__name__)

# Default TTL for dead-provider exclusion. 5 minutes covers most transient
# outages (Gemini brown-outs, OpenAI backend hiccups) without making
# operators wait hours for a recovered provider. Shorter values increase
# probe traffic; longer values delay recovery.
DEFAULT_EXCLUSION_TTL_SEC = 300.0


def _provider_name(provider: Any, config: LLMConfig | None = None) -> str:
    return str(
        getattr(provider, "provider_name", "")
        or (
            getattr(config, "provider", "")
            if config is not None
            else ""
        )
        or getattr(provider, "name", "")
        or ""
    )


def _settings_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    items = getattr(raw, "items", None)
    if callable(items):
        try:
            return {str(k): v for k, v in items()}
        except Exception:  # noqa: BLE001
            return {}
    return {}


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
        self._provider_policy_allowlist: frozenset[str] | None = None
        self._provider_policy_denylist: frozenset[str] = frozenset()
        self._provider_policy_source: str = ""
        # Time-bounded exclusion: provider_name → unix timestamp when it was
        # marked dead. reprobe_excluded_providers() re-tests entries older
        # than a TTL and removes them if they respond. Replaces the earlier
        # boot-permanent exclusion which left recovered providers locked out
        # for the whole process lifetime.
        self._dead_at: dict[str, float] = {}

        # Inject self into every provider that can self-report rate-limit /
        # quota failures at runtime. Enables FrugalGPT-on-rate-limit: when
        # a provider starts 429-ing mid-task, circuit trips immediately and
        # subsequent nodes route elsewhere without waiting for probe cycles.
        for _p in self._providers.values():
            if hasattr(_p, "_pool_ref"):
                _p._pool_ref = self

    def set_provider_policy(
        self,
        *,
        allowlist: frozenset[str] | None,
        denylist: frozenset[str],
        source: str,
    ) -> None:
        """Install the effective runtime provider policy for resolve() guards."""
        self._provider_policy_allowlist = allowlist
        self._provider_policy_denylist = denylist
        self._provider_policy_source = source

    def _effective_provider_policy(self) -> Any:
        from sage.pipeline_v2.provider_policy import (
            ProviderPolicy,
            provider_policy_from_env,
        )

        if self._provider_policy_source:
            return ProviderPolicy(
                allowlist=self._provider_policy_allowlist,
                denylist=self._provider_policy_denylist,
                source=self._provider_policy_source,
            )
        return provider_policy_from_env()

    def _enforce_provider_policy(
        self,
        *,
        model_id: str,
        provider_name: str,
    ) -> None:
        from sage.pipeline_v2.provider_policy import ProviderPolicyViolation

        policy = self._effective_provider_policy()
        reason = policy.violation_reason(provider_name)
        if reason is None:
            return
        raise ProviderPolicyViolation(
            "provider policy violation: "
            f"source={policy.source}; model_id={model_id!r}; "
            f"provider_id={provider_name!r}; reason={reason}"
        )

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
        from sage.pipeline_v2.provider_policy import ProviderPolicyViolation

        results: dict[str, bool] = {}
        probe_msg = [Message(role=Role.USER, content="hi")]

        for name, provider in list(self._providers.items()):
            try:
                model_id = getattr(provider, "model_id", "") or getattr(provider, "model_string", "")
                self._enforce_provider_policy(model_id=model_id, provider_name=name)
                cfg = _Cfg(provider=name, model=model_id, max_tokens=10)
                await asyncio.wait_for(
                    provider.generate(messages=probe_msg, config=cfg),
                    timeout=timeout,
                )
                results[name] = True
                self.record_success(name)
            except ProviderPolicyViolation as exc:
                results[name] = False
                self._dead_at[name] = time.time()
                log.info("Health check skipped for policy-blocked provider %s: %s", name, exc)
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
                    self._dead_at[name] = time.time()
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
                    # Successful probe clears any prior exclusion.
                    self._dead_at.pop(name, None)
                    log.info(
                        "Health check OK for %s (API error on probe, but reachable: %s)",
                        name, exc_name,
                    )
            else:
                # Success path — ensure exclusion cleared.
                self._dead_at.pop(name, None)

        alive = sum(1 for v in results.values() if v)
        log.info("Provider health check: %d/%d alive %s", alive, len(results), results)
        return results

    def get_dead_providers(self, ttl_sec: float = DEFAULT_EXCLUSION_TTL_SEC) -> list[str]:
        """Return providers currently marked DEAD and still within TTL.

        Entries older than ttl_sec are considered expired — they will be
        re-probed on the next ``reprobe_excluded_providers()`` call and
        removed from the dead list if they respond.
        """
        now = time.time()
        return [
            name for name, dead_since in self._dead_at.items()
            if (now - dead_since) < ttl_sec
        ]

    async def reprobe_excluded_providers(
        self,
        timeout: float = 5.0,
        ttl_sec: float = DEFAULT_EXCLUSION_TTL_SEC,
    ) -> dict[str, bool]:
        """Re-test any provider that has been DEAD longer than ttl_sec.

        Providers marked DEAD are not permanently excluded — outages
        recover (Gemini brown-outs, quota resets at midnight UTC, OpenAI
        backend hiccups). Call this at the start of each task batch (or
        cron-style on a short interval) so ModelAssigner sees a current
        view of provider health.

        Parameters
        ----------
        timeout : float
            Per-provider probe timeout (short by design — we already
            probed once at boot; this is a cheap re-verify).
        ttl_sec : float
            Minimum age before re-probing. Within TTL: stay DEAD.
            After TTL: probe; if OK, remove from dead list.

        Returns
        -------
        dict[str, bool]
            {provider_name: alive_now} for every provider that was in
            the dead list at entry. Providers not in the dead list are
            omitted.
        """
        import asyncio
        from sage.llm.base import Message, Role, LLMConfig as _Cfg
        from sage.pipeline_v2.provider_policy import ProviderPolicyViolation

        now = time.time()
        to_reprobe = [
            name for name, dead_since in list(self._dead_at.items())
            if (now - dead_since) >= ttl_sec
        ]
        if not to_reprobe:
            return {}

        results: dict[str, bool] = {}
        probe_msg = [Message(role=Role.USER, content="hi")]
        for name in to_reprobe:
            provider = self._providers.get(name)
            if provider is None:
                # Provider gone entirely (unloaded) — drop from dead list
                self._dead_at.pop(name, None)
                continue
            try:
                model_id = getattr(provider, "model_id", "") or getattr(provider, "model_string", "")
                self._enforce_provider_policy(model_id=model_id, provider_name=name)
                cfg = _Cfg(provider=name, model=model_id, max_tokens=10)
                await asyncio.wait_for(
                    provider.generate(messages=probe_msg, config=cfg),
                    timeout=timeout,
                )
                results[name] = True
                self.record_success(name)
                self._dead_at.pop(name, None)
                log.info("Reprobe RECOVERED provider %s — removed from exclusion list", name)
            except ProviderPolicyViolation as exc:
                results[name] = False
                log.info("Reprobe skipped for policy-blocked provider %s: %s", name, exc)
            except Exception as exc:
                exc_str = str(exc).lower()
                exc_name = type(exc).__name__
                # Use the same classifier as health_check — recoverable
                # API errors (400/401 with wrong probe params) should
                # recover the provider since it's reachable.
                is_connection_error = any(s in exc_str for s in [
                    "connection", "dns", "ssl", "timeout", "getaddrinfo",
                    "refused", "unreachable", "network",
                ])
                is_quota_exhaustion = (
                    "429" in exc_str or "ratelimit" in exc_name.lower()
                ) and any(s in exc_str for s in [
                    "insufficient_quota", "quota", "billing", "credit",
                    "exceeded your current quota", "payment",
                ])
                if is_connection_error or is_quota_exhaustion:
                    results[name] = False
                    self._dead_at[name] = now  # reset TTL — still dead
                    log.info("Reprobe STILL DEAD for %s: %s", name, exc_name)
                else:
                    results[name] = True
                    self.record_success(name)
                    self._dead_at.pop(name, None)
                    log.info(
                        "Reprobe RECOVERED provider %s (API-layer error on probe: %s) "
                        "— reachable, removed from exclusion list",
                        name, exc_name,
                    )
        return results

    async def refresh_exclusion_list(
        self,
        model_assigner: Any = None,
        ttl_sec: float = DEFAULT_EXCLUSION_TTL_SEC,
        timeout: float = 5.0,
    ) -> list[str]:
        """Reprobe expired dead providers and push the current dead list to
        the Rust ``ModelAssigner`` if one is provided.

        Intended to be called at the start of each task batch (e.g. from
        ``SWEBenchBench.generate_patches`` before the first instance, or
        from ``CognitiveOrchestrationPipeline.run`` before Stage 4). Fast
        path: no dead entries older than ttl_sec → returns early without
        touching the network.

        Returns the final list of provider names still considered dead.
        """
        await self.reprobe_excluded_providers(timeout=timeout, ttl_sec=ttl_sec)
        dead = list(self._dead_at.keys())
        if model_assigner is not None and hasattr(model_assigner, "exclude_providers"):
            try:
                model_assigner.exclude_providers(dead)
            except Exception as exc:  # noqa: BLE001 - fallback logging
                log.warning("refresh_exclusion_list: assigner update failed: %s", exc)
        return dead

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
            # B2 bug 1 (2026-05-12 canary): registry._profile_from_toml stamps
            # provider="unknown" when a curated model_profiles.toml entry has
            # no `provider` key. That truthy sentinel must NOT win over the
            # resolvable fallback below — it leaked into node_started
            # provider_id and tripped the canary provider_gate.
            if pname and pname != "unknown":
                return pname
        mid = model_id.lower()
        if mid.startswith("gemini"):
            return "google"
        if mid.startswith("gpt-"):
            return "openai"
        if mid.startswith("grok"):
            return "xai"
        if mid.startswith("deepseek"):
            return "deepseek"
        if "minimax" in mid:
            return "minimax"
        if mid.startswith("kimi"):
            return "kimi"
        if "qwen" in mid:
            return "openrouter"
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

    def _catalog_runtime_resolution(
        self,
        model_id: str,
    ) -> tuple[str, Any | None, dict[str, Any], str]:
        """Resolve catalog aliases to executable ids before provider lookup."""
        if self._registry is None or not hasattr(self._registry, "get"):
            return model_id, None, {}, ""

        profile = self._registry.get(model_id)
        if profile is None:
            return model_id, None, {}, ""

        extra = _settings_dict(getattr(profile, "runtime_settings", {}) or {})
        if getattr(profile, "runtime_selectable", True) is not False:
            return model_id, profile, extra, ""

        replacement = str(getattr(profile, "runtime_replacement", "") or "").strip()
        if not replacement:
            return model_id, profile, extra, ""

        replacement_profile = self._registry.get(replacement)
        if replacement_profile is None:
            return model_id, profile, extra, ""

        runtime_extra = _settings_dict(
            getattr(profile, "runtime_replacement_settings", {}) or {}
        )
        if not runtime_extra:
            runtime_extra = _settings_dict(
                getattr(replacement_profile, "runtime_settings", {}) or {}
            )
        runtime_extra["alias_from"] = model_id
        return replacement, replacement_profile, runtime_extra, model_id

    def resolve(self, model_id: str) -> tuple[LLMProvider, LLMConfig]:
        """Resolve model_id to (provider, config). Falls back to default.

        Resolution order:
        1. Check active provider policy before returning a provider.
        2. Return cached result if already resolved.
        3. Look up model profile in registry via ``registry.get(model_id)``.
        4. Match profile's provider name against injected ``providers`` dict.
        5. Fall back to default_provider on any miss or non-policy error.

        Parameters
        ----------
        model_id:
            Fully-qualified model identifier (e.g. "gemini-2.5-flash").

        Returns
        -------
        tuple[LLMProvider, LLMConfig]
            Returns a valid pair unless an active provider policy blocks the
            inferred provider or fallback provider.
        """
        default_provider_name = _provider_name(self._default, self._default_config)
        if not model_id:
            self._enforce_provider_policy(
                model_id=model_id,
                provider_name=default_provider_name,
            )
            return (
                self._default,
                self._default_config or LLMConfig(provider="default", model="default"),
            )

        requested_model_id = model_id

        if requested_model_id in self._cache:
            cached_provider, cached_config = self._cache[requested_model_id]
            cached_provider_name = _provider_name(cached_provider, cached_config)
            self._enforce_provider_policy(
                model_id=model_id,
                provider_name=cached_provider_name,
            )
            return cached_provider, cached_config

        try:
            model_id, profile, runtime_extra, alias_from = self._catalog_runtime_resolution(model_id)

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
                        self._enforce_provider_policy(
                            model_id=model_id,
                            provider_name=pname,
                        )
                        inferred = self._providers.get(pname)
                        if inferred is not None:
                            log.debug(
                                "ProviderPool: inferred provider=%s for model_id=%s",
                                pname, model_id,
                            )
                            result = (inferred, LLMConfig(provider=pname, model=model_id))
                            self._cache[requested_model_id] = result
                            return result

                log.debug(
                    "ProviderPool: model_id=%s not found in registry, using default",
                    model_id,
                )
                self._enforce_provider_policy(
                    model_id=model_id,
                    provider_name=default_provider_name,
                )
                return (
                    self._default,
                    self._default_config or LLMConfig(provider="default", model=model_id),
                )

            provider_name: str = getattr(profile, "provider", "")
            self._enforce_provider_policy(
                model_id=model_id,
                provider_name=provider_name,
            )
            cw = getattr(profile, "context_window", 128000) or 128000
            config = LLMConfig(
                provider=provider_name,
                model=model_id,
                context_window=cw,
                extra=dict(runtime_extra),
            )
            if alias_from:
                log.info(
                    "ProviderPool: model alias rewritten requested=%s runtime=%s",
                    alias_from,
                    model_id,
                )

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
                self._enforce_provider_policy(
                    model_id=model_id,
                    provider_name=default_provider_name,
                )
                result = (self._default, config)
            else:
                result = (provider, config)

            # Don't cache when circuit is open — allow recovery after cooldown
            if not circuit_open:
                self._cache[requested_model_id] = result
            return result

        except Exception as exc:
            from sage.pipeline_v2.provider_policy import ProviderPolicyViolation

            if isinstance(exc, ProviderPolicyViolation):
                raise
            log.warning(
                "ProviderPool: resolve(%s) failed: %s, using default", model_id, exc
            )
            self._enforce_provider_policy(
                model_id=model_id,
                provider_name=default_provider_name,
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
