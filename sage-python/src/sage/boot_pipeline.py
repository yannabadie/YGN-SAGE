"""Boot sub-module: pipeline, controller, quality estimator, and ToolForge."""
from __future__ import annotations

import logging
import os
import asyncio
import concurrent.futures
from collections import Counter
from typing import Any

__all__ = ["init_pipeline"]

_log = logging.getLogger("sage.boot")


def _discover_models(registry: Any, use_mock_llm: bool) -> None:
    """Auto-discover available models at boot (HTTP health-check calls)."""
    if use_mock_llm:
        return

    # NOTE: The ThreadPoolExecutor pattern below is intentional and safe.
    # registry.refresh() only performs HTTP health-check calls (no shared
    # state mutation).  When a running event loop already exists (e.g. in
    # Jupyter or async test harnesses), we cannot call asyncio.run() on
    # the same thread, so we delegate to a separate thread with its own
    # event loop.  This avoids "cannot run nested event loop" errors.
    try:
        try:
            _running_loop = asyncio.get_running_loop()
        except RuntimeError:
            _running_loop = None
        if _running_loop and _running_loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: asyncio.run(registry.refresh())).result(timeout=30)
        else:
            # Wrap in timeout to prevent hanging on slow provider discovery
            async def _refresh_with_timeout():
                try:
                    await asyncio.wait_for(registry.refresh(), timeout=30)
                except asyncio.TimeoutError:
                    _log.warning("Provider discovery timed out (30s) -- using cached/TOML models only")
            asyncio.run(_refresh_with_timeout())
        # Log per-provider summary
        available = registry.list_available()
        provider_counts = Counter(p.provider for p in available)
        total = len(registry.profiles)
        avail = len(available)
        summary_parts = [f"{name}: {count}" for name, count in sorted(provider_counts.items())]
        _log.info(
            "Boot: discovered %d models (%d available) -- %s",
            total, avail, ", ".join(summary_parts) if summary_parts else "none",
        )
    except (RuntimeError, TimeoutError, OSError) as e:
        _log.warning("Boot: model discovery failed (%s), continuing with legacy routing", e)


def _build_capability_matrix(
    registry: Any,
    use_mock_llm: bool,
) -> tuple[Any, dict[str, Any]]:
    """Build CapabilityMatrix and runtime adapters from discovered providers.

    Returns:
        (capability_matrix, runtime_adapters) tuple.
    """
    _cap_matrix = None
    _runtime_adapters: dict[str, Any] = {}

    if use_mock_llm:
        return _cap_matrix, _runtime_adapters

    from sage.providers.capabilities import CapabilityMatrix as _CapMatrix
    from sage.providers.connector import PROVIDER_CONFIGS
    from sage.providers.openai_compat import OpenAICompatProvider

    _cap_matrix = _CapMatrix()
    _discovered_providers = {p.provider for p in registry.list_available()}

    for _cfg in PROVIDER_CONFIGS:
        _pname = _cfg["provider"]
        if _pname not in _discovered_providers:
            continue
        _api_key = os.environ.get(_cfg["api_key_env"], "")
        # Fallback for legacy env var spelling
        if not _api_key and _pname == "deepseek":
            _api_key = os.environ.get("DEEP_SEEK_API_KEY", "")
        if not _api_key:
            continue
        if _cfg.get("sdk") == "google-genai":
            from sage.llm.google import GoogleProvider
            _runtime_adapters[_pname] = GoogleProvider(api_key=_api_key)
        else:
            _runtime_adapters[_pname] = OpenAICompatProvider(
                api_key=_api_key,
                base_url=_cfg.get("base_url"),
                provider_name=_pname,
            )
    # Codex CLI provider (uses subprocess, not API key)
    if "codex" in _discovered_providers:
        try:
            from sage.llm.codex import CodexProvider
            _runtime_adapters["codex"] = CodexProvider()
            _log.info("Boot: Codex CLI provider added to runtime adapters")
        except (ImportError, RuntimeError) as e:
            _log.warning("Boot: Codex provider init failed (%s)", e)

    _cap_matrix.populate_from_providers(
        list(_discovered_providers), adapters=_runtime_adapters,
    )

    return _cap_matrix, _runtime_adapters


def init_pipeline(
    router: Any,
    engine: Any,
    provider: Any,
    llm_config: Any,
    bandit: Any,
    rust_registry: Any,
    py_model_registry: Any,
    registry: Any,
    event_bus: Any,
    use_mock_llm: bool,
    consolidator: Any,
    working_memory: Any,
    episodic_memory: Any,
    tool_registry: Any,
    memory_compressor: Any,
) -> dict[str, Any]:
    """Initialize the CognitiveOrchestrationPipeline and supporting components.

    Returns:
        dict with keys: pipeline, controller, registry, capability_matrix,
        runtime_adapters, quality_estimator, tool_forge.
    """
    # ModelRegistry: always created (even in mock mode) so callers can inspect it
    from sage.providers.registry import ModelRegistry
    if registry is None:
        registry = ModelRegistry()

    _discover_models(registry, use_mock_llm)
    _cap_matrix, _runtime_adapters = _build_capability_matrix(registry, use_mock_llm)

    # Model assigner: Rust-first with Python fallback
    model_assigner = None
    try:
        from sage_core import ModelAssigner as RustModelAssigner  # type: ignore[import-not-found]
        if rust_registry:
            model_assigner = RustModelAssigner(rust_registry)
    except ImportError:
        pass
    if model_assigner is None:
        try:
            from sage.llm.model_assigner import ModelAssigner as PyModelAssigner
            if py_model_registry:
                model_assigner = PyModelAssigner(py_model_registry)
        except (ImportError, RuntimeError):
            pass

    # Provider pool: wraps default provider + registry for per-node resolution
    _provider_pool = None
    if provider and registry:
        try:
            from sage.llm.provider_pool import ProviderPool
            _provider_pool = ProviderPool(
                default_provider=provider,
                registry=registry,
                default_config=llm_config,
                providers=_runtime_adapters,
            )
            _log.info("ProviderPool: %d live providers -- %s", len(_runtime_adapters), list(_runtime_adapters.keys()))

            # Health check: probe all providers, open circuit for dead ones
            import asyncio
            try:
                _loop = asyncio.get_event_loop()
                if _loop.is_running():
                    # Already in async context — schedule but don't block
                    _log.debug("Skipping sync health check (event loop running)")
                else:
                    health = _loop.run_until_complete(_provider_pool.health_check(timeout=8.0))
                    dead = [k for k, v in health.items() if not v]
                    if dead:
                        _log.warning("Dead providers excluded: %s", dead)
            except RuntimeError:
                _log.debug("Health check skipped (no event loop)")
        except (ImportError, RuntimeError) as exc:
            _log.warning("ProviderPool init failed: %s", exc)

    # Pipeline: 5-stage orchestration (optional -- None if deps missing)
    _pipeline = None
    if model_assigner and _provider_pool:
        try:
            from sage.pipeline import CognitiveOrchestrationPipeline
            _pipeline = CognitiveOrchestrationPipeline(
                router=router,
                engine=engine,
                assigner=model_assigner,
                provider_pool=_provider_pool,
                bandit=bandit,
                quality_estimator=None,  # Populated dynamically if available
                event_bus=event_bus,
                llm_provider=provider,
                llm_config=llm_config,
                consolidator=consolidator,
                working_memory=working_memory,
                episodic_memory=episodic_memory,
            )
            _log.info("CognitiveOrchestrationPipeline initialized")
        except (ImportError, RuntimeError) as exc:
            _log.warning("Pipeline init failed: %s -- using legacy path", exc)

    # TopologyController (Phase C -- runtime adaptation)
    _controller = None
    _qe = None
    if model_assigner:
        try:
            from sage.topology_controller import TopologyController
            _pv = None
            try:
                from sage.contracts.policy import PolicyVerifier
                _pv = PolicyVerifier
            except ImportError:
                pass
            # QualityEstimator: instantiate for controller quality scoring
            try:
                from sage.quality_estimator import QualityEstimator
                _qe = QualityEstimator()
            except (ImportError, RuntimeError):
                pass
            # PRM: from agent_loop if available -- we don't have agent_loop here
            # so we pass None and let the caller wire it if needed
            _controller = TopologyController(
                assigner=model_assigner,
                quality_estimator=_qe,
                prm=None,
                policy_verifier=_pv,
                embedder=memory_compressor.embedder if memory_compressor else None,
                event_bus=event_bus,
            )
            _log.info("TopologyController initialized (Phase C)")
        except (ImportError, RuntimeError) as exc:
            _log.warning("TopologyController init failed: %s", exc)

    # Pass controller to pipeline
    if _pipeline and _controller:
        _pipeline.controller = _controller

    # Wire QualityEstimator into pipeline Stage 5 LEARN for bandit feedback
    # (ETH-SRI ICLR '25, PILOT 2508.21141: bandit must learn from actual quality)
    _pipeline_qe = _qe
    if not _pipeline_qe:
        try:
            from sage.quality_estimator import QualityEstimator
            _pipeline_qe = QualityEstimator()
        except (ImportError, RuntimeError):
            pass
    if _pipeline and _pipeline_qe:
        _pipeline.quality_estimator = _pipeline_qe

    # ToolForge: autonomous tool synthesis (UCT + SMITH pattern)
    _tool_forge = None
    if _pipeline and provider and tool_registry:
        try:
            from sage.tools.forge import ToolForge
            _tool_forge = ToolForge(
                registry=tool_registry,
                llm_provider=provider,
                llm_config=llm_config,
                event_bus=event_bus,
            )
            _pipeline.tool_forge = _tool_forge
            _log.info("ToolForge initialized (autonomous tool synthesis)")
        except (ImportError, RuntimeError) as exc:
            _log.debug("ToolForge init failed: %s", exc)

    return {
        "pipeline": _pipeline,
        "controller": _controller,
        "registry": registry,
        "capability_matrix": _cap_matrix,
        "runtime_adapters": _runtime_adapters,
        "quality_estimator": _pipeline_qe,
        "tool_forge": _tool_forge,
    }
