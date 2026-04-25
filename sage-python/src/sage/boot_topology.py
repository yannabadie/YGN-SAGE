"""Boot sub-module: topology engine, bandit, and evolution initialization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

__all__ = ["init_topology"]

_log = logging.getLogger("sage.boot")

# Rust Cognitive Engine (primary routing when sage_core is compiled)
try:
    from sage_core import SystemRouter as RustSystemRouter
    from sage_core import ModelRegistry as RustModelRegistry
    from sage_core import TopologyEngine as RustTopologyEngine  # Phase 6
    from sage_core import ContextualBandit as RustBandit  # Phase 6
    _HAS_RUST_ROUTER = True
except ImportError:
    _log.info(
        "sage_core not available -- topology engine using Python fallbacks"
    )
    _HAS_RUST_ROUTER = False


def _find_cards_toml() -> str | None:
    """Search for cards.toml in standard locations."""
    for _cards_dir in [
        Path.cwd() / "sage-core" / "config" / "cards.toml",
        Path.cwd().parent / "sage-core" / "config" / "cards.toml",
        Path(__file__).resolve().parent.parent.parent.parent / "sage-core" / "config" / "cards.toml",
        Path.home() / ".sage" / "cards.toml",
        Path.cwd() / "config" / "cards.toml",
    ]:
        if _cards_dir.exists():
            return str(_cards_dir)
    return None


def _init_rust_router(cards_toml: str | None) -> tuple[Any, Any]:
    """Initialize Rust SystemRouter and ModelRegistry.

    Returns:
        (rust_router, rust_registry) -- either may be None.
    """
    rust_router = None
    rust_registry = None

    if _HAS_RUST_ROUTER:
        try:
            if cards_toml:
                rust_registry = RustModelRegistry.from_toml_file(cards_toml)
                rust_router = RustSystemRouter(rust_registry)
                _log.info(
                    "Boot: Rust SystemRouter active (%d models from %s)",
                    len(rust_registry), cards_toml,
                )
            else:
                _log.info("Boot: cards.toml not found, using Python AdaptiveRouter")
        except (ImportError, RuntimeError) as e:
            _log.warning(
                "Boot: Rust SystemRouter init failed (%s), using Python AdaptiveRouter", e,
            )

    return rust_router, rust_registry


def _init_py_registry(rust_registry: Any, cards_toml: str | None) -> Any:
    """Python ModelRegistry fallback -- used when Rust is unavailable."""
    from sage.llm.model_registry import ModelCardCatalog as PyModelCardCatalog

    py_model_registry = None
    if rust_registry is None and cards_toml:
        try:
            py_model_registry = PyModelCardCatalog.from_toml_file(cards_toml)
            _log.info(
                "Boot: Python ModelRegistry active (%d models from %s)",
                len(py_model_registry), cards_toml,
            )
        except (IOError, OSError, ValueError) as e:
            _log.warning("Boot: Python ModelRegistry init failed (%s)", e)
    return py_model_registry


def init_topology(
    rust_registry: Any,
    metacognition: Any,
) -> dict[str, Any]:
    """Initialize topology engine, bandit, and restore state.

    Returns:
        dict with keys: rust_router, rust_registry, py_model_registry,
        topology_engine, bandit, cards_toml.
    """

    cards_toml = _find_cards_toml()
    rust_router, rust_reg = _init_rust_router(cards_toml)
    py_model_registry = _init_py_registry(rust_reg, cards_toml)

    # If caller provided a pre-built rust_registry, use it; otherwise use ours
    if rust_registry is not None:
        rust_reg = rust_registry

    # Phase 6: Rust TopologyEngine (6-path generation + learning loop)
    rust_topology_engine = None
    rust_bandit = None
    if _HAS_RUST_ROUTER:
        try:
            rust_topology_engine = RustTopologyEngine()
            rust_bandit = RustBandit(0.995, 0.1)
            if rust_router and rust_bandit:
                try:
                    rust_router.set_bandit(rust_bandit)
                    _log.info("Boot: Bandit wired into SystemRouter for integrated routing")
                except (ImportError, RuntimeError) as e:
                    _log.debug("Boot: Failed to wire bandit into router (%s)", e)
            # Warm-start bandit arms from ModelCard affinities
            if rust_reg and rust_bandit:
                try:
                    cards = rust_reg.all_models()
                    templates = ["sequential", "avr", "parallel", "debate"]
                    model_ids = [c.id for c in cards]
                    # Build affinities in row-major: [model0_tmpl0, model0_tmpl1, ..., modelN_tmplT]
                    affinities: list[float] = []
                    for c in cards:
                        for t in templates:
                            if t in ("sequential", "avr"):
                                affinities.append(c.s2_affinity)
                            elif t in ("parallel", "debate"):
                                affinities.append(c.s3_affinity)
                            else:
                                affinities.append(max(c.s1_affinity, c.s2_affinity, c.s3_affinity))
                    rust_bandit.warm_start_from_affinities(model_ids, templates, affinities)
                    _log.info(
                        "Boot: Bandit warm-started with %d models x %d templates (%d arms)",
                        len(model_ids), len(templates), len(model_ids) * len(templates),
                    )
                except (ImportError, RuntimeError) as e:
                    _log.debug("Boot: Bandit warm-start failed (%s)", e)
            _log.info(
                "Boot: Phase 6 active -- TopologyEngine + ContextualBandit ready"
            )
        except (ImportError, RuntimeError) as e:
            _log.warning("Boot: Phase 6 TopologyEngine init failed (%s)", e)

    # P1: Restore persisted bandit + MAP-Elites state from previous session
    _sage_state_dir = str(Path.home() / ".sage")
    if rust_topology_engine is not None:
        try:
            if hasattr(rust_topology_engine, 'load_state'):
                bandit_arms, archive_cells = rust_topology_engine.load_state(_sage_state_dir)
                if bandit_arms > 0 or archive_cells > 0:
                    _log.info(
                        "Boot: Restored persisted state -- %d bandit arms, %d archive cells from %s",
                        bandit_arms, archive_cells, _sage_state_dir,
                    )
        except (IOError, OSError, RuntimeError) as e:
            _log.debug("Boot: No persisted state loaded (%s)", e)

    # P1: Register atexit handler to save bandit + MAP-Elites state at shutdown
    if rust_topology_engine is not None and hasattr(rust_topology_engine, 'save_state'):
        import atexit

        def _save_engine_state(engine=rust_topology_engine, state_dir=_sage_state_dir):
            try:
                engine.save_state(state_dir)
                _log.info("Shutdown: Saved engine state to %s", state_dir)
            except (IOError, OSError, RuntimeError) as exc:
                _log.warning("Shutdown: Failed to save engine state (%s)", exc)

        atexit.register(_save_engine_state)
        _log.info("Boot: atexit handler registered for engine state persistence")

    # Bootstrap S-MMU with template topologies on cold start (P5)
    if rust_topology_engine is not None and rust_topology_engine.smmu_chunk_count() == 0:
        _bootstrap_systems = [1, 2, 3]  # S1=sequential, S2=avr, S3=debate
        _bootstrapped = 0
        for _sys in _bootstrap_systems:
            try:
                _result = rust_topology_engine.generate(
                    f"bootstrap_s{_sys}", None, _sys, 0.0,
                )
                rust_topology_engine.cache_topology(_result.topology)
                rust_topology_engine.record_outcome(
                    _result.topology.id,
                    f"bootstrap_s{_sys}",
                    ["bootstrap", f"s{_sys}"],
                    None,
                    0.5,  # neutral quality
                    0.0,
                    0.0,
                )
                _bootstrapped += 1
            except (ImportError, RuntimeError):
                pass
        if _bootstrapped > 0:
            _log.info(
                "S-MMU bootstrapped with %d template topologies (%d chunks)",
                _bootstrapped, rust_topology_engine.smmu_chunk_count(),
            )

    return {
        "rust_router": rust_router,
        "rust_registry": rust_reg,
        "py_model_registry": py_model_registry,
        "topology_engine": rust_topology_engine,
        "bandit": rust_bandit,
        "cards_toml": cards_toml,
    }
