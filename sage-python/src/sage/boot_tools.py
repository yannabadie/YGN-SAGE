"""Boot sub-module: tool registry, sandbox, and kNN router initialization."""
from __future__ import annotations

import logging
from typing import Any

from sage.sandbox.manager import SandboxManager
from sage.strategy.adaptive_router import AdaptiveRouter
from sage.tools.registry import ToolRegistry

__all__ = ["init_tools", "check_sandbox_availability"]

_log = logging.getLogger("sage.boot")


def check_sandbox_availability() -> bool:
    """Check if any code execution sandbox is available. Warns if not."""
    has_wasm = False
    has_subprocess = False
    has_docker = False

    try:
        from sage_core import ToolExecutor
        te = ToolExecutor()
        has_wasm = te.has_wasm() or te.has_wasi()
        # tree-sitter + subprocess are always available when ToolExecutor loads
        has_subprocess = True
    except (ImportError, RuntimeError):
        pass

    if not has_subprocess:
        try:
            import shutil
            has_docker = shutil.which("docker") is not None
        except ImportError:
            pass

    available = has_wasm or has_subprocess or has_docker
    if not available:
        _log.warning(
            "Code execution unavailable (no sage_core, no Docker). "
            "Tool execution will fail unless allow_local=True."
        )
    elif not has_wasm:
        _log.info(
            "Sandbox: tree-sitter + subprocess (no Wasm component loaded). "
            "Load a .wasm module via ToolExecutor.load_component() for full isolation."
        )
    return available


def init_tools(
    event_bus: Any,
    provider: Any,
    use_mock_llm: bool,
) -> tuple[ToolRegistry, SandboxManager, Any]:
    """Initialize tool registry, sandbox manager, and kNN router.

    Returns:
        (tool_registry, sandbox_manager, knn_router) tuple.
    """
    tool_registry = ToolRegistry()

    # Runtime tool synthesis -- sandboxed (SEC-01/SEC-02 fixed).
    # Tools execute in subprocess isolation, not in-process exec().
    from sage.tools.meta import create_python_tool, create_bash_tool
    tool_registry.register(create_python_tool)
    tool_registry.register(create_bash_tool)

    # Sandbox manager for S2 empirical validation
    # SECURITY: local host execution disabled by default.
    # Set SAGE_ALLOW_LOCAL_EXEC=1 for benchmarks that need code execution.
    import os
    _allow_local = os.environ.get("SAGE_ALLOW_LOCAL_EXEC") == "1"
    sandbox_manager = SandboxManager(allow_local=_allow_local)
    if _allow_local:
        _log.info("Sandbox: allow_local=True (SAGE_ALLOW_LOCAL_EXEC=1)")

    # Stage 0.5: kNN router (arXiv 2505.12601 -- kNN on embeddings beats complex routers)
    _knn_router = None
    try:
        from sage.strategy.knn_router import KnnRouter
        _knn_router = KnnRouter()
        if not _knn_router.is_ready:
            # Try building from ground truth on-the-fly
            if _knn_router.build_from_ground_truth():
                _log.info(
                    "Boot: kNN router built from ground truth (%d exemplars, %s)",
                    _knn_router.exemplar_count, _knn_router.embedder_backend,
                )
            else:
                _knn_router = None
        else:
            _log.info(
                "Boot: kNN router loaded (%d exemplars, %s)",
                _knn_router.exemplar_count, _knn_router.embedder_backend,
            )
    except (ImportError, RuntimeError) as e:
        _log.info("Boot: kNN router unavailable (%s)", e)

    return tool_registry, sandbox_manager, _knn_router


def init_metacognition(
    provider: Any,
    use_mock_llm: bool,
    knn_router: Any,
) -> AdaptiveRouter:
    """Initialize AdaptiveRouter with kNN and optional Rust SIMD exemplars.

    Returns:
        Configured AdaptiveRouter instance.
    """
    metacognition = AdaptiveRouter(
        llm_provider=provider if not use_mock_llm else None,
        knn_router=knn_router,
    )

    # Load kNN exemplars into Rust AdaptiveRouter for native SIMD kNN search
    if knn_router is not None and knn_router.is_ready and metacognition.has_rust:
        try:
            import numpy as np
            emb = knn_router._exemplar_embeddings
            labels = knn_router._exemplar_labels
            if emb is not None and labels is not None:
                flat_emb = emb.flatten().tolist()
                flat_labels = labels.astype(np.uint8).tolist()
                assert metacognition._rust is not None  # narrowed by has_rust above
                n = metacognition._rust.load_exemplars(flat_emb, flat_labels)
                if n > 0:
                    _log.info("Boot: Rust kNN loaded %d exemplars (native SIMD search)", n)
        except (ImportError, RuntimeError, ValueError) as e:
            _log.info("Boot: Rust kNN exemplar load failed (%s), using Python kNN", e)

    return metacognition
