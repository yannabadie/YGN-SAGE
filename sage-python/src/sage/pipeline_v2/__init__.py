"""`pipeline_v2` — backing modules for the `CognitiveOrchestrationPipeline` façade.

This package owns the runtime: classify, decompose, select_topology,
assign_models, execute, and learn modules, the orchestrator
(`run_internal` body), the constructor body, the `PipelineContext`
dataclass, and the helper modules (bandit_attribution / runtime_events
/ memory_gate / topology_helpers / costing). `sage.pipeline` is now
a thin public façade.

Module-level lazy `__getattr__` (PEP 562) resolves the public class
names at attribute-access time so the eager `sage.pipeline →
sage.pipeline_v2 → sage.pipeline` cycle (introduced when
`sage.pipeline` re-exports `PipelineContext` from
`pipeline_v2.context`) is broken at module load.

Public surface:

  - ``from sage.pipeline_v2 import CognitiveOrchestrationPipeline``
  - ``from sage.pipeline_v2 import Pipeline``  (alias)
  - ``from sage.pipeline_v2 import PipelineContext``

All three reference the same class objects as the corresponding
imports from ``sage.pipeline``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only imports so static analyzers see the public surface.
    # Runtime resolution goes through `__getattr__` below.
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext  # noqa: F401

# `Pipeline` is the canonical alias the façade exposes. cgpro
# round-3 backward-compat lock: `from sage.pipeline_v2 import Pipeline`
# must reference the same class object as
# `from sage.pipeline import CognitiveOrchestrationPipeline`.

_PUBLIC_LAZY_NAMES = ("CognitiveOrchestrationPipeline", "Pipeline", "PipelineContext")


def __getattr__(name: str) -> Any:
    """PEP 562 module-level lazy attribute resolution.

    Deferring the `from sage.pipeline import ...` to attribute-access
    time avoids the otherwise-circular dependency
    `sage.pipeline → sage.pipeline_v2 → sage.pipeline`.
    """
    if name == "CognitiveOrchestrationPipeline" or name == "Pipeline":
        from sage.pipeline import CognitiveOrchestrationPipeline as _Cog
        return _Cog
    if name == "PipelineContext":
        from sage.pipeline import PipelineContext as _Ctx
        return _Ctx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the lazy names in `dir(pipeline_v2)` and tab-completion."""
    return sorted(_PUBLIC_LAZY_NAMES)


__all__ = list(_PUBLIC_LAZY_NAMES)
