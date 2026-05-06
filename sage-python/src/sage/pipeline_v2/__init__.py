"""`pipeline_v2` — Phase 2.1 façade backing modules.

Cycle-13 K Phase 2.1 (cgpro `cgpro_phase21_facade_rewrite_20260506`,
2026-05-06): this package now hosts the bulk of the legacy
`CognitiveOrchestrationPipeline` body. The 5 stage modules
(classify/decompose/select_topology/assign_models/execute/learn) +
the orchestrator + the helper modules
(bandit_attribution / runtime_events / memory_gate / topology_helpers /
costing) own the runtime; `sage.pipeline` keeps a thin façade plus
6 `_stage_*` transition seams that 27 test files rely on as their
runtime interception contract. cgpro round-4 OPTION_3 reclassified
the seam removal + final ``pipeline.py < 300 LOC`` target to
Phase 2.2.

Phase 2.1 Step E0 (this file): the previous top-level
``from sage.pipeline import CognitiveOrchestrationPipeline,
PipelineContext`` is replaced by a PEP 562 module-level
``__getattr__`` that resolves the public names lazily at attribute-
access time. Reason: Phase 2.1 Step E1 moves the ``PipelineContext``
dataclass source to ``pipeline_v2/context.py`` and adds
``from sage.pipeline_v2.context import PipelineContext`` to
``sage.pipeline``. The legacy eager import from inside this
``__init__`` would then close a circular: ``sage.pipeline →
sage.pipeline_v2 → sage.pipeline``. The PEP 562 form defers the
lookup to call time, and the actual symbol resolution falls back
to whichever module currently owns the name (``sage.pipeline``
during Step D, ``sage.pipeline_v2.context`` after Step E1).

Public surface preserved:

  - ``from sage.pipeline_v2 import CognitiveOrchestrationPipeline``
  - ``from sage.pipeline_v2 import Pipeline``  (alias)
  - ``from sage.pipeline_v2 import PipelineContext``

The Phase 2.1 acceptance contract (cgpro round-4 amended) requires
all three to remain stable references to the same class objects
that are imported from ``sage.pipeline``.
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

    cgpro Phase 2.1 round-4 critical garde-fou: deferring the
    `from sage.pipeline import ...` to attribute-access time avoids
    a circular import once Step E1 lands. Module-level dunder
    `__getattr__` is the canonical PEP 562 way to expose attributes
    on demand without forcing the dependency at module load.
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
