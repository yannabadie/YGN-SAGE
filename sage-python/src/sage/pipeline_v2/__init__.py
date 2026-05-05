"""Cycle-12 Phase A — additive wrapper package for pipeline.py decomposition.

Per ADR-015 + ADR-016 + cgpro 2026-05-05 DESIGN lock
(`cgpro_pi_mono_pivot_20260505`):

  - Phase A (this commit chain): create `pipeline_v2/` modules that
    expose the 6 stage functions as standalone callables. Each module
    function takes `(pipeline, ctx)` (or with extra kwargs for execute)
    and the body delegates to the legacy `pipeline._stage_<X>` method.
  - Phase B (subsequent commits): move the BODIES into the modules.
    Each `pipeline._stage_<X>` becomes a 1-line delegator that
    LOCAL-IMPORTS the new module function. NO top-level import of
    `pipeline_v2` from `pipeline.py` (circular).
  - Phase C (later session, NOT this one per cgpro): real façade
    rewrite that retires the `_stage_*` stubs entirely. Requires
    refactoring `pipeline.run()` callsites away from `self._stage_*`.

What's in this package today (Phase A):

  classify.py           — wraps Stage 0
  decompose.py          — wraps Stage 1
  select_topology.py    — wraps Stage 2
  assign_models.py      — wraps Stage 3
  execute.py            — wraps Stage 4
  learn.py              — wraps Stage 5
  context.py            — re-exports `PipelineContext` (cgpro: do NOT
                          move the dataclass; re-export only).
  runtime_events.py     — placeholder; helpers stay in pipeline.py for
                          now (cgpro: move stage bodies first; helper
                          ownership later).
  bandit_attribution.py — placeholder; lifecycle helpers stay in
                          pipeline.py for now (same reason).

What this package is NOT (yet):

  - Not a façade. `Pipeline = CognitiveOrchestrationPipeline` is a
    re-export alias for naming consistency, NOT a wrapper class.
  - Not a circular-import risk. Modules MUST NOT import
    `sage.pipeline_v2` at top level when called from inside
    `sage.pipeline`. Use local imports (`from sage.pipeline_v2.<x>
    import <fn>`) inside delegator methods.
  - Not a behavior change. 25 P9 phase 1 acceptance-gate tests
    pass byte-identically through the entire decomposition.
"""
from __future__ import annotations

# Re-exports: no module logic. Importing this module MUST NOT cause
# `sage.pipeline` to be loaded transitively in a way that creates a
# cycle. The line below is safe because `sage.pipeline` does not
# import `sage.pipeline_v2` at module scope.
from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Naming alias: lets future code write `from sage.pipeline_v2 import
# Pipeline` once the façade rewrite (Phase C) lands. For Phase A/B,
# this is just a name.
Pipeline = CognitiveOrchestrationPipeline

__all__ = [
    "CognitiveOrchestrationPipeline",
    "Pipeline",
    "PipelineContext",
]
