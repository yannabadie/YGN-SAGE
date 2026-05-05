"""Phase A wrapper for Stage 5 (LEARN).

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. The legacy method is async and returns None.

Phase B note: Stage 5 is the gate for invariants 2 (oracle evidence)
and 6 (bandit attribution). The 25 P9 phase 1 tests
(`test_pipeline_v2_oracle_gate_invariant.py` +
`test_pipeline_v2_bandit_attribution_invariant.py`) are the
verification gate when the body moves.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


async def learn(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> None:
    """Delegate to legacy `pipeline._stage_learn`. Phase A."""
    return await pipeline._stage_learn(ctx)


__all__ = ["learn"]
