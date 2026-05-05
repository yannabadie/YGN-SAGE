"""Phase A wrapper for Stage 0 (CLASSIFY).

Phase A (this commit): function delegates to the legacy
`pipeline._stage_classify` method. No body movement yet.
Phase B (subsequent commit): the body moves into this file and
`pipeline._stage_classify` becomes a 1-line delegator that local-
imports `classify` from here.

Contract: the function returns the same `PipelineContext` the
legacy method returns, byte-identically. The 25 P9 phase 1
characterization tests are the verification gate — they MUST pass
both through `pipeline._stage_classify(ctx)` AND through
`classify(pipeline, ctx)` in Phase A and Phase B.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def classify(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Delegate to legacy `pipeline._stage_classify`. Phase A."""
    return pipeline._stage_classify(ctx)


__all__ = ["classify"]
