"""Phase A wrapper for Stage 1 (DECOMPOSE).

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. The legacy method is async; the wrapper preserves that.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


async def decompose(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Delegate to legacy `pipeline._stage_decompose`. Phase A."""
    return await pipeline._stage_decompose(ctx)


__all__ = ["decompose"]
