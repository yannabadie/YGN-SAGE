"""Phase A wrapper for Stage 2 (SELECT_TOPOLOGY).

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. The legacy method is sync.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def select_topology(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Delegate to legacy `pipeline._stage_select_topology`. Phase A."""
    return pipeline._stage_select_topology(ctx)


__all__ = ["select_topology"]
