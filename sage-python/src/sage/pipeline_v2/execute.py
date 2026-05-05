"""Phase A wrapper for Stage 4 (EXECUTE).

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. The legacy method is async and takes optional event_log +
run_frame_builder kwargs.

Phase B note (cgpro 2026-05-05 DESIGN lock): execute is the LAST
stage to move (largest blast radius — bypass mutation, FrugalGPT
cascade, multi-agent error fallback, controller decisions). P6-A
Phase B (AgentLoop bypass factory swap) is a SEPARATE commit that
lands BEFORE the execute body move, so the move itself stays
mechanical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


async def execute(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    event_log: Any | None = None,
    run_frame_builder: Any | None = None,
) -> "PipelineContext":
    """Delegate to legacy `pipeline._stage_execute`. Phase A."""
    return await pipeline._stage_execute(
        ctx, event_log=event_log, run_frame_builder=run_frame_builder,
    )


__all__ = ["execute"]
