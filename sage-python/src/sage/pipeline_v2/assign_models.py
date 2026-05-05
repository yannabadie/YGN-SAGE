"""Phase A wrapper for Stage 3 (ASSIGN_MODELS).

See pipeline_v2/__init__.py module docstring for the Phase A/B/C
plan. The legacy method is sync.

Phase B note (cgpro 2026-05-05 DESIGN lock): when the body moves
in here, leave `_load_model_catalog` in `sage.pipeline` — it uses
`Path(__file__).parent.parent.parent / "config" / "cards.toml"`
which depends on the module's filesystem location. Moving it
breaks the path resolution silently.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def assign_models(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> "PipelineContext":
    """Delegate to legacy `pipeline._stage_assign_models`. Phase A."""
    return pipeline._stage_assign_models(ctx)


__all__ = ["assign_models"]
