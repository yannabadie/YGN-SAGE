"""Phase A re-export of `PipelineContext`. Source of truth stays in `sage.pipeline`.

Per cgpro 2026-05-05 DESIGN lock: do NOT move the dataclass. Moving
changes its `__module__`, repr, and pickle behavior — bench / dashboards
/ observability consumers may compare against the literal module path.
For Phase A and Phase B, `pipeline_v2/context.py` re-exports only.
A future Phase C session may decide to move it; that's a separate ADR.
"""
from __future__ import annotations

from sage.pipeline import PipelineContext

__all__ = ["PipelineContext"]
