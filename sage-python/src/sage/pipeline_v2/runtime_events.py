"""Cycle-13 K Phase 2.1 — runtime-event helper module.

cgpro DESIGN_LOCKED 2026-05-06 (`cgpro_phase21_facade_rewrite_20260506`)
gradually promotes this module from the previous "intentionally empty"
placeholder to its real home for the EventBus emission helpers and the
~12 `_runtime_emit_*` builders defined on `CognitiveOrchestrationPipeline`.

Step A3 (this commit, ~17 LOC): the basic `_emit` helper. The 11
remaining `_runtime_emit_*` helpers (topology_selected, model_assigned,
edge/node summaries, graph_digest, final_status, final_node_count, ...)
will land in Step B4 via codex IMPLEMENT. Each module function takes
the host `pipeline` instance as the first positional argument.

Method form is preserved on `CognitiveOrchestrationPipeline` as
1-line LOCAL-import delegators so:

  - existing call sites in `pipeline_v2/execute.py:{153,162,346}`
    invoking `self._emit(stage, data)` continue working unchanged
  - call sites internal to `pipeline.py` (`self._emit("CLASSIFY", ...)`
    etc.) continue working byte-identical
  - `AgentLoop._emit`, `TopologyController._emit`, `ToolForge._emit`
    are unrelated methods on different classes and are NOT touched
    by this module — only `CognitiveOrchestrationPipeline._emit`
    is delegated here

Logger uses ``sage.pipeline`` per cgpro Q7 trap "logger name drift" —
modules carved out of `pipeline.py` keep the legacy logger name so
trace-grep continuity is preserved across the refactor.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline


log = logging.getLogger("sage.pipeline")


def emit(
    pipeline: "CognitiveOrchestrationPipeline",
    stage: str,
    data: dict[str, Any],
) -> None:
    """Emit a PIPELINE-tagged AgentEvent on the pipeline's EventBus, if available.

    No-op when the pipeline has no event_bus or when AgentEvent
    construction fails (defensive ImportError / RuntimeError).
    """
    event_bus = pipeline.event_bus
    if event_bus and hasattr(event_bus, "emit"):
        try:
            from sage.agent_loop import AgentEvent

            event_bus.emit(
                AgentEvent(
                    type="PIPELINE",
                    step=0,
                    timestamp=time.time(),
                    meta={"stage": stage, **data},
                )
            )
        except (ImportError, RuntimeError):
            pass


__all__ = ["emit"]
