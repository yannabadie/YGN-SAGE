"""Memory gate + budget-exceeded emit helpers.

Three module functions, all taking the host `pipeline` instance as the
first positional argument:

  - `build_write_gate(pipeline)` — constructs the per-task
    `CompositeWriteGate` (or returns None on failure under fail-open
    governance; re-raises under `SAGE_STRICT_GOVERNANCE=1` per A0b
    2026-04-23, ALIRE2 §6).
  - `record_to_memory(pipeline, ctx, *, is_training_evidence=None)`
    — TypeError-fallback aware episodic-memory writes.
  - `emit_budget_exceeded(pipeline, ctx)` — emits
    `EXECUTE_BUDGET_EXCEEDED` through `pipeline._emit(...)` so the
    EventBus seam (Q3a) stays authoritative.

Logger uses ``sage.pipeline`` so trace-grep continuity is preserved
across the refactor.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sage.events import EXECUTE_BUDGET_EXCEEDED

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


log = logging.getLogger("sage.pipeline")


def emit_budget_exceeded(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
) -> None:
    """Emit `EXECUTE_BUDGET_EXCEEDED` carrying any cost-tracker stats.

    Defensive: a `cost_tracker.stats()` exception is swallowed so a
    broken telemetry path never masks the higher-priority abort.
    """
    cost_tracker = getattr(ctx, "cost_tracker", None)
    data: dict[str, Any] = {"reason": "task budget exceeded"}
    if cost_tracker is not None and hasattr(cost_tracker, "stats"):
        try:
            data.update(cost_tracker.stats())
        except Exception:  # noqa: BLE001 - telemetry must not mask abort
            pass
    pipeline._emit(EXECUTE_BUDGET_EXCEEDED, data)


def build_write_gate(pipeline: "CognitiveOrchestrationPipeline") -> Any:
    """Construct a fresh CompositeWriteGate (Rust if available, Python fallback).

    Governance (A0b, 2026-04-23, ALIRE2 §6): in normal mode a failure
    here logs-and-returns-None (memory writes continue ungated — the
    pre-A0b default that keeps dev smoke runs resilient to a broken
    Rust build). Under ``SAGE_STRICT_GOVERNANCE=1`` the failure is
    re-raised so the caller aborts the pipeline — the posture we
    want in production / audit runs where "continuing ungated"
    silently falsifies the governance claim.
    """
    from sage.memory.write_gate import create_composite_write_gate
    from sage.pipeline import _is_strict_governance

    try:
        return create_composite_write_gate(**pipeline._gate_config)
    except Exception as exc:
        if _is_strict_governance():
            log.error(
                "CompositeWriteGate init failed under SAGE_STRICT_GOVERNANCE=1; "
                "aborting pipeline: %s", exc,
            )
            raise
        log.debug("CompositeWriteGate init failed, memory writes ungated: %s", exc)
        return None


def record_to_memory(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    *,
    is_training_evidence: bool | None = None,
) -> None:
    """Write execution trace to Tier 0 (working memory) and Tier 1 (episodic).

    This closes the gap where the pipeline bypassed memory entirely.
    The consolidator (Stage 5) then migrates episodic→semantic→causal.
    """
    # Tier 0: Working memory S-MMU events
    if pipeline.working_memory:
        try:
            pipeline.working_memory.add_event("TASK", ctx.task[:500])
            pipeline.working_memory.add_event(
                "TOPOLOGY",
                f"system=S{ctx.system}, nodes={ctx.topology.node_count() if ctx.topology and hasattr(ctx.topology, 'node_count') else 0}, "
                f"assignments={ctx.assignments}",
            )
            pipeline.working_memory.add_event(
                "RESULT",
                ctx.result[:500] if ctx.result else "empty",
            )
            pipeline.working_memory.add_event(
                "METRICS",
                f"latency={ctx.latency_ms:.0f}ms, cost={ctx.cost:.4f}, path={getattr(pipeline, '_last_path', 'pipeline')}",
            )
            # Compact to Arrow chunk for S-MMU graph storage
            # In-run observability: log the STM → Arrow tier transition so
            # smoke runs can track S-MMU paging frequency per task. The 4
            # events added above (TASK/TOPOLOGY/RESULT/METRICS) guarantee
            # the threshold is met on every pipeline.run().
            events_before = pipeline.working_memory.event_count()
            if events_before >= 4:
                pipeline.working_memory.compact_to_arrow()
                log.info(
                    "memory.smmu.tier_transition from=stm to=arrow "
                    "events=%d task_id=%s",
                    events_before, pipeline._task_count,
                )
        except (RuntimeError, IOError) as exc:
            log.debug("Memory write (Tier 0) failed: %s", exc)

    # Tier 1: Episodic memory (persistent SQLite)
    if pipeline.episodic_memory:
        try:
            entry: dict[str, Any] = {
                "task": ctx.task[:200],
                "system": ctx.system,
                "topology_nodes": ctx.topology.node_count() if ctx.topology and hasattr(ctx.topology, 'node_count') else 0,
                "assignments": str(ctx.assignments),
                "result_len": len(ctx.result) if ctx.result else 0,
                "latency_ms": ctx.latency_ms,
                "cost": ctx.cost,
            }
            metadata: dict[str, Any] = {}
            if is_training_evidence is not None:
                entry["is_training_evidence"] = is_training_evidence
                metadata["is_training_evidence"] = is_training_evidence
                if ctx.oracle_verdict is not None:
                    metadata["oracle_verdict"] = ctx.oracle_verdict.to_dict()
            content = json.dumps(entry, default=str)
            # Use sync add if available, else async
            if hasattr(pipeline.episodic_memory, 'add'):
                try:
                    if metadata:
                        pipeline.episodic_memory.add(
                            key=f"pipeline-{pipeline._task_count}",
                            content=content,
                            metadata=metadata,
                        )
                    else:
                        pipeline.episodic_memory.add(
                            key=f"pipeline-{pipeline._task_count}",
                            content=content,
                        )
                except TypeError:
                    pipeline.episodic_memory.add(
                        key=f"pipeline-{pipeline._task_count}",
                        content=content,
                    )
            elif hasattr(pipeline.episodic_memory, 'add_episode'):
                try:
                    if metadata:
                        pipeline.episodic_memory.add_episode(
                            key=f"pipeline-{pipeline._task_count}",
                            content=content,
                            metadata=metadata,
                        )
                    else:
                        pipeline.episodic_memory.add_episode(
                            key=f"pipeline-{pipeline._task_count}",
                            content=content,
                        )
                except TypeError:
                    pipeline.episodic_memory.add_episode(
                        key=f"pipeline-{pipeline._task_count}",
                        content=content,
                    )
        except (RuntimeError, IOError) as exc:
            log.debug("Memory write (Tier 1) failed: %s", exc)


__all__ = ["build_write_gate", "emit_budget_exceeded", "record_to_memory"]
