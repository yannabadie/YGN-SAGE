"""Orchestrator — `run_internal` body home.

The `run_internal(pipeline, task, ...)` async function is the
canonical orchestrator entry point; the thin
`CognitiveOrchestrationPipeline._run_internal` method is a 1-line
wrapper that calls it (keeps subclass-override surfaces intact for
the few benchmark adapters that override the private method).

Critical garde-fou: existing tests monkeypatch
`sage.pipeline._new_runtime_run_id` and `sage.pipeline.time.monotonic`
(see `test_run_frame.py`, `test_oracle_stack.py`). The orchestrator
MUST NOT take the naive top-level `from sage.pipeline import
_new_runtime_run_id` shortcut — that would cache the bound symbol at
import time and defeat the monkeypatches. Instead we resolve the
names DYNAMICALLY through `from sage import pipeline as pipeline_mod`
at function-call time so each call walks the attribute lookup that
the test fixture intercepts.

Pattern:

    async def run_internal(pipeline, task, ...):
        from sage import pipeline as pipeline_mod
        from sage.pipeline_v2.classify import classify
        # ... other five stage module imports ...
        # ALL references to legacy module-level symbols via pipeline_mod
        # for monkeypatch correctness:
        run_id = pipeline_mod._new_runtime_run_id()
        budget = pipeline_mod._resolve_task_budget_usd(...)
        ctx = pipeline_mod.PipelineContext(...)
        t0 = pipeline_mod.time.monotonic()
        # Stage entry points are module functions called with the pipeline
        # instance as first argument:
        ctx = classify(pipeline, ctx)
        ...

Logger uses ``sage.pipeline`` so trace-grep continuity is preserved
across the refactor.
"""
from __future__ import annotations

import inspect as _inspect
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline
    from sage.runtime.run_frame import RunFrame


log = logging.getLogger("sage.pipeline")


def _set_cli_progress_stage(pipeline: Any, stage: str) -> None:
    """Update ``pipeline._cli_progress_state.stage`` if a CLI is attached.

    Stage C of cycle-13 K post-Phase-2.2 (cgpro
    `cgpro_cli_protocol_gaps_20260507`). The CLI driver in
    ``sage.cli.run`` attaches a ``_CliProgressState`` to the pipeline at
    runtime via ``setattr`` so the heartbeat can report the current
    high-level stage. Runs without a CLI never see this attribute and
    the helper is a no-op.
    """
    state = getattr(pipeline, "_cli_progress_state", None)
    if state is not None:
        state.stage = stage


async def run_internal(
    pipeline: "CognitiveOrchestrationPipeline",
    task: str,
    budget_usd: float | None = None,
    system_hint: int | None = None,
    *,
    emit_run_frame_summary: bool = False,
    bench_evaluator: Any = None,
) -> "tuple[str, RunFrame]":
    """Execute the full pipeline (classify → decompose → select_topology → assign_models → execute → learn).

      - References to module-level symbols (`_new_runtime_run_id`,
        `_resolve_task_budget_usd`, `PipelineContext`, `time.monotonic`)
        go through `pipeline_mod.<X>` so test monkeypatches on
        `sage.pipeline.*` keep firing.
      - The stage entry points are module functions in
        `sage.pipeline_v2.<stage>`, locally imported and called with
        the pipeline instance as first argument.
      - Order final_result -> oracle_verdict -> learn -> run_frame_summary
        is preserved byte-identical.

    Args:
        pipeline: The CognitiveOrchestrationPipeline instance hosting
            this run.
        task: The user's task.
        budget_usd: Task-level spend cap for the run. ``None`` uses the
            constructor/env value; ``0`` means unlimited.
        system_hint: Optional override for Stage 0 routing (1, 2, or 3).
            Benchmark adapters use this when they already know the task
            complexity (e.g. SWE-bench tasks are always S3). When set,
            the Rust SystemRouter still runs (so we keep the model
            assignment + bandit posteriors), but `ctx.system` is forced
            to the hint afterwards.
    """
    # cgpro round-4 critical garde-fou: dynamic lookup via the live
    # `sage.pipeline` module so test monkeypatches on
    # `sage.pipeline._new_runtime_run_id` and `sage.pipeline.time.monotonic`
    # keep firing. Top-level `from sage.pipeline import _new_runtime_run_id`
    # would freeze the symbol at import time and defeat the patch.
    from sage import pipeline as pipeline_mod
    from sage.contracts.cost_tracker import CostTracker
    from sage.observability.spans import sage_span
    from sage.pipeline_v2 import memory_gate as memory_gate_mod
    from sage.pipeline_v2 import runtime_events as runtime_events_mod
    from sage.pipeline_v2.assign_models import assign_models
    from sage.pipeline_v2.classify import classify
    from sage.pipeline_v2.decompose import decompose
    from sage.pipeline_v2.execute import execute
    from sage.pipeline_v2.learn import learn
    from sage.pipeline_v2.select_topology import select_topology
    from sage.runtime.event_log import (
        EventLogUnavailable,
        RuntimeEventLog,
        current_event_log,
        install_event_log,
    )
    from sage.runtime.event_log.redaction import _hash_text
    from sage.runtime.oracle import EvidenceRef, OracleVerdict, oracle_enabled
    from sage.runtime.run_frame.builder import _RunFrameBuilder

    # Cycle-13 E Tier 2.1 smoke discovery 2026-05-05: when called via
    # `sage run --jsonl` (cycle-12 prelude `d09bed4d`), the CLI installs
    # its own RuntimeEventLog with a stdout-mirror tee BEFORE calling
    # pipeline.run(). The previous unconditional construction here
    # shadowed the CLI's eventlog with a fresh-disabled one (no
    # trace_dir kwarg + SAGE_TRACE_JSONL_DIR env unset =>
    # writer.py:162 sets disabled=True, all emit_* become no-ops),
    # so no runtime events ever reached the CLI's stdout. Prefer
    # the externally-installed eventlog when present; fall back to
    # creating a fresh one for direct-Python callers (the
    # historical default).
    event_log = current_event_log()
    if event_log is None:
        event_log = RuntimeEventLog(run_id=pipeline_mod._new_runtime_run_id())
    run_frame_builder = _RunFrameBuilder(
        run_id=event_log.run_id,
        task_id=event_log.run_id,
        task_hash=_hash_text(task),
    )
    run_frame_builder.capture_feature_flags()
    token = install_event_log(event_log)
    final_emitted = False
    ctx: "pipeline_mod.PipelineContext | None" = None
    t0 = pipeline_mod.time.monotonic()
    _span_attrs: dict[str, Any] = {"gen_ai.request.model": ""}
    try:
        with sage_span("sage.pipeline.run", op="invoke_agent", **_span_attrs):
            effective_budget_usd = (
                pipeline.budget_usd
                if budget_usd is None
                else pipeline_mod._resolve_task_budget_usd(budget_usd)
            )
            ctx = pipeline_mod.PipelineContext(task=task, budget=effective_budget_usd)
            # Cost tracker is ALWAYS created so the CLI ``set_budget`` command
            # can tighten the budget mid-run even when the run starts unlimited
            # (cycle-13 K post-Phase-2.2 cgpro `cgpro_cli_protocol_gaps_20260507`
            # Stage B lock 2026-05-07). ``CostTracker(budget_usd=0.0)`` is the
            # unlimited sentinel: ``is_over_budget`` stays False and
            # ``remaining`` stays ``inf`` until ``tighten_remaining_budget``
            # rebases the cap. Always-on tracker is bookkeeping for unlimited
            # runs (cost.record is a no-op contract for is_over_budget).
            ctx.cost_tracker = CostTracker(budget_usd=effective_budget_usd)
            # Expose the active context so ``pipeline.tighten_budget()`` can
            # find the per-run tracker. Cleared in ``finally`` below.
            pipeline._active_context = ctx

            pipeline._last_routing_decision = None
            pipeline._last_runtime_routing_source = "default"
            pipeline._last_runtime_routing_confidence = None
            pipeline._last_runtime_routing_model_id = ""
            event_log.emit_task_started(ctx.task)

            # P1-3 (REVIEW4 2026-05-08): prompt-injection detection at
            # pipeline ingress — before classify/decompose/topology.
            # Default: log-only, emits events, does not interrupt.
            # Strict (SAGE_PROMPT_INJECTION_STRICT=1): refuses the task
            # before any orchestration begins.
            try:
                from sage.security.prompt_injection import detect as _pi_detect
                _pi_matches = _pi_detect(task)
                for _m in _pi_matches:
                    event_log.emit_prompt_injection_detected(
                        pattern_name=_m.pattern_name,
                        match_text=_m.match_text[:200],
                        span_start=_m.start,
                        span_end=_m.end,
                        severity=_m.severity,
                        parent_event_id=None,
                    )
                if _pi_matches and os.environ.get("SAGE_PROMPT_INJECTION_STRICT") == "1":
                    log.warning(
                        "Stage 0: prompt injection detected (%d patterns), "
                        "refusing under SAGE_PROMPT_INJECTION_STRICT=1",
                        len(_pi_matches),
                    )
                    ctx.result = "[sage: prompt injection detected — task refused]"
                    event_log.emit_final_result(
                        status="failure",
                        output="",
                        total_cost_usd=0.0,
                        total_latency_ms=0.0,
                        node_count=0,
                    )
                    return ctx.result, run_frame_builder.finalize()
            except ImportError:
                pass  # security module not available — non-blocking

            # G-series (2026-04-19): rebuild write gate per task so entries from a
            # previous task don't persist as novelty penalties or exact-dedup hits
            # on content in THIS task. Rust gate has no in-place reset yet.
            pipeline.write_gate = memory_gate_mod.build_write_gate(pipeline)

            # Stage 0: CLASSIFY
            _set_cli_progress_stage(pipeline, "classify")
            ctx = classify(pipeline, ctx)
            if system_hint in (1, 2, 3) and ctx.system != system_hint:
                log.info(
                    "Stage 0: system_hint=S%d overrides router S%d",
                    system_hint, ctx.system,
                )
                ctx.system = system_hint
            routing_source = getattr(pipeline, "_last_runtime_routing_source", "default")
            routing_confidence = getattr(pipeline, "_last_runtime_routing_confidence", None)
            routing_model_id = getattr(pipeline, "_last_runtime_routing_model_id", "")
            routing_seq = event_log.emit_routing_decision(
                routing_source=routing_source,
                system=ctx.system,
                domain=ctx.domain,
                confidence=routing_confidence,
                model_id=routing_model_id,
            )
            run_frame_builder.record_routing_decision(
                seq=routing_seq,
                routing_source=routing_source,
                system=ctx.system,
                domain=ctx.domain,
                confidence=routing_confidence,
                model_id=routing_model_id,
            )
            pipeline._emit("CLASSIFY", {"system": ctx.system, "domain": ctx.domain})

            # Stage 1: DECOMPOSE (S2/S3 only)
            _set_cli_progress_stage(pipeline, "decompose")
            ctx = await decompose(pipeline, ctx)
            dag_node_count = 0
            if ctx.task_dag is not None:
                if hasattr(ctx.task_dag, "node_count"):
                    dag_node_count = ctx.task_dag.node_count
                elif hasattr(ctx.task_dag, "node_ids"):
                    dag_node_count = len(list(ctx.task_dag.node_ids))
            pipeline._emit(
                "DECOMPOSE",
                {
                    "dag_nodes": dag_node_count,
                    "features": (
                        {
                            "omega": ctx.dag_features.omega,
                            "delta": ctx.dag_features.delta,
                            "gamma": ctx.dag_features.gamma,
                        }
                        if ctx.dag_features
                        else {}
                    ),
                },
            )

            # Stage 2: SELECT TOPOLOGY
            _set_cli_progress_stage(pipeline, "select_topology")
            ctx = select_topology(pipeline, ctx)
            topo_nodes = (
                ctx.topology.node_count()
                if ctx.topology and hasattr(ctx.topology, "node_count")
                else 0
            )
            runtime_events_mod.runtime_emit_topology_selected(
                pipeline,
                ctx,
                event_log,
                run_frame_builder,
                reason="initial",
            )
            pipeline._emit("SELECT_TOPOLOGY", {"node_count": topo_nodes})

            # Stage 3: ASSIGN MODELS
            _set_cli_progress_stage(pipeline, "assign_models")
            ctx = assign_models(pipeline, ctx)
            runtime_events_mod.runtime_emit_model_assigned(pipeline, ctx, event_log, run_frame_builder)
            pipeline._emit(
                "ASSIGN_MODELS", {"assignments": ctx.assignments, "domain": ctx.domain}
            )

            # Stage 4: EXECUTE
            _set_cli_progress_stage(pipeline, "execute")
            ctx = await execute(
                pipeline,
                ctx,
                event_log=event_log,
                run_frame_builder=run_frame_builder,
            )
            ctx.latency_ms = (pipeline_mod.time.monotonic() - t0) * 1000

            # cgpro 2026-04-29 R6.1a verify Path E: bench-result feedback
            # seam. Synchronous-eval benches (BigCodeBench, EvalPlus, etc.)
            # attach an evaluator via run_with_bench_evaluator(); we call
            # it on the executed output BEFORE final_result + oracle so
            # _exact_oracle has bench_result["passed"] available. Fail-
            # closed: any exception leaves ctx.bench_result=None and the
            # oracle abstains via _exact_oracle's None-guard.
            if bench_evaluator is not None and ctx.bench_result is None:
                try:
                    candidate = bench_evaluator(ctx.result or "")
                    if _inspect.isawaitable(candidate):
                        candidate = await candidate
                    if isinstance(candidate, Mapping):
                        ctx.bench_result = candidate
                    else:
                        log.warning(
                            "bench_evaluator returned %r; expected Mapping. "
                            "Oracle will abstain.",
                            type(candidate).__name__,
                        )
                except Exception as _eval_exc:  # noqa: BLE001 - fail-closed
                    log.warning(
                        "bench_evaluator raised %s: %s; oracle will abstain.",
                        type(_eval_exc).__name__,
                        _eval_exc,
                    )

            oracle_on = oracle_enabled()
            final_status = runtime_events_mod.runtime_final_status(pipeline, ctx)

            if oracle_on:
                final_seq = event_log.emit_final_result(
                    status=final_status,
                    output=ctx.result or "",
                    total_cost_usd=float(ctx.cost or 0.0),
                    total_latency_ms=ctx.latency_ms,
                    node_count=runtime_events_mod.runtime_final_node_count(pipeline, ctx),
                )
                run_frame_builder.record_final_result(
                    seq=final_seq,
                    status=final_status,
                )
                final_emitted = True
                try:
                    from sage.runtime import oracle as oracle_stack

                    verdict = oracle_stack.evaluate(
                        run_frame_builder.snapshot_view(),
                        final_output=ctx.result or "",
                        bench_result=ctx.bench_result,
                        config=pipeline._oracle_config,
                    )
                except Exception as exc:  # noqa: BLE001 - oracle must fail closed
                    log.warning("OracleStack failed; collapsing to Abstain: %s", exc)
                    verdict = OracleVerdict(
                        trainable=False,
                        verdict_source="abstain",
                        quality_label="unknown",
                        score=None,
                        confidence=1.0,
                        reason_codes=("oracle_exception", type(exc).__name__),
                        evidence=(EvidenceRef(run_id=event_log.run_id),),
                    )
                oracle_seq = event_log.emit_oracle_verdict(
                    parent_event_id=final_seq,
                    verdict=verdict,
                )
                run_frame_builder.record_oracle_verdict(
                    seq=oracle_seq,
                    verdict=verdict,
                )
                ctx.oracle_verdict = verdict

                memory_gate_mod.record_to_memory(
                    pipeline,
                    ctx,
                    is_training_evidence=verdict.trainable,
                )
                _set_cli_progress_stage(pipeline, "learn")
                await learn(pipeline, ctx)
                pipeline._emit("LEARN", {"latency_ms": ctx.latency_ms})
            else:
                # Legacy OFF mode: keep the R7 execution/learn/final order.
                memory_gate_mod.record_to_memory(pipeline, ctx)
                _set_cli_progress_stage(pipeline, "learn")
                await learn(pipeline, ctx)
                pipeline._emit("LEARN", {"latency_ms": ctx.latency_ms})

                # Expose full context before final_result, preserving R7 order.
                pipeline.last_context = ctx

                if pipeline._agent_loop is not None and ctx.cost:
                    pipeline._agent_loop.total_cost_usd = float(ctx.cost)

                final_seq = event_log.emit_final_result(
                    status=final_status,
                    output=ctx.result or "",
                    total_cost_usd=float(ctx.cost or 0.0),
                    total_latency_ms=ctx.latency_ms,
                    node_count=runtime_events_mod.runtime_final_node_count(pipeline, ctx),
                )
                run_frame_builder.record_final_result(
                    seq=final_seq,
                    status=final_status,
                )
                final_emitted = True

            if oracle_on:
                pipeline.last_context = ctx
                if pipeline._agent_loop is not None and ctx.cost:
                    pipeline._agent_loop.total_cost_usd = float(ctx.cost)

            frame = run_frame_builder.finalize()
            if emit_run_frame_summary and frame.final_result_seq is not None:
                try:
                    event_log.emit_run_frame_summary(
                        parent_event_id=frame.final_result_seq,
                        summary=frame.to_summary_dict(redacted=True),
                    )
                except (EventLogUnavailable, OSError, IOError, ValueError):
                    pass
            return ctx.result, frame
    except Exception:
        if not final_emitted:
            latency_ms = (pipeline_mod.time.monotonic() - t0) * 1000
            if ctx is not None:
                ctx.latency_ms = latency_ms
            final_seq = event_log.emit_final_result(
                status="failure",
                output=(ctx.result if ctx is not None else "") or "",
                total_cost_usd=float((ctx.cost if ctx is not None else 0.0) or 0.0),
                total_latency_ms=latency_ms,
                node_count=runtime_events_mod.runtime_final_node_count(pipeline, ctx),
            )
            run_frame_builder.record_final_result(seq=final_seq, status="failure")
        raise
    finally:
        # Clear the active-context hook so a stale ctx can't be mutated
        # after the run finishes (Stage B set_budget lock).
        pipeline._active_context = None
        token.var.reset(token)
        event_log.close()


__all__ = ["run_internal"]
