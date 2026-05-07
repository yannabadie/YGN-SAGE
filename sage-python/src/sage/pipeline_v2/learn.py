"""Stage 5 - LEARN.

Per ADR-015 + cgpro 2026-05-05 DESIGN lock, cycle-12 Phase B
moves the body of `CognitiveOrchestrationPipeline._stage_learn`
here as a module-level function. Legacy method becomes a 1-line
LOCAL-import async delegator.

Helpers stay on `CognitiveOrchestrationPipeline`; the body accesses
them via `self.<helper>(...)` after the `self = pipeline` shim.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sage.runtime.oracle import oracle_enabled

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

# Per-module logger MUST stay "sage.pipeline" for trace-grep
# compatibility with the cycle-9 ledger consumers.
log = logging.getLogger("sage.pipeline")


async def learn(
    pipeline: "CognitiveOrchestrationPipeline", ctx: "PipelineContext",
) -> None:
    """Stage 5: Record outcome for learning.

    Quality signal for bandit feedback. Research background: ETH-SRI's
    Cascade Routing (arxiv 2410.10347, ICLR 2025) established that
    quality estimation — not the routing algorithm — is the bottleneck
    in LLM-routing bandits. A "PILOT 2508.21141" citation previously
    sat here but that arxiv ID is dead (see 2026-04-22 audit
    verification D.1); removed rather than kept as a ghost reference.

    - Empty result: quality = 0.0 (definitively bad, bandit learns from it)
    - QualityEstimator returns float: use it
    - QualityEstimator returns None: abstain — bandit does NOT record
    - No estimator: abstain — bandit does NOT record
    """
    self = pipeline
    from sage.observability.spans import sage_span
    with sage_span("sage.learn", op="sage.learn"):
        import re

        quality: float | None = None
        oracle_on = oracle_enabled()
        oracle_trainable = False

        if oracle_on:
            verdict = getattr(ctx, "oracle_verdict", None)
            if verdict is not None and verdict.trainable:
                quality = verdict.score
                oracle_trainable = quality is not None
        # Empty result => total failure, bandit must learn from it
        elif not ctx.result or not ctx.result.strip():
            quality = 0.0
        elif self.quality_estimator:
            try:
                quality = self.quality_estimator.estimate(
                    ctx.task, ctx.result, ctx.latency_ms
                )
            except (ImportError, RuntimeError):
                quality = None  # cannot assess — abstain

        # PRM lightweight scoring (Phase C) — 6th formal signal
        # Guard: only call PRM on structured content (<think>, assert, code)
        # Only blend when quality is known (not None)
        _STRUCTURED = re.compile(r'<think>|```|assert\s|def\s+test_', re.IGNORECASE)
        if (
            not oracle_on
            and self.prm
            and quality is not None
            and ctx.result
            and _STRUCTURED.search(ctx.result)
        ):
            try:
                r_path, _ = self.prm.calculate_r_path(ctx.result)
                if r_path >= 0.0:  # valid score (negative = penalty for no reasoning)
                    quality = 0.8 * quality + 0.2 * r_path
                    log.debug("PRM blended quality: %.2f (estimator + PRM)", quality)
            except (RuntimeError, ValueError) as exc:
                log.warning("PRM scoring failed in LEARN: %s", exc)

        # Only record to bandit when quality is known and attribution is causal.
        from sage.pipeline_v2 import bandit_attribution as _bandit_attr
        if quality is not None:
            _bandit_attr.record_bandit_outcome_checked(self, ctx, quality)
        else:
            _bandit_attr.cancel_bandit_decision(self, ctx)
            _bandit_attr.clear_bandit_decision(self, ctx)

        # Evolution feedback: record outcome in TopologyEngine archive
        # Feeds MAP-Elites + CMA-ME + S-MMU bridge for future topology selection
        if self.engine and quality is not None and ctx.topology is not None:
            try:
                topology_id = ctx.topology_id or getattr(ctx.topology, 'id', '')
                if topology_id and hasattr(self.engine, 'record_outcome'):
                    keywords = list(set(
                        w.lower() for w in re.findall(r'\b\w{4,}\b', ctx.task)
                    ))[:10]
                    # Compute real task embedding for S-MMU retrieval
                    task_embedding = None
                    try:
                        from sage.memory.embedder import Embedder
                        _embedder = Embedder()
                        if _embedder.is_semantic:
                            task_embedding = _embedder.embed(ctx.task[:500])
                    except (ImportError, RuntimeError):
                        pass  # Embedding unavailable, degrade gracefully

                    # In-run observability: read archive cell count before +
                    # after record_outcome so we can attribute MAP-Elites growth
                    # to a specific topology_id. The H4 incident (dc51976)
                    # proved this delta is frequently 0 even when record_outcome
                    # returns cleanly — the log would have caught that bypass
                    # immediately, so we keep it structured going forward.
                    cells_before = 0
                    if hasattr(self.engine, 'archive_cell_count'):
                        try:
                            cells_before = int(self.engine.archive_cell_count())
                        except (RuntimeError, TypeError):
                            cells_before = 0
                    self.engine.record_outcome(
                        topology_id,
                        ctx.task[:200],
                        keywords,
                        task_embedding,  # real embedding instead of None
                        quality,
                        ctx.cost,
                        ctx.latency_ms,
                    )
                    cells_after = cells_before
                    if hasattr(self.engine, 'archive_cell_count'):
                        try:
                            cells_after = int(self.engine.archive_cell_count())
                        except (RuntimeError, TypeError):
                            cells_after = cells_before
                    if cells_after > cells_before:
                        # Get descriptor shape for the new cell when available
                        coverage = 0.0
                        if hasattr(self.engine, 'archive_coverage'):
                            try:
                                coverage = float(self.engine.archive_coverage())
                            except (RuntimeError, TypeError):
                                coverage = 0.0
                        log.info(
                            "memory.archive.grow cells=%d delta=%d "
                            "coverage=%.3f topology=%s quality=%.2f",
                            cells_after, cells_after - cells_before,
                            coverage, topology_id[:8] if topology_id else "unknown",
                            quality,
                        )
                    log.debug(
                        "Evolution: recorded outcome for topology %s (quality=%.2f)",
                        topology_id[:8], quality,
                    )
            except (ImportError, RuntimeError) as exc:
                log.debug("Evolution feedback failed: %s", exc)

        # H1 audit fix (2026-04-19): Online evolution gate.
        #
        # Architecture/architecture.md and evolution/README.md both claim
        # "Online evolution: Rust should_evolve() gates evolve() in agent
        # loop (SA-3 complete)". The Rust impl exists at
        # `sage-core/src/topology/engine.rs:644 (should_evolve)` and
        # `:668 (evolve)`, both exposed via PyO3, but no Python call site
        # invoked them — same class of bypass as the G-series write-gate
        # (built, tested in isolation, never wired). `_auto_evolve=True`
        # is set on AgentLoop in boot.py:332 but no code reads that flag.
        #
        # Wiring at the end of LEARN is the correct hook: by here we've
        # just appended an outcome to the archive, so should_evolve()
        # has fresh data. Constants come from sage.constants — single
        # source of truth, already covered by test_online_evolution.py.
        allow_training_updates = (not oracle_on) or oracle_trainable
        if allow_training_updates and self.engine and hasattr(self.engine, "should_evolve"):
            try:
                from sage.constants import (
                    EVOLUTION_MIN_OUTCOMES,
                    EVOLUTION_COOLDOWN_OUTCOMES,
                    EVOLUTION_ONLINE_POP_SIZE,
                    EVOLUTION_ONLINE_GENERATIONS,
                )
                decision = self.engine.should_evolve(
                    EVOLUTION_MIN_OUTCOMES, EVOLUTION_COOLDOWN_OUTCOMES,
                )
                # In-run observability: log every should_evolve() call, not
                # just the True branch. Without this, False branches are
                # invisible -- we can't tell if the gate was even evaluated
                # on a given task, which is exactly the class of silent-bypass
                # that H1 (2cd840e) and H4 (dc51976) were. Archive context
                # travels with the log so post-run analysis can correlate.
                cells = 0
                coverage = 0.0
                if hasattr(self.engine, "archive_cell_count"):
                    try:
                        cells = int(self.engine.archive_cell_count())
                    except (RuntimeError, TypeError):
                        cells = 0
                if hasattr(self.engine, "archive_coverage"):
                    try:
                        coverage = float(self.engine.archive_coverage())
                    except (RuntimeError, TypeError):
                        coverage = 0.0
                log.info(
                    "evolution.should_evolve.decision decision=%s cells=%d "
                    "coverage=%.3f min_outcomes=%d cooldown=%d",
                    "true" if decision else "false", cells, coverage,
                    EVOLUTION_MIN_OUTCOMES, EVOLUTION_COOLDOWN_OUTCOMES,
                )
                if decision:
                    # Read cells before evolve to show the effect of the call
                    cells_pre_evolve = cells
                    self.engine.evolve(
                        pop_size=EVOLUTION_ONLINE_POP_SIZE,
                        generations=EVOLUTION_ONLINE_GENERATIONS,
                    )
                    cells_post_evolve = cells_pre_evolve
                    if hasattr(self.engine, "archive_cell_count"):
                        try:
                            cells_post_evolve = int(self.engine.archive_cell_count())
                        except (RuntimeError, TypeError):
                            cells_post_evolve = cells_pre_evolve
                    log.info(
                        "evolution.evolve.called pop_size=%d generations=%d "
                        "cells_before=%d cells_after=%d",
                        EVOLUTION_ONLINE_POP_SIZE, EVOLUTION_ONLINE_GENERATIONS,
                        cells_pre_evolve, cells_post_evolve,
                    )
                    # Per-child operator logs. Rust's TopologyEngine tracks
                    # the Thompson-sampled operator name for every mutation
                    # attempt inside evolve() (see pyo3_wrappers.rs
                    # PyTopologyEngine.drain_last_applied_ops). We drain the
                    # buffer here and emit one `evolution.mutation.applied`
                    # line per attempt. Closes gap #3 from the 0bcb92b
                    # pillar-logging pass (per-mutation-operator observability).
                    #
                    # tier: best-effort — the pipeline's most recent routing
                    # decision. evolve() mutations aren't bound to a specific
                    # routing tier (they operate on cached topologies), so we
                    # emit the current-task tier purely for correlation in
                    # post-run analysis.
                    if hasattr(self.engine, "drain_last_applied_ops"):
                        try:
                            applied_ops = list(self.engine.drain_last_applied_ops())
                        except (RuntimeError, TypeError):
                            applied_ops = []
                        if applied_ops:
                            import hashlib as _hashlib
                            tier = ""
                            rd = getattr(self, "_last_routing_decision", None)
                            if rd is not None:
                                tier = getattr(rd, "llm_tier", "") or ""
                            topo_id = ctx.topology_id or getattr(
                                ctx.topology, "id", "",
                            )
                            parent_cell = cells_pre_evolve
                            for child_idx, op_name in enumerate(applied_ops):
                                # child_hash: stable short hash mixing the
                                # topology id, child index, and op name.
                                # Opaque but reproducible per-run identifier.
                                h_src = f"{topo_id}:{child_idx}:{op_name}".encode(
                                    "utf-8",
                                )
                                child_hash = _hashlib.blake2b(
                                    h_src, digest_size=4,
                                ).hexdigest()
                                log.info(
                                    "evolution.mutation.applied op=%s "
                                    "parent_cell=%d child_hash=%s tier=%s",
                                    op_name, parent_cell, child_hash,
                                    tier or "unknown",
                                )
            except (ImportError, RuntimeError, AttributeError) as exc:
                # Engine without these methods (e.g. test stub) → silent skip.
                log.debug("Online evolution gate skipped: %s", exc)

        # ── Periodic maintenance ───────────────────────────────────────────
        self._task_count += 1

        # Inter-tier consolidation: episodic → semantic → causal (MAGMA 2601.03236)
        from sage.constants import CONSOLIDATION_INTERVAL_STEPS
        if (allow_training_updates
                and self._task_count % CONSOLIDATION_INTERVAL_STEPS == 0
                and self.consolidator is not None):
            try:
                consolidation_result = await self.consolidator.consolidate()
                # In-run observability: emit a structured log regardless of
                # whether any entries were processed, so smoke runs can see
                # consolidation ran on the scheduled interval. The old DEBUG
                # log was silent for zero-processed passes — now every firing
                # is accounted for.
                processed = getattr(consolidation_result, 'processed', 0)
                entities = getattr(consolidation_result, 'entities_added', 0)
                edges = getattr(consolidation_result, 'causal_edges_added', 0)
                log.info(
                    "memory.consolidation.fired processed=%d entities=%d "
                    "causal_edges=%d task_id=%d",
                    processed, entities, edges, self._task_count,
                )
            except (RuntimeError, IOError):
                pass  # Best-effort, never blocks pipeline

        # Bandit + MAP-Elites state persistence (crash-safe, WAL write ~5ms)
        #
        # Cycle-11 follow-up (2026-05-05, advisor-flagged 2026-05-04):
        # symmetric with the atexit handler at boot_topology.py:185.
        # Both call sites preflight epoch consistency before writing
        # state so directive #8 (A14 posterior epoch guard is fail-
        # closed) holds across the run lifecycle, not just at session
        # boundaries. Without this preflight, the periodic flush
        # would (a) keep writing state under SAGE_BOOT_BYPASS_EPOCH_GUARD=1
        # while atexit correctly skips, (b) overwrite a contaminated
        # marker and erase forensic evidence, (c) deepen a manifest-
        # less state dir making the next boot fail-close.
        from sage.constants import BANDIT_FLUSH_INTERVAL
        if (self._task_count % BANDIT_FLUSH_INTERVAL == 0
                and self.engine and hasattr(self.engine, 'save_state')):
            try:
                from pathlib import Path
                from sage.posterior_epoch import (
                    ensure_clean_epoch_before_save,
                    is_a14_epoch_guard_error,
                )
                state_dir = Path.home() / ".sage"
                # Preflight: matches boot_topology.py:185 atexit
                # handler ordering. Raises on bypass active /
                # contaminated marker / missing manifest / mismatch.
                ensure_clean_epoch_before_save(state_dir)
                self.engine.save_state(str(state_dir))
                log.debug("Periodic state flush (%d tasks)", self._task_count)
            except (RuntimeError, IOError) as exc:
                if is_a14_epoch_guard_error(exc):
                    # Visible warning: A14 guard fired. Operator
                    # needs this signal to know periodic saves are
                    # skipping. Atexit logs at .info; periodic at
                    # .warning since it can recur many times in a
                    # session and ops should investigate sooner.
                    log.warning(
                        "Periodic state flush blocked by A14 epoch "
                        "guard: %s", exc,
                    )
                # else: silent best-effort behaviour preserved
                # (e.g. transient I/O errors, manifest verification
                # failures, malformed state on disk).


__all__ = ["learn"]
