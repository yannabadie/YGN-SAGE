"""Cycle-11 follow-up: pipeline.py periodic state save epoch preflight.

Closes the asymmetry advisor-flagged 2026-05-04: the mid-pipeline
``engine.save_state(Path.home() / ".sage")`` call at
``pipeline.py:3075`` did NOT call ``ensure_clean_epoch_before_save``
the way the atexit handler at ``boot_topology.py:185`` does. This
left a hole in directive #8 (A14 posterior epoch guard is fail-
closed): under ``SAGE_BOOT_BYPASS_EPOCH_GUARD=1``, the periodic
flush kept writing state (the atexit handler correctly skipped via
``_epoch_bypass_active`` guard); under a contaminated marker, the
periodic flush would overwrite the contamination evidence; with
state files present but no manifest, the next boot would fail-
closed on the load side.

Maps to runtime-integrity-ledger.md invariant 3 (Posterior epoch).
The atexit handler binds save to "epoch=1 + manifest matches"; the
periodic flush must do the same for the same invariant to hold
across the run lifecycle, not just at session boundaries.

What this test locks
====================
After the fix, the periodic flush at ``pipeline.py:_stage_learn``
calls ``ensure_clean_epoch_before_save(state_dir)`` BEFORE
``self.engine.save_state(...)``. Symmetric with the atexit handler.

  1. Bypass active (``SAGE_BOOT_BYPASS_EPOCH_GUARD=1``) → save_state
     NOT called. Atexit-symmetric: see ``boot_topology.py:178-179``
     ``a14_epoch_guard_bypass_save_disabled`` warning path.

  2. Contaminated marker file present → save_state NOT called.
     The marker is the "operator-readable poison-pill" from
     ledger invariant 4 (Contaminated backup); writing fresh state
     on top of it would erase forensic evidence.

  3. State files present without manifest (the cycle-11 P4
     shutdown-error scenario) → save_state NOT called. Otherwise
     the next boot's load preflight at ``boot_topology.py:170``
     fail-closes on missing-manifest.

  4. Clean state dir (no files at all) → save_state IS called and
     completes; the preflight writes the clean epoch marker
     (``_write_clean_epoch_marker`` at posterior_epoch.py:122)
     during the cold-start path, so the dir ends up consistent.

  5. Subsequent flush after a clean save → preflight sees status
     ``"match"`` + manifest valid → save_state called again.
     Idempotent under repeated calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sage.constants import BANDIT_FLUSH_INTERVAL
from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline, PipelineContext
from sage.pipeline_v2.learn import learn
from sage.posterior_epoch import (
    CONTAMINATED_MARKER_FILENAME,
    POSTERIOR_EPOCH_FILENAME,
    REQUIRED_POSTERIOR_EPOCH,
    write_topology_state_manifest,
)


def _build_pipeline_for_periodic_save(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured_save_calls: list[str],
) -> Pipeline:
    """Surgical Pipeline stub focused on the periodic-flush block in _stage_learn.

    Embedder + bandit + oracle are stubbed out; the only side-effect
    we care about is whether ``self.engine.save_state(state_dir)``
    was called.
    """
    monkeypatch.setenv("SAGE_ORACLE", "0")

    import sage.memory.embedder as embedder_mod

    class _StubEmbedder:
        is_semantic = False

        def embed(self, text: str) -> list[float]:
            return []

    monkeypatch.setattr(embedder_mod, "Embedder", _StubEmbedder)

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._llm_tier = ""
    pipeline.controller = None
    pipeline.llm_provider = None
    pipeline.llm_config = None
    pipeline.provider_pool = None
    pipeline.assigner = None
    pipeline.event_bus = None
    pipeline.tool_registry = None
    pipeline._agent_loop = None
    pipeline.write_gate = None
    pipeline.episodic_memory = None
    pipeline.semantic_memory = None
    pipeline.memory_agent = None
    pipeline.causal_memory = None
    pipeline._emit = MagicMock()
    pipeline._emit_budget_exceeded = MagicMock()
    pipeline._emit_bandit_attribution_mismatch = MagicMock()
    pipeline._on_topology_evolve = None
    pipeline.harness_config = None
    pipeline._harness_patcher = None
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.prm = None
    pipeline._estimate_topology_cost = MagicMock(return_value=0.0)
    pipeline.router = None
    pipeline._topology_cache = {}
    pipeline._apply_topology_budget_and_cache = MagicMock()
    pipeline._log_topology_structure = MagicMock()
    pipeline._last_routing_decision = None
    pipeline._last_runtime_routing_source = ""
    pipeline._last_runtime_routing_confidence = None
    pipeline._last_runtime_routing_model_id = ""
    pipeline.bandit = None
    pipeline.consolidator = None
    pipeline.quality_estimator = MagicMock()
    pipeline.quality_estimator.estimate = MagicMock(return_value=0.5)
    pipeline._rust_router = None  # Stage-5 bandit attribution skipped (no decision_id)

    # Pre-flush state: _task_count = N-1 so this _stage_learn call
    # increments to N (multiple of BANDIT_FLUSH_INTERVAL → fires).
    pipeline._task_count = BANDIT_FLUSH_INTERVAL - 1

    # Engine stub: save_state records the state_dir argument so we
    # can assert whether it was called.
    class _StubEngine:
        def save_state(self, state_dir: str) -> None:
            captured_save_calls.append(state_dir)

    pipeline.engine = _StubEngine()
    return pipeline


def _build_minimal_ctx() -> PipelineContext:
    """Minimal ctx for _stage_learn: result + topology=None + no decision_id."""
    ctx = PipelineContext(task="periodic save preflight test")
    ctx.system = 1
    ctx.domain = "code"
    ctx.result = "stub output"
    ctx.cost = 0.001
    ctx.latency_ms = 50.0
    ctx.topology = None
    ctx.bandit_decision_id = ""  # no decision → bandit recorder branch skipped
    ctx.executed_template = "single_agent"
    ctx.executed_model_id = "stub-model"
    return ctx


# ─────────────────────────────────────────────────────────────────
# Test 1: bypass active blocks the periodic save
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_periodic_save_blocks_when_bypass_env_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SAGE_BOOT_BYPASS_EPOCH_GUARD=1`` → periodic save MUST not fire.

    Atexit-symmetric: ``boot_topology.py:178-179`` skips
    ``atexit.register(_save_engine_state)`` when bypass is active.
    The periodic flush has the same invariant — under bypass, the
    save would proceed against an unverified state dir, which is
    exactly what directive #8 fail-close is meant to prevent.
    """
    monkeypatch.setenv("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")
    monkeypatch.setenv("SAGE_BOOT_BYPASS_REASON", "test")
    monkeypatch.setenv("SAGE_OPERATOR_ID", "test-operator")

    captured: list[str] = []
    pipeline = _build_pipeline_for_periodic_save(
        monkeypatch, captured_save_calls=captured,
    )
    ctx = _build_minimal_ctx()

    await learn(pipeline, ctx)

    assert captured == [], (
        f"Periodic save fired under SAGE_BOOT_BYPASS_EPOCH_GUARD=1: "
        f"{captured!r}. Bypass MUST disable saves (atexit handler "
        f"skips registration; periodic flush must skip the call). "
        f"This is directive #8 fail-closed."
    )


# ─────────────────────────────────────────────────────────────────
# Test 2: contaminated marker present blocks the periodic save
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_periodic_save_blocks_on_contaminated_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contaminated marker file present → save_state MUST not fire.

    The marker is the operator-readable poison-pill from ledger
    invariant 4 (Contaminated backup). If periodic save proceeded,
    fresh state files would land on top of contaminated files,
    erasing the forensic evidence the marker exists to preserve.
    """
    state_dir = Path(os.environ["HOME"]) / ".sage"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / CONTAMINATED_MARKER_FILENAME).write_text(
        json.dumps({
            "contaminated": True,
            "reason": "test marker",
            "audit_dump_sha256": "abc123",
        }),
        encoding="utf-8",
    )

    captured: list[str] = []
    pipeline = _build_pipeline_for_periodic_save(
        monkeypatch, captured_save_calls=captured,
    )
    ctx = _build_minimal_ctx()

    await learn(pipeline, ctx)

    assert captured == [], (
        f"Periodic save fired with _CONTAMINATED.json present: "
        f"{captured!r}. Writing fresh state on top of a contaminated "
        f"marker erases the forensic evidence the marker exists to "
        f"preserve."
    )


# ─────────────────────────────────────────────────────────────────
# Test 3: state files without manifest blocks the periodic save
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_periodic_save_blocks_on_state_files_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State files exist but no manifest → save_state MUST not fire.

    This is the cycle-11 P4 shutdown-error scenario observed in
    test sweeps. If the periodic save proceeded here, it would
    write more state files into a dir whose manifest is missing,
    making the next boot's load preflight fail-close. Fail closed
    NOW so the operator notices instead of at session start hours
    later.
    """
    state_dir = Path(os.environ["HOME"]) / ".sage"
    state_dir.mkdir(parents=True, exist_ok=True)
    # Plant fake state files like the Rust engine.save_state would
    # have written. Crucially: NO topology_state_manifest.json and
    # NO posterior_epoch.json — exactly the cycle-11 P4 shutdown
    # error string.
    (state_dir / "bandit_state.db").write_bytes(b"fake bandit db")
    (state_dir / "archive_state.db").write_bytes(b"fake archive db")
    (state_dir / "engine_extras.json").write_text("{}", encoding="utf-8")

    captured: list[str] = []
    pipeline = _build_pipeline_for_periodic_save(
        monkeypatch, captured_save_calls=captured,
    )
    ctx = _build_minimal_ctx()

    await learn(pipeline, ctx)

    assert captured == [], (
        f"Periodic save fired with state files present but no "
        f"manifest: {captured!r}. This is the cycle-11 P4 "
        f"shutdown-error scenario — fresh save would deepen the "
        f"manifest-less hole; the next load preflight would fail-"
        f"close. Block here instead."
    )


# ─────────────────────────────────────────────────────────────────
# Test 4: clean state dir → periodic save proceeds and writes epoch
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_periodic_save_proceeds_on_clean_state_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean state dir → save_state IS called; preflight writes epoch marker.

    The cold-start path: empty dir, no markers, no state files.
    ``ensure_clean_epoch_before_save`` writes the clean epoch
    marker and returns; ``engine.save_state`` then runs. The dir
    ends up consistent (epoch=1 marker present + state files +
    manifest from Rust ``save_state`` itself).

    Sanity check that the preflight isn't TOO aggressive — the
    happy path must still proceed.
    """
    state_dir = Path(os.environ["HOME"]) / ".sage"
    # Autouse fixture should have wiped this; double-check.
    assert not state_dir.exists() or not any(state_dir.iterdir()), (
        f"State dir not clean at test start: contents="
        f"{list(state_dir.iterdir()) if state_dir.exists() else 'n/a'}"
    )

    captured: list[str] = []
    pipeline = _build_pipeline_for_periodic_save(
        monkeypatch, captured_save_calls=captured,
    )
    ctx = _build_minimal_ctx()

    await learn(pipeline, ctx)

    assert len(captured) == 1, (
        f"Periodic save did not fire on clean state: {captured!r}. "
        f"The preflight is over-aggressive — clean cold-start must "
        f"still permit the save."
    )
    assert captured[0] == str(state_dir), (
        f"save_state was called with the wrong state_dir: "
        f"{captured[0]!r}, expected {str(state_dir)!r}"
    )

    # The preflight wrote the clean epoch marker via _write_clean_epoch_marker.
    epoch_path = state_dir / POSTERIOR_EPOCH_FILENAME
    assert epoch_path.exists(), (
        f"Clean epoch marker was not written by the preflight: "
        f"{epoch_path} does not exist."
    )
    payload = json.loads(epoch_path.read_text(encoding="utf-8"))
    assert payload["epoch"] == REQUIRED_POSTERIOR_EPOCH, (
        f"Epoch marker has wrong epoch value: {payload['epoch']!r}, "
        f"expected {REQUIRED_POSTERIOR_EPOCH}."
    )


# ─────────────────────────────────────────────────────────────────
# Test 5: subsequent flush after a clean save is idempotent
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_periodic_save_idempotent_after_first_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a successful save with manifest, next periodic flush also fires.

    The preflight reads ``status == "match"`` (epoch marker exists)
    and ``state_files`` present → calls
    ``verify_state_files_against_manifest``. Manifest exists from the
    previous save → verification passes → save proceeds. Subsequent
    flushes don't accumulate failures.

    Without this test, the fix could pass tests 1-4 but break the
    second flush (e.g. by leaving state in an inconsistent partial
    write).
    """
    state_dir = Path(os.environ["HOME"]) / ".sage"
    state_dir.mkdir(parents=True, exist_ok=True)
    # Pre-populate as if a prior save_state had run: epoch marker,
    # state files, AND a valid manifest for those state files.
    (state_dir / "bandit_state.db").write_bytes(b"prior bandit db")
    (state_dir / "archive_state.db").write_bytes(b"prior archive db")
    (state_dir / "engine_extras.json").write_text("{}", encoding="utf-8")
    (state_dir / POSTERIOR_EPOCH_FILENAME).write_text(
        json.dumps({
            "epoch": REQUIRED_POSTERIOR_EPOCH,
            "started_utc": "2026-05-05T00:00:00+00:00",
            "reason": "test setup",
            "policy": {},
        }),
        encoding="utf-8",
    )
    write_topology_state_manifest(state_dir, writer="test-setup")

    captured: list[str] = []
    pipeline = _build_pipeline_for_periodic_save(
        monkeypatch, captured_save_calls=captured,
    )
    ctx = _build_minimal_ctx()

    await learn(pipeline, ctx)

    assert len(captured) == 1, (
        f"Subsequent flush after a clean save did not fire: "
        f"{captured!r}. Preflight must accept status='match' + "
        f"valid manifest as a permitted save state."
    )
