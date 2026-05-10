"""Cycle-13 B — Rust `save_state` topology_state_manifest.json contract regression.

cgpro NEXT_BLOCK_ID = B (post-push 2026-05-06, conv `cgpro_pi_mono_pivot_20260505`):
  "Post-save manifest gap fix (advisor 2026-05-04 + caught empirically tonight)
  — engine.save_state writes the manifest in Rust at engine.rs:1031, but
  ~/.sage/ empirically lacks it after recent saves. Each successful pipeline
  run leaves contaminated state for the next boot per directive #8 fail-closed.
  Operator workaround = python -m sage.ops.a14_reset per run."

## Root cause caught by this test

The fix at `engine.rs:1031` (calling `posterior_epoch::write_topology_state_manifest`)
landed in commit `f9521616` 2026-04-30. The installed `sage_core.cp313-win_amd64.pyd`
on this dev machine was dated 2026-04-27 — built BEFORE the fix. Stale binary →
state files (bandit_state.db, archive_state.db, engine_extras.json) written but
manifest absent → next boot fails per directive #8 fail-closed → operator forced
to run `python -m sage.ops.a14_reset` per cycle. This test prevents the
recurrence.

## What this test asserts

Direct `sage_core.TopologyEngine` instance + `save_state()` call → manifest file
must exist + must contain {manifest_type, epoch, writer, state_files}.

Assumptions:
- CI builds fresh wheels every time, so the test passes on CI.
- Local dev with stale `.pyd` will FAIL with a clear error message pointing at
  the rebuild command.

## Why this is a Python test (not Rust)

The Rust source has its own unit tests for `write_topology_state_manifest`. But
those tests run against the SOURCE, not against the BINARY a Python user has
installed. This test runs the Python boundary — exactly the surface where the
gap manifested operationally.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sage.posterior_epoch import write_topology_state_manifest

import sage_core


_REBUILD_HINT = (
    "If this test fails on your dev box but passes on CI, your installed "
    "`sage_core.cp313-*.pyd` is likely STALE relative to the Rust source. "
    "Rebuild via:\n"
    "    cd sage-core && maturin develop --features smt,onnx\n"
    "(Per CLAUDE.md 'Quick Commands' block. CI builds fresh wheels per "
    "commit so this test passes on every CI run.)"
)


def _write_epoch_marker(state_dir: Path, reason: str) -> None:
    (state_dir / "posterior_epoch.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "started_utc": "2026-05-10T00:00:00Z",
                "reason": reason,
                "policy": "test",
                "audit_dump": "",
                "commit_at_reset": "",
                "predecessor_state": "",
                "first_clean_run_after": None,
            }
        ),
        encoding="utf-8",
    )


def _engine_with_smmu_outcome() -> tuple[sage_core.TopologyEngine, int]:
    engine = sage_core.TopologyEngine()
    result = engine.generate("Write a merge sort function", None, 2, 0.0)
    topology = result.topology
    engine.cache_topology(topology)
    engine.record_outcome(
        topology.id,
        "merge sort completed successfully",
        ["sort", "merge", "algorithm"],
        None,
        0.93,
        0.008,
        120.0,
    )
    chunk_count = engine.smmu_chunk_count()
    assert chunk_count > 0, "fixture must populate the internal PyO3 S-MMU"
    return engine, chunk_count


def test_save_state_writes_topology_state_manifest(tmp_path: Path) -> None:
    """Direct contract: TopologyEngine.save_state must write the manifest.

    Pre-fix (2026-04-30 cycle-8 step 2 A14 VERIFY round-1, commit f9521616):
    save_state wrote bandit_state.db + archive_state.db + engine_extras.json
    but NOT topology_state_manifest.json. Next boot failed per directive #8.

    Post-fix: save_state writes ALL FOUR files atomically. Manifest contains
    `manifest_type=YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST`, `epoch=1`,
    `writer=TopologyEngine::save_state`, and SHA256 fingerprints of every
    state file.
    """
    state_dir = tmp_path / "save_state_probe"
    state_dir.mkdir()

    # save_state's preflight (validate_epoch_for_save) needs posterior_epoch.json
    # to exist. Write a minimal valid one.
    (state_dir / "posterior_epoch.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "started_utc": "2026-05-06T00:00:00Z",
                "reason": "regression test for save_state manifest contract",
                "policy": "test",
                "audit_dump": "",
                "commit_at_reset": "",
                "predecessor_state": "",
                "first_clean_run_after": None,
            }
        ),
        encoding="utf-8",
    )

    engine = sage_core.TopologyEngine()
    engine.save_state(str(state_dir))

    manifest_path = state_dir / "topology_state_manifest.json"
    assert manifest_path.exists(), (
        f"topology_state_manifest.json was NOT written by save_state. "
        f"State dir contents: {sorted(p.name for p in state_dir.iterdir())}.\n\n"
        f"{_REBUILD_HINT}"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_type"] == "YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST", (
        f"unexpected manifest_type: {manifest.get('manifest_type')!r}"
    )
    assert manifest["epoch"] == 1, f"expected epoch=1, got {manifest.get('epoch')!r}"
    assert manifest["writer"] == "TopologyEngine::save_state", (
        f"unexpected writer: {manifest.get('writer')!r}"
    )

    # state_files list MUST cover every persisted A14 state file
    # (bandit_state.db, archive_state.db, engine_extras.json on a fresh
    # engine — wal/shm sidecars only exist after sqlite WAL flush).
    state_files_in_manifest = {entry["name"] for entry in manifest["state_files"]}
    expected_minimum = {"bandit_state.db", "archive_state.db", "engine_extras.json"}
    missing = expected_minimum - state_files_in_manifest
    assert not missing, (
        f"manifest missing entries for {sorted(missing)}. "
        f"Got: {sorted(state_files_in_manifest)}.\n\n{_REBUILD_HINT}"
    )

    # Per cgpro deep VERIFY 2026-05-06 Q3: assert each manifest entry's
    # sha256 + size_bytes match the file's CURRENT bytes on disk. The
    # manifest is a provenance binding: write_topology_state_manifest()
    # reads each file's bytes AT manifest-write time. Just after
    # save_state() returns (no concurrent writer in this test), the
    # save-time fingerprint and the current fingerprint must match
    # byte-for-byte. Without this, a future code path could write a
    # syntactically-valid manifest decoupled from the actual state
    # files and pass the previous (loose) shape-only test.
    for entry in manifest["state_files"]:
        name = entry["name"]
        path = state_dir / name
        assert path.is_file(), f"manifest references non-existent {name}"

        actual_bytes = path.read_bytes()
        expected_sha = entry.get("sha256", "")
        actual_sha = hashlib.sha256(actual_bytes).hexdigest()
        assert (
            len(expected_sha) == 64
            and all(c in "0123456789abcdef" for c in expected_sha)
        ), f"invalid sha256 hex for {name}: {expected_sha!r}"
        assert expected_sha == actual_sha, (
            f"manifest sha256 mismatch for {name}: "
            f"manifest={expected_sha} actual={actual_sha}.\n\n"
            f"Save-time fingerprint must equal post-save current "
            f"fingerprint. {_REBUILD_HINT}"
        )

        expected_size = entry.get("size_bytes", 0)
        actual_size = len(actual_bytes)
        assert isinstance(expected_size, int) and expected_size > 0, (
            f"invalid size_bytes for {name}: {expected_size!r}"
        )
        assert expected_size == actual_size, (
            f"manifest size_bytes mismatch for {name}: "
            f"manifest={expected_size} actual={actual_size}"
        )


def test_save_state_writes_smmu_snapshot_and_manifest_binding(tmp_path: Path) -> None:
    """TopologyEngine.save_state must persist its internal PyO3-owned S-MMU.

    Standalone MultiViewMMU.save_json/load_json is not enough for runtime
    integrity: the S-MMU that feeds TopologyEngine's smmu_hit path is owned
    inside the PyO3 wrapper. A save_state claim is only truthful if that
    wrapper-owned S-MMU is written to state and fingerprinted in the A14
    manifest.
    """
    state_dir = tmp_path / "save_state_smmu_probe"
    state_dir.mkdir()
    _write_epoch_marker(state_dir, "regression test for save_state smmu contract")

    engine, chunk_count = _engine_with_smmu_outcome()
    engine.save_state(str(state_dir))

    smmu_path = state_dir / "smmu_state.json"
    assert smmu_path.exists(), (
        "save_state did not persist the internal PyO3 S-MMU as "
        "smmu_state.json. Standalone MultiViewMMU.save_json evidence is "
        "insufficient for TopologyEngine.save_state.\n\n"
        f"{_REBUILD_HINT}"
    )

    smmu_snapshot = json.loads(smmu_path.read_text(encoding="utf-8"))
    assert smmu_snapshot["version"] == 1
    assert len(smmu_snapshot["multi_view_mmu"]["chunks"]) == chunk_count
    assert len(smmu_snapshot["topology_bridge"]["chunk_meta"]) == chunk_count

    manifest = json.loads(
        (state_dir / "topology_state_manifest.json").read_text(encoding="utf-8")
    )
    smmu_entries = [
        entry for entry in manifest["state_files"] if entry["name"] == "smmu_state.json"
    ]
    assert len(smmu_entries) == 1, (
        "topology_state_manifest.json must include exactly one smmu_state.json "
        f"entry. Got: {[entry['name'] for entry in manifest['state_files']]}"
    )
    entry = smmu_entries[0]
    actual = smmu_path.read_bytes()
    assert entry["size_bytes"] == len(actual)
    assert entry["sha256"] == hashlib.sha256(actual).hexdigest()


def test_load_state_restores_internal_smmu(tmp_path: Path) -> None:
    """Fresh PyTopologyEngine.load_state must restore S-MMU chunks."""
    state_dir = tmp_path / "load_state_smmu_probe"
    state_dir.mkdir()
    _write_epoch_marker(state_dir, "regression test for load_state smmu contract")

    engine, chunk_count = _engine_with_smmu_outcome()
    engine.save_state(str(state_dir))

    restored = sage_core.TopologyEngine()
    assert restored.smmu_chunk_count() == 0
    restored.load_state(str(state_dir))
    assert restored.smmu_chunk_count() == chunk_count


def test_load_state_fails_closed_on_smmu_manifest_hash_mismatch(tmp_path: Path) -> None:
    """A tampered S-MMU snapshot must trip A14 before state is accepted."""
    state_dir = tmp_path / "smmu_hash_mismatch_probe"
    state_dir.mkdir()
    _write_epoch_marker(state_dir, "regression test for smmu manifest hash mismatch")

    engine, _chunk_count = _engine_with_smmu_outcome()
    engine.save_state(str(state_dir))

    smmu_path = state_dir / "smmu_state.json"
    smmu_path.write_text(
        smmu_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    restored = sage_core.TopologyEngine()
    with pytest.raises(Exception) as exc:
        restored.load_state(str(state_dir))

    message = str(exc.value)
    assert "smmu_state.json" in message
    assert "sha256" in message or "size" in message
    assert restored.smmu_chunk_count() == 0


def test_load_state_restores_functional_smmu_hit_path(tmp_path: Path) -> None:
    """Restored S-MMU must be usable by the live smmu_hit generation path."""
    state_dir = tmp_path / "smmu_hit_after_reload_probe"
    state_dir.mkdir()
    _write_epoch_marker(state_dir, "regression test for live smmu hit after reload")

    engine = sage_core.TopologyEngine()
    embedding = [1.0, 0.0, 0.0, 0.0]

    first = engine.generate("debug rust memory graph baseline", None, 2, 0.0)
    engine.cache_topology(first.topology)
    engine.record_outcome(
        first.topology.id,
        "debug rust memory graph baseline",
        ["debug", "rust", "memory"],
        embedding,
        0.95,
        0.005,
        100.0,
    )

    second = engine.generate("debug rust memory graph followup", None, 2, 0.0)
    engine.cache_topology(second.topology)
    engine.record_outcome(
        second.topology.id,
        "debug rust memory graph followup",
        ["debug", "rust", "memory"],
        embedding,
        0.70,
        0.02,
        120.0,
    )
    assert engine.smmu_chunk_count() == 2

    engine.save_state(str(state_dir))

    restored = sage_core.TopologyEngine()
    restored.load_state(str(state_dir))

    result = restored.generate_with_options(
        "debug rust memory graph next",
        None,
        2,
        0.0,
        True,
        False,
        False,
        False,
        False,
    )
    assert result.source == "smmu_hit"
    assert result.topology.node_count() > 0


def test_save_state_refuses_smmu_write_under_a14_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forensic A14 bypass remains load-only and must not write S-MMU state."""
    state_dir = tmp_path / "smmu_bypass_save_probe"
    state_dir.mkdir()
    _write_epoch_marker(state_dir, "regression test for bypass save refusal")

    engine, _chunk_count = _engine_with_smmu_outcome()
    monkeypatch.setenv("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")

    with pytest.raises(Exception) as exc:
        engine.save_state(str(state_dir))

    assert "bypass" in str(exc.value).lower()
    assert not (state_dir / "smmu_state.json").exists()


def test_load_old_checkpoint_without_smmu_clears_warm_smmu(tmp_path: Path) -> None:
    """Backward-compatible absence means cold S-MMU, not retained warm memory."""
    state_dir = tmp_path / "old_checkpoint_without_smmu_probe"
    state_dir.mkdir()
    _write_epoch_marker(state_dir, "regression test for old checkpoint compatibility")

    source, _chunk_count = _engine_with_smmu_outcome()
    source.save_state(str(state_dir))
    (state_dir / "smmu_state.json").unlink()
    write_topology_state_manifest(state_dir, writer="test-old-checkpoint")

    warm, _warm_chunk_count = _engine_with_smmu_outcome()
    assert warm.smmu_chunk_count() > 0
    warm.load_state(str(state_dir))
    assert warm.smmu_chunk_count() == 0


def test_save_state_manifest_survives_round_trip_load(tmp_path: Path) -> None:
    """The manifest written by save_state must be readable by load_state.

    Per directive #8 fail-closed boot guard: a save+load cycle MUST keep
    the runtime in normal-state (epoch=1 + manifest valid + state files
    present). If save writes manifest but load can't parse it, we still
    fail the next boot — a subtler form of the same gap.
    """
    state_dir = tmp_path / "save_load_probe"
    state_dir.mkdir()

    (state_dir / "posterior_epoch.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "started_utc": "2026-05-06T00:00:00Z",
                "reason": "regression test for save+load round-trip",
                "policy": "test",
                "audit_dump": "",
                "commit_at_reset": "",
                "predecessor_state": "",
                "first_clean_run_after": None,
            }
        ),
        encoding="utf-8",
    )

    engine_save = sage_core.TopologyEngine()
    engine_save.save_state(str(state_dir))

    # Manifest written; reload via fresh engine.
    engine_load = sage_core.TopologyEngine()
    # load_state must NOT raise an A14 epoch guard error (i.e. the manifest
    # must be valid + state files unchanged).
    try:
        engine_load.load_state(str(state_dir))
    except Exception as exc:
        pytest.fail(
            f"load_state failed after save_state — manifest contract broken: "
            f"{exc}\n\n{_REBUILD_HINT}"
        )
