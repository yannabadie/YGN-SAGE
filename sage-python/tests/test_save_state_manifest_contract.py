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

import sage_core


_REBUILD_HINT = (
    "If this test fails on your dev box but passes on CI, your installed "
    "`sage_core.cp313-*.pyd` is likely STALE relative to the Rust source. "
    "Rebuild via:\n"
    "    cd sage-core && maturin develop --features smt,onnx\n"
    "(Per CLAUDE.md 'Quick Commands' block. CI builds fresh wheels per "
    "commit so this test passes on every CI run.)"
)


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
