import json
import shutil
from pathlib import Path

import pytest

from sage.ops import a14_reset
from sage.posterior_epoch import (
    A14_EPOCH_GUARD_ERROR_PREFIX,
    CONTAMINATED_MARKER_FILENAME,
    POSTERIOR_EPOCH_FILENAME,
    TOPOLOGY_STATE_MANIFEST_FILENAME,
    check_posterior_epoch_for_boot,
    ensure_clean_epoch_before_save,
    write_topology_state_manifest,
)


def _touch(path: Path, content: str = "legacy-state") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_reset(state_dir: Path, audit_dir: Path, reason: str) -> Path:
    reset_id = "pre_a14_20260429"
    rc = a14_reset.main(
        [
            "--state-dir",
            str(state_dir),
            "--audit-dir",
            str(audit_dir),
            "--reason",
            reason,
            "--reset-id",
            reset_id,
        ],
    )
    assert rc == 0
    return state_dir / "contaminated" / reset_id


def _seed_a14_state(state_dir: Path) -> None:
    _touch(state_dir / "bandit_state.db", "bandit")
    _touch(state_dir / "archive_state.db", "archive")
    _touch(state_dir / "engine_extras.json", '{"cma": true}')
    _touch(state_dir / "bandit_state.db-wal", "wal")
    _touch(state_dir / "episodic.db", "memory-tier")


def _assert_a14_failure(state_dir: Path) -> str:
    with pytest.raises(RuntimeError) as exc_info:
        check_posterior_epoch_for_boot(state_dir)
    message = str(exc_info.value)
    assert message.startswith(A14_EPOCH_GUARD_ERROR_PREFIX)
    return message


def test_a14_reset_round_trip(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)

    backup_dir = _run_reset(state_dir, audit_dir, "A14 reset under test")

    active_epoch = json.loads((state_dir / POSTERIOR_EPOCH_FILENAME).read_text())
    assert active_epoch["epoch"] == 1
    assert active_epoch["reason"] == "A14 reset under test"
    assert "pre_a14_20260429" in active_epoch["predecessor_state"]
    assert not (state_dir / "bandit_state.db").exists()
    assert not (state_dir / "archive_state.db").exists()
    assert (state_dir / "episodic.db").exists()

    assert (backup_dir / "bandit_state.db").exists()
    assert (backup_dir / "archive_state.db").exists()
    assert (backup_dir / "engine_extras.json").exists()
    marker = json.loads((backup_dir / CONTAMINATED_MARKER_FILENAME).read_text())
    assert marker["marker_type"] == "YGN-SAGE_A14_CONTAMINATED_TOPOLOGY_STATE"
    assert marker["contaminated"] is True
    assert marker["target_epoch"] == 1
    assert marker["reason"] == "A14 reset under test"
    assert "episodic.db" not in marker["state_files"]

    manifest_path = audit_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    assert {artifact["name"] for artifact in manifest["artifacts"]} >= {
        "bandit_state.db",
        "archive_state.db",
        "engine_extras.json",
    }
    assert marker["audit_dump"] == str(manifest_path)
    assert marker["audit_dump_sha256"] == a14_reset.sha256_file(manifest_path)


def test_post_reset_boot_fail_closed_on_restore(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)
    backup_dir = _run_reset(state_dir, audit_dir, "A14 reset under test")

    (state_dir / POSTERIOR_EPOCH_FILENAME).unlink()
    shutil.copy2(backup_dir / "bandit_state.db", state_dir / "bandit_state.db")

    message = _assert_a14_failure(state_dir)
    assert "epoch_file=missing" in message


def test_a14_reset_retries_windows_final_backup_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)

    real_replace = a14_reset.os.replace
    calls = {"final": 0}

    def flaky_replace(src: object, dst: object) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        if (
            src_path.is_dir()
            and src_path.name.startswith(".tmp_pre_a14_20260429_")
            and dst_path.name == "pre_a14_20260429"
            and calls["final"] < 2
        ):
            calls["final"] += 1
            exc = PermissionError("Access is denied")
            exc.winerror = 5  # type: ignore[attr-defined]
            raise exc
        real_replace(src, dst)

    monkeypatch.setattr(a14_reset.os, "name", "nt", raising=False)
    monkeypatch.setattr(a14_reset.os, "replace", flaky_replace)

    backup_dir = _run_reset(state_dir, audit_dir, "A14 reset under test")

    assert calls["final"] == 2
    assert backup_dir.exists()
    assert (backup_dir / "bandit_state.db").exists()
    assert (state_dir / POSTERIOR_EPOCH_FILENAME).exists()


def test_a14_reset_preserves_temp_and_poison_marks_on_final_rename_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)

    real_replace = a14_reset.os.replace

    def failing_final_replace(src: object, dst: object) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        if (
            src_path.is_dir()
            and src_path.name.startswith(".tmp_pre_a14_20260429_")
            and dst_path.name == "pre_a14_20260429"
        ):
            exc = PermissionError("Access is denied")
            exc.winerror = 5  # type: ignore[attr-defined]
            raise exc
        real_replace(src, dst)

    monkeypatch.setattr(a14_reset.os, "name", "nt", raising=False)
    monkeypatch.setattr(a14_reset.os, "replace", failing_final_replace)

    with pytest.raises(PermissionError):
        _run_reset(state_dir, audit_dir, "A14 reset under test")

    preserved = list((state_dir / "contaminated").glob(".tmp_pre_a14_20260429_*"))
    assert len(preserved) == 1
    assert (preserved[0] / "bandit_state.db").exists()

    marker_path = state_dir / CONTAMINATED_MARKER_FILENAME
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["marker_type"] == "YGN-SAGE_A14_FAILED_RESET_POISON"
    assert marker["failure_stage"] == "backup_finalize"
    assert marker["preserved_temp_backup_dir"] == str(preserved[0])

    message = _assert_a14_failure(state_dir)
    assert "poison pill marker present" in message


def test_a14_reset_final_backup_retry_is_windows_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = tmp_path / "target"
    calls = {"replace": 0}

    def failing_replace(src: object, dst: object) -> None:  # noqa: ARG001
        calls["replace"] += 1
        exc = PermissionError("Access is denied")
        exc.winerror = 5  # type: ignore[attr-defined]
        raise exc

    monkeypatch.setattr(a14_reset.os, "name", "posix", raising=False)
    monkeypatch.setattr(a14_reset.os, "replace", failing_replace)

    with pytest.raises(PermissionError):
        a14_reset._replace_dir_atomic_with_retry(
            source,
            target,
            attempts=3,
            delay_seconds=0,
        )

    assert calls["replace"] == 1


def test_post_reset_boot_fail_closed_on_restore_over_valid_epoch(tmp_path: Path) -> None:
    """cgpro 2026-04-30 cycle-8 step 2 A14 round-1 trap: contaminated DB
    copied back over a valid epoch=1 marker MUST fail-closed.
    """
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)
    backup_dir = _run_reset(state_dir, audit_dir, "A14 reset under test")

    # Important: DO NOT unlink posterior_epoch.json. The active marker is
    # left in place at epoch=1 (the post-reset honest state).
    assert json.loads((state_dir / POSTERIOR_EPOCH_FILENAME).read_text())["epoch"] == 1

    # Operator copies DB-only from contaminated backup back into ~/.sage.
    shutil.copy2(backup_dir / "bandit_state.db", state_dir / "bandit_state.db")

    with pytest.raises(RuntimeError) as exc:
        check_posterior_epoch_for_boot(state_dir)

    assert str(exc.value).startswith(A14_EPOCH_GUARD_ERROR_PREFIX)
    assert "topology_state_manifest" in str(exc.value)


def test_post_reset_boot_fail_closed_on_whole_backup_restore(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)
    backup_dir = _run_reset(state_dir, audit_dir, "A14 reset under test")

    for filename in ("bandit_state.db", CONTAMINATED_MARKER_FILENAME):
        shutil.copy2(backup_dir / filename, state_dir / filename)

    message = _assert_a14_failure(state_dir)
    assert "poison pill marker present" in message


def test_fresh_install_cold_start_then_save_then_second_boot_loads(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    first = check_posterior_epoch_for_boot(state_dir)
    assert first.state_files == ()

    ensure_clean_epoch_before_save(state_dir)
    _touch(state_dir / "bandit_state.db")
    write_topology_state_manifest(state_dir, writer="test")
    second = check_posterior_epoch_for_boot(state_dir)

    assert second.file_epoch == 1
    assert second.epoch_status == "match"


def test_legacy_20260429_backup_marker_written(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "contaminated_pre_a14_20260429"
    audit_dir = tmp_path / ".tmp" / "a14_reset_20260429"
    _touch(legacy_dir / "bandit_state.db")
    audit_dir.mkdir(parents=True)
    (audit_dir / "MANIFEST.json").write_text(
        json.dumps({"artifacts": []}),
        encoding="utf-8",
    )

    rc = a14_reset.main(
        [
            "--mark-existing-contaminated-dir",
            str(legacy_dir),
            "--audit-dir",
            str(audit_dir),
            "--reason",
            "mark legacy backup",
        ],
    )

    assert rc == 0
    marker = json.loads((legacy_dir / CONTAMINATED_MARKER_FILENAME).read_text())
    assert marker["reset_id"] == "pre_a14_20260429"
    assert marker["reason"] == "mark legacy backup"
    assert marker["audit_dump_sha256"] == a14_reset.sha256_file(
        audit_dir / "MANIFEST.json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cycle-13 B Q4-bis (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=I, conv
# `cgpro_pi_mono_pivot_20260505`): orphaned `.<name>.<id>.tmp` cleanup
# follow-up to the Rust fix at `bc662d9a`.
# ─────────────────────────────────────────────────────────────────────────────


def _write_orphan_tmp(state_dir: Path, name: str) -> Path:
    """Helper: create a `.<name>.<id>.tmp` orphan under `state_dir`."""
    state_dir.mkdir(parents=True, exist_ok=True)
    orphan = state_dir / f".{name}.01ABCDEFGHJKMNPQRSTVWXYZ12.tmp"
    orphan.write_text("orphan", encoding="utf-8")
    return orphan


def test_orphan_tmp_files_lists_atomic_written_orphans(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Source-of-truth-ed via imported constants — if a future rename
    # touches POSTERIOR_EPOCH_FILENAME / TOPOLOGY_STATE_MANIFEST_FILENAME
    # / CONTAMINATED_MARKER_FILENAME, the test follows.
    orphan_a = _write_orphan_tmp(state_dir, POSTERIOR_EPOCH_FILENAME)
    orphan_b = _write_orphan_tmp(state_dir, TOPOLOGY_STATE_MANIFEST_FILENAME)
    orphan_c = _write_orphan_tmp(state_dir, CONTAMINATED_MARKER_FILENAME)

    # Decoy: a `.tmp` file NOT matching any atomic-written name must
    # NOT be listed (the helper is conservative, only known names).
    (state_dir / ".unrelated.txt.zzz.tmp").write_text("decoy", encoding="utf-8")
    # Decoy: legitimate non-tmp file (no prefix dot) must NOT be listed.
    (state_dir / "posterior_epoch.json").write_text("real", encoding="utf-8")

    listed = a14_reset._orphan_tmp_files(state_dir)

    assert orphan_a.name in listed
    assert orphan_b.name in listed
    assert orphan_c.name in listed
    assert ".unrelated.txt.zzz.tmp" not in listed
    assert "posterior_epoch.json" not in listed
    assert listed == sorted(listed), "must be sorted for forensic determinism"


def test_orphan_tmp_files_includes_topology_state_manifest(tmp_path: Path) -> None:
    """Cycle-13 B Q4-bis cgpro pre-commit HARD_STOP catch (2026-05-06):
    `topology_state_manifest.json` is the PRIMARY orphan class — the
    Rust `write_bytes_atomic` leak this whole follow-up exists to
    address. Pre-cgpro-pre-commit `_ATOMIC_WRITTEN_NAMES` had the
    string literal `"topology_state_manifest.json"` (correct value
    but not source-of-truth-ed); the test must EXPLICITLY assert
    this filename is detected so a future rename of the Python
    constant (without touching `_ATOMIC_WRITTEN_NAMES`) fails the
    test loud."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    orphan = _write_orphan_tmp(state_dir, TOPOLOGY_STATE_MANIFEST_FILENAME)

    listed = a14_reset._orphan_tmp_files(state_dir)
    assert orphan.name in listed, (
        f"topology_state_manifest.json orphan MUST be detected — "
        f"this is the Rust write_bytes_atomic leak class (engine.rs:1031). "
        f"Got: {listed}"
    )

    removed = a14_reset._cleanup_orphaned_tmp_files(state_dir)
    assert orphan.name in removed
    assert not orphan.exists()


def test_orphan_tmp_files_empty_for_missing_state_dir(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist"
    assert a14_reset._orphan_tmp_files(nonexistent) == []


def test_cleanup_orphaned_tmp_files_removes_them(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    orphan_a = _write_orphan_tmp(state_dir, "posterior_epoch.json")
    orphan_b = _write_orphan_tmp(state_dir, "topology_state_manifest.json")

    removed = a14_reset._cleanup_orphaned_tmp_files(state_dir)

    assert orphan_a.name in removed
    assert orphan_b.name in removed
    assert not orphan_a.exists()
    assert not orphan_b.exists()
    # Idempotent: a second run finds nothing, returns [].
    assert a14_reset._cleanup_orphaned_tmp_files(state_dir) == []


def test_cleanup_orphaned_tmp_files_empty_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # No orphans, no errors, returns [].
    assert a14_reset._cleanup_orphaned_tmp_files(state_dir) == []


def test_run_reset_records_cleaned_orphans_in_audit_manifest(tmp_path: Path) -> None:
    """End-to-end: orphaned .tmp present at reset time gets cleaned
    AND recorded in the audit manifest's `cleaned_orphan_tmp_files`
    field. Forensic surface for operators."""
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)

    orphan = _write_orphan_tmp(state_dir, "posterior_epoch.json")
    assert orphan.exists()

    _run_reset(state_dir, audit_dir, "orphan cleanup smoke")

    # Orphan was removed from state_dir during reset.
    assert not orphan.exists()
    # Manifest records the cleanup.
    manifest = json.loads((audit_dir / "MANIFEST.json").read_text())
    assert "cleaned_orphan_tmp_files" in manifest
    assert orphan.name in manifest["cleaned_orphan_tmp_files"]


def test_run_reset_records_empty_cleaned_orphans_when_none(tmp_path: Path) -> None:
    """Manifest's `cleaned_orphan_tmp_files` is `[]` (not absent) when
    no orphans existed — operators can rely on the field always being
    present for forensic queries."""
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    _seed_a14_state(state_dir)

    _run_reset(state_dir, audit_dir, "no orphan smoke")

    manifest = json.loads((audit_dir / "MANIFEST.json").read_text())
    assert manifest["cleaned_orphan_tmp_files"] == []


def test_dry_run_lists_orphans_without_cleaning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--dry-run` reports orphan `.tmp` files but does NOT remove
    them. Operators can preview the cleanup before acting."""
    state_dir = tmp_path / "state"
    audit_dir = tmp_path / "audit"
    state_dir.mkdir()

    orphan = _write_orphan_tmp(state_dir, "topology_state_manifest.json")

    rc = a14_reset.main(
        [
            "--state-dir",
            str(state_dir),
            "--audit-dir",
            str(audit_dir),
            "--reason",
            "dry run preview",
            "--reset-id",
            "pre_a14_20260506",
            "--dry-run",
        ],
    )
    assert rc == 0
    # Orphan still present after dry-run.
    assert orphan.exists()
    # stdout payload includes the orphan list.
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert orphan.name in payload["orphan_tmp_files_to_clean"]
