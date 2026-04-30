import json
import shutil
from pathlib import Path

import pytest

from sage.ops import a14_reset
from sage.posterior_epoch import (
    A14_EPOCH_GUARD_ERROR_PREFIX,
    CONTAMINATED_MARKER_FILENAME,
    POSTERIOR_EPOCH_FILENAME,
    check_posterior_epoch_for_boot,
    ensure_clean_epoch_before_save,
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
