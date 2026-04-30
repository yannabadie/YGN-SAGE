import json
from pathlib import Path

import pytest

from sage.posterior_epoch import (
    A14_EPOCH_GUARD_ERROR_PREFIX,
    CONTAMINATED_MARKER_FILENAME,
    POSTERIOR_EPOCH_FILENAME,
    REQUIRED_POSTERIOR_EPOCH,
    check_posterior_epoch_for_boot,
    ensure_clean_epoch_before_save,
    is_a14_epoch_guard_error,
)


def _touch(path: Path) -> None:
    path.write_text("legacy-state", encoding="utf-8")


def _write_epoch(state_dir: Path, payload: object) -> None:
    (state_dir / POSTERIOR_EPOCH_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _assert_a14_error(exc_info: pytest.ExceptionInfo[BaseException]) -> str:
    message = str(exc_info.value)
    assert message.startswith(A14_EPOCH_GUARD_ERROR_PREFIX)
    return message


def test_python_preflight_missing_epoch_with_state_raises(tmp_path: Path) -> None:
    _touch(tmp_path / "bandit_state.db")

    with pytest.raises(RuntimeError) as exc_info:
        check_posterior_epoch_for_boot(tmp_path)

    message = _assert_a14_error(exc_info)
    assert "epoch_file=missing" in message


def test_python_preflight_epoch_mismatch_raises(tmp_path: Path) -> None:
    _touch(tmp_path / "archive_state.db")
    _write_epoch(tmp_path, {"epoch": 2})

    with pytest.raises(RuntimeError) as exc_info:
        check_posterior_epoch_for_boot(tmp_path)

    message = _assert_a14_error(exc_info)
    assert "epoch mismatch: file=2 required=1" in message


def test_python_preflight_malformed_with_state_raises(tmp_path: Path) -> None:
    _touch(tmp_path / "engine_extras.json")
    _write_epoch(tmp_path, {"epoch": "1"})

    with pytest.raises(RuntimeError) as exc_info:
        check_posterior_epoch_for_boot(tmp_path)

    message = _assert_a14_error(exc_info)
    assert "posterior_epoch.json malformed" in message


def test_python_preflight_contaminated_marker_raises(tmp_path: Path) -> None:
    (tmp_path / CONTAMINATED_MARKER_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        check_posterior_epoch_for_boot(tmp_path)

    message = _assert_a14_error(exc_info)
    assert "poison pill marker present" in message


def test_python_preflight_missing_epoch_no_state_cold_start_ok(tmp_path: Path) -> None:
    result = check_posterior_epoch_for_boot(tmp_path)

    assert result.state_files == ()
    assert result.file_epoch is None
    assert result.epoch_status == "missing"


def test_python_preflight_ignores_memory_tier_dbs(tmp_path: Path) -> None:
    for filename in ("episodic.db", "semantic.db", "causal.db", "evolution_memory.db"):
        _touch(tmp_path / filename)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    result = check_posterior_epoch_for_boot(tmp_path)

    assert result.state_files == ()
    assert result.epoch_status == "missing"


def test_ensure_clean_epoch_before_save_creates_marker_on_first_save(
    tmp_path: Path,
) -> None:
    ensure_clean_epoch_before_save(tmp_path)

    payload = json.loads((tmp_path / POSTERIOR_EPOCH_FILENAME).read_text())
    assert payload["epoch"] == REQUIRED_POSTERIOR_EPOCH
    assert payload["reason"] == (
        "auto-created clean topology posterior epoch before first save_state"
    )


def test_ensure_clean_epoch_before_save_refuses_missing_epoch_with_state(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / "bandit_state.db-shm")

    with pytest.raises(RuntimeError) as exc_info:
        ensure_clean_epoch_before_save(tmp_path)

    message = _assert_a14_error(exc_info)
    assert "epoch_file=missing" in message


def test_ensure_clean_epoch_before_save_refuses_malformed_or_mismatched_epoch(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    _write_epoch(malformed, {"epoch": "1"})

    with pytest.raises(RuntimeError) as malformed_exc:
        ensure_clean_epoch_before_save(malformed)
    assert "posterior_epoch.json malformed" in _assert_a14_error(malformed_exc)

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    _write_epoch(mismatched, {"epoch": 0})

    with pytest.raises(RuntimeError) as mismatched_exc:
        ensure_clean_epoch_before_save(mismatched)
    assert "epoch mismatch: file=0 required=1" in _assert_a14_error(mismatched_exc)


def test_is_a14_epoch_guard_error_matches_prefix_only() -> None:
    assert is_a14_epoch_guard_error(
        OSError("contaminated_pre_a14_state: missing marker"),
    )
    assert not is_a14_epoch_guard_error(
        OSError("wrapper: contaminated_pre_a14_state: missing marker"),
    )
    assert not is_a14_epoch_guard_error(RuntimeError("other error"))
