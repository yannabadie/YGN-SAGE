"""A14 posterior epoch guard for topology learning state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

REQUIRED_POSTERIOR_EPOCH = 1
A14_EPOCH_GUARD_ERROR_PREFIX = "contaminated_pre_a14_state:"
A14_BYPASS_ENV = "SAGE_BOOT_BYPASS_EPOCH_GUARD"
POSTERIOR_EPOCH_FILENAME = "posterior_epoch.json"
CONTAMINATED_MARKER_FILENAME = "_CONTAMINATED.json"

_TOPOLOGY_STATE_FILES: tuple[str, ...] = (
    "bandit_state.db",
    "bandit_state.db-wal",
    "bandit_state.db-shm",
    "archive_state.db",
    "archive_state.db-wal",
    "archive_state.db-shm",
    "engine_extras.json",
)

_EpochStatus = Literal["missing", "match", "mismatch", "malformed"]

_log = logging.getLogger("sage.posterior_epoch")


@dataclass(frozen=True)
class EpochCheck:
    state_dir: Path
    state_files: tuple[str, ...]
    epoch_status: _EpochStatus
    file_epoch: int | None
    bypass_used: bool = False


def check_posterior_epoch_for_boot(state_dir: Path) -> EpochCheck:
    state_dir = Path(state_dir).expanduser()
    state_files = _topology_state_files(state_dir)
    status, file_epoch, malformed_reason = _read_epoch_status(state_dir)

    if (state_dir / CONTAMINATED_MARKER_FILENAME).exists():
        return _allow_bypass_or_raise(
            state_dir,
            state_files,
            status,
            file_epoch,
            _poison_pill_error(state_dir),
        )

    if not state_files:
        if status != "missing":
            _log.warning(
                "a14_epoch_guard_epoch_without_state layer=python state_dir=%s "
                "required_epoch=%s file_epoch=%s",
                state_dir,
                REQUIRED_POSTERIOR_EPOCH,
                _file_epoch_for_log(status, file_epoch),
            )
        return EpochCheck(state_dir, state_files, status, file_epoch)

    if status == "match":
        return EpochCheck(state_dir, state_files, status, file_epoch)
    if status == "missing":
        error = _missing_epoch_error(state_dir, state_files)
    elif status == "mismatch":
        error = _epoch_mismatch_error(state_dir, state_files, file_epoch)
    else:
        error = _malformed_epoch_error(state_dir, state_files, malformed_reason)

    return _allow_bypass_or_raise(state_dir, state_files, status, file_epoch, error)


def is_a14_epoch_guard_error(exc: BaseException) -> bool:
    return str(exc).startswith("contaminated_pre_a14_state:")


def ensure_clean_epoch_before_save(state_dir: Path) -> None:
    state_dir = Path(state_dir).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_files = _topology_state_files(state_dir)

    if (state_dir / CONTAMINATED_MARKER_FILENAME).exists():
        raise RuntimeError(_poison_pill_error(state_dir))

    status, file_epoch, malformed_reason = _read_epoch_status(state_dir)
    if status == "match":
        return
    if status == "missing" and not state_files:
        _write_clean_epoch_marker(state_dir)
        return
    if status == "missing":
        raise RuntimeError(_missing_epoch_error(state_dir, state_files))
    if status == "mismatch":
        raise RuntimeError(_epoch_mismatch_error(state_dir, state_files, file_epoch))
    raise RuntimeError(_malformed_epoch_error(state_dir, state_files, malformed_reason))


def _topology_state_files(state_dir: Path) -> tuple[str, ...]:
    return tuple(name for name in _TOPOLOGY_STATE_FILES if (state_dir / name).exists())


def _read_epoch_status(state_dir: Path) -> tuple[_EpochStatus, int | None, str | None]:
    epoch_path = state_dir / POSTERIOR_EPOCH_FILENAME
    if not epoch_path.exists():
        return ("missing", None, None)
    try:
        raw = epoch_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return ("malformed", None, _short_reason(str(exc)))

    if not isinstance(payload, dict):
        return ("malformed", None, "posterior_epoch.json must contain an object")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        return ("malformed", None, "epoch must be an integer")
    if epoch == REQUIRED_POSTERIOR_EPOCH:
        return ("match", epoch, None)
    return ("mismatch", epoch, None)


def _allow_bypass_or_raise(
    state_dir: Path,
    state_files: tuple[str, ...],
    status: _EpochStatus,
    file_epoch: int | None,
    error: str,
) -> EpochCheck:
    if _bypass_enabled():
        _log.warning(
            "a14_epoch_guard_bypass layer=python state_dir=%s required_epoch=%s "
            "file_epoch=%s state_files=%s operator=%s reason=%s",
            state_dir,
            REQUIRED_POSTERIOR_EPOCH,
            _file_epoch_for_log(status, file_epoch),
            ",".join(state_files),
            _operator_id(),
            _bypass_reason(),
        )
        return EpochCheck(state_dir, state_files, status, file_epoch, bypass_used=True)
    raise RuntimeError(error)


def _write_clean_epoch_marker(state_dir: Path) -> None:
    payload = {
        "epoch": REQUIRED_POSTERIOR_EPOCH,
        "started_utc": _utc_now(),
        "reason": "auto-created clean topology posterior epoch before first save_state",
        "policy": (
            "all bandit/MAP-Elites updates for this state are post-A14 clean-epoch updates"
        ),
    }
    _write_json_atomic(state_dir / POSTERIOR_EPOCH_FILENAME, payload)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _missing_epoch_error(state_dir: Path, state_files: tuple[str, ...]) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} explicit epoch required when topology state exists; "
        f"state_dir={state_dir}; state_files={','.join(state_files)}; "
        f"epoch_file=missing; required_epoch={REQUIRED_POSTERIOR_EPOCH}; "
        f"bypass_env={A14_BYPASS_ENV}"
    )


def _epoch_mismatch_error(
    state_dir: Path,
    state_files: tuple[str, ...],
    file_epoch: int | None,
) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} epoch mismatch: file={file_epoch} "
        f"required={REQUIRED_POSTERIOR_EPOCH}; state_dir={state_dir}; "
        f"state_files={','.join(state_files)}; bypass_env={A14_BYPASS_ENV}"
    )


def _malformed_epoch_error(
    state_dir: Path,
    state_files: tuple[str, ...],
    reason: str | None,
) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} posterior_epoch.json malformed: "
        f"{_short_reason(reason or 'unknown')}; state_dir={state_dir}; "
        f"state_files={','.join(state_files)}; required_epoch={REQUIRED_POSTERIOR_EPOCH}; "
        f"bypass_env={A14_BYPASS_ENV}"
    )


def _poison_pill_error(state_dir: Path) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} poison pill marker present in state dir: "
        f"{CONTAMINATED_MARKER_FILENAME}; state_dir={state_dir}; bypass_env={A14_BYPASS_ENV}"
    )


def _file_epoch_for_log(status: _EpochStatus, file_epoch: int | None) -> str:
    if status == "missing":
        return "missing"
    if status == "malformed":
        return "malformed"
    return str(file_epoch)


def _short_reason(reason: str) -> str:
    return reason.replace("\r", " ").replace("\n", " ").replace(";", " ")[:180]


def _bypass_enabled() -> bool:
    return os.environ.get(A14_BYPASS_ENV) == "1"


def _operator_id() -> str:
    return (
        os.environ.get("SAGE_OPERATOR_ID")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "unknown"
    )


def _bypass_reason() -> str:
    return os.environ.get("SAGE_BOOT_BYPASS_REASON") or "unspecified"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
