"""A14 posterior epoch guard for topology learning state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
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
TOPOLOGY_STATE_MANIFEST_FILENAME = "topology_state_manifest.json"
TOPOLOGY_STATE_MANIFEST_TYPE = "YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST"
SMMU_STATE_FILENAME = "smmu_state.json"

_TOPOLOGY_STATE_FILES: tuple[str, ...] = (
    "bandit_state.db",
    "bandit_state.db-wal",
    "bandit_state.db-shm",
    "archive_state.db",
    "archive_state.db-wal",
    "archive_state.db-shm",
    "engine_extras.json",
    SMMU_STATE_FILENAME,
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
        manifest_error = verify_state_files_against_manifest(
            state_dir,
            load_topology_state_manifest(state_dir),
        )
        if manifest_error is None:
            return EpochCheck(state_dir, state_files, status, file_epoch)
        return _allow_bypass_or_raise(
            state_dir,
            state_files,
            status,
            file_epoch,
            manifest_error,
        )
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
    if _bypass_enabled():
        raise RuntimeError(_bypass_save_error(state_dir))

    state_dir.mkdir(parents=True, exist_ok=True)
    state_files = _topology_state_files(state_dir)

    if (state_dir / CONTAMINATED_MARKER_FILENAME).exists():
        raise RuntimeError(_poison_pill_error(state_dir))

    status, file_epoch, malformed_reason = _read_epoch_status(state_dir)
    if status == "match":
        if state_files:
            manifest_error = verify_state_files_against_manifest(
                state_dir,
                load_topology_state_manifest(state_dir),
            )
            if manifest_error is not None:
                raise RuntimeError(manifest_error)
        return
    if status == "missing" and not state_files:
        _write_clean_epoch_marker(state_dir)
        return
    if status == "missing":
        raise RuntimeError(_missing_epoch_error(state_dir, state_files))
    if status == "mismatch":
        raise RuntimeError(_epoch_mismatch_error(state_dir, state_files, file_epoch))
    raise RuntimeError(_malformed_epoch_error(state_dir, state_files, malformed_reason))


def load_topology_state_manifest(state_dir: Path) -> dict[str, object] | str:
    manifest_path = Path(state_dir) / TOPOLOGY_STATE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return _manifest_missing_error(Path(state_dir), _topology_state_files(Path(state_dir)))
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _manifest_malformed_error(Path(state_dir), _short_reason(str(exc)))
    if not isinstance(payload, dict):
        return _manifest_malformed_error(
            Path(state_dir),
            "topology_state_manifest.json must contain an object",
        )
    return payload


def verify_state_files_against_manifest(
    state_dir: Path,
    manifest: dict[str, object] | str,
) -> str | None:
    state_dir = Path(state_dir)
    state_files = _topology_state_files(state_dir)
    if isinstance(manifest, str):
        return manifest

    manifest_type = manifest.get("manifest_type")
    if manifest_type != TOPOLOGY_STATE_MANIFEST_TYPE:
        return _manifest_malformed_error(
            state_dir,
            "manifest_type must be YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST",
        )

    epoch = manifest.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        return _manifest_malformed_error(state_dir, "epoch must be an integer")
    if epoch != REQUIRED_POSTERIOR_EPOCH:
        return _manifest_epoch_mismatch_error(state_dir, state_files, epoch)

    raw_entries = manifest.get("state_files")
    if not isinstance(raw_entries, list):
        return _manifest_malformed_error(state_dir, "state_files must be a list")

    entries: dict[str, tuple[str, int]] = {}
    allowed_names = set(_TOPOLOGY_STATE_FILES)
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            return _manifest_malformed_error(state_dir, "state_files entries must be objects")
        name = raw_entry.get("name")
        sha256 = raw_entry.get("sha256")
        size_bytes = raw_entry.get("size_bytes")
        if not isinstance(name, str) or not name:
            return _manifest_malformed_error(state_dir, "state_files[].name must be a string")
        if name not in allowed_names:
            return _manifest_file_set_error(state_dir, state_files, sorted([*entries, name]))
        if name in entries:
            return _manifest_malformed_error(state_dir, f"duplicate state file entry: {name}")
        if not isinstance(sha256, str) or not _is_sha256_hex(sha256):
            return _manifest_malformed_error(
                state_dir,
                f"state_files[].sha256 malformed for {name}",
            )
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            return _manifest_malformed_error(
                state_dir,
                f"state_files[].size_bytes malformed for {name}",
            )
        entries[name] = (sha256, size_bytes)

    manifest_names = sorted(entries)
    expected_names = sorted(state_files)
    if manifest_names != expected_names:
        return _manifest_file_set_error(state_dir, state_files, manifest_names)

    for name in state_files:
        path = state_dir / name
        expected_sha256, expected_size = entries[name]
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            return _manifest_malformed_error(state_dir, f"stat {name}: {_short_reason(str(exc))}")
        if actual_size != expected_size:
            return _manifest_size_mismatch_error(
                state_dir,
                name,
                expected_size,
                actual_size,
            )
        try:
            actual_sha256 = _sha256_file(path)
        except OSError as exc:
            return _manifest_malformed_error(state_dir, f"read {name}: {_short_reason(str(exc))}")
        if actual_sha256 != expected_sha256:
            return _manifest_sha256_mismatch_error(
                state_dir,
                name,
                expected_sha256,
                actual_sha256,
            )
    return None


def write_topology_state_manifest(
    state_dir: Path,
    *,
    writer: str = "TopologyEngine::save_state",
) -> None:
    state_dir = Path(state_dir)
    state_files = _topology_state_files(state_dir)
    entries = [
        {
            "name": name,
            "sha256": _sha256_file(state_dir / name),
            "size_bytes": (state_dir / name).stat().st_size,
        }
        for name in state_files
    ]
    payload = {
        "manifest_type": TOPOLOGY_STATE_MANIFEST_TYPE,
        "epoch": REQUIRED_POSTERIOR_EPOCH,
        "state_generation_id": uuid4().hex,
        "created_at_utc": _utc_now(),
        "writer": writer,
        "state_files": entries,
    }
    _write_json_atomic(state_dir / TOPOLOGY_STATE_MANIFEST_FILENAME, payload)


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
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if fd != -1:
            os.close(fd)
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _bypass_save_error(state_dir: Path) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} save disabled while {A14_BYPASS_ENV}=1; "
        f"state_dir={state_dir}"
    )


def _manifest_missing_error(state_dir: Path, state_files: tuple[str, ...]) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} state files present but "
        f"{TOPOLOGY_STATE_MANIFEST_FILENAME} missing; state_dir={state_dir}; "
        f"state_files={','.join(state_files)}; required_epoch={REQUIRED_POSTERIOR_EPOCH}; "
        f"bypass_env={A14_BYPASS_ENV}"
    )


def _manifest_malformed_error(state_dir: Path, reason: str) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} {TOPOLOGY_STATE_MANIFEST_FILENAME} malformed: "
        f"{_short_reason(reason)}; state_dir={state_dir}; "
        f"required_epoch={REQUIRED_POSTERIOR_EPOCH}; bypass_env={A14_BYPASS_ENV}"
    )


def _manifest_epoch_mismatch_error(
    state_dir: Path,
    state_files: tuple[str, ...],
    epoch: int,
) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} {TOPOLOGY_STATE_MANIFEST_FILENAME} epoch mismatch: "
        f"file={epoch} required={REQUIRED_POSTERIOR_EPOCH}; state_dir={state_dir}; "
        f"state_files={','.join(state_files)}; bypass_env={A14_BYPASS_ENV}"
    )


def _manifest_file_set_error(
    state_dir: Path,
    state_files: tuple[str, ...],
    manifest_names: list[str],
) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} {TOPOLOGY_STATE_MANIFEST_FILENAME} "
        f"state file set mismatch; state_dir={state_dir}; "
        f"state_files={','.join(state_files)}; manifest_files={','.join(manifest_names)}; "
        f"required_epoch={REQUIRED_POSTERIOR_EPOCH}; bypass_env={A14_BYPASS_ENV}"
    )


def _manifest_size_mismatch_error(
    state_dir: Path,
    name: str,
    expected: int,
    actual: int,
) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} {TOPOLOGY_STATE_MANIFEST_FILENAME} "
        f"size_bytes mismatch on {name}; expected={expected} actual={actual}; "
        f"state_dir={state_dir}; bypass_env={A14_BYPASS_ENV}"
    )


def _manifest_sha256_mismatch_error(
    state_dir: Path,
    name: str,
    expected: str,
    actual: str,
) -> str:
    return (
        f"{A14_EPOCH_GUARD_ERROR_PREFIX} {TOPOLOGY_STATE_MANIFEST_FILENAME} "
        f"sha256 mismatch on {name}; expected={expected} actual={actual}; "
        f"state_dir={state_dir}; bypass_env={A14_BYPASS_ENV}"
    )


def _file_epoch_for_log(status: _EpochStatus, file_epoch: int | None) -> str:
    if status == "missing":
        return "missing"
    if status == "malformed":
        return "malformed"
    return str(file_epoch)


def _short_reason(reason: str) -> str:
    return reason.replace("\r", " ").replace("\n", " ").replace(";", " ")[:180]


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


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
