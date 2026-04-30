"""A14 reset utility for topology posterior state."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence
from uuid import uuid4

from sage.posterior_epoch import (
    CONTAMINATED_MARKER_FILENAME,
    POSTERIOR_EPOCH_FILENAME,
    REQUIRED_POSTERIOR_EPOCH,
)

_A14_TOPOLOGY_STATE_FILES: tuple[str, ...] = (
    "bandit_state.db",
    "bandit_state.db-wal",
    "bandit_state.db-shm",
    "archive_state.db",
    "archive_state.db-wal",
    "archive_state.db-shm",
    "engine_extras.json",
)
_RESET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RESTORE_POLICY = (
    "forensic-only; do not copy into ~/.sage; load only with "
    "SAGE_BOOT_BYPASS_EPOCH_GUARD=1 in a non-default state dir"
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reason = args.reason.strip()
    if not reason:
        raise SystemExit("--reason must be non-empty")

    mark_existing = (
        Path(args.mark_existing_contaminated_dir).expanduser().resolve()
        if args.mark_existing_contaminated_dir
        else None
    )
    reset_id = args.reset_id or _default_reset_id(mark_existing)
    _validate_reset_id(reset_id)
    audit_dir = Path(args.audit_dir or _default_audit_dir(reset_id)).expanduser()

    if mark_existing is not None:
        _mark_existing_contaminated_dir(mark_existing, reset_id, audit_dir, reason)
        return 0

    state_dir = Path(args.state_dir).expanduser().resolve()
    if args.dry_run:
        _print_json(
            {
                "dry_run": True,
                "state_dir": str(state_dir),
                "reset_id": reset_id,
                "audit_dir": str(audit_dir),
                "state_files": _existing_state_files(state_dir),
            },
        )
        return 0

    _run_reset(state_dir, reset_id, audit_dir, reason)
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="~/.sage")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--reset-id")
    parser.add_argument("--audit-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mark-existing-contaminated-dir")
    return parser.parse_args(argv)


def _run_reset(state_dir: Path, reset_id: str, audit_dir: Path, reason: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with _ResetLock(state_dir):
        state_files = _existing_state_files(state_dir)
        commit = _git_commit()
        manifest_path = _write_audit_manifest(state_dir, audit_dir, reset_id, reason, commit)
        manifest_sha = sha256_file(manifest_path)

        backup_root = state_dir / "contaminated"
        backup_root.mkdir(parents=True, exist_ok=True)
        final_backup_dir = backup_root / reset_id
        if final_backup_dir.exists():
            raise SystemExit(f"backup dir already exists: {final_backup_dir}")

        temp_backup_dir = backup_root / f".tmp_{reset_id}_{uuid4().hex}"
        temp_backup_dir.mkdir()
        moved_files: list[str] = []
        try:
            _assert_same_filesystem_for_atomic_reset(
                state_dir,
                backup_root,
                temp_backup_dir,
            )
            for filename in state_files:
                source = state_dir / filename
                if source.exists():
                    os.replace(source, temp_backup_dir / filename)
                    moved_files.append(filename)

            _write_contaminated_marker(
                temp_backup_dir,
                reset_id,
                reason,
                moved_files,
                manifest_path,
                manifest_sha,
                commit,
            )
            os.replace(temp_backup_dir, final_backup_dir)
            _write_active_epoch(state_dir, reset_id, final_backup_dir, manifest_path, reason, commit)
        except Exception:
            if temp_backup_dir.exists():
                shutil.rmtree(temp_backup_dir, ignore_errors=True)
            raise


def _write_audit_manifest(
    state_dir: Path,
    audit_dir: Path,
    reset_id: str,
    reason: str,
    commit: str,
) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = audit_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    artifacts: list[dict[str, object]] = []

    for filename in _existing_state_files(state_dir):
        source = state_dir / filename
        audit_copy = artifacts_dir / filename
        shutil.copy2(source, audit_copy)
        artifacts.append(
            {
                "name": filename,
                "source_path": str(source),
                "audit_path": str(audit_copy),
                "sha256": sha256_file(audit_copy),
                "size_bytes": audit_copy.stat().st_size,
            },
        )

    manifest_path = audit_dir / "MANIFEST.json"
    manifest = {
        "reset_id": reset_id,
        "created_at_utc": _utc_now(),
        "reason": reason,
        "state_dir": str(state_dir),
        "source_epoch": 0,
        "source_epoch_status": "legacy_pre_a14_unknown",
        "target_epoch": REQUIRED_POSTERIOR_EPOCH,
        "commit_at_reset": commit,
        "artifacts": artifacts,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _write_contaminated_marker(
    backup_dir: Path,
    reset_id: str,
    reason: str,
    state_files: list[str],
    manifest_path: Path | None,
    manifest_sha: str | None,
    commit: str,
) -> None:
    marker = _contaminated_marker_payload(
        reset_id,
        reason,
        state_files,
        manifest_path,
        manifest_sha,
        commit,
    )
    _write_json_atomic(backup_dir / CONTAMINATED_MARKER_FILENAME, marker)


def _write_active_epoch(
    state_dir: Path,
    reset_id: str,
    final_backup_dir: Path,
    manifest_path: Path,
    reason: str,
    commit: str,
) -> None:
    payload = {
        "epoch": REQUIRED_POSTERIOR_EPOCH,
        "started_utc": _utc_now(),
        "reason": reason,
        "predecessor_state": f"moved to {final_backup_dir}",
        "audit_dump": str(manifest_path),
        "first_clean_run_after": None,
        "policy": (
            "all bandit/MAP-Elites updates after this point come from "
            "oracle.trainable=True verdicts only"
        ),
        "commit_at_reset": commit,
    }
    _write_json_atomic(state_dir / POSTERIOR_EPOCH_FILENAME, payload)


def _mark_existing_contaminated_dir(
    contaminated_dir: Path,
    reset_id: str,
    audit_dir: Path,
    reason: str,
) -> None:
    if not contaminated_dir.is_dir():
        raise SystemExit(f"contaminated dir does not exist: {contaminated_dir}")
    manifest_path = audit_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"audit manifest missing: {manifest_path}")
    _write_contaminated_marker(
        contaminated_dir,
        reset_id,
        reason,
        _existing_state_files(contaminated_dir),
        manifest_path,
        sha256_file(manifest_path),
        _git_commit(),
    )


def _contaminated_marker_payload(
    reset_id: str,
    reason: str,
    state_files: list[str],
    manifest_path: Path | None,
    manifest_sha: str | None,
    commit: str,
) -> dict[str, object]:
    if manifest_path is not None and manifest_sha is None:
        raise ValueError("audit_dump_sha256 is required when audit_dump is non-null")
    return {
        "marker_type": "YGN-SAGE_A14_CONTAMINATED_TOPOLOGY_STATE",
        "contaminated": True,
        "do_not_restore_without_manual_override": True,
        "reset_id": reset_id,
        "source_epoch": 0,
        "source_epoch_status": "legacy_pre_a14_unknown",
        "target_epoch": REQUIRED_POSTERIOR_EPOCH,
        "moved_at_utc": _utc_now(),
        "reason": reason,
        "state_files": state_files,
        "audit_dump": str(manifest_path) if manifest_path is not None else None,
        "audit_dump_sha256": manifest_sha,
        "commit_at_reset": commit,
        "restore_policy": _RESTORE_POLICY,
    }


def _existing_state_files(state_dir: Path) -> list[str]:
    return [filename for filename in _A14_TOPOLOGY_STATE_FILES if (state_dir / filename).exists()]


def _assert_same_filesystem_for_atomic_reset(
    state_dir: Path,
    backup_root: Path,
    temp_backup_dir: Path,
) -> None:
    state_dir_dev = state_dir.stat().st_dev
    backup_root_dev = backup_root.stat().st_dev
    temp_backup_dev = temp_backup_dir.stat().st_dev
    if not (state_dir_dev == backup_root_dev == temp_backup_dev):
        raise RuntimeError(
            "a14_reset: cross-filesystem detected; "
            f"state_dir.st_dev={state_dir_dev} backup_root.st_dev={backup_root_dev} "
            f"temp_backup_dir.st_dev={temp_backup_dev}; "
            "os.replace atomicity not guaranteed",
        )


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


def _default_reset_id(mark_existing: Path | None) -> str:
    if mark_existing is not None:
        name = mark_existing.name
        if name.startswith("contaminated_"):
            candidate = name.removeprefix("contaminated_")
            if _RESET_ID_RE.fullmatch(candidate):
                return candidate
        if _RESET_ID_RE.fullmatch(name):
            return name
    return f"pre_a14_{datetime.now(UTC):%Y%m%d}"


def _default_audit_dir(reset_id: str) -> str:
    match = re.search(r"(\d{8})$", reset_id)
    stamp = match.group(1) if match else f"{datetime.now(UTC):%Y%m%d}"
    return f".tmp/a14_reset_{stamp}"


def _validate_reset_id(reset_id: str) -> None:
    if not _RESET_ID_RE.fullmatch(reset_id):
        raise SystemExit("reset-id must match [a-z0-9][a-z0-9_-]{0,63}")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


class _ResetLock:
    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / ".a14_reset.lock"
        self.fd: int | None = None

    def __enter__(self) -> _ResetLock:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self.fd = os.open(self.path, flags)
            os.write(self.fd, str(os.getpid()).encode("ascii"))
        except FileExistsError as exc:
            raise SystemExit(f"A14 reset lock already exists: {self.path}") from exc
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
