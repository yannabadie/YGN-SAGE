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
import time
from typing import Sequence
from uuid import uuid4

from sage.posterior_epoch import (
    CONTAMINATED_MARKER_FILENAME,
    POSTERIOR_EPOCH_FILENAME,
    REQUIRED_POSTERIOR_EPOCH,
    TOPOLOGY_STATE_MANIFEST_FILENAME,
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

# Cycle-13 B Q4-bis (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=I):
# files written through atomic-rename (`write_bytes_atomic` in Rust at
# `sage-core/src/topology/posterior_epoch.rs` + `_write_json_atomic`
# in this module) leave `.<name>.<id>.tmp` files behind on rename
# failure. Pre `bc662d9a` Rust fix the .tmp was leaked; pre this
# Python fix nothing cleaned them up. Operators on long-lived state
# dirs can accumulate stale tmps. List the files YGN-SAGE writes
# atomically; the cleanup glob `.<name>.*.tmp` matches both ulid
# (Rust write_bytes_atomic) and uuid4-hex (Python _write_json_atomic)
# id formats — both are alphanumeric and the trailing `.tmp` is
# unambiguous. Source of truth for the contaminated marker filename
# is `sage.posterior_epoch.CONTAMINATED_MARKER_FILENAME`; it is
# imported above and reused here so a future rename only happens in
# one place.
_ATOMIC_WRITTEN_NAMES: tuple[str, ...] = (
    POSTERIOR_EPOCH_FILENAME,
    TOPOLOGY_STATE_MANIFEST_FILENAME,
    CONTAMINATED_MARKER_FILENAME,
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
                "orphan_tmp_files_to_clean": _orphan_tmp_files(state_dir),
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
        # Cycle-13 B Q4-bis: clean up orphaned `.<name>.<id>.tmp` files
        # left by pre-`bc662d9a` atomic-rename failures BEFORE we
        # gather the canonical state file list. The cleanup is
        # idempotent + best-effort; the resulting list is recorded in
        # the audit manifest for forensic tracking. Cleaning at this
        # point (post-lock, pre-state-gather) means concurrent reset
        # attempts can't race over the same orphans, and operators
        # can audit which orphan files this reset removed via the
        # manifest's `cleaned_orphan_tmp_files` field.
        cleaned_orphans = _cleanup_orphaned_tmp_files(state_dir)

        state_files = _existing_state_files(state_dir)
        commit = _git_commit()
        manifest_path = _write_audit_manifest(
            state_dir,
            audit_dir,
            reset_id,
            reason,
            commit,
            cleaned_orphan_tmp_files=cleaned_orphans,
        )
        manifest_sha = sha256_file(manifest_path)

        backup_root = state_dir / "contaminated"
        backup_root.mkdir(parents=True, exist_ok=True)
        final_backup_dir = backup_root / reset_id
        if final_backup_dir.exists():
            raise SystemExit(f"backup dir already exists: {final_backup_dir}")

        temp_backup_dir = backup_root / f".tmp_{reset_id}_{uuid4().hex}"
        temp_backup_dir.mkdir()
        moved_files: list[str] = []
        finalized_backup = False
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
            _replace_dir_atomic_with_retry(temp_backup_dir, final_backup_dir)
            finalized_backup = True
            _write_active_epoch(state_dir, reset_id, final_backup_dir, manifest_path, reason, commit)
        except Exception as exc:
            if moved_files:
                _write_failed_reset_poison_marker(
                    state_dir=state_dir,
                    reset_id=reset_id,
                    reason=reason,
                    moved_files=moved_files,
                    manifest_path=manifest_path,
                    manifest_sha=manifest_sha,
                    commit=commit,
                    temp_backup_dir=temp_backup_dir,
                    final_backup_dir=final_backup_dir if finalized_backup else None,
                    exc=exc,
                )
            elif temp_backup_dir.exists():
                shutil.rmtree(temp_backup_dir, ignore_errors=True)
            raise


def _replace_dir_atomic_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 6,
    delay_seconds: float = 0.02,
) -> None:
    """Atomically rename a backup dir, retrying transient Windows handle races."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if (
                attempt == attempts - 1
                or not _is_windows_transient_replace_error(exc, target)
            ):
                raise
            time.sleep(delay_seconds * (2**attempt))


def _is_windows_transient_replace_error(exc: OSError, target: Path) -> bool:
    if os.name != "nt":
        return False
    if target.exists():
        return False
    return getattr(exc, "winerror", None) in {5, 32}


def _write_failed_reset_poison_marker(
    *,
    state_dir: Path,
    reset_id: str,
    reason: str,
    moved_files: list[str],
    manifest_path: Path,
    manifest_sha: str,
    commit: str,
    temp_backup_dir: Path,
    final_backup_dir: Path | None,
    exc: BaseException,
) -> None:
    marker = _contaminated_marker_payload(
        reset_id,
        reason,
        moved_files,
        manifest_path,
        manifest_sha,
        commit,
    )
    marker.update(
        {
            "marker_type": "YGN-SAGE_A14_FAILED_RESET_POISON",
            "reset_failure": True,
            "failure_stage": (
                "active_epoch_write" if final_backup_dir is not None else "backup_finalize"
            ),
            "preserved_temp_backup_dir": (
                str(temp_backup_dir) if temp_backup_dir.exists() else None
            ),
            "final_backup_dir": (
                str(final_backup_dir)
                if final_backup_dir is not None and final_backup_dir.exists()
                else None
            ),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    )
    try:
        _write_json_atomic(state_dir / CONTAMINATED_MARKER_FILENAME, marker)
    except Exception:
        # Preserve the original reset failure; the moved temp/final backup remains
        # in place for operator forensics even if the poison marker write fails.
        pass


def _write_audit_manifest(
    state_dir: Path,
    audit_dir: Path,
    reset_id: str,
    reason: str,
    commit: str,
    cleaned_orphan_tmp_files: list[str] | None = None,
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
    manifest: dict[str, object] = {
        "reset_id": reset_id,
        "created_at_utc": _utc_now(),
        "reason": reason,
        "state_dir": str(state_dir),
        "source_epoch": 0,
        "source_epoch_status": "legacy_pre_a14_unknown",
        "target_epoch": REQUIRED_POSTERIOR_EPOCH,
        "commit_at_reset": commit,
        "artifacts": artifacts,
        # Cycle-13 B Q4-bis: forensic record of orphaned .tmp files
        # this reset cleaned. Empty list when none were present (the
        # common case post-bc662d9a Rust fix). Pre-fix dirs may show
        # `.posterior_epoch.json.<ulid>.tmp` and similar.
        "cleaned_orphan_tmp_files": list(cleaned_orphan_tmp_files or []),
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


def _orphan_tmp_files(state_dir: Path) -> list[str]:
    """List orphaned `.<name>.<id>.tmp` files in `state_dir`.

    Cycle-13 B Q4-bis (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=I).
    Files in `_ATOMIC_WRITTEN_NAMES` are written via atomic-rename
    (`.<name>.<id>.tmp` -> rename -> `<name>`); a rename failure pre
    `bc662d9a` left the `.tmp` behind. This helper finds them so
    `_cleanup_orphaned_tmp_files` can remove them or `--dry-run` can
    list them without acting.

    Glob pattern: `.<name>.*.tmp`. The dot prefix + `.tmp` suffix
    bracket is unambiguous; `*` matches both ulid (26-char Crockford
    base32 uppercase, written by Rust `write_bytes_atomic`) and
    uuid4-hex (32-char lowercase, written by Python
    `_write_json_atomic`) id shapes. Returns sorted basenames
    relative to `state_dir`. Non-existent state_dir or non-directory
    inputs return [].
    """
    if not state_dir.is_dir():
        return []
    found: list[str] = []
    for name in _ATOMIC_WRITTEN_NAMES:
        for orphan in state_dir.glob(f".{name}.*.tmp"):
            if orphan.is_file():
                found.append(orphan.name)
    return sorted(found)


def _cleanup_orphaned_tmp_files(state_dir: Path) -> list[str]:
    """Best-effort cleanup of orphaned `.<name>.<id>.tmp` files.

    Cycle-13 B Q4-bis follow-up to Rust commit `bc662d9a` (closure-
    wrapped `write_bytes_atomic` cleanup): the Rust fix prevents
    future leaks; this Python ops cleanup removes PRE-FIX artifacts
    already present in operator state dirs.

    Each `unlink()` failure is silently swallowed (permission error /
    in-use file / TOCTOU race) — the primary reset operation must
    not be blocked by best-effort cleanup of pre-existing artifacts.
    Per cgpro deep VERIFY 2026-05-06 Q2: cleanup failures are
    secondary; the primary error (or success) is what callers need
    to see.

    Returns: sorted list of basenames actually removed.
    """
    removed: list[str] = []
    for orphan_name in _orphan_tmp_files(state_dir):
        orphan_path = state_dir / orphan_name
        try:
            orphan_path.unlink()
            removed.append(orphan_name)
        except OSError:
            # Best-effort: silently swallow per Q2 verdict.
            pass
    return removed


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
