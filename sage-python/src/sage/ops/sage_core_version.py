"""Inspect installed `sage_core` Rust binary build metadata.

Cycle-13 B Q1 (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=G follow-up
to Rust commit `b035973e`): `sage_core` now exposes 4 module-level
attributes (`__commit_sha__`, `__build_timestamp__`,
`__build_profile__`, `__version__`) populated at compile time. This
helper consumes them so operators can detect a stale binary BEFORE
running into the silent contract violations the cycle-13 B chain
just closed (engine.rs:1031 manifest-write fix shipped 2026-04-30
but a 2026-04-27 .pyd was still installed on a dev box).

Usage:
    # CLI: print JSON, exit 0 if installed binary matches source HEAD
    # (or `--allow-unknown` is set + binary commit_sha is "unknown").
    # Exit 1 on drift.
    python -m sage.ops.sage_core_version

    # Programmatic:
    from sage.ops import sage_core_version
    info = sage_core_version.check_freshness()
    if not info["matches"]:
        ...

The check is INFORMATIONAL: it does NOT fail boot, does NOT modify
state, and does NOT call the network. It runs `git rev-parse HEAD`
locally to find the source HEAD; when git is absent (PyPI install,
no `.git`) the comparison is unknown and exit code 0 is returned
unless `--strict` is set.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("sage.ops.sage_core_version")

_UNKNOWN = "unknown"

# Per cgpro HARD_STOP 2026-05-06 (conv `cgpro_pi_mono_pivot_20260505`):
# `get_source_head_sha()` MUST validate that the cwd's git root is a
# YGN-SAGE checkout before returning HEAD. Otherwise an operator
# running `python -m sage.ops.sage_core_version` from inside an
# unrelated git repo would compare the YGN-SAGE wheel's commit_sha
# against THAT repo's HEAD and falsely flag the wheel as stale.
#
# Sentinels: files that MUST exist in the git toplevel for it to be
# a YGN-SAGE checkout. Both files have existed since cycle-1; renaming
# either would break this guard, which is the desired behavior — the
# guard's job is to refuse comparison against a foreign repo.
_YGN_SAGE_SENTINEL_FILES: tuple[str, ...] = (
    "sage-core/Cargo.toml",
    "sage-python/src/sage/__init__.py",
)


def get_build_info() -> dict[str, str]:
    """Read the 4 build-info attributes from the installed sage_core.

    Falls back to `"unknown"` for any attribute the binary doesn't
    expose (older wheels predating the cycle-13 B Q1 fix shipped at
    Rust commit `b035973e`).
    """
    try:
        import sage_core
    except ImportError:
        return {
            "commit_sha": _UNKNOWN,
            "build_timestamp": _UNKNOWN,
            "build_profile": _UNKNOWN,
            "version": _UNKNOWN,
            "module_path": _UNKNOWN,
        }

    return {
        "commit_sha": str(getattr(sage_core, "__commit_sha__", _UNKNOWN)),
        "build_timestamp": str(getattr(sage_core, "__build_timestamp__", _UNKNOWN)),
        "build_profile": str(getattr(sage_core, "__build_profile__", _UNKNOWN)),
        "version": str(getattr(sage_core, "__version__", _UNKNOWN)),
        "module_path": str(getattr(sage_core, "__file__", _UNKNOWN)),
    }


def _git_toplevel() -> str | None:
    """Return absolute path of cwd's git toplevel, or None when absent."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    path = result.stdout.strip()
    return path or None


def _is_ygn_sage_checkout(toplevel: str) -> bool:
    """Validate `toplevel` looks like a YGN-SAGE git checkout via sentinels."""
    root = Path(toplevel)
    if not root.is_dir():
        return False
    return all((root / sentinel).is_file() for sentinel in _YGN_SAGE_SENTINEL_FILES)


def get_source_head_sha() -> str:
    """Return YGN-SAGE source repo's HEAD commit SHA, or `"unknown"`.

    Per cgpro HARD_STOP 2026-05-06: validates cwd's git toplevel is a
    YGN-SAGE checkout before returning HEAD. Otherwise an operator
    running this from another git repo would falsely flag the
    YGN-SAGE wheel as stale (the SHA would be the unrelated repo's).

    Returns `"unknown"` when:
      - git binary absent (FileNotFoundError).
      - cwd not in a git repo (CalledProcessError).
      - subprocess timeout.
      - git toplevel exists but doesn't have YGN-SAGE sentinel files
        (an unrelated repo's checkout).
      - HEAD lookup itself fails despite valid toplevel (corrupt
        ref, ENOENT race).

    Network-free.
    """
    toplevel = _git_toplevel()
    if toplevel is None or not _is_ygn_sage_checkout(toplevel):
        return _UNKNOWN
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=toplevel,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return _UNKNOWN
    sha = result.stdout.strip()
    return sha or _UNKNOWN


def _humanize_timestamp(unix_seconds_str: str) -> str:
    """Convert UNIX seconds string to ISO-8601 UTC. `unknown` -> `unknown`.

    Per cgpro deep VERIFY 2026-05-06 Q3: guards against OverflowError
    (year > 9999 / negative absurd values) and OSError (platform
    timestamp out-of-range). Returns `"unknown"` instead of crashing
    the ops CLI on corrupt build metadata.
    """
    if unix_seconds_str == _UNKNOWN:
        return _UNKNOWN
    try:
        ts = int(unix_seconds_str)
    except ValueError:
        return _UNKNOWN
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return _UNKNOWN


def check_freshness() -> dict[str, Any]:
    """Return a freshness report comparing installed binary vs source HEAD.

    Output dict (always present, never KeyError):
      - `commit_sha`: from `sage_core.__commit_sha__`.
      - `build_timestamp`: UNIX seconds string.
      - `build_timestamp_iso`: ISO-8601 UTC for human display.
      - `build_profile`: `"release"` / `"debug"` / other.
      - `version`: from `Cargo.toml [package].version`.
      - `module_path`: filesystem path to sage_core .pyd / .so.
      - `source_head_sha`: from `git rev-parse HEAD` in cwd.
      - `matches`: True | False | None.
        True = both SHAs known AND equal -> binary is fresh.
        False = both known AND different -> binary is STALE, rebuild.
        None = at least one is `unknown` -> can't compare.

    Treats `__build_profile__` as informational per cgpro Q4 — a
    `debug`-profile binary is not flagged as stale even though it is
    not the canonical release artifact.
    """
    info: dict[str, Any] = dict(get_build_info())
    info["build_timestamp_iso"] = _humanize_timestamp(info["build_timestamp"])
    info["source_head_sha"] = get_source_head_sha()

    binary_sha = info["commit_sha"]
    source_sha = info["source_head_sha"]
    if binary_sha == _UNKNOWN or source_sha == _UNKNOWN:
        info["matches"] = None
    else:
        info["matches"] = binary_sha == source_sha
    return info


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit 0 = fresh / unknown, 1 = stale (under --strict)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when matches=None (unknown source or binary SHA). "
        "Default: only exit 1 on confirmed stale (matches=False).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress JSON output. Useful for shell guards: "
        "`python -m sage.ops.sage_core_version --quiet || echo 'rebuild needed'`.",
    )
    args = parser.parse_args(argv)

    info = check_freshness()
    if not args.quiet:
        print(json.dumps(info, indent=2, sort_keys=True))

    matches = info["matches"]
    if matches is True:
        return 0
    if matches is False:
        return 1
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
