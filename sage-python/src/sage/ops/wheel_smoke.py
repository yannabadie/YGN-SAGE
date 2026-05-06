"""Post-install wheel smoke for `sage_core`.

Cycle-13 B Q2 follow-up (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=H):
asserts the runtime contract on the INSTALLED `sage_core` wheel —
not on the source repo. Source-current pytest tests
(`test_save_state_manifest_contract.py`,
`test_sage_core_version.py`) pass on every CI run because CI builds
fresh wheels per commit. But a published TestPyPI / PyPI wheel can
still fail the contract if the build pipeline has a glitch (build.rs
git-resolve falls through to "unknown"; embedded RustPython artifact
missing; manifest write at `engine.rs:1031` lost in a rebase, etc).

This smoke is the LAST CHECK before a wheel becomes operationally
trusted: install + import + assert. Runs in a fresh venv after
`pip install <wheel>` in CI (wheels.yml + release-test.yml).
Requires NO pytest, NO source repo — pure stdlib + sage_core.

Exit code:
  0 = all checks passed.
  1 = at least one check failed; structured JSON report goes to
      stderr explaining each failure for CI logs.

Usage (CI):
  python -m sage.ops.wheel_smoke

Programmatic:
  from sage.ops import wheel_smoke
  report = wheel_smoke.run()
  assert report["ok"], report["failures"]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("sage.ops.wheel_smoke")

# Reuse the source-of-truth file names from sage.posterior_epoch.
# When that module's constants move, this smoke follows automatically.
from sage.posterior_epoch import (  # noqa: E402
    POSTERIOR_EPOCH_FILENAME,
    REQUIRED_POSTERIOR_EPOCH,
    TOPOLOGY_STATE_MANIFEST_FILENAME,
)

_REQUIRED_BUILD_INFO_ATTRS: tuple[str, ...] = (
    "__commit_sha__",
    "__build_timestamp__",
    "__build_profile__",
    "__version__",
)

# Canonical pyclasses that MUST be exposed by the default-feature wheel.
# Mirrors the existing wheels.yml inline smoke list (kept in sync).
_REQUIRED_DEFAULT_SYMBOLS: tuple[str, ...] = (
    "TopologyEngine",
    "SystemRouter",
    "ContextualBandit",
    "ModelAssigner",
    "ModelRegistry",
    "MultiViewMMU",
    "WorkingMemory",
    "ToolExecutor",
)


def _check_sage_core_imports() -> dict[str, Any]:
    """Phase 1: sage_core imports cleanly."""
    try:
        import sage_core  # noqa: F401

        return {"ok": True, "module_path": getattr(sage_core, "__file__", "?")}
    except ImportError as exc:
        return {"ok": False, "error": f"ImportError: {exc}"}


def _check_build_info_attrs() -> dict[str, Any]:
    """Phase 2: 4 build-info attributes present + at least commit_sha
    + version are not "unknown" (the build went through build.rs).

    Per cgpro deep VERIFY 2026-05-06 (cycle-13 G round 1): build.rs
    falls back to "unknown" for `__commit_sha__` only when git is
    absent at build time. CI builds always have git available — a
    PyPI wheel with `__commit_sha__` == "unknown" indicates a build
    pipeline glitch (the CI runner lost git access, or the build
    cwd isn't a git checkout).
    """
    import sage_core

    failures: list[str] = []
    values: dict[str, str] = {}
    for attr in _REQUIRED_BUILD_INFO_ATTRS:
        if not hasattr(sage_core, attr):
            failures.append(f"missing attribute: sage_core.{attr}")
            continue
        values[attr] = str(getattr(sage_core, attr))

    # commit_sha + version must NOT be "unknown" — those are CI smell.
    # build_timestamp can be a UNIX seconds string (which won't match
    # the literal "unknown"). build_profile is informational.
    if values.get("__commit_sha__") == "unknown":
        failures.append(
            "sage_core.__commit_sha__ == 'unknown' — CI build cwd has no "
            "git, OR `SAGE_CORE_COMMIT_SHA_OVERRIDE` env not set"
        )
    if values.get("__version__") in ("unknown", "0.0.0", ""):
        failures.append(f"sage_core.__version__ implausible: {values.get('__version__')!r}")

    return {
        "ok": not failures,
        "failures": failures,
        "values": values,
    }


def _check_required_symbols() -> dict[str, Any]:
    """Phase 3: canonical pyclasses are present in the default wheel."""
    import sage_core

    missing = [s for s in _REQUIRED_DEFAULT_SYMBOLS if not hasattr(sage_core, s)]
    return {
        "ok": not missing,
        "required": list(_REQUIRED_DEFAULT_SYMBOLS),
        "missing": missing,
    }


def _check_save_state_manifest_contract() -> dict[str, Any]:
    """Phase 4: TopologyEngine().save_state(tmp) writes the manifest
    with byte-exact SHA256 binding to the state files.

    Same contract as `test_save_state_manifest_contract.py` but
    runnable without pytest (uses stdlib + sage_core directly).
    Structured failure report so CI logs explain WHICH part broke
    if the wheel is broken.
    """
    import sage_core

    failures: list[str] = []
    details: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="sage_wheel_smoke_") as tmp_str:
        state_dir = Path(tmp_str)

        # save_state's preflight needs a valid posterior_epoch.json.
        epoch_payload = {
            "epoch": REQUIRED_POSTERIOR_EPOCH,
            "started_utc": "2026-05-06T00:00:00Z",
            "reason": "wheel_smoke contract assertion",
            "policy": "smoke",
            "audit_dump": "",
            "commit_at_reset": "",
            "predecessor_state": "",
            "first_clean_run_after": None,
        }
        (state_dir / POSTERIOR_EPOCH_FILENAME).write_text(
            json.dumps(epoch_payload), encoding="utf-8"
        )

        try:
            engine = sage_core.TopologyEngine()
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "failures": [f"TopologyEngine() construction: {exc!r}"],
            }

        try:
            engine.save_state(str(state_dir))
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "failures": [f"engine.save_state(): {exc!r}"],
            }

        manifest_path = state_dir / TOPOLOGY_STATE_MANIFEST_FILENAME
        if not manifest_path.exists():
            failures.append(
                f"{TOPOLOGY_STATE_MANIFEST_FILENAME} NOT WRITTEN by save_state — "
                "this is the cycle-13 B Rust manifest-write gap class. "
                "Wheel is STALE / broken. Rebuild source."
            )
            details["state_dir_contents"] = sorted(p.name for p in state_dir.iterdir())
            return {"ok": False, "failures": failures, "details": details}

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "failures": [f"manifest read/parse: {exc!r}"],
            }

        # Canonical fields per Rust posterior_epoch::TopologyStateManifest.
        if manifest.get("manifest_type") != "YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST":
            failures.append(
                f"manifest_type mismatch: {manifest.get('manifest_type')!r}"
            )
        if manifest.get("epoch") != REQUIRED_POSTERIOR_EPOCH:
            failures.append(f"manifest epoch != {REQUIRED_POSTERIOR_EPOCH}")
        if manifest.get("writer") != "TopologyEngine::save_state":
            failures.append(f"manifest writer mismatch: {manifest.get('writer')!r}")

        state_files = manifest.get("state_files") or []
        if not state_files:
            failures.append("manifest state_files list is empty")
        else:
            # SHA256 binding: every entry must equal the file's actual bytes.
            for entry in state_files:
                name = entry.get("name", "<missing>")
                file_path = state_dir / name
                if not file_path.is_file():
                    failures.append(
                        f"manifest references non-existent state file: {name}"
                    )
                    continue
                actual_bytes = file_path.read_bytes()
                actual_sha = hashlib.sha256(actual_bytes).hexdigest()
                if entry.get("sha256") != actual_sha:
                    failures.append(
                        f"sha256 mismatch for {name}: "
                        f"manifest={entry.get('sha256')} actual={actual_sha}"
                    )
                if entry.get("size_bytes") != len(actual_bytes):
                    failures.append(
                        f"size_bytes mismatch for {name}: "
                        f"manifest={entry.get('size_bytes')} actual={len(actual_bytes)}"
                    )

        details["state_files_count"] = len(state_files)
        details["manifest_path"] = str(manifest_path)

    return {"ok": not failures, "failures": failures, "details": details}


def run() -> dict[str, Any]:
    """Execute all phases and return a structured report.

    Report shape:
      {
        "ok": bool,
        "phases": {
          "imports": {"ok": bool, ...},
          "build_info": {"ok": bool, "failures": [...], "values": {...}},
          "symbols": {"ok": bool, "missing": [...]},
          "save_state_contract": {"ok": bool, "failures": [...], ...},
        },
        "failures": [str, ...],   # flattened across phases for CI logs.
      }
    """
    phases: dict[str, dict[str, Any]] = {}
    phases["imports"] = _check_sage_core_imports()
    if not phases["imports"]["ok"]:
        # If sage_core can't import, all subsequent phases are vacuous.
        return {
            "ok": False,
            "phases": phases,
            "failures": [phases["imports"].get("error", "import failed")],
        }
    phases["build_info"] = _check_build_info_attrs()
    phases["symbols"] = _check_required_symbols()
    phases["save_state_contract"] = _check_save_state_manifest_contract()

    flattened_failures: list[str] = []
    for phase_name, phase_data in phases.items():
        if not phase_data.get("ok"):
            for f in phase_data.get("failures") or []:
                flattened_failures.append(f"[{phase_name}] {f}")
            for missing in phase_data.get("missing") or []:
                flattened_failures.append(f"[{phase_name}] missing symbol: {missing}")
            if "error" in phase_data:
                flattened_failures.append(f"[{phase_name}] {phase_data['error']}")

    return {
        "ok": not flattened_failures,
        "phases": phases,
        "failures": flattened_failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress JSON report on success. Failures always print to stderr.",
    )
    args = parser.parse_args(argv)

    report = run()
    if report["ok"]:
        if not args.quiet:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    # Failure path: structured JSON to stderr so CI logs surface it.
    print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
    print(
        f"\nwheel_smoke FAILED: {len(report['failures'])} failures.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
