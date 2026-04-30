#!/usr/bin/env python3
"""Generate single-source-of-truth project status snapshot.

Runs pytest --collect-only on sage-python and sage-discover, parses cargo
test --list on sage-core, and writes docs/status/current.json. Downstream
consumers (README, CLAUDE.md, Dashboard) read from this JSON instead of
maintaining hand-edited counts.

cgpro 2026-04-30 architect review Q-C verdict: "le repo a besoin d'un single
source of truth: un script scripts/status_snapshot.py ou CI artifact qui
écrit docs/status/current.json, puis README/Dashboard/CLAUDE consomment ce
JSON." This is that script.

Usage:
  python scripts/status_snapshot.py            # write docs/status/current.json
  python scripts/status_snapshot.py --check    # exit 1 if file is stale vs live counts
  python scripts/status_snapshot.py --print    # print to stdout, do not write
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> tuple[int, str, str]:
    """Run a command, capture stdout/stderr/returncode."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"


def _count_pytest_collect(cwd: Path) -> dict[str, Any]:
    """Run `pytest --collect-only -q` and return total/per-file counts."""
    if not (cwd / "tests").exists() and not list(cwd.glob("tests*")):
        return {"available": False}
    rc, out, err = _run(
        ["python", "-m", "pytest", "tests/", "--collect-only", "-q",
         "--no-header"],
        cwd,
        timeout=180,
    )
    if rc != 0:
        return {"available": False, "error": err.splitlines()[-3:] if err else []}
    match = re.search(r"(\d+)\s+tests?\s+collected", out)
    total = int(match.group(1)) if match else None
    return {"available": True, "total": total}


def _count_cargo_tests(cwd: Path, features: str = "smt,cognitive,sandbox,cranelift,tool-executor") -> dict[str, Any]:
    """Run `cargo test --features {features} --lib --no-run` to count tests."""
    if not (cwd / "Cargo.toml").exists():
        return {"available": False}
    # Use --list flavour: cargo test -- --list (after --no-run)
    rc, out, err = _run(
        ["cargo", "test", f"--features={features}", "--lib", "--", "--list"],
        cwd,
        timeout=600,
    )
    if rc != 0:
        # Fallback: just try a build to confirm available
        rc2, _, _ = _run(["cargo", "check", f"--features={features}", "--lib"], cwd, timeout=300)
        return {"available": rc2 == 0, "list_failed": True, "error": err.splitlines()[-3:] if err else []}
    # Count "test" lines in --list output
    test_count = sum(1 for line in out.splitlines() if line.strip().endswith(": test"))
    return {"available": True, "total": test_count, "features": features}


def _git_meta(cwd: Path) -> dict[str, Any]:
    """Capture commit SHA, branch, and dirty flag."""
    sha_rc, sha, _ = _run(["git", "rev-parse", "HEAD"], cwd)
    branch_rc, branch, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    dirty_rc, dirty, _ = _run(["git", "status", "--porcelain"], cwd)
    return {
        "commit_sha": sha.strip() if sha_rc == 0 else "unknown",
        "branch": branch.strip() if branch_rc == 0 else "unknown",
        "dirty": bool(dirty.strip()) if dirty_rc == 0 else None,
    }


def _build_snapshot() -> dict[str, Any]:
    """Build the full status snapshot."""
    sage_python = REPO_ROOT / "sage-python"
    sage_core = REPO_ROOT / "sage-core"
    sage_discover = REPO_ROOT / "sage-discover"

    git = _git_meta(REPO_ROOT)
    snapshot: dict[str, Any] = {
        "schema_version": "v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "sage_python_tests": _count_pytest_collect(sage_python),
        "sage_core_tests": _count_cargo_tests(sage_core),
        "sage_discover_tests": _count_pytest_collect(sage_discover),
        "notes": (
            "Counts are pytest --collect-only / cargo test --list. They reflect "
            "the test surface available, not necessarily PASSING tests. For pass "
            "counts, run the full suite and consult CI artifacts."
        ),
    }
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if docs/status/current.json is stale vs live counts")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="Print to stdout, do not write")
    args = parser.parse_args()

    snapshot = _build_snapshot()
    out_path = REPO_ROOT / "docs" / "status" / "current.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_text = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    if args.print_only:
        print(canonical_text, end="")
        return 0

    if args.check:
        if not out_path.exists():
            print(f"STALE: {out_path} does not exist", file=sys.stderr)
            return 1
        existing = out_path.read_text(encoding="utf-8")
        # Strip generated_at_utc + commit_sha for stale-check comparison (those
        # change every run; we care about test counts being equal)
        def _strip_volatile(text: str) -> str:
            obj = json.loads(text)
            obj.pop("generated_at_utc", None)
            obj.get("git", {}).pop("commit_sha", None)
            obj.get("git", {}).pop("dirty", None)
            return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
        if _strip_volatile(existing) != _strip_volatile(canonical_text):
            print(f"STALE: {out_path} test counts differ from live counts.", file=sys.stderr)
            print("Run `python scripts/status_snapshot.py` to refresh.", file=sys.stderr)
            return 1
        print(f"OK: {out_path} matches live counts.")
        return 0

    out_path.write_text(canonical_text, encoding="utf-8")
    print(f"Wrote {out_path}")
    py_total = snapshot["sage_python_tests"].get("total", "?")
    rust_total = snapshot["sage_core_tests"].get("total", "?")
    discover_total = snapshot["sage_discover_tests"].get("total", "?")
    print(f"  sage-python: {py_total} tests collected")
    print(f"  sage-core: {rust_total} tests listed")
    print(f"  sage-discover: {discover_total} tests collected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
