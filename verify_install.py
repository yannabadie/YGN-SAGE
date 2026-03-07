#!/usr/bin/env python3
"""YGN-SAGE installation verifier — checks all dependencies and runs smoke tests.

Usage:
    python verify_install.py           # Full check
    python verify_install.py --quick   # Skip slow tests
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time


def _check(name: str, check_fn, *, required: bool = True) -> bool:
    """Run a check and print result."""
    try:
        result = check_fn()
        print(f"  [OK]  {name}: {result}")
        return True
    except Exception as e:
        tag = "FAIL" if required else "SKIP"
        print(f"  [{tag}] {name}: {e}")
        return not required


def _import_check(module: str) -> str:
    mod = importlib.import_module(module)
    version = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
    return str(version)


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:200] or f"exit code {result.returncode}")
    return result.stdout.strip()[:100]


def main() -> int:
    quick = "--quick" in sys.argv
    failed = 0
    total = 0

    print("=" * 60)
    print("  YGN-SAGE Installation Verification")
    print("=" * 60)

    # 1. Python version
    print("\n[1/6] Python Environment")
    total += 1
    if not _check("Python >= 3.12", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"):
        failed += 1
    if sys.version_info < (3, 12):
        print("       WARNING: Python 3.12+ recommended")

    # 2. Core Python dependencies
    print("\n[2/6] Core Python Dependencies")
    core_deps = ["pytest", "z3", "aiosqlite", "pyarrow"]
    for dep in core_deps:
        total += 1
        if not _check(dep, lambda d=dep: _import_check(d)):
            failed += 1

    # 3. Optional Python dependencies
    print("\n[3/6] Optional Python Dependencies")
    opt_deps = [
        ("google.genai", False),
        ("openai", False),
        ("fastapi", False),
        ("ruff", False),
    ]
    for dep, req in opt_deps:
        total += 1
        if not _check(dep, lambda d=dep: _import_check(d), required=req):
            failed += 1

    # 4. Rust core
    print("\n[4/6] Rust Core")
    total += 1
    if not _check("cargo", lambda: _run(["cargo", "--version"])):
        failed += 1
    total += 1
    if not _check("sage-core build", lambda: _run(
        ["cargo", "check", "--no-default-features"],
        cwd="sage-core",
    )):
        failed += 1

    # 5. Python test suite
    print("\n[5/6] Python Test Suite")
    if not quick:
        total += 1
        t0 = time.time()
        if not _check("sage-python tests", lambda: _run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line", "-x"],
            cwd="sage-python",
            timeout=300,
        )):
            failed += 1
        else:
            dt = time.time() - t0
            print(f"       Completed in {dt:.1f}s")
    else:
        print("  [SKIP] Skipped (--quick mode)")

    # 6. Rust test suite
    print("\n[6/6] Rust Test Suite")
    if not quick:
        total += 1
        if not _check("sage-core tests", lambda: _run(
            ["cargo", "test", "--no-default-features"],
            cwd="sage-core",
            timeout=120,
        )):
            failed += 1
    else:
        print("  [SKIP] Skipped (--quick mode)")

    # Summary
    print("\n" + "=" * 60)
    passed = total - failed
    print(f"  Result: {passed}/{total} checks passed")
    if failed:
        print(f"  {failed} check(s) FAILED")
    else:
        print("  All checks passed!")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
