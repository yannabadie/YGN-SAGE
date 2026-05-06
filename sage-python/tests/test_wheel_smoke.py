"""Tests for `sage.ops.wheel_smoke` — post-install wheel contract assertion.

Cycle-13 B Q2 follow-up. The smoke runs against the locally-installed
`sage_core`, NOT a mock. So these tests use the REAL imported sage_core
and verify the smoke phases evaluate correctly.

When CI runs this on a fresh wheel (per cycle-13 H wiring), the smoke
asserts the runtime contract. When pytest runs locally, the same
smoke executes against whatever sage_core is currently installed —
catches the same broken-wheel class earlier.
"""
from __future__ import annotations

import json

import pytest

from sage.ops import wheel_smoke


# ── End-to-end smoke (against the real installed sage_core) ──────────────────


def test_run_returns_ok_against_installed_sage_core() -> None:
    """Sanity: against the wheel installed in the CI/dev environment,
    the full smoke MUST pass. If this fails, either the wheel is
    broken (the cycle-13 B class) OR the test environment is
    misconfigured (no sage_core at all)."""
    report = wheel_smoke.run()
    assert report["ok"], (
        "wheel_smoke FAILED against the installed sage_core. "
        "This means the wheel is broken or the source is ahead of "
        "the binary. Failures:\n"
        + "\n".join(f"  - {f}" for f in report["failures"])
    )

    # All 4 phases ran.
    assert "imports" in report["phases"]
    assert "build_info" in report["phases"]
    assert "symbols" in report["phases"]
    assert "save_state_contract" in report["phases"]


def test_run_reports_phase_structure() -> None:
    report = wheel_smoke.run()
    # Structural shape — used by CI log parsers.
    assert isinstance(report["ok"], bool)
    assert isinstance(report["failures"], list)
    assert isinstance(report["phases"], dict)


# ── _check_sage_core_imports ─────────────────────────────────────────────────


def test_check_sage_core_imports_ok() -> None:
    """sage_core IS importable in the test env."""
    result = wheel_smoke._check_sage_core_imports()
    assert result["ok"] is True
    assert result["module_path"]


# ── _check_build_info_attrs ──────────────────────────────────────────────────


def test_check_build_info_attrs_ok_for_well_built_wheel() -> None:
    """The locally-installed wheel was built with build.rs git
    resolution (since cycle-13 G commit `b035973e`). The 4 attrs
    must be present + commit_sha != 'unknown'."""
    result = wheel_smoke._check_build_info_attrs()
    assert result["ok"], f"build_info phase failed: {result['failures']}"
    values = result["values"]
    assert values["__commit_sha__"] != "unknown"
    assert len(values["__commit_sha__"]) >= 40, (
        f"commit_sha looks invalid: {values['__commit_sha__']!r}"
    )
    assert values["__version__"] == "0.1.0"
    assert values["__build_profile__"] in ("release", "debug")


def test_check_build_info_attrs_fails_when_commit_sha_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a wheel ships with __commit_sha__ == 'unknown' (CI build
    cwd had no git access), the smoke MUST fail. This is the canary
    that catches build-pipeline glitches."""
    import sage_core

    monkeypatch.setattr(sage_core, "__commit_sha__", "unknown", raising=False)
    result = wheel_smoke._check_build_info_attrs()
    assert result["ok"] is False
    assert any("__commit_sha__" in f and "unknown" in f for f in result["failures"]), (
        f"expected __commit_sha__='unknown' failure, got: {result['failures']}"
    )


def test_check_build_info_attrs_fails_when_attr_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-`b035973e` wheel won't have __commit_sha__ at all. Smoke
    MUST fail loudly."""
    import sage_core

    monkeypatch.delattr(sage_core, "__commit_sha__", raising=False)
    result = wheel_smoke._check_build_info_attrs()
    assert result["ok"] is False
    assert any("missing attribute" in f for f in result["failures"])


# ── _check_required_symbols ──────────────────────────────────────────────────


def test_check_required_symbols_ok() -> None:
    result = wheel_smoke._check_required_symbols()
    assert result["ok"], (
        f"installed wheel missing required pyclasses: {result['missing']}"
    )
    assert result["missing"] == []


# ── _check_save_state_manifest_contract ──────────────────────────────────────


def test_check_save_state_manifest_contract_ok() -> None:
    """The locally-installed wheel must satisfy the manifest contract.

    This is the cycle-13 B regression test (commit `32d39bdf`)
    re-cast as a runtime assertion: TopologyEngine().save_state(tmp)
    writes the manifest with byte-exact SHA256 binding.

    If this fails on CI, the wheel SHIPS BROKEN — block publish.
    """
    result = wheel_smoke._check_save_state_manifest_contract()
    assert result["ok"], (
        "wheel manifest contract broken. Failures:\n"
        + "\n".join(f"  - {f}" for f in result["failures"])
    )
    # state_files count: TopologyEngine fresh save writes 3 files
    # (bandit_state.db + archive_state.db + engine_extras.json).
    assert result["details"]["state_files_count"] >= 3


# ── main() / CLI ─────────────────────────────────────────────────────────────


def test_main_exit_0_on_pass(capsys: pytest.CaptureFixture[str]) -> None:
    rc = wheel_smoke.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == "", (
        "no stderr output expected on success path; "
        f"got: {captured.err!r}"
    )
    payload = json.loads(captured.out)
    assert payload["ok"] is True


def test_main_quiet_suppresses_success_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = wheel_smoke.main(["--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_exit_1_on_failure_with_stderr_report(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force a failure and assert the structured JSON report goes to
    stderr (CI log capture)."""
    import sage_core

    monkeypatch.setattr(sage_core, "__commit_sha__", "unknown", raising=False)
    rc = wheel_smoke.main([])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err, "failure path MUST write structured report to stderr"
    # Stderr is JSON-prefix + "wheel_smoke FAILED" suffix; the JSON
    # must be parseable up to the trailing log line.
    json_blob = captured.err.split("\nwheel_smoke FAILED")[0]
    payload = json.loads(json_blob)
    assert payload["ok"] is False
    assert any("__commit_sha__" in f for f in payload["failures"])
