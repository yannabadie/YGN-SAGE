"""Tests for sage.bench.modal_billing.

Block ``canary-stage-timing-budget`` slice 5 (cgpro DESIGN 2026-05-11).

Covers ``capture_modal_app_cost_usd`` happy paths + every failure mode
that should produce ``status="manual_required"`` without raising:

- Modal CLI missing (FileNotFoundError → manual_required).
- CLI timeout.
- CLI nonzero exit.
- CLI returns invalid JSON.
- CLI returns non-list JSON.
- No rows match the requested app_id.
- A matching row has an unparseable Cost cell.
- Happy path: single row.
- Happy path: multiple rows summed.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from sage.bench.modal_billing import (
    MODAL_BILLING_TIMEOUT_S,
    capture_modal_app_cost_usd,
)


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_runner(*, returncode: int, stdout: str, stderr: str = "") -> Any:
    def _run(argv, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeCompletedProcess(returncode=returncode, stdout=stdout, stderr=stderr)

    return _run


def test_happy_path_single_matching_row() -> None:
    rows = [
        {
            "Object ID": "ap-target",
            "Description": "swe-bench-pro-eval",
            "Cost": "0.01352370",
        },
        {
            "Object ID": "ap-other",
            "Description": "unrelated",
            "Cost": "1.5",
        },
    ]
    runner = _make_runner(returncode=0, stdout=json.dumps(rows))
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        for_range="today",
        runner=runner,
    )
    assert result["status"] == "measured"
    assert result["modal_cost_usd"] == pytest.approx(0.01352370)
    assert result["modal_app_id"] == "ap-target"
    assert result["n_rows_matched"] == 1
    assert result["rows"][0]["Description"] == "swe-bench-pro-eval"
    assert result["failure_reason"] is None


def test_happy_path_sums_multiple_matching_rows() -> None:
    """A workspace may have multiple billing intervals for the same app
    within --for today (hourly resolution). The function MUST sum them.
    """
    rows = [
        {"Object ID": "ap-target", "Cost": "0.20"},
        {"Object ID": "ap-target", "Cost": "0.30"},
        {"Object ID": "ap-other", "Cost": "10.00"},
    ]
    runner = _make_runner(returncode=0, stdout=json.dumps(rows))
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "measured"
    assert result["modal_cost_usd"] == pytest.approx(0.50)
    assert result["n_rows_matched"] == 2


def test_modal_cli_missing_returns_manual_required() -> None:
    def _missing(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("modal not on PATH")

    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=_missing,
    )
    assert result["status"] == "manual_required"
    assert result["modal_cost_usd"] is None
    assert result["failure_reason"] == "modal_cli_not_installed"
    assert result["modal_app_id"] == "ap-target"


def test_modal_cli_timeout_returns_manual_required() -> None:
    def _timeout(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=argv, timeout=30)

    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=_timeout,
    )
    assert result["status"] == "manual_required"
    assert result["failure_reason"] == "modal_cli_timeout"


def test_modal_cli_nonzero_exit_captures_stderr() -> None:
    runner = _make_runner(returncode=2, stdout="", stderr="auth error: token expired")
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "manual_required"
    assert result["failure_reason"] == "modal_cli_nonzero_exit"


def test_modal_cli_invalid_json_returns_manual_required() -> None:
    runner = _make_runner(returncode=0, stdout="not-json")
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "manual_required"
    assert result["failure_reason"] is not None
    assert result["failure_reason"].startswith("modal_cli_json_decode_error:")


def test_modal_cli_unexpected_shape_returns_manual_required() -> None:
    """A future CLI schema change to a dict (vs current list) must not
    raise; it must surface as manual_required.
    """
    runner = _make_runner(returncode=0, stdout=json.dumps({"unexpected": "shape"}))
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "manual_required"
    assert result["failure_reason"] == "modal_cli_unexpected_shape_not_list"


def test_no_matching_row_returns_manual_required_with_reason() -> None:
    """CLI succeeded but no row references our app_id (e.g. running
    early in the day before the first sandbox finished billing). The
    cost is genuinely unknown and the caller needs the dashboard.
    """
    rows = [{"Object ID": "ap-other-app", "Cost": "0.50"}]
    runner = _make_runner(returncode=0, stdout=json.dumps(rows))
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "manual_required"
    assert result["failure_reason"] == "no_rows_for_app_id"
    assert result["n_rows_matched"] == 0


def test_unparseable_cost_cell_returns_manual_required() -> None:
    rows = [{"Object ID": "ap-target", "Cost": "not-a-number"}]
    runner = _make_runner(returncode=0, stdout=json.dumps(rows))
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "manual_required"
    assert result["failure_reason"] == "unparseable_cost_field"
    assert result["n_rows_matched"] == 1
    assert result["rows"][0]["Cost"] == "not-a-number"


def test_cost_field_accepts_numeric_for_future_schema() -> None:
    """If Modal flips the Cost field from string to numeric, the parser
    must keep working — protects against silent regression.
    """
    rows = [{"Object ID": "ap-target", "Cost": 0.42}]
    runner = _make_runner(returncode=0, stdout=json.dumps(rows))
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        runner=runner,
    )
    assert result["status"] == "measured"
    assert result["modal_cost_usd"] == pytest.approx(0.42)


def test_for_range_is_echoed_in_result() -> None:
    runner = _make_runner(returncode=0, stdout="[]")
    result = capture_modal_app_cost_usd(
        app_id="ap-target",
        for_range="last week",
        runner=runner,
    )
    assert result["for_range"] == "last week"


def test_module_constants_exposed() -> None:
    """Sanity: the module exports the timeout constant so callers can
    pass a different bound without monkeypatching.
    """
    assert MODAL_BILLING_TIMEOUT_S > 0
    assert isinstance(MODAL_BILLING_TIMEOUT_S, float)
