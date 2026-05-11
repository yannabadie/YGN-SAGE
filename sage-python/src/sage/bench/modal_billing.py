"""Modal billing capture for SWE-bench Pro grader runs.

Block ``canary-stage-timing-budget`` slice 5 (cgpro DESIGN 2026-05-11,
conv ``cgpro_ygn_sage_global_analysis_20260510``).

The B2 final close requires ``modal_cost_usd`` to be either ``measured``
(read from Modal billing) or ``manual_required`` (dashboard lookup with
the app + sandbox IDs captured). Hand-waving the cost as "small" is
not acceptable — Modal bills actual CPU-seconds + image-pull bandwidth.

This module wraps the public ``modal billing report --for <range>
--json`` CLI which returns a list of per-app cost rows::

    [
      {
        "Object ID": "ap-...",
        "Description": "swe-bench-pro-eval",
        "Environment": "main",
        "Interval Start": "2026-05-11T00:00:00",
        "Cost": "0.01352370"
      },
      ...
    ]

Usage::

    from sage.bench.modal_billing import capture_modal_app_cost_usd
    result = capture_modal_app_cost_usd(
        app_id="ap-LhzIeBC5TBPQ4BumsWJEeD",
        for_range="today",
    )
    # result = {
    #   "status": "measured",
    #   "modal_cost_usd": 0.01352370,
    #   "modal_app_id": "ap-...",
    #   "for_range": "today",
    #   "rows": [...]
    # }

The function never raises on Modal CLI failure; it returns
``status="manual_required"`` and the failure reason in the result dict.
That keeps the caller's gate logic uniform: a missing measurement
becomes an explicit manual-lookup TODO, not a stack trace.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

__all__ = ["MODAL_BILLING_TIMEOUT_S", "capture_modal_app_cost_usd"]

log = logging.getLogger(__name__)

# Conservative subprocess timeout. The CLI typically returns < 5s; we
# allow more in case the workspace has many apps or the network is slow.
MODAL_BILLING_TIMEOUT_S: float = 30.0


def _run_modal_billing(
    for_range: str,
    *,
    extra_args: list[str] | None = None,
    runner: Any = subprocess.run,
    timeout_s: float = MODAL_BILLING_TIMEOUT_S,
) -> dict[str, Any]:
    """Invoke the Modal CLI and return a structured result.

    Returns a dict with keys:

    - ``ok`` (bool)
    - ``rows`` (list[dict] | None)
    - ``failure_reason`` (str | None)
    - ``stderr`` (str | None — captured for diagnostics)

    The ``runner`` indirection makes the function unit-testable without
    actually shelling out to the Modal CLI.
    """
    argv = ["modal", "billing", "report", "--for", for_range, "--json"]
    if extra_args:
        argv.extend(extra_args)

    try:
        completed = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "rows": None,
            "failure_reason": "modal_cli_not_installed",
            "stderr": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "rows": None,
            "failure_reason": "modal_cli_timeout",
            "stderr": None,
        }

    if completed.returncode != 0:
        return {
            "ok": False,
            "rows": None,
            "failure_reason": "modal_cli_nonzero_exit",
            "stderr": (completed.stderr or "")[:2000],
        }

    try:
        rows = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "rows": None,
            "failure_reason": f"modal_cli_json_decode_error:{exc.msg}",
            "stderr": (completed.stderr or "")[:2000],
        }

    if not isinstance(rows, list):
        return {
            "ok": False,
            "rows": None,
            "failure_reason": "modal_cli_unexpected_shape_not_list",
            "stderr": (completed.stderr or "")[:2000],
        }

    return {
        "ok": True,
        "rows": rows,
        "failure_reason": None,
        "stderr": None,
    }


def _coerce_cost_field(value: Any) -> float | None:
    """Parse a ``"Cost"`` cell into ``float`` or return ``None``.

    Modal's JSON output ships cost as a string (e.g. ``"0.01352370"``).
    A future schema change to numeric is tolerated.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def capture_modal_app_cost_usd(
    *,
    app_id: str,
    for_range: str = "today",
    runner: Any = subprocess.run,
    timeout_s: float = MODAL_BILLING_TIMEOUT_S,
) -> dict[str, Any]:
    """Return a structured cost record for ``app_id`` over ``for_range``.

    Result shape (always returned, never raises)::

        {
          "status": "measured" | "manual_required",
          "modal_cost_usd": float | None,
          "modal_app_id": str,
          "for_range": str,
          "n_rows_matched": int,
          "rows": list[dict],          # matching rows, possibly empty
          "failure_reason": str | None,
        }

    - ``status="measured"`` only when the Modal CLI returned at least
      one matching row AND all of those rows' Cost fields parsed to a
      finite float. The total is the sum across matching rows.
    - ``status="manual_required"`` in every other case (CLI missing,
      timeout, nonzero exit, JSON parse error, no matching rows, or any
      unparseable Cost cell). Caller is expected to follow up via the
      Modal web dashboard.

    ``modal_app_id`` echoes back the input so downstream artefacts can
    record the exact id queried.
    """
    cli_result = _run_modal_billing(
        for_range,
        runner=runner,
        timeout_s=timeout_s,
    )

    if not cli_result["ok"]:
        return {
            "status": "manual_required",
            "modal_cost_usd": None,
            "modal_app_id": app_id,
            "for_range": for_range,
            "n_rows_matched": 0,
            "rows": [],
            "failure_reason": cli_result["failure_reason"],
        }

    all_rows: list[dict[str, Any]] = cli_result["rows"] or []
    matching_rows = [
        row for row in all_rows if isinstance(row, dict) and row.get("Object ID") == app_id
    ]

    if not matching_rows:
        return {
            "status": "manual_required",
            "modal_cost_usd": None,
            "modal_app_id": app_id,
            "for_range": for_range,
            "n_rows_matched": 0,
            "rows": [],
            "failure_reason": "no_rows_for_app_id",
        }

    total = 0.0
    for row in matching_rows:
        parsed = _coerce_cost_field(row.get("Cost"))
        if parsed is None:
            return {
                "status": "manual_required",
                "modal_cost_usd": None,
                "modal_app_id": app_id,
                "for_range": for_range,
                "n_rows_matched": len(matching_rows),
                "rows": matching_rows,
                "failure_reason": "unparseable_cost_field",
            }
        total += parsed

    return {
        "status": "measured",
        "modal_cost_usd": total,
        "modal_app_id": app_id,
        "for_range": for_range,
        "n_rows_matched": len(matching_rows),
        "rows": matching_rows,
        "failure_reason": None,
    }
