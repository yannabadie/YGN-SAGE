"""Tests for the ablation runner's post-processing helpers.

Covers the synthetic pass-rate computation added 2026-04-17, which lets
decide_next_phase.py read a real pass_rate from the per-config report
files without running the Docker harness. The "valid diff" proxy is
intentionally conservative — it catches the common failure modes
(empty string, F2 sentinel, short placeholder) and rewards real unified
diffs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import importlib.util

_module_path = _SCRIPTS / "run_swebench_ablation.py"
_spec = importlib.util.spec_from_file_location("run_swebench_ablation", _module_path)
runner_module = importlib.util.module_from_spec(_spec)
# Register in sys.modules so the @dataclass decorator inside the module
# can find the module via cls.__module__ during class creation.
sys.modules["run_swebench_ablation"] = runner_module
_spec.loader.exec_module(runner_module)

_synth_report_from_predictions = runner_module._synth_report_from_predictions


def _write_preds(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        "\n".join(json.dumps(e) for e in entries),
        encoding="utf-8",
    )
    return p


def test_valid_diff_counts_as_pass(tmp_path):
    entries = [
        {
            "instance_id": "t1",
            "model_patch": (
                "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
                "@@ -1,3 +1,3 @@\n-a\n+b\n c\n" + ("x" * 50)
            ),
        }
    ]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="full", dataset="lite",
    )
    assert report["total"] == 1
    assert report["passed"] == 1
    assert report["pass_rate"] == 1.0


def test_empty_patch_counts_as_fail(tmp_path):
    entries = [{"instance_id": "t1", "model_patch": ""}]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="full", dataset="lite",
    )
    assert report["passed"] == 0
    assert report["pass_rate"] == 0.0


def test_sentinel_counts_as_fail(tmp_path):
    """The F2 sentinel is explicitly NOT a diff — must score 0."""
    entries = [
        {"instance_id": "t1",
         "model_patch": "[sage: agent exited after 20 steps with no content]"},
        {"instance_id": "t2",
         "model_patch": "Agent finished at step 5"},
    ]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="bare", dataset="lite",
    )
    assert report["total"] == 2
    assert report["passed"] == 0


def test_short_diff_counts_as_fail(tmp_path):
    """A 50-char 'diff' doesn't have room for a meaningful hunk — fail."""
    entries = [{"instance_id": "t1", "model_patch": "diff --git a/x b/x\n--- a/x\n+++ b/x"}]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="full", dataset="lite",
    )
    assert report["passed"] == 0


def test_mixed_results_rate(tmp_path):
    good = "diff --git a/x b/x\n" + ("z" * 200)
    entries = [
        {"instance_id": "t1", "model_patch": good},
        {"instance_id": "t2", "model_patch": ""},
        {"instance_id": "t3", "model_patch": good},
    ]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="full", dataset="lite",
    )
    assert report["total"] == 3
    assert report["passed"] == 2
    assert abs(report["pass_rate"] - 2 / 3) < 1e-9


def test_missing_file_returns_zero(tmp_path):
    missing = tmp_path / "nope.jsonl"
    report = _synth_report_from_predictions(
        missing, wall_time_s=5.0, config_name="x", dataset="lite",
    )
    assert report["total"] == 0
    assert report["pass_rate"] == 0.0


def test_per_task_results_recorded(tmp_path):
    good = "diff --git a/x b/x\n" + ("z" * 200)
    entries = [
        {"instance_id": "ok", "model_patch": good},
        {"instance_id": "bad", "model_patch": ""},
    ]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="full", dataset="lite",
    )
    assert [r["task_id"] for r in report["results"]] == ["ok", "bad"]
    assert [r["passed"] for r in report["results"]] == [True, False]
    assert report["results"][0]["patch_len"] > report["results"][1]["patch_len"]


def test_report_shape_matches_decide_next_phase_contract(tmp_path):
    """decide_next_phase reads `pass_rate` OR `resolved_rate` — must ship one."""
    entries = [{"instance_id": "t1", "model_patch": ""}]
    report = _synth_report_from_predictions(
        _write_preds(tmp_path, entries),
        wall_time_s=10.0, config_name="full", dataset="lite",
    )
    assert "pass_rate" in report
    assert "_note" in report and "synthetic" in report["_note"].lower()
