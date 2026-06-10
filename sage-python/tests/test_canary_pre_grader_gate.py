"""Tests for sage-python/scripts/canary_pre_grader_gate.py.

Block ``canary-stage-timing-budget`` slice 2 (cgpro DESIGN 2026-05-11).
Covers the resolution order documented at the top of the script:

1. ``patch != ""`` → ``pass:non_empty_patch``.
2. Pre-recorded ``_reason_code`` in predictions.jsonl wins when valid.
3. Timeout → ``categorize_timeout`` derives the code from events.
4. Non-timeout → ``classify_non_timeout_empty_patch`` handles budget /
   verifier / fallback.
5. Anything else → ``fail:no_allowed_reason_code``.

Also asserts:

- Missing predictions.jsonl degrades to events-only resolution.
- ``_diff_verifier_outcome=None`` does NOT short-circuit to pass; the
  fallback returns ``no_patch_extracted`` which is itself a pass — but
  via a named code, not a silent "None means OK".
- Overall ``gate_status`` flips to ``FAIL`` when any instance fails.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "sage-python" / "scripts" / "canary_pre_grader_gate.py"


def _load_gate_module() -> ModuleType:
    """Import the gate script by path so we can call run_gate / main."""
    spec = importlib.util.spec_from_file_location(
        "canary_pre_grader_gate", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _write_predictions(
    tmp_path: Path,
    predictions: list[dict],
    annotations: list[dict] | None = None,
) -> tuple[Path, Path, Path]:
    """Write predictions.json + predictions.jsonl + per_task/ and return paths."""
    predictions_json = tmp_path / "predictions.json"
    predictions_jsonl = tmp_path / "predictions.jsonl"
    events_dir = tmp_path / "per_task"
    events_dir.mkdir()

    predictions_json.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    if annotations is not None:
        predictions_jsonl.write_text(
            "\n".join(json.dumps(a) for a in annotations),
            encoding="utf-8",
        )
    return predictions_json, predictions_jsonl, events_dir


def _write_events(events_dir: Path, instance_id: str, events: list[dict]) -> Path:
    path = events_dir / f"{instance_id}.events.jsonl"
    path.write_text(
        "\n".join(json.dumps(ev) for ev in events),
        encoding="utf-8",
    )
    return path


def test_non_empty_patch_passes(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[
            {"instance_id": "inst-1", "patch": "diff --git a/foo b/foo\n+x\n", "prefix": "sage"}
        ],
        annotations=[],
    )
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    assert result["gate_status"] == "PASS"
    assert result["n_pass"] == 1
    assert result["n_fail"] == 0
    [entry] = result["per_instance"]
    assert entry["verdict"] == "pass:non_empty_patch"
    assert entry["patch_present"] is True
    assert entry["reason_code"] is None


def test_pre_recorded_reason_code_wins(tmp_path: Path) -> None:
    """A valid ``_reason_code`` field in the jsonl row short-circuits
    derivation. This is the path slice 4 will use to record the canary's
    own classification.
    """
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-1", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-1",
                "patch": "",
                "_reason_code": "task_budget_exhausted",
                "_timeout": False,
            }
        ],
    )
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    assert result["gate_status"] == "PASS"
    [entry] = result["per_instance"]
    assert entry["verdict"] == "pass:task_budget_exhausted"
    assert entry["reason_code"] == "task_budget_exhausted"
    assert entry["evidence"]["source"] == "pre_recorded"


def test_pre_recorded_invalid_reason_code_falls_through(tmp_path: Path) -> None:
    """A non-member ``_reason_code`` must not be accepted — fall through
    to derivation.
    """
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-1", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-1",
                "patch": "",
                "_reason_code": "made_up_code",
                "_timeout": False,
                "_diff_verifier_outcome": None,
            }
        ],
    )
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    [entry] = result["per_instance"]
    # Falls through to non-timeout classifier with no budget / no verifier
    # signal → no_patch_extracted (the named fallback).
    assert entry["reason_code"] == "no_patch_extracted"
    assert entry["evidence"]["source"] == "classify_non_timeout_empty_patch"


def test_timeout_provider_call_timeout_passes(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-vuls", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-vuls",
                "patch": "",
                "_timeout": True,
                "_latency_ms": 300_000,
            }
        ],
    )
    _write_events(
        events_dir,
        "inst-vuls",
        events=[
            {"event_type": "routing_decision", "payload": {"model_id": "deepseek-v4-flash"}},
            {
                "event_type": "model_assigned",
                "payload": {
                    "model_id": "deepseek-v4-flash",
                    "provider_id": "deepseek",
                },
            },
            {"event_type": "cli_progress", "payload": {"stage": "execute", "elapsed_ms": 60_000}},
            {"event_type": "node_started", "payload": {"node_id": "node-0"}},
            {"event_type": "cli_progress", "payload": {"stage": "execute", "elapsed_ms": 200_000}},
            {"event_type": "cli_progress", "payload": {"stage": "execute", "elapsed_ms": 290_000}},
        ],
    )
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    assert result["gate_status"] == "PASS"
    [entry] = result["per_instance"]
    assert entry["verdict"] == "pass:provider_call_timeout"
    assert entry["reason_code"] == "provider_call_timeout"
    assert entry["evidence"]["source"] == "categorize_timeout"
    assert entry["evidence"]["provider_attempted"] is True


def test_timeout_scoring_boot_impossible_passes(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-2", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-2",
                "patch": "",
                "_timeout": True,
                "_latency_ms": 60_000,
            }
        ],
    )
    # No events at all — pipeline never produced routing / progress.
    _write_events(events_dir, "inst-2", events=[])
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    [entry] = result["per_instance"]
    assert entry["reason_code"] == "scoring_boot_impossible"
    assert entry["verdict"] == "pass:scoring_boot_impossible"


def test_non_timeout_diff_verifier_no_patch_to_verify(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-3", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-3",
                "patch": "",
                "_timeout": False,
                "_diff_verifier_outcome": "no_patch_to_verify",
            }
        ],
    )
    _write_events(events_dir, "inst-3", events=[])
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    [entry] = result["per_instance"]
    assert entry["reason_code"] == "no_patch_to_verify"
    assert entry["evidence"]["source"] == "classify_non_timeout_empty_patch"


def test_non_timeout_budget_exhausted_dominates(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-4", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-4",
                "patch": "",
                "_timeout": False,
                "_budget_exhausted": True,
                "_diff_verifier_outcome": "no_patch_to_verify",  # would otherwise win
            }
        ],
    )
    _write_events(events_dir, "inst-4", events=[])
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    [entry] = result["per_instance"]
    assert entry["reason_code"] == "task_budget_exhausted"


def test_non_timeout_diff_verifier_none_falls_back_but_passes_via_named_code(
    tmp_path: Path,
) -> None:
    """Per cgpro DESIGN: ``_diff_verifier_outcome=None`` must NOT be
    treated as pass. It falls back to ``no_patch_extracted`` which IS a
    pass — but via a named member of EMPTY_PATCH_REASON_CODES, not via
    silent acceptance of None.
    """
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-5", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-5",
                "patch": "",
                "_timeout": False,
                "_diff_verifier_outcome": None,
            }
        ],
    )
    _write_events(events_dir, "inst-5", events=[])
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    [entry] = result["per_instance"]
    # Named code, not None.
    assert entry["reason_code"] == "no_patch_extracted"
    assert entry["verdict"] == "pass:no_patch_extracted"
    assert entry["evidence"]["diff_verifier_outcome"] is None


def test_empty_patch_no_annotation_no_events_fails(tmp_path: Path) -> None:
    """If neither predictions.jsonl nor events.jsonl give the gate
    anything to work with, the result must FAIL — silent passing here
    would defeat the gate's purpose.
    """
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-6", "patch": "", "prefix": "sage"}],
        annotations=None,  # no jsonl file at all
    )
    # No events file either.
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    [entry] = result["per_instance"]
    # No annotation → annotation is None → not timeout → classifier
    # called with budget=False, verifier=None → no_patch_extracted
    # (named fallback, pass).
    assert entry["reason_code"] == "no_patch_extracted"
    assert result["gate_status"] == "PASS"


def test_timeout_with_no_recognized_signal_fails(tmp_path: Path) -> None:
    """A timeout-flagged row whose events are empty + categorize_timeout
    output is somehow NOT in EMPTY_PATCH_REASON_CODES would fail. This
    is defensive — categorize_timeout currently always returns a member,
    but the gate must not silently accept future drift.
    """
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-7", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-7",
                "patch": "",
                "_timeout": True,
                "_latency_ms": 60_000,
            }
        ],
    )
    # Empty events → categorize_timeout returns scoring_boot_impossible
    # (a valid member). To get a FAIL, monkeypatch categorize_timeout to
    # return an invalid code.
    original = gate.categorize_timeout
    try:
        gate.categorize_timeout = lambda **kw: {  # type: ignore[assignment]
            "reason_code": "fictional_drift",
            "last_stage": None,
            "elapsed_ms_by_stage": {},
            "provider_attempted": False,
            "model_id_final": None,
            "provider_final": None,
        }
        _write_events(events_dir, "inst-7", events=[])
        result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    finally:
        gate.categorize_timeout = original

    [entry] = result["per_instance"]
    assert entry["reason_code"] is None
    assert entry["verdict"] == "fail:no_allowed_reason_code"
    assert entry["evidence"]["rejected_reason"] == "fictional_drift"
    assert result["gate_status"] == "FAIL"


def test_overall_status_is_fail_when_any_instance_fails(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[
            {"instance_id": "good", "patch": "diff --git a/x b/x\n+y\n", "prefix": "sage"},
            {"instance_id": "bad", "patch": "", "prefix": "sage"},
        ],
        annotations=[
            {
                "instance_id": "bad",
                "patch": "",
                "_timeout": True,
                "_latency_ms": 60_000,
            }
        ],
    )
    original = gate.categorize_timeout
    try:
        gate.categorize_timeout = lambda **kw: {  # type: ignore[assignment]
            "reason_code": "fictional_drift",
            "last_stage": None,
            "elapsed_ms_by_stage": {},
            "provider_attempted": False,
            "model_id_final": None,
            "provider_final": None,
        }
        _write_events(events_dir, "bad", events=[])
        result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    finally:
        gate.categorize_timeout = original

    assert result["n_pass"] == 1
    assert result["n_fail"] == 1
    assert result["gate_status"] == "FAIL"


def test_main_writes_gate_result_json_and_returns_exit_code(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[
            {"instance_id": "ok", "patch": "diff --git a/a b/a\n+1\n", "prefix": "sage"}
        ],
        annotations=[],
    )
    output_path = tmp_path / "gate_result.json"
    exit_code = gate.main(
        [
            "--predictions",
            str(pj),
            "--events-dir",
            str(events_dir),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "PASS"
    assert payload["schema_version"] == "canary_pre_grader_gate_v1"
    assert "non_empty_patch" in payload["per_instance"][0]["verdict"]
    # allowed_reason_codes must be a sorted list of the 8 enum members
    # (repo_unavailable added by RESOLUTION_UNBLOCKERS 2026-06-10).
    assert payload["allowed_reason_codes"] == sorted(
        {
            "scoring_boot_impossible",
            "provider_call_timeout",
            "reasoner_thinking_overflow",
            "stage_deadlock",
            "no_patch_extracted",
            "task_budget_exhausted",
            "no_patch_to_verify",
            "repo_unavailable",
        }
    )


def test_main_exits_nonzero_on_failure(tmp_path: Path) -> None:
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "bad", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "bad",
                "patch": "",
                "_timeout": True,
                "_latency_ms": 60_000,
            }
        ],
    )
    output_path = tmp_path / "gate_result.json"
    original = gate.categorize_timeout
    try:
        gate.categorize_timeout = lambda **kw: {  # type: ignore[assignment]
            "reason_code": "fictional_drift",
            "last_stage": None,
            "elapsed_ms_by_stage": {},
            "provider_attempted": False,
            "model_id_final": None,
            "provider_final": None,
        }
        _write_events(events_dir, "bad", events=[])
        exit_code = gate.main(
            [
                "--predictions",
                str(pj),
                "--events-dir",
                str(events_dir),
                "--output",
                str(output_path),
            ]
        )
    finally:
        gate.categorize_timeout = original

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "FAIL"


def test_real_canary_n1_artefact_passes(tmp_path: Path) -> None:
    """Smoke test against the real B2 step 4 canary artefacts shipped
    earlier today. They MUST pass the gate because the empty patch came
    from a provider-call timeout that A5 categorize_timeout classifies
    correctly. If this test ever flips to fail, the canary instrumentation
    has drifted.
    """
    canary_dir = (
        _REPO_ROOT
        / "docs"
        / "benchmarks"
        / "2026-05-11-canary-n1-graded-prep-5a7ed115"
    )
    if not canary_dir.exists():
        pytest.skip(f"Canary fixture {canary_dir} not present")

    result = gate.run_gate(
        predictions_path=canary_dir / "predictions.json",
        events_dir=canary_dir / "per_task",
    )
    assert result["gate_status"] == "PASS", result
    [entry] = result["per_instance"]
    assert entry["reason_code"] == "provider_call_timeout"


def test_pre_recorded_repo_unavailable_passes(tmp_path: Path) -> None:
    """RESOLUTION_UNBLOCKERS (review blocker, 2026-06-10): a fail-closed
    repo-skip prediction carries the pre-recorded repo_unavailable reason
    and must PASS the gate as an INFRA classification — never fall through
    to no_allowed_reason_code."""
    pj, _, events_dir = _write_predictions(
        tmp_path,
        predictions=[{"instance_id": "inst-skip", "patch": "", "prefix": "sage"}],
        annotations=[
            {
                "instance_id": "inst-skip",
                "patch": "",
                "_reason_code": "repo_unavailable",
                "_timeout": False,
            }
        ],
    )
    result = gate.run_gate(predictions_path=pj, events_dir=events_dir)
    assert result["gate_status"] == "PASS"
    [entry] = result["per_instance"]
    assert entry["verdict"] == "pass:repo_unavailable"
    assert entry["evidence"]["source"] == "pre_recorded"
