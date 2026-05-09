"""Tests for the SWE-bench Pro Arm D dry-run runner."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "run_dryrun_arm_d.py"
).resolve()
_SPEC = importlib.util.spec_from_file_location("run_dryrun_arm_d", _SCRIPT_PATH)
arm_d = importlib.util.module_from_spec(_SPEC)
sys.modules["run_dryrun_arm_d"] = arm_d
assert _SPEC.loader is not None
_SPEC.loader.exec_module(arm_d)


class _FakeFormatModule:
    def format_patch(
        self,
        instance_id: str,
        patch: str,
        prefix: str | None = None,
    ) -> dict[str, str]:
        record = {"instance_id": instance_id, "patch": patch}
        if prefix is not None:
            record["prefix"] = prefix
        return record

    def write_predictions(
        self,
        records: list[dict[str, str]],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(records, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def test_validate_learning_evidence_records_pass(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, Any]] = []
    source_trace = tmp_path / "source"
    source_trace.mkdir()
    (source_trace / "01CANARYRUN000000000001.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )

    def fake_run(cmd, *, text, capture_output, timeout):
        calls.append(
            {
                "cmd": cmd,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(cmd, 0, '{"ok":true,"records":2}', "")

    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._validate_learning_evidence(
        source_trace,
        "01CANARYRUN000000000001",
        archive_trace_dir=tmp_path / "archive",
        expect_default_pipeline_learn=True,
    )

    assert result["status"] == "pass"
    assert result["validator_stdout"] == '{"ok":true,"records":2}'
    assert (tmp_path / "archive" / "01CANARYRUN000000000001.jsonl").is_file()
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--run-id") + 1] == "01CANARYRUN000000000001"
    assert cmd[cmd.index("--mode") + 1] == "evidence-boundary"
    assert "--expect-default-pipeline-learn" in cmd


def test_validate_learning_evidence_fails_closed_on_schema_error(
    monkeypatch,
    tmp_path,
) -> None:
    source_trace = tmp_path / "source"
    source_trace.mkdir()

    def fake_run(cmd, *, text, capture_output, timeout):
        return subprocess.CompletedProcess(cmd, 1, "", "missing sidecar")

    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._validate_learning_evidence(
        source_trace,
        "01CANARYRUN000000000002",
        archive_trace_dir=tmp_path / "archive",
        expect_default_pipeline_learn=False,
    )

    assert result["status"] == "no_go"
    assert result["reason_code"] == "validator_failed"
    assert result["validator_stderr"] == "missing sidecar"


def test_run_rejects_mock_learning_evidence(tmp_path) -> None:
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "task-1", "problem_statement": "x"}]),
        encoding="utf-8",
    )

    exit_code = asyncio.run(
        arm_d.run(
            instances_json,
            tmp_path / "out",
            mock=True,
            limit=1,
            budget_usd=1.0,
            tier="budget",
            prefix="test",
            claim_default_pipeline_learning_evidence=True,
            expect_default_pipeline_learn=False,
        )
    )

    assert exit_code == 2


def test_run_returns_failure_when_learning_evidence_gate_fails(
    monkeypatch,
    tmp_path,
) -> None:
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "task-1", "problem_statement": "x"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(arm_d, "_load_format_patch_module", _FakeFormatModule)

    async def fake_run_one_task(*args, **kwargs):
        return {
            "summary": {
                "instance_id": "task-1",
                "exit_code": 0,
                "latency_ms": 1,
                "total_cost_usd": 0.01,
                "extracted_patch_present": False,
                "extracted_patch_chars": 0,
                "mock": False,
                "learning_evidence_boundary": {
                    "claimed": True,
                    "status": "no_go",
                    "reason_code": "validator_failed",
                },
            },
            "record": {"instance_id": "task-1", "patch": "", "prefix": "test"},
        }

    monkeypatch.setattr(arm_d, "_run_one_task", fake_run_one_task)

    exit_code = asyncio.run(
        arm_d.run(
            instances_json,
            tmp_path / "out",
            mock=False,
            limit=1,
            budget_usd=1.0,
            tier="budget",
            prefix="test",
            claim_default_pipeline_learning_evidence=True,
            expect_default_pipeline_learn=True,
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    predictions = json.loads((tmp_path / "out" / "predictions.json").read_text())
    assert exit_code == 3
    assert summary["learning_evidence_gate"]["status"] == "NO_GO"
    assert summary["learning_evidence_gate"]["failed"] == 1
    assert summary["learning_evidence_gate"]["expect_default_pipeline_learn"] is True
    assert set(predictions[0]) == {"instance_id", "patch", "prefix"}
