"""Tests for the SWE-bench Pro Arm D dry-run runner."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

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


def test_run_writes_predictions_jsonl_with_canary_audit_fields(
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
                "extracted_patch_present": True,
                "extracted_patch_chars": 12,
                "mock": False,
                "_verifier_repair_budget_usd": 0.0,
                "_diff_verifier_mismatches": [],
                "_diff_verifier_outcome": "observe_not_run",
                "model_id_final": "deepseek-v4-flash",
                "provider_final": "deepseek",
                "learning_evidence_boundary": {
                    "claimed": False,
                    "status": "skipped",
                    "reason_code": "not_claimed",
                },
            },
            "record": {"instance_id": "task-1", "patch": "diff --git\n", "prefix": "test"},
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
        )
    )

    assert exit_code == 0
    predictions_jsonl = tmp_path / "out" / "predictions.jsonl"
    assert predictions_jsonl.is_file()
    rows = [
        json.loads(line)
        for line in predictions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert {
        "_verifier_repair_budget_usd",
        "_diff_verifier_mismatches",
        "_diff_verifier_outcome",
        "model_id_final",
        "provider_final",
        "total_cost_usd",
    } <= set(rows[0])
    assert rows[0]["patch"] == "diff --git\n"


def test_run_global_budget_uses_canary_cap_not_legacy_five_dollars(
    monkeypatch,
    tmp_path,
) -> None:
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps(
            [
                {"instance_id": "task-1", "problem_statement": "x"},
                {"instance_id": "task-2", "problem_statement": "y"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arm_d, "_load_format_patch_module", _FakeFormatModule)
    calls = 0

    async def fake_run_one_task(*args, **kwargs):
        nonlocal calls
        calls += 1
        instance_id = f"task-{calls}"
        return {
            "summary": {
                "instance_id": instance_id,
                "exit_code": 0,
                "latency_ms": 1,
                "total_cost_usd": 5.01 if calls == 1 else 0.25,
                "extracted_patch_present": False,
                "extracted_patch_chars": 0,
                "mock": False,
                "learning_evidence_boundary": {
                    "claimed": False,
                    "status": "skipped",
                    "reason_code": "not_claimed",
                },
            },
            "record": {"instance_id": instance_id, "patch": "", "prefix": "test"},
        }

    monkeypatch.setattr(arm_d, "_run_one_task", fake_run_one_task)

    exit_code = asyncio.run(
        arm_d.run(
            instances_json,
            tmp_path / "out",
            mock=False,
            limit=2,
            budget_usd=5.0,
            global_budget_usd=25.0,
            tier="budget",
            prefix="test",
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert exit_code == 0
    assert calls == 2
    assert summary["tasks_run"] == 2
    assert summary["budget"]["global_budget_usd"] == 25.0


def test_run_blocks_task_that_would_exceed_global_budget(
    monkeypatch,
    tmp_path,
) -> None:
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps(
            [
                {"instance_id": "task-1", "problem_statement": "x"},
                {"instance_id": "task-2", "problem_statement": "y"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arm_d, "_load_format_patch_module", _FakeFormatModule)
    calls = 0

    async def fake_run_one_task(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "summary": {
                "instance_id": "task-1",
                "exit_code": 0,
                "latency_ms": 1,
                "total_cost_usd": 2.0,
                "extracted_patch_present": False,
                "extracted_patch_chars": 0,
                "mock": False,
                "learning_evidence_boundary": {
                    "claimed": False,
                    "status": "skipped",
                    "reason_code": "not_claimed",
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
            limit=2,
            budget_usd=5.0,
            global_budget_usd=6.0,
            tier="budget",
            prefix="test",
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert exit_code == 0
    assert calls == 1
    assert summary["tasks_run"] == 1
    assert summary["budget"]["stop_reasons"] == [
        "task_budget_would_exceed_global_cap"
    ]


def test_timeout_task_result_recovers_cost_from_node_completed_events(tmp_path: Path) -> None:
    """cgpro NEXT_BLOCK_ID=COST_TRACKING_ZERO_COST_RUNNER_TIMEOUT_FIX
    regression: when the runner times out but a `node_completed`
    event has already recorded a real `cost_usd > 0`, the timeout
    summary MUST surface that cost (via
    `_observed_event_cost_usd` from the event audit) instead of
    hardcoding $0.00. The real-canary at HEAD `7d631f76` exposed
    this bug: trace had cost_usd=$0.136 from node_completed, summary
    reported $0.00.
    """
    instance_id = "task-cost-recovery"
    per_task_events = tmp_path / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True, exist_ok=True)
    # Seed events with model_assigned + node_started + node_completed
    # carrying cost_usd at the top level (mirrors RuntimeEventLog
    # _Node event shape).
    seed_events = [
        {
            "event_type": "task_started",
            "seq": 0,
            "parent_event_id": 0,
        },
        {
            "event_type": "model_assigned",
            "seq": 1,
            "parent_event_id": 0,
            "node_id": "0",
            "node_role": "actor",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "model_assigned": True,
        },
        {
            "event_type": "node_started",
            "seq": 2,
            "parent_event_id": 1,
            "node_id": "0",
            "node_role": "actor",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
        },
        {
            "event_type": "node_completed",
            "seq": 3,
            "parent_event_id": 2,
            "node_id": "0",
            "node_role": "actor",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "cost_usd": 0.13637598,
            "latency_ms": 200700.0,
            "payload": "model output text...",
        },
    ]
    with per_task_events.open("w", encoding="utf-8", newline="\n") as handle:
        for e in seed_events:
            handle.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")

    result = arm_d._timeout_task_result(
        task={"instance_id": instance_id},
        output_dir=tmp_path,
        prefix="test",
        fmt_module=_FakeFormatModule(),
        task_timeout_s=300.0,
        expect_default_pipeline_learn=False,
    )
    summary = result["summary"]
    assert summary["timeout"] is True
    assert summary["total_cost_usd"] == pytest.approx(0.13637598, rel=1e-6), (
        "timeout summary MUST recover cost from node_completed event "
        "instead of hardcoding $0.00 — cgpro 2026-05-12 fix"
    )
    assert summary["_total_cost_usd_source"] == "event_audit_observed_event_cost_usd"
    # LLM execution evidence + non-zero cost → no integrity warning
    assert summary["cost_integrity_warning"] is None


def test_timeout_task_result_emits_cost_integrity_warning_when_evidence_but_zero_cost(tmp_path: Path) -> None:
    """cgpro acceptance criterion #5: when LLM execution evidence
    exists (node_started / model_assigned events with a model_id)
    but no cost_usd > 0 was recorded, the timeout summary MUST
    surface a `cost_integrity_warning` so downstream budget audits
    don't treat this as a clean $0.00 spend.
    """
    instance_id = "task-zero-cost-warning"
    per_task_events = tmp_path / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True, exist_ok=True)
    # node_started but NO node_completed (i.e. agent ran but timed
    # out before any node completed). LLM execution evidence
    # present (model_id observed) but no cost_usd.
    seed_events = [
        {
            "event_type": "task_started",
            "seq": 0,
            "parent_event_id": 0,
        },
        {
            "event_type": "model_assigned",
            "seq": 1,
            "parent_event_id": 0,
            "node_id": "0",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "model_assigned": True,
        },
        {
            "event_type": "node_started",
            "seq": 2,
            "parent_event_id": 1,
            "node_id": "0",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
        },
        # No node_completed — agent timed out mid-call
    ]
    with per_task_events.open("w", encoding="utf-8", newline="\n") as handle:
        for e in seed_events:
            handle.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")

    result = arm_d._timeout_task_result(
        task={"instance_id": instance_id},
        output_dir=tmp_path,
        prefix="test",
        fmt_module=_FakeFormatModule(),
        task_timeout_s=300.0,
        expect_default_pipeline_learn=False,
    )
    summary = result["summary"]
    assert summary["total_cost_usd"] == 0.0
    assert summary["_total_cost_usd_source"] == "no_cost_evidence"
    warning = summary["cost_integrity_warning"]
    assert warning is not None, (
        "cost_integrity_warning MUST fire when LLM execution evidence "
        "exists but recorded cost is zero — cgpro DESIGN_LOCK 2026-05-12 acceptance #5"
    )
    assert warning["reason_code"] == "llm_execution_observed_zero_cost"


def test_timeout_task_result_no_warning_when_no_llm_evidence(tmp_path: Path) -> None:
    """Defensive: if the trace has NO LLM execution evidence (no
    model_assigned, no node_started, no node_completed), the
    integrity warning should NOT fire — $0.00 cost is legitimately
    correct, not a missing-attribution case.
    """
    instance_id = "task-no-llm-evidence"
    per_task_events = tmp_path / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True, exist_ok=True)
    seed_events = [
        {
            "event_type": "task_started",
            "seq": 0,
            "parent_event_id": 0,
        },
        # No model_assigned, no node events — agent never got past
        # classification before timeout.
    ]
    with per_task_events.open("w", encoding="utf-8", newline="\n") as handle:
        for e in seed_events:
            handle.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")

    result = arm_d._timeout_task_result(
        task={"instance_id": instance_id},
        output_dir=tmp_path,
        prefix="test",
        fmt_module=_FakeFormatModule(),
        task_timeout_s=300.0,
        expect_default_pipeline_learn=False,
    )
    summary = result["summary"]
    assert summary["total_cost_usd"] == 0.0
    assert summary["cost_integrity_warning"] is None


def test_run_timeout_writes_fail_closed_task_artifacts(monkeypatch, tmp_path) -> None:
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "task-1", "problem_statement": "x"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(arm_d, "_load_format_patch_module", _FakeFormatModule)

    async def fake_run_one_task(*args, **kwargs):
        await asyncio.sleep(1)
        raise AssertionError("timeout should cancel this task")

    monkeypatch.setattr(arm_d, "_run_one_task", fake_run_one_task)

    exit_code = asyncio.run(
        arm_d.run(
            instances_json,
            tmp_path / "out",
            mock=False,
            limit=1,
            budget_usd=5.0,
            global_budget_usd=25.0,
            task_timeout_s=0.01,
            tier="budget",
            prefix="test",
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert exit_code == 0
    assert summary["timeout_gate"] == {"status": "PASS", "timeouts": 1}
    assert summary["task_summaries"][0]["timeout"] is True
    assert summary["task_summaries"][0]["learning_evidence_boundary"]["status"] == "no_go"
    assert (tmp_path / "out" / "per_task" / "task-1.events.jsonl").is_file()


def test_run_sage_cli_sends_prompt_as_jsonl_command(monkeypatch, tmp_path) -> None:
    class _FakeStdin:
        def __init__(self) -> None:
            self.data = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.data += data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _AsyncBytesStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.stdout = _AsyncBytesStream(
                [
                    b'{"event_type":"final_result","payload":{"result":"done"}}\n',
                    (
                        b'{"event_type":"cli_complete","run_id":"RUN1",'
                        b'"payload":{"outcome":"success","total_cost_usd":0.01,'
                        b'"trace_dir":"trace"}}\n'
                    ),
                ]
            )
            self.stderr = _AsyncBytesStream([])
            self.returncode = 0
            self.terminated = False
            self.killed = False

        async def wait(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    created: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        proc = _FakeProcess()
        created["proc"] = proc
        created["args"] = args
        return proc

    monkeypatch.setattr(arm_d.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(
        arm_d._run_sage_cli(
            "fix the bug",
            budget_usd=5.0,
            output_events_path=tmp_path / "events.jsonl",
            tier="budget",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )

    args = created["args"]
    assert "--provider-allowlist" in args
    assert args[args.index("--provider-allowlist") + 1] == "google,deepseek"
    assert "--provider-denylist" in args
    assert args[args.index("--provider-denylist") + 1] == "openai"
    written = created["proc"].stdin.data.decode("utf-8")
    command = json.loads(written)
    assert command == {
        "command": "prompt",
        "args": {"task": "fix the bug", "budget_usd": 5.0},
    }
    assert written.endswith("\n")
    assert created["proc"].stdin.closed is True
    assert result["total_cost_usd"] == 0.01


def test_run_writes_launch_manifest_and_blocks_unfrozen_manifest(
    tmp_path,
) -> None:
    manifest = tmp_path / "cycle-13-canary-manifest.md"
    manifest.write_text(
        "# Canary\n\n| Parameter | Value |\n|-----------|-------|\n"
        "| Commit SHA | `<SET_AT_LAUNCH>` |\n",
        encoding="utf-8",
    )
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            task_timeout_s=120.0,
            tier="budget",
            prefix="test",
            manifest_path=manifest,
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )

    launch = json.loads((tmp_path / "out" / "launch_manifest.json").read_text())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert exit_code == 0
    assert (tmp_path / "out" / "launch_manifest.md").read_text(encoding="utf-8") == (
        manifest.read_text(encoding="utf-8")
    )
    assert launch["inputs"]["manifest"]["sha256"]
    assert launch["inputs"]["instances_json"]["sha256"]
    assert isinstance(launch["repo"]["dirty"], bool)
    assert "status_short" in launch["repo"]
    assert launch["manifest_gate"]["status"] == "BLOCKED"
    assert "manifest_commit_not_frozen" in launch["manifest_gate"]["reasons"]
    assert summary["launch_manifest_path"] == "launch_manifest.json"
    assert summary["acceptance_gate_results"]["manifest_gate"]["status"] == "BLOCKED"
    assert summary["canary_decision"] == "BLOCKED"


def test_run_sage_cli_extracts_final_model_and_provider_from_events(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeStdin:
        def write(self, data: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    class _AsyncBytesStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.stdout = _AsyncBytesStream(
                [
                    (
                        b'{"event_type":"model_assigned",'
                        b'"payload":{"model_id":"deepseek-v4-flash",'
                        b'"provider_id":"deepseek"}}\n'
                    ),
                    (
                        b'{"event_type":"node_started",'
                        b'"payload":{"model_id":"deepseek-v4-flash",'
                        b'"provider_id":"deepseek"}}\n'
                    ),
                    b'{"event_type":"final_result","payload":{"result":"done"}}\n',
                    (
                        b'{"event_type":"cli_complete","run_id":"RUN1",'
                        b'"payload":{"outcome":"success","total_cost_usd":0.01,'
                        b'"trace_dir":"trace"}}\n'
                    ),
                ]
            )
            self.stderr = _AsyncBytesStream([])
            self.returncode = 0

        async def wait(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess()

    monkeypatch.setattr(arm_d.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(
        arm_d._run_sage_cli(
            "task",
            budget_usd=5.0,
            output_events_path=tmp_path / "events.jsonl",
            tier="budget",
        )
    )

    assert result["model_id_final"] == "deepseek-v4-flash"
    assert result["provider_final"] == "deepseek"


def test_run_provider_gate_blocks_denylisted_provider(monkeypatch, tmp_path) -> None:
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
                "extracted_patch_present": True,
                "extracted_patch_chars": 12,
                "mock": False,
                "provider_final": "openai",
                "model_id_final": "gpt-5.5-pro",
                "_assigned_providers": ["openai"],
                "_execution_providers": ["openai"],
                "learning_evidence_boundary": {
                    "claimed": False,
                    "status": "skipped",
                    "reason_code": "not_claimed",
                },
            },
            "record": {"instance_id": "task-1", "patch": "diff --git\n", "prefix": "test"},
        }

    monkeypatch.setattr(arm_d, "_run_one_task", fake_run_one_task)

    exit_code = asyncio.run(
        arm_d.run(
            instances_json,
            tmp_path / "out",
            mock=False,
            limit=1,
            budget_usd=5.0,
            global_budget_usd=25.0,
            tier="budget",
            prefix="test",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert exit_code == 0
    assert summary["acceptance_gate_results"]["provider_gate"]["status"] == "NO_GO"
    gate = summary["acceptance_gate_results"]["provider_gate"]
    assert gate["observed_providers"] == ["openai"]
    assert gate["execution_denied_providers"] == ["openai"]
    assert summary["canary_decision"] == "NO_GO"


def test_run_provider_gate_uses_all_observed_providers(monkeypatch, tmp_path) -> None:
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
                "exit_code": None,
                "latency_ms": 120000,
                "total_cost_usd": 0.0,
                "extracted_patch_present": False,
                "extracted_patch_chars": 0,
                "mock": False,
                "timeout": True,
                "provider_final": "deepseek",
                "model_id_final": "deepseek-v4-flash",
                "_observed_providers": ["deepseek", "openai", "openrouter"],
                "_assigned_providers": ["deepseek", "openai", "openrouter"],
                "_execution_providers": ["deepseek", "openai", "openrouter"],
                "_observed_model_ids": [
                    "deepseek-v4-flash",
                    "gpt-5.4",
                    "qwen/qwen3.5-plus-02-15",
                ],
                "_observed_event_cost_usd": 0.02672,
                "learning_evidence_boundary": {
                    "claimed": True,
                    "status": "no_go",
                    "reason_code": "task_timeout",
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            tier="budget",
            prefix="test",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    gate = summary["acceptance_gate_results"]["provider_gate"]
    assert exit_code == 0
    assert gate["status"] == "NO_GO"
    assert gate["observed_providers"] == ["deepseek", "openai", "openrouter"]
    assert gate["execution_denied_providers"] == ["openai"]
    assert gate["execution_outside_allowlist"] == ["openai", "openrouter"]


def test_run_provider_gate_accepts_runtime_policy_block(monkeypatch, tmp_path) -> None:
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
                "exit_code": 1,
                "latency_ms": 1,
                "total_cost_usd": 0.0,
                "extracted_patch_present": False,
                "extracted_patch_chars": 0,
                "mock": False,
                "provider_final": "openai",
                "model_id_final": "gpt-5.5-pro",
                "_assigned_providers": ["openai"],
                "_execution_providers": [],
                "_provider_policy_failure_seen": True,
                "learning_evidence_boundary": {
                    "claimed": False,
                    "status": "skipped",
                    "reason_code": "not_claimed",
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            tier="budget",
            prefix="test",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )

    gate = json.loads((tmp_path / "out" / "summary.json").read_text())[
        "acceptance_gate_results"
    ]["provider_gate"]
    assert exit_code == 0
    assert gate["status"] == "PASS"
    assert gate["reason"] == "runtime_provider_policy_enforced"
    assert gate["assigned_denied_providers"] == ["openai"]
    assert gate["execution_denied_providers"] == []


def test_run_writes_aggregate_events_jsonl(tmp_path) -> None:
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            task_timeout_s=120.0,
            tier="budget",
            prefix="test",
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    events_path = tmp_path / "out" / "events.jsonl"
    assert exit_code == 0
    assert summary["events_path"] == "events.jsonl"
    assert events_path.is_file()
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["event_type"] == "synthetic_mock"


def test_timeout_task_result_appends_runner_timeout_to_partial_events(
    tmp_path,
) -> None:
    output_dir = tmp_path / "out"
    instance_id = "task-1"
    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True)
    per_task_events.write_text(
        '{"event_type":"cli_started"}\n'
        '{"event_type":"model_assigned","model_id":"gpt-5.4",'
        '"provider_id":"openai"}\n',
        encoding="utf-8",
        newline="\n",
    )

    result = arm_d._timeout_task_result(
        {"instance_id": instance_id},
        output_dir,
        prefix="test",
        fmt_module=_FakeFormatModule(),
        task_timeout_s=120.0,
        expect_default_pipeline_learn=True,
    )

    rows = [
        json.loads(line)
        for line in per_task_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event_type"] for row in rows] == [
        "cli_started",
        "model_assigned",
        "runner_timeout",
    ]
    assert result["summary"]["model_id_final"] == "gpt-5.4"
    assert result["summary"]["provider_final"] == "openai"
    assert result["summary"]["_observed_providers"] == ["openai"]
    assert result["summary"]["_assigned_providers"] == ["openai"]
    assert result["summary"]["_execution_providers"] == []
    assert result["summary"]["learning_evidence_boundary"][
        "expect_default_pipeline_learn"
    ] is True


def test_timeout_task_result_includes_a5_timeout_categorization(tmp_path) -> None:
    """Block A5: _timeout_task_result must populate `timeout_categorization`
    with last_stage / elapsed_ms_by_stage / provider_attempted /
    reason_code derived from the per-task event log content.
    """
    output_dir = tmp_path / "out"
    instance_id = "task-a5"
    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True)
    # Reasoner overflow scenario: routing + model_assigned, then cli_progress
    # frames stuck in `decompose` covering the bulk of a 60s run.
    per_task_events.write_text(
        '{"event_type":"cli_started"}\n'
        '{"event_type":"routing_decision","payload":{"model_id":"deepseek-v4-pro"}}\n'
        '{"event_type":"model_assigned","payload":{"model_id":"deepseek-v4-pro",'
        '"provider_id":"deepseek"}}\n'
        '{"event_type":"cli_progress","payload":{"stage":"decompose","elapsed_ms":10000}}\n'
        '{"event_type":"cli_progress","payload":{"stage":"decompose","elapsed_ms":25000}}\n'
        '{"event_type":"cli_progress","payload":{"stage":"decompose","elapsed_ms":45000}}\n'
        '{"event_type":"cli_progress","payload":{"stage":"decompose","elapsed_ms":58000}}\n',
        encoding="utf-8",
        newline="\n",
    )

    result = arm_d._timeout_task_result(
        {"instance_id": instance_id},
        output_dir,
        prefix="test",
        fmt_module=_FakeFormatModule(),
        task_timeout_s=60.0,
        expect_default_pipeline_learn=True,
    )

    summary = result["summary"]
    assert summary["timeout"] is True
    assert "timeout_categorization" in summary
    cat = summary["timeout_categorization"]
    assert cat["reason_code"] == "reasoner_thinking_overflow"
    assert cat["last_stage"] == "decompose"
    # Stage duration = last - first elapsed_ms in decompose.
    assert cat["elapsed_ms_by_stage"]["decompose"] == 48_000
    # provider_attempted MUST be False — only model_assigned, no node_started.
    assert cat["provider_attempted"] is False
    # Categorization should also surface model_id_final and provider_final
    # at the summary top level (consistent with event_audit fallback).
    assert summary["model_id_final"] == "deepseek-v4-pro"
    assert summary["provider_final"] == "deepseek"


def test_timeout_task_result_categorization_scoring_boot_impossible(tmp_path) -> None:
    """A5: empty events file (no routing, no progress, no assignment) →
    reason_code=scoring_boot_impossible.
    """
    output_dir = tmp_path / "out"
    instance_id = "task-a5-boot"
    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True)
    # Empty file — pipeline never produced any usable signal before timeout.
    per_task_events.write_text("", encoding="utf-8", newline="\n")

    result = arm_d._timeout_task_result(
        {"instance_id": instance_id},
        output_dir,
        prefix="test",
        fmt_module=_FakeFormatModule(),
        task_timeout_s=60.0,
        expect_default_pipeline_learn=True,
    )

    cat = result["summary"]["timeout_categorization"]
    # The runner_timeout event was appended AFTER the categorization read,
    # but in this branch the events file was empty even when the
    # categorization reads it — the runner_timeout line itself is not
    # one of {cli_progress, model_assigned, node_started, routing_decision}
    # so it does not change the verdict.
    assert cat["reason_code"] == "scoring_boot_impossible"
    assert cat["last_stage"] is None
    assert cat["provider_attempted"] is False


def test_event_audit_extracts_top_level_runtime_fields_and_cost(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        '{"event_type":"routing_decision","model_id":"gpt-5.5-pro"}\n'
        '{"event_type":"model_assigned","model_id":"gpt-5.4",'
        '"provider_id":"openai"}\n'
        '{"event_type":"node_completed","model_id":"deepseek-v4-flash",'
        '"provider_id":"deepseek","cost_usd":0.02672}\n',
        encoding="utf-8",
        newline="\n",
    )

    audit = arm_d._event_audit_from_file(events_path)

    assert audit["model_id_final"] == "deepseek-v4-flash"
    assert audit["provider_final"] == "deepseek"
    assert audit["_observed_model_ids"] == [
        "deepseek-v4-flash",
        "gpt-5.4",
        "gpt-5.5-pro",
    ]
    assert audit["_observed_providers"] == ["deepseek", "openai"]
    assert audit["_assigned_providers"] == ["openai"]
    assert audit["_execution_providers"] == ["deepseek"]
    assert audit["_provider_policy_failure_seen"] is False
    assert audit["_observed_event_cost_usd"] == 0.02672


# ─────────────────────────────────────────────────────────────────────────────
# Block `canary-stage-timing-budget` slice 3 (cgpro DESIGN 2026-05-11):
# named timeout profiles + explicit-override reporting in launch_manifest
# and summary.json.
# ─────────────────────────────────────────────────────────────────────────────


def test_timeout_profiles_enum_includes_default_and_graded_patch_generation() -> None:
    """Both profiles required by cgpro DESIGN must exist with the
    documented timeout values. ``default`` preserves the historical
    120s for plumbing smokes; ``graded_patch_generation`` is the
    900s mid-point of cgpro's 600-1200s envelope.
    """
    assert arm_d._TIMEOUT_PROFILES == {
        "default": 120.0,
        "graded_patch_generation": 900.0,
    }
    assert arm_d._DEFAULT_PROFILE == "default"


def test_run_records_default_profile_metadata_when_unflagged(tmp_path) -> None:
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            task_timeout_s=120.0,
            tier="budget",
            prefix="test",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )
    assert exit_code == 0
    launch = json.loads((tmp_path / "out" / "launch_manifest.json").read_text())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert launch["budget"]["effective_profile"] == "default"
    assert launch["budget"]["profile_timeout_default_s"] == 120.0
    assert launch["budget"]["profile_timeout_override"] is False
    assert launch["budget"]["task_timeout_s"] == 120.0
    assert summary["budget"]["effective_profile"] == "default"
    assert summary["budget"]["profile_timeout_default_s"] == 120.0
    assert summary["budget"]["profile_timeout_override"] is False


def test_run_records_graded_patch_generation_profile(tmp_path) -> None:
    """Caller passes ``profile="graded_patch_generation"`` + the matching
    timeout (the resolver in main() does this for the CLI; here we mimic
    the resolved state). The launch_manifest + summary must reflect that
    the timeout came from the profile, not an explicit override.
    """
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            task_timeout_s=900.0,
            profile="graded_patch_generation",
            profile_timeout_override=False,
            tier="budget",
            prefix="test",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )
    assert exit_code == 0
    launch = json.loads((tmp_path / "out" / "launch_manifest.json").read_text())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert launch["budget"]["effective_profile"] == "graded_patch_generation"
    assert launch["budget"]["profile_timeout_default_s"] == 900.0
    assert launch["budget"]["profile_timeout_override"] is False
    assert launch["budget"]["task_timeout_s"] == 900.0
    assert summary["budget"]["effective_profile"] == "graded_patch_generation"
    assert summary["budget"]["task_timeout_s"] == 900.0


def test_run_records_profile_override_when_explicit_timeout_given(tmp_path) -> None:
    """When the CLI resolver sets ``profile_timeout_override=True`` because
    ``--task-timeout-s`` was explicitly given, the manifest must surface
    that fact + still record the *profile* the user named so post-hoc
    analysis can spot the divergence between profile and actual run.
    """
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
            budget_usd=5.0,
            global_budget_usd=25.0,
            task_timeout_s=600.0,  # explicit, NOT a profile default
            profile="graded_patch_generation",
            profile_timeout_override=True,
            tier="budget",
            prefix="test",
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )
    assert exit_code == 0
    launch = json.loads((tmp_path / "out" / "launch_manifest.json").read_text())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert launch["budget"]["task_timeout_s"] == 600.0
    assert launch["budget"]["profile_timeout_default_s"] == 900.0  # profile's notional default
    assert launch["budget"]["profile_timeout_override"] is True
    assert summary["budget"]["task_timeout_s"] == 600.0
    assert summary["budget"]["profile_timeout_override"] is True


def test_cli_parses_profile_flag_and_resolves_timeout(monkeypatch, tmp_path) -> None:
    """Smoke the argparse + resolver path: invoking main() with --profile
    graded_patch_generation and no --task-timeout-s must call run() with
    task_timeout_s=900 + profile_timeout_override=False. We monkeypatch
    arm_d.run with an async stub that captures kwargs and lets asyncio.run
    await it normally.
    """
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "task-1", "problem_statement": "x"}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.md"
    manifest.write_text("# canary", encoding="utf-8")

    captured: dict[str, Any] = {}

    async def _capture_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(arm_d, "run", _capture_run)

    exit_code = arm_d.main(
        [
            "--instances-json",
            str(instances_json),
            "--output-dir",
            str(tmp_path / "out"),
            "--mock",
            "--profile",
            "graded_patch_generation",
            "--manifest-path",
            str(manifest),
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["task_timeout_s"] == 900.0
    assert captured["kwargs"]["profile"] == "graded_patch_generation"
    assert captured["kwargs"]["profile_timeout_override"] is False


def test_run_sage_cli_terminates_subprocess_after_cli_complete(
    monkeypatch, tmp_path
) -> None:
    """B2 step 4 follow-up (2026-05-11): the SAGE CLI subprocess MAY
    hold stdout open after emitting ``cli_complete``. The on-disk
    evidence: protonmail task in ``2026-05-11-canary-n5-graded`` ran a
    full success pipeline in 186s wall but the runner kept waiting on
    ``async for raw in proc.stdout`` for another ~707s before
    ``asyncio.wait_for(timeout=900)`` fired.

    Fix: as soon as ``cli_complete`` is parsed, break out of the
    stdout loop and ``proc.terminate()`` the subprocess instead of
    waiting for natural EOF.

    This test stubs a subprocess that emits ``cli_complete`` and then
    yields a third frame the runner MUST NOT consume. If the fix
    regresses, the third frame would be read and recorded.
    """
    import asyncio as _asyncio  # local alias to avoid clashing with arm_d.asyncio monkeypatch

    class _FakeStdin:
        def __init__(self) -> None:
            self.data = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.data += data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _CliCompleteThenHangStdout:
        """Emits final_result + cli_complete, then BLOCKS forever.

        Mimics the real post-cli_complete hang. If the runner does
        NOT break out of the stdout loop, this iterator will park the
        test indefinitely (caught by pytest-asyncio's default timeout
        or by a fail-fast assertion below).
        """

        def __init__(self) -> None:
            self._delivered = 0

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            if self._delivered == 0:
                self._delivered = 1
                return b'{"event_type":"final_result","payload":{"result":"done"}}\n'
            if self._delivered == 1:
                self._delivered = 2
                return (
                    b'{"event_type":"cli_complete","run_id":"RUN1",'
                    b'"payload":{"outcome":"success","total_cost_usd":0.01,'
                    b'"trace_dir":"trace"}}\n'
                )
            # Past cli_complete: simulate the hang. If the fix is broken
            # and the runner keeps reading, await forever.
            await _asyncio.sleep(3600)
            raise AssertionError("runner read past cli_complete — fix regressed")

    class _AsyncBytesStreamLocal:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.stdout = _CliCompleteThenHangStdout()
            self.stderr = _AsyncBytesStreamLocal([])
            self._return: int | None = None
            self.terminated = False
            self.killed = False

        @property
        def returncode(self) -> int | None:
            return self._return

        async def wait(self) -> int:
            # Until terminate() / kill() is called, simulate the hang.
            while self._return is None:
                await _asyncio.sleep(0.01)
            return self._return

        def terminate(self) -> None:
            self.terminated = True
            # Allow the next ``wait()`` to resolve.
            self._return = -15

        def kill(self) -> None:
            self.killed = True
            self._return = -9

    created: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        proc = _FakeProcess()
        created["proc"] = proc
        return proc

    monkeypatch.setattr(
        arm_d.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    # Hard wall-clock cap: the fix must make this complete in ~5s
    # (matching the terminate timeout in the runner). 30s gives plenty
    # of margin while still failing fast if the hang regresses.
    result = asyncio.run(
        asyncio.wait_for(
            arm_d._run_sage_cli(
                "fix the bug",
                budget_usd=5.0,
                output_events_path=tmp_path / "events.jsonl",
                tier="budget",
                provider_allowlist=("google",),
                provider_denylist=("openai",),
            ),
            timeout=30.0,
        )
    )

    assert result["cli_complete_payload"] is not None
    assert result["cli_complete_payload"]["outcome"] == "success"
    proc = created["proc"]
    assert proc.terminated is True, "fix must call terminate() after cli_complete"
    # exit_code returned by the runner is the post-terminate returncode.
    assert result["exit_code"] == -15


def test_run_one_task_uses_swebench_template_prompt(monkeypatch, tmp_path) -> None:
    """Slice 7 (2026-05-11): the canary MUST send the canonical
    SWEBENCH_SYSTEM_TEMPLATE (or its SR variant) to the agent, not a
    bare "Produce a unified diff" string. Empirical evidence: the
    first real N=5 graded canary run (commit 77b6dd98) produced 0/5
    patches because the agent received only the problem_statement +
    a single-line instruction, returned synthesizer reasoning text,
    and the extractor found no `diff --git` header.

    This test stubs the SAGE CLI subprocess + extractor + Pro format
    module, then asserts the prompt sent to ``_run_sage_cli`` starts
    with the canonical SWE-bench instruction sentence.
    """
    instance = {
        "instance_id": "test-instance-1",
        "repo": "owner/repo",
        "base_commit": "deadbeef",
        "problem_statement": "Bug in `foo` when `bar` is None.",
    }

    captured_prompts: list[str] = []

    async def _capture_cli(prompt, *, budget_usd, output_events_path, tier,
                          provider_allowlist=(), provider_denylist=(), cwd=None, **_ignored):
        captured_prompts.append(prompt)
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        output_events_path.write_text("", encoding="utf-8")
        return {
            "exit_code": 0,
            "latency_ms": 100,
            "final_result_payload": {"result": "no diff here, just reasoning"},
            "cli_complete_payload": {
                "outcome": "success",
                "total_cost_usd": 0.0,
                "trace_dir": str(tmp_path / "trace"),
            },
            "run_id": "RUN-X",
            "model_id_final": "fake-model",
            "provider_final": "fake-provider",
            "total_cost_usd": 0.0,
        }

    monkeypatch.setattr(arm_d, "_run_sage_cli", _capture_cli)
    # Stub repo setup so the test doesn't try a real git clone over
    # the network. Slice 8 path: _setup_repo_for_canary is called
    # before the CLI subprocess; tests don't need a real checkout.
    monkeypatch.setattr(
        arm_d,
        "_setup_repo_for_canary",
        lambda inst: {
            "repo_context_status": "ready",
            "repo_dir": str(tmp_path / "fake_repo"),
            "repo_url": f"https://github.com/{inst['repo']}.git",
            "base_commit": inst["base_commit"],
            "checkout_sha": inst["base_commit"],
            "clone_elapsed_ms": 0,
            "fetch_fallback_used": False,
            "failure_reason": None,
        },
    )
    monkeypatch.setattr(
        arm_d, "_cleanup_repo_dir", lambda repo_dir, *, tmp_root=None: "removed"
    )

    fmt_module = _FakeFormatModule()

    asyncio.run(
        arm_d._run_one_task(
            instance,
            tmp_path / "out",
            mock=False,
            budget_usd=5.0,
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
            prefix="test",
            fmt_module=fmt_module,
            claim_default_pipeline_learning_evidence=False,
            expect_default_pipeline_learn=False,
        )
    )

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    # The SWEBENCH_SYSTEM_TEMPLATE opens with this sentence — the
    # load-bearing instruction that tells the agent to emit a diff.
    assert (
        "You are an expert software engineer" in prompt
    ), f"Prompt missing SWEBENCH_SYSTEM_TEMPLATE opener:\n{prompt[:300]}"
    # And carries the strict Patch Format section.
    assert "## Patch Format" in prompt
    # AND embeds the instance's problem_statement verbatim.
    assert "Bug in `foo` when `bar` is None." in prompt
    # AND identifies the repo + base_commit, which the agent needs.
    assert "owner/repo" in prompt
    assert "deadbeef" in prompt


def test_run_one_task_extracts_fenced_diff_via_swebench_extractor(
    monkeypatch, tmp_path
) -> None:
    """Slice 7 (2026-05-11): the local Tier-2.1 dumb extractor missed
    diffs wrapped in markdown ```diff fences. The slice-7 upgrade
    delegates to ``swebench_bench._extract_patch`` which handles all
    three formats (raw, fenced, embedded). Smoke that a fenced diff
    in the agent reply is now extracted to a non-empty patch.
    """
    instance = {
        "instance_id": "test-instance-2",
        "repo": "owner/repo",
        "base_commit": "deadbeef",
        "problem_statement": "Fix the off-by-one.",
    }

    fake_response = (
        "Looking at the code, the loop bound is wrong. Here is the fix:\n\n"
        "```diff\n"
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -10,3 +10,3 @@\n"
        " def f(xs):\n"
        "-    for i in range(len(xs) + 1):\n"
        "+    for i in range(len(xs)):\n"
        "         yield xs[i]\n"
        "```\n\n"
        "That removes the off-by-one."
    )

    async def _fake_cli(prompt, *, budget_usd, output_events_path, tier,
                       provider_allowlist=(), provider_denylist=(), cwd=None, **_ignored):
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        output_events_path.write_text("", encoding="utf-8")
        return {
            "exit_code": 0,
            "latency_ms": 100,
            "final_result_payload": {"result": fake_response},
            "cli_complete_payload": {
                "outcome": "success",
                "total_cost_usd": 0.01,
                "trace_dir": str(tmp_path / "trace"),
            },
            "run_id": "RUN-Y",
            "model_id_final": "fake-model",
            "provider_final": "fake-provider",
            "total_cost_usd": 0.01,
        }

    monkeypatch.setattr(arm_d, "_run_sage_cli", _fake_cli)
    monkeypatch.setattr(
        arm_d,
        "_setup_repo_for_canary",
        lambda inst: {
            "repo_context_status": "ready",
            "repo_dir": str(tmp_path / "fake_repo"),
            "repo_url": f"https://github.com/{inst['repo']}.git",
            "base_commit": inst["base_commit"],
            "checkout_sha": inst["base_commit"],
            "clone_elapsed_ms": 0,
            "fetch_fallback_used": False,
            "failure_reason": None,
        },
    )
    monkeypatch.setattr(
        arm_d, "_cleanup_repo_dir", lambda repo_dir, *, tmp_root=None: "removed"
    )
    fmt_module = _FakeFormatModule()

    result = asyncio.run(
        arm_d._run_one_task(
            instance,
            tmp_path / "out",
            mock=False,
            budget_usd=5.0,
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
            prefix="test",
            fmt_module=fmt_module,
            claim_default_pipeline_learning_evidence=False,
            expect_default_pipeline_learn=False,
        )
    )

    summary = result["summary"]
    record = result["record"]
    assert summary["extracted_patch_present"] is True
    chars = summary["extracted_patch_chars"]
    assert chars > 0, f"expected non-empty patch, got chars={chars}"
    # The Pro record carries the extracted diff verbatim.
    assert "diff --git" in record.get("patch", "")


def test_run_sage_cli_natural_eof_path_still_works(monkeypatch, tmp_path) -> None:
    """The break-on-cli_complete fix MUST NOT regress the cooperative
    path: when the subprocess closes stdout naturally after
    cli_complete (well-behaved SAGE CLI), the runner reaches the loop
    end without needing to terminate, awaits proc.wait() once, and
    returns. terminate() must NOT be called in this case.
    """
    class _AsyncBytesStreamLocal:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    class _FakeStdin:
        def __init__(self) -> None:
            self.data = b""

        def write(self, data: bytes) -> None:
            self.data += data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            pass

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.stdout = _AsyncBytesStreamLocal(
                [
                    b'{"event_type":"final_result","payload":{"result":"done"}}\n',
                    (
                        b'{"event_type":"cli_complete","run_id":"R","'
                        b'payload":{"outcome":"success","total_cost_usd":0.0,'
                        b'"trace_dir":"t"}}\n'
                    ),
                ]
            )
            self.stderr = _AsyncBytesStreamLocal([])
            self.returncode = 0
            self.terminated = False
            self.killed = False

        async def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    created: dict[str, Any] = {}

    async def fake_create(*args, **kwargs):
        proc = _FakeProcess()
        created["proc"] = proc
        return proc

    monkeypatch.setattr(arm_d.asyncio, "create_subprocess_exec", fake_create)

    result = asyncio.run(
        arm_d._run_sage_cli(
            "fix the bug",
            budget_usd=5.0,
            output_events_path=tmp_path / "events.jsonl",
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
        )
    )
    assert result["cli_complete_payload"]["outcome"] == "success"
    # Subprocess returned 0 naturally → terminate must NOT have fired.
    # Per the fix: terminate runs only when ``proc.returncode is None``
    # immediately after seeing cli_complete. Here returncode is 0
    # because wait() returns 0 — but the check is on .returncode pre-wait.
    # _FakeProcess.returncode is 0 from init, so terminate path is skipped.
    assert created["proc"].terminated is False
    assert created["proc"].killed is False
    assert result["exit_code"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Block `canary-real-repo-context` slice 8 (cgpro DESIGN 2026-05-11):
# per-task repo checkout so SWE-bench Pro tools see real source.
# ─────────────────────────────────────────────────────────────────────────────


class _FakeCompletedProcess:
    """Stand-in for ``subprocess.CompletedProcess`` in unit tests."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mock_subprocess_factory(script: list[Any]) -> Any:
    """Build a fake ``subprocess.run`` from a list of return values.

    Each script entry is one of:
    - ``_FakeCompletedProcess`` — returned for the next call.
    - ``"timeout"`` — raises ``subprocess.TimeoutExpired``.
    - callable — invoked with the kwargs and must return a
      ``_FakeCompletedProcess``.
    """
    calls: list[tuple[Any, ...]] = []
    idx = [0]

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        argv = args[0] if args else kwargs.get("args")
        calls.append(tuple(argv))
        if idx[0] >= len(script):
            raise AssertionError(f"subprocess.run called more times than scripted: {argv}")
        entry = script[idx[0]]
        idx[0] += 1
        if entry == "timeout":
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
        if callable(entry):
            return entry(*args, **kwargs)
        return entry

    return _fake_run, calls


def test_setup_repo_for_canary_happy_path(monkeypatch, tmp_path) -> None:
    """Shallow clone + detached checkout works first try → status=ready,
    checkout_sha populated from rev-parse, no fetch fallback used.
    """
    script = [
        _FakeCompletedProcess(0),                # clone
        _FakeCompletedProcess(0),                # checkout --detach <base>
        _FakeCompletedProcess(0, stdout=b"abc123def456\n"),  # rev-parse HEAD
    ]
    fake_run, calls = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    instance = {
        "instance_id": "x",
        "repo": "owner/proj",
        "base_commit": "abc123def456",
    }
    result = arm_d._setup_repo_for_canary(instance)

    assert result["repo_context_status"] == "ready"
    assert result["repo_url"] == "https://github.com/owner/proj.git"
    assert result["base_commit"] == "abc123def456"
    assert result["checkout_sha"] == "abc123def456"
    assert result["fetch_fallback_used"] is False
    assert result["repo_dir"] is not None
    assert result["repo_dir"].endswith("proj") or "proj" in result["repo_dir"]
    assert result["clone_elapsed_ms"] >= 0
    assert result["failure_reason"] is None
    # 3 subprocess calls: clone, checkout, rev-parse.
    assert len(calls) == 3
    assert calls[0][0] == "git" and "clone" in calls[0]
    assert calls[1][0] == "git" and "checkout" in calls[1]
    assert calls[2][0] == "git" and "rev-parse" in calls[2]

    # Cleanup the tempdir the function created.
    arm_d._cleanup_repo_dir(result["repo_dir"], tmp_root=os.path.dirname(result["repo_dir"]))


def test_setup_repo_for_canary_fetch_fallback(monkeypatch, tmp_path) -> None:
    """Shallow clone OK, first checkout fails (commit not in shallow
    clone), fetch --depth 1 origin <base> succeeds, second checkout
    succeeds → status=ready, fetch_fallback_used=True.
    """
    script = [
        _FakeCompletedProcess(0),                # clone OK
        _FakeCompletedProcess(1, stderr=b"unknown revision"),  # first checkout fails
        _FakeCompletedProcess(0),                # fetch fallback OK
        _FakeCompletedProcess(0),                # second checkout OK
        _FakeCompletedProcess(0, stdout=b"deadbeef\n"),  # rev-parse
    ]
    fake_run, calls = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    instance = {
        "instance_id": "y",
        "repo": "owner/proj",
        "base_commit": "deadbeef",
    }
    result = arm_d._setup_repo_for_canary(instance)

    assert result["repo_context_status"] == "ready"
    assert result["fetch_fallback_used"] is True
    assert result["checkout_sha"] == "deadbeef"
    assert len(calls) == 5
    assert "fetch" in calls[2]
    arm_d._cleanup_repo_dir(result["repo_dir"], tmp_root=os.path.dirname(result["repo_dir"]))


def test_setup_repo_for_canary_missing_inputs() -> None:
    """No repo or no base_commit → status=missing_inputs, repo_dir=None,
    no subprocess calls.
    """
    r1 = arm_d._setup_repo_for_canary({"instance_id": "z"})
    assert r1["repo_context_status"] == "missing_inputs"
    assert r1["repo_dir"] is None

    r2 = arm_d._setup_repo_for_canary({"instance_id": "z", "repo": "x/y"})
    assert r2["repo_context_status"] == "missing_inputs"
    assert r2["repo_dir"] is None


def test_setup_repo_for_canary_clone_failed(monkeypatch) -> None:
    """Clone fails AND the targeted-fetch fallback fails too → clone_failed
    with the combined failure chain (both causes auditable)."""
    script = [
        _FakeCompletedProcess(128, stderr=b"could not resolve host"),  # clone
        _FakeCompletedProcess(0),                    # git init OK
        _FakeCompletedProcess(0),                    # remote add OK
        _FakeCompletedProcess(128, stderr=b"could not resolve host"),  # fetch
    ]
    fake_run, _ = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._setup_repo_for_canary({
        "instance_id": "z",
        "repo": "ghost/repo",
        "base_commit": "abc",
    })
    assert result["repo_context_status"] == "clone_failed"
    assert "could not resolve host" in (result["failure_reason"] or "")
    assert "targeted_fetch_failed" in (result["failure_reason"] or "")
    assert result["repo_dir"] is None


def test_setup_repo_for_canary_fetch_failed(monkeypatch) -> None:
    """Clone OK, checkout fails, fetch also fails → status=fetch_failed."""
    script = [
        _FakeCompletedProcess(0),                # clone OK
        _FakeCompletedProcess(1, stderr=b"unknown revision"),  # checkout fails
        _FakeCompletedProcess(128, stderr=b"object not found upstream"),  # fetch fails
    ]
    fake_run, _ = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._setup_repo_for_canary({
        "instance_id": "z",
        "repo": "x/y",
        "base_commit": "abc",
    })
    assert result["repo_context_status"] == "fetch_failed"
    assert result["fetch_fallback_used"] is True


def test_setup_repo_for_canary_timeout(monkeypatch) -> None:
    """A clone timeout now triggers the targeted-fetch fallback
    (RESOLUTION_UNBLOCKERS 2026-06-10); when the fallback ALSO fails,
    the status stays timeout with the combined failure chain."""
    script = [
        "timeout",                                   # clone times out
        _FakeCompletedProcess(0),                    # git init OK
        _FakeCompletedProcess(0),                    # remote add OK
        _FakeCompletedProcess(128, stderr=b"fetch refused"),  # targeted fetch fails
    ]
    fake_run, _ = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._setup_repo_for_canary({
        "instance_id": "z",
        "repo": "x/y",
        "base_commit": "abc",
    })
    assert result["repo_context_status"] == "timeout"
    assert "git_clone_timeout" in (result["failure_reason"] or "")
    assert "targeted_fetch_failed" in (result["failure_reason"] or "")


def test_setup_repo_targeted_fetch_recovers_clone_timeout(monkeypatch) -> None:
    """RESOLUTION_UNBLOCKERS criterion 2 (teleport class, 2026-06-10): a
    clone that blows the 180s budget recovers via git init + single-commit
    fetch, then the normal checkout/rev-parse chain runs."""
    script = [
        "timeout",                                   # clone times out
        _FakeCompletedProcess(0),                    # git init OK
        _FakeCompletedProcess(0),                    # remote add OK
        _FakeCompletedProcess(0),                    # targeted fetch OK
        _FakeCompletedProcess(0),                    # checkout --detach OK
        _FakeCompletedProcess(0, stdout=b"abc\n"),   # rev-parse
    ]
    fake_run, calls = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._setup_repo_for_canary({
        "instance_id": "z",
        "repo": "x/y",
        "base_commit": "abc",
    })
    assert result["repo_context_status"] == "ready"
    assert result["fetch_fallback_used"] is True
    assert result["checkout_sha"] == "abc"
    assert "fetch" in calls[3]
    arm_d._cleanup_repo_dir(
        result["repo_dir"], tmp_root=os.path.dirname(result["repo_dir"])
    )


def test_setup_repo_targeted_fetch_recovers_clone_failure(monkeypatch) -> None:
    """A non-timeout clone failure (e.g. transient HTTP 500) also goes
    through the targeted-fetch recovery."""
    script = [
        _FakeCompletedProcess(128, stderr=b"early EOF"),  # clone fails
        _FakeCompletedProcess(0),                    # git init OK
        _FakeCompletedProcess(0),                    # remote add OK
        _FakeCompletedProcess(0),                    # targeted fetch OK
        _FakeCompletedProcess(0),                    # checkout --detach OK
        _FakeCompletedProcess(0, stdout=b"abc\n"),   # rev-parse
    ]
    fake_run, _ = _mock_subprocess_factory(script)
    monkeypatch.setattr(arm_d.subprocess, "run", fake_run)

    result = arm_d._setup_repo_for_canary({
        "instance_id": "z",
        "repo": "x/y",
        "base_commit": "abc",
    })
    assert result["repo_context_status"] == "ready"
    assert result["fetch_fallback_used"] is True
    arm_d._cleanup_repo_dir(
        result["repo_dir"], tmp_root=os.path.dirname(result["repo_dir"])
    )


def test_cleanup_repo_dir_removes_tree(tmp_path) -> None:
    """Cleanup removes the tempdir tree and drops it from the cleanup
    registry. Subsequent atexit handler is then a no-op for that path.
    """
    real_tmp = tempfile.mkdtemp(prefix="sage_canary_repo_test_")
    arm_d._CANARY_REPO_TMPDIRS.add(real_tmp)
    # Drop a sentinel file so we can verify the directory existed.
    Path(real_tmp, "marker.txt").write_text("x", encoding="utf-8")
    assert os.path.exists(real_tmp)

    status = arm_d._cleanup_repo_dir(
        repo_dir=os.path.join(real_tmp, "sub"),
        tmp_root=real_tmp,
    )
    assert status == "removed"
    assert not os.path.exists(real_tmp)
    assert real_tmp not in arm_d._CANARY_REPO_TMPDIRS


def test_cleanup_repo_dir_handles_missing_path() -> None:
    """Path does not exist → status=missing, no exception."""
    status = arm_d._cleanup_repo_dir(None)
    assert status == "missing"
    status2 = arm_d._cleanup_repo_dir("/no/such/path/here", tmp_root="/no/such/path/here")
    assert status2 == "missing"


def test_run_one_task_passes_cwd_and_records_repo_context(
    monkeypatch, tmp_path
) -> None:
    """End-to-end shape check: when repo setup succeeds, the canary
    must pass ``cwd=repo_dir`` to ``_run_sage_cli`` AND record the
    full repo_context block in the per-task summary.
    """
    instance = {
        "instance_id": "test-cwd",
        "repo": "owner/proj",
        "base_commit": "abc123",
        "problem_statement": "Bug in foo.",
    }

    captured_cwd: dict[str, Any] = {}

    async def _fake_cli(prompt, *, budget_usd, output_events_path, tier,
                       provider_allowlist=(), provider_denylist=(), cwd=None, **_ignored):
        captured_cwd["value"] = cwd
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        output_events_path.write_text("", encoding="utf-8")
        return {
            "exit_code": 0,
            "latency_ms": 100,
            "final_result_payload": {"result": "diff --git a/x b/x\n+y\n"},
            "cli_complete_payload": {
                "outcome": "success",
                "total_cost_usd": 0.0,
                "trace_dir": str(tmp_path / "trace"),
            },
            "run_id": "RUN-Z",
            "model_id_final": "fake-model",
            "provider_final": "fake-provider",
            "total_cost_usd": 0.0,
        }

    monkeypatch.setattr(arm_d, "_run_sage_cli", _fake_cli)

    fake_repo_dir = str(tmp_path / "fake_repo_canary" / "proj")
    cleanup_log: list[Any] = []

    def _fake_setup(inst):
        return {
            "repo_context_status": "ready",
            "repo_dir": fake_repo_dir,
            "repo_url": "https://github.com/owner/proj.git",
            "base_commit": "abc123",
            "checkout_sha": "abc123",
            "clone_elapsed_ms": 1234,
            "fetch_fallback_used": False,
            "failure_reason": None,
        }

    def _fake_cleanup(repo_dir, *, tmp_root=None):
        cleanup_log.append((repo_dir, tmp_root))
        return "removed"

    monkeypatch.setattr(arm_d, "_setup_repo_for_canary", _fake_setup)
    monkeypatch.setattr(arm_d, "_cleanup_repo_dir", _fake_cleanup)

    result = asyncio.run(
        arm_d._run_one_task(
            instance,
            tmp_path / "out",
            mock=False,
            budget_usd=5.0,
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
            prefix="test",
            fmt_module=_FakeFormatModule(),
            claim_default_pipeline_learning_evidence=False,
            expect_default_pipeline_learn=False,
        )
    )

    # cwd was forwarded to _run_sage_cli.
    assert captured_cwd["value"] == fake_repo_dir
    # Cleanup ran with the right tmp_root (parent of repo_dir).
    assert len(cleanup_log) == 1
    assert cleanup_log[0][0] == fake_repo_dir
    assert cleanup_log[0][1] == os.path.dirname(fake_repo_dir)
    # Per-task summary carries the full repo_context block.
    summary = result["summary"]
    rc = summary["repo_context"]
    assert rc["status"] == "ready"
    assert rc["repo_url"] == "https://github.com/owner/proj.git"
    assert rc["base_commit"] == "abc123"
    assert rc["checkout_sha"] == "abc123"
    assert rc["clone_elapsed_ms"] == 1234
    assert rc["fetch_fallback_used"] is False
    assert rc["subprocess_cwd"] == fake_repo_dir
    assert rc["repo_dir_cleanup_status"] == "removed"
    assert rc["failure_reason"] is None
    # Extracted patch lands in the record (proves end-to-end with cwd).
    assert "diff --git" in result["record"]["patch"]


def _repo_setup_failure_stub(inst):
    return {
        "repo_context_status": "clone_failed",
        "repo_dir": None,
        "tmp_root": None,
        "repo_url": "https://github.com/x/y.git",
        "base_commit": "deadbeef",
        "checkout_sha": None,
        "clone_elapsed_ms": 30000,
        "fetch_fallback_used": False,
        "failure_reason": "git_clone_exit=128 stderr=could not resolve host",
    }


def test_run_one_task_fails_closed_when_repo_setup_fails(monkeypatch, tmp_path) -> None:
    """RESOLUTION_UNBLOCKERS criterion 1 (cgpro Q2, 2026-06-10): when the
    repo worktree cannot be set up, the canary must SKIP generation
    entirely — no paid blind patch. The 2026-06-10 graded N=5 proved the
    old warn-and-continue default produces plausible-looking patches that
    break the build (teleport: $0.21 spent, patch applied, build failed).
    """
    instance = {
        "instance_id": "fail",
        "repo": "x/y",
        "base_commit": "deadbeef",
        "problem_statement": "Bug in bar.",
    }

    cli_called: dict[str, Any] = {"value": False}

    async def _fake_cli(*args, **kwargs):
        cli_called["value"] = True
        raise AssertionError("generation must not run when repo is unavailable")

    monkeypatch.setattr(arm_d, "_run_sage_cli", _fake_cli)
    monkeypatch.setattr(arm_d, "_setup_repo_for_canary", _repo_setup_failure_stub)
    monkeypatch.setattr(
        arm_d, "_cleanup_repo_dir", lambda repo_dir, *, tmp_root=None: "missing"
    )

    result = asyncio.run(
        arm_d._run_one_task(
            instance,
            tmp_path / "out",
            mock=False,
            budget_usd=5.0,
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
            prefix="test",
            fmt_module=_FakeFormatModule(),
            claim_default_pipeline_learning_evidence=True,
            expect_default_pipeline_learn=True,
        )
    )

    assert cli_called["value"] is False
    summary = result["summary"]
    assert summary["generation_skipped"] is True
    assert summary["patch_empty_reason"] == "repo_unavailable"
    assert summary["extracted_patch_present"] is False
    assert summary["total_cost_usd"] == 0.0
    assert summary["_total_cost_usd_source"] == "no_cost_evidence"
    assert summary["_diff_verifier_outcome"] == "skipped_no_patch"
    assert summary["timeout"] is False
    # Infra failure must not fail the learning gate: explicit skipped.
    assert summary["learning_evidence_boundary"]["status"] == "skipped"
    rc = summary["repo_context"]
    assert rc["status"] == "clone_failed"
    assert "could not resolve host" in (rc["failure_reason"] or "")
    # Prediction record still produced (empty patch, Pro shape).
    assert result["record"]["patch"] == ""


def test_run_one_task_blind_generation_escape_flag(monkeypatch, tmp_path) -> None:
    """The diagnostic escape `--allow-blind-generation-on-repo-failure`
    restores the old observable-failure behavior (subprocess runs with
    inherited cwd). OFF by default on every paid/graded run."""
    instance = {
        "instance_id": "fail",
        "repo": "x/y",
        "base_commit": "deadbeef",
        "problem_statement": "Bug in bar.",
    }

    captured_cwd: dict[str, Any] = {"value": "<unset>"}

    async def _fake_cli(prompt, *, budget_usd, output_events_path, tier,
                       provider_allowlist=(), provider_denylist=(), cwd=None, **_ignored):
        captured_cwd["value"] = cwd
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        output_events_path.write_text("", encoding="utf-8")
        return {
            "exit_code": 0,
            "latency_ms": 50,
            "final_result_payload": {"result": "no diff"},
            "cli_complete_payload": {
                "outcome": "success",
                "total_cost_usd": 0.0,
                "trace_dir": str(tmp_path / "trace"),
            },
            "run_id": "RUN-W",
            "model_id_final": "fake",
            "provider_final": "fake",
            "total_cost_usd": 0.0,
        }

    monkeypatch.setattr(arm_d, "_run_sage_cli", _fake_cli)
    monkeypatch.setattr(arm_d, "_setup_repo_for_canary", _repo_setup_failure_stub)
    monkeypatch.setattr(
        arm_d, "_cleanup_repo_dir", lambda repo_dir, *, tmp_root=None: "missing"
    )

    result = asyncio.run(
        arm_d._run_one_task(
            instance,
            tmp_path / "out",
            mock=False,
            budget_usd=5.0,
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
            prefix="test",
            fmt_module=_FakeFormatModule(),
            claim_default_pipeline_learning_evidence=False,
            expect_default_pipeline_learn=False,
            allow_blind_generation_on_repo_failure=True,
        )
    )

    # Escape hatch: subprocess ran with inherited cwd, failure observable.
    assert captured_cwd["value"] is None
    summary = result["summary"]
    assert summary.get("generation_skipped") is not True
    rc = summary["repo_context"]
    assert rc["status"] == "clone_failed"
    assert rc["subprocess_cwd"] is None


def test_provider_gate_ignores_generation_skipped_tasks() -> None:
    """A fail-closed repo skip never executed anything — it must not trip
    the provider gate's missing-provider check. An EXECUTED task without
    provider_final must still trip it (no weakening)."""
    skipped = {
        "instance_id": "skipped-task",
        "generation_skipped": True,
        "provider_final": None,
        "model_id_final": None,
    }
    executed_ok = {
        "instance_id": "ok-task",
        "provider_final": "google",
        "model_id_final": "gemini-3.1-pro-preview",
        "_execution_providers": ["google"],
        "_assigned_providers": ["google"],
    }
    gate = arm_d._provider_gate(
        [skipped, executed_ok],
        mock=False,
        provider_allowlist=("google", "deepseek"),
        provider_denylist=(),
    )
    assert gate["status"] == "PASS"
    assert gate["missing_provider_or_model"] == []

    executed_missing = {
        "instance_id": "executed-no-provider",
        "provider_final": None,
        "model_id_final": None,
    }
    gate2 = arm_d._provider_gate(
        [executed_missing],
        mock=False,
        provider_allowlist=("google", "deepseek"),
        provider_denylist=(),
    )
    assert gate2["status"] == "NO_GO"
    assert gate2["missing_provider_or_model"] == ["executed-no-provider"]


# ─────────────────────────────────────────────────────────────────────────────
# Block `canary-patch-focused-prompt-profile` slice 9 (cgpro DESIGN 2026-05-11):
# canary-local SWE-bench prompt profile dispatcher.
# ─────────────────────────────────────────────────────────────────────────────


def test_prompt_profile_enum_includes_canonical_and_patch_focused() -> None:
    """Both profiles required by cgpro DESIGN must exist and the
    default must be ``canonical`` so existing callers stay
    byte-compatible.
    """
    assert arm_d._PROMPT_PROFILES == ("canonical", "patch_focused")
    assert arm_d._DEFAULT_PROMPT_PROFILE == "canonical"


def test_build_prompt_canonical_matches_render_swebench_prompt() -> None:
    """The default profile MUST forward to
    ``sage.input.swebench.render_swebench_prompt(normalize_swebench(...))``
    byte-for-byte. If a future refactor diverges, this catches it.
    """
    from sage.input.swebench import normalize_swebench, render_swebench_prompt

    instance = {
        "instance_id": "x",
        "repo": "owner/proj",
        "base_commit": "abc123",
        "problem_statement": "Bug in foo.",
    }
    text, meta = arm_d._build_prompt(instance, "canonical")
    expected = render_swebench_prompt(normalize_swebench(instance))
    assert text == expected
    assert meta["prompt_profile"] == "canonical"
    assert meta["topology_override_used"] is False
    assert meta["system_hint_forced"] is False
    # Hash present and matches SHA-256(text bytes)
    import hashlib
    assert meta["prompt_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_build_prompt_patch_focused_drops_mandatory_workflow() -> None:
    """The patch_focused profile MUST NOT carry the canonical
    "MUST make at least THREE distinct tool calls" mandate.
    """
    instance = {
        "instance_id": "y",
        "repo": "owner/proj",
        "base_commit": "abc123",
        "problem_statement": "Bug in bar.",
    }
    text, meta = arm_d._build_prompt(instance, "patch_focused")
    # The canonical template's mandate sentence is verbatim absent.
    assert "MUST make at least THREE distinct tool calls" not in text
    assert "Mandatory Workflow" not in text
    # And the meta marks the profile + no-overrides contract.
    assert meta["prompt_profile"] == "patch_focused"
    assert meta["topology_override_used"] is False
    assert meta["system_hint_forced"] is False


def test_build_prompt_patch_focused_keeps_repo_grounding(monkeypatch) -> None:
    """The patch_focused profile MUST tell the agent it has the repo
    checked out at the working directory. Per cgpro DESIGN
    NON_GOALS: 'Do not revert to a bare "produce a diff" prompt with
    no repo-grounding.'
    """
    instance = {
        "instance_id": "z",
        "repo": "owner/proj",
        "base_commit": "abc123",
        "problem_statement": "Bug in baz.",
    }
    text, _ = arm_d._build_prompt(instance, "patch_focused")
    # Must mention the repo + base commit + the working-directory
    # statement that proves repo-grounding.
    assert "owner/proj" in text
    assert "abc123" in text
    assert "repo is checked out" in text
    # And the issue description.
    assert "Bug in baz." in text


def test_build_prompt_patch_focused_keeps_strict_diff_contract() -> None:
    """Per cgpro DESIGN STOP_CONDITIONS: the new prompt MUST make
    the final-answer contract STRICTER, not looser. ``diff --git``
    headers, ``--- a/`` / ``+++ b/`` paths, and the fenced ```diff
    block requirement must remain.
    """
    instance = {
        "instance_id": "w",
        "repo": "owner/proj",
        "base_commit": "abc123",
        "problem_statement": "Strict format test.",
    }
    text, _ = arm_d._build_prompt(instance, "patch_focused")
    assert "diff --git a/" in text
    assert "--- a/" in text
    assert "+++ b/" in text
    assert "```diff" in text
    # No-output-after-the-block rule (the canary extractor strips
    # everything after the fence anyway, but the prompt must say so).
    assert "Reasoning text AFTER" in text or "Output ONLY" in text


def test_build_prompt_unknown_profile_raises() -> None:
    """Defense against typos / future regression: an unknown profile
    name must raise ValueError, not silently fall back.
    """
    instance = {
        "instance_id": "a",
        "repo": "x/y",
        "base_commit": "abc",
        "problem_statement": "test",
    }
    try:
        arm_d._build_prompt(instance, "made_up_profile")
    except ValueError as exc:
        assert "made_up_profile" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown profile")


def test_run_one_task_propagates_prompt_metadata_to_summary(monkeypatch, tmp_path) -> None:
    """When the canary runs in patch_focused mode, the per-task summary
    must record ``prompt_metadata`` with the right profile + sha256 +
    no-overrides flags so the gate / post-run analysis can verify the
    acceptance criteria (topology_override_used=False everywhere).
    """
    instance = {
        "instance_id": "test-meta",
        "repo": "owner/proj",
        "base_commit": "abc123",
        "problem_statement": "Bug in foo.",
    }

    captured_prompts: list[str] = []

    async def _capture_cli(prompt, *, budget_usd, output_events_path, tier,
                          provider_allowlist=(), provider_denylist=(), cwd=None, **_ignored):
        captured_prompts.append(prompt)
        output_events_path.parent.mkdir(parents=True, exist_ok=True)
        output_events_path.write_text("", encoding="utf-8")
        return {
            "exit_code": 0,
            "latency_ms": 100,
            "final_result_payload": {"result": "no diff"},
            "cli_complete_payload": {
                "outcome": "success",
                "total_cost_usd": 0.0,
                "trace_dir": str(tmp_path / "trace"),
            },
            "run_id": "RUN-M",
            "model_id_final": "fake",
            "provider_final": "fake",
            "total_cost_usd": 0.0,
        }

    monkeypatch.setattr(arm_d, "_run_sage_cli", _capture_cli)
    monkeypatch.setattr(
        arm_d,
        "_setup_repo_for_canary",
        lambda inst: {
            "repo_context_status": "ready",
            "repo_dir": str(tmp_path / "fake_repo"),
            "repo_url": f"https://github.com/{inst['repo']}.git",
            "base_commit": inst["base_commit"],
            "checkout_sha": inst["base_commit"],
            "clone_elapsed_ms": 0,
            "fetch_fallback_used": False,
            "failure_reason": None,
        },
    )
    monkeypatch.setattr(
        arm_d, "_cleanup_repo_dir", lambda repo_dir, *, tmp_root=None: "removed"
    )

    result = asyncio.run(
        arm_d._run_one_task(
            instance,
            tmp_path / "out",
            mock=False,
            budget_usd=5.0,
            tier="budget",
            provider_allowlist=("google",),
            provider_denylist=("openai",),
            prefix="test",
            fmt_module=_FakeFormatModule(),
            claim_default_pipeline_learning_evidence=False,
            expect_default_pipeline_learn=False,
            swebench_prompt_profile="patch_focused",
        )
    )

    # The forwarded prompt must NOT contain the mandate clause.
    assert len(captured_prompts) == 1
    assert "MUST make at least THREE distinct tool calls" not in captured_prompts[0]

    pm = result["summary"]["prompt_metadata"]
    assert pm["prompt_profile"] == "patch_focused"
    assert pm["topology_override_used"] is False
    assert pm["system_hint_forced"] is False
    assert len(pm["prompt_sha256"]) == 64  # SHA-256 hex


def test_cli_swebench_prompt_profile_flag_plumbs(monkeypatch, tmp_path) -> None:
    """CLI smoke: --swebench-prompt-profile patch_focused reaches
    run() as the matching kwarg.
    """
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "t", "problem_statement": "x"}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "m.md"
    manifest.write_text("# canary", encoding="utf-8")

    captured: dict[str, Any] = {}

    async def _capture(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(arm_d, "run", _capture)

    exit_code = arm_d.main(
        [
            "--instances-json",
            str(instances_json),
            "--output-dir",
            str(tmp_path / "out"),
            "--mock",
            "--swebench-prompt-profile",
            "patch_focused",
            "--manifest-path",
            str(manifest),
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["swebench_prompt_profile"] == "patch_focused"


def test_cli_default_prompt_profile_is_canonical(monkeypatch, tmp_path) -> None:
    """Without ``--swebench-prompt-profile``, the default is
    ``canonical`` so existing callers stay byte-compatible.
    """
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "t", "problem_statement": "x"}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "m.md"
    manifest.write_text("# canary", encoding="utf-8")

    captured: dict[str, Any] = {}

    async def _capture(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(arm_d, "run", _capture)

    exit_code = arm_d.main(
        [
            "--instances-json",
            str(instances_json),
            "--output-dir",
            str(tmp_path / "out"),
            "--mock",
            "--manifest-path",
            str(manifest),
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["swebench_prompt_profile"] == "canonical"


def test_setup_repo_for_canary_real_git_fixture(tmp_path) -> None:
    """Real-git end-to-end smoke. Creates a local git repo with two
    commits + a clone-source bare repo, then asks
    ``_setup_repo_for_canary`` to clone & check out the older commit
    (which is NOT in a shallow clone of HEAD → forces fetch fallback).
    Skipped if git is not on PATH (e.g. sandboxed CI without git).
    """
    if shutil.which("git") is None:
        pytest.skip("git not installed; skip real-git fixture")

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git_env = {"GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path), **os.environ}

    def _g(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(upstream),
            capture_output=True,
            check=True,
            env=git_env,
            timeout=60,
        )

    subprocess.run(
        ["git", "init", "-b", "main", str(upstream)],
        check=True, capture_output=True, env=git_env, timeout=30,
    )
    _g("config", "user.email", "test@example.com")
    _g("config", "user.name", "test")
    (upstream / "file1.txt").write_text("v1", encoding="utf-8")
    _g("add", "file1.txt")
    _g("commit", "-m", "first")
    commit1 = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(upstream),
            env=git_env,
            timeout=10,
        )
        .decode("utf-8")
        .strip()
    )
    (upstream / "file2.txt").write_text("v2", encoding="utf-8")
    _g("add", "file2.txt")
    _g("commit", "-m", "second")
    commit2 = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(upstream),
            env=git_env,
            timeout=10,
        )
        .decode("utf-8")
        .strip()
    )
    assert commit1 != commit2

    # Patch the URL builder to point at the file:// upstream so
    # _setup_repo_for_canary can clone without network. We do this by
    # rewriting the instance dict's ``repo`` to a value that produces
    # the right URL — easier: monkeypatch the f-string composition via
    # an env var would be intrusive; instead we expose the file URL
    # through a closure over subprocess.run.
    file_url = (upstream.resolve().as_uri())

    real_run = subprocess.run

    def _rewriting_run(*args, **kwargs):
        argv = list(args[0]) if args else list(kwargs.get("args") or [])
        if argv and argv[0] == "git" and len(argv) >= 5 and "clone" in argv:
            # Replace the github.com URL with our local file:// URL.
            for i, a in enumerate(argv):
                if isinstance(a, str) and a.startswith("https://github.com/"):
                    argv[i] = file_url
                    break
            if args:
                args = (argv, *args[1:])
            else:
                kwargs["args"] = argv
        return real_run(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(arm_d.subprocess, "run", _rewriting_run)

        result = arm_d._setup_repo_for_canary({
            "instance_id": "fixture",
            "repo": "fake/fixture",  # rewritten by _rewriting_run
            "base_commit": commit1,  # OLDER commit → likely needs fetch fallback
        })

    assert result["repo_context_status"] == "ready", result
    assert result["checkout_sha"] == commit1
    # Either the shallow clone happened to include commit1 (small repo
    # — sometimes git includes both), or we used the fetch fallback.
    # Both are valid; the contract is "checkout_sha == base_commit".
    assert result["fetch_fallback_used"] in (True, False)
    assert os.path.isdir(result["repo_dir"])
    assert (Path(result["repo_dir"]) / "file1.txt").read_text(encoding="utf-8") == "v1"

    arm_d._cleanup_repo_dir(result["repo_dir"], tmp_root=os.path.dirname(result["repo_dir"]))


def test_cli_explicit_task_timeout_marks_profile_override(monkeypatch, tmp_path) -> None:
    """When --task-timeout-s is passed alongside --profile, the explicit
    value wins and profile_timeout_override flips True. Both pieces of
    metadata reach run().
    """
    instances_json = tmp_path / "instances.json"
    instances_json.write_text(
        json.dumps([{"instance_id": "task-1", "problem_statement": "x"}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.md"
    manifest.write_text("# canary", encoding="utf-8")

    captured: dict[str, Any] = {}

    async def _capture_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(arm_d, "run", _capture_run)

    exit_code = arm_d.main(
        [
            "--instances-json",
            str(instances_json),
            "--output-dir",
            str(tmp_path / "out"),
            "--mock",
            "--profile",
            "graded_patch_generation",
            "--task-timeout-s",
            "600",
            "--manifest-path",
            str(manifest),
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["task_timeout_s"] == 600.0
    assert captured["kwargs"]["profile"] == "graded_patch_generation"
    assert captured["kwargs"]["profile_timeout_override"] is True


def test_load_grader_gate_accepts_ready_modal_decision(tmp_path: Path) -> None:
    """Latent integration bug regression: the preflight script emits
    decision string ``READY_MODAL`` for the remote-Modal grading path
    (added 2026-05-10 commit a7474306), but the gate in
    `_load_grader_gate` was added earlier (ec0b775e) with the
    speculatively-typed string ``READY_REMOTE_MODAL``. Result: every
    Modal preflight ever produced was BLOCKED with
    `grader_preflight_not_ready`, which silently blocked B2 N=5
    graded launches on Modal hosts. The gate now accepts both the
    boolean ``modal_grading_ready`` field and the actual decision
    string emitted by the live preflight tool.
    """
    preflight = tmp_path / "grader_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "swebench_pro_grader_preflight_v1",
                "decision": "READY_MODAL",
                "local_grading_ready": False,
                "modal_grading_ready": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    gate = arm_d._load_grader_gate(preflight)

    assert gate["status"] == "PASS"
    assert gate["reason"] is None
    assert gate["decision"] == "READY_MODAL"


def test_load_grader_gate_accepts_local_docker_decision(tmp_path: Path) -> None:
    """Local-docker grading path still passes — backward compatibility
    proof for the multi-string decision allowlist.
    """
    preflight = tmp_path / "grader_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "swebench_pro_grader_preflight_v1",
                "decision": "READY_LOCAL_DOCKER",
                "local_grading_ready": True,
                "modal_grading_ready": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    gate = arm_d._load_grader_gate(preflight)

    assert gate["status"] == "PASS"
    assert gate["decision"] == "READY_LOCAL_DOCKER"


def test_load_grader_gate_blocks_no_go_decisions(tmp_path: Path) -> None:
    """NO_GO_* decisions remain BLOCKED — the broader allowlist
    must not accidentally accept failure decisions.
    """
    preflight = tmp_path / "grader_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "swebench_pro_grader_preflight_v1",
                "decision": "NO_GO_GRADER_REPO_DIRTY",
                "local_grading_ready": False,
                "modal_grading_ready": False,
                "blockers": ["grader_repo_dirty"],
            }
        ),
        encoding="utf-8",
    )

    gate = arm_d._load_grader_gate(preflight)

    assert gate["status"] == "BLOCKED"
    assert gate["reason"] == "grader_preflight_not_ready"
    assert gate["decision"] == "NO_GO_GRADER_REPO_DIRTY"
    assert "grader_repo_dirty" in gate["blockers"]


# ---------------------------------------------------------------------------
# B2_RERUN_UNBLOCKERS bug 1 — gate contract locks (cgpro required tests 2-3)
# ---------------------------------------------------------------------------

def test_provider_gate_still_no_go_on_unknown_provider() -> None:
    """B2 contract test 2: fixing provider ATTRIBUTION must not weaken the
    gate — an execution provider of "unknown" still yields NO_GO via
    execution_outside_allowlist (the exact 2026-05-12 canary task #3 shape)."""
    gate = arm_d._provider_gate(
        [
            {
                "instance_id": "task-3",
                "provider_final": "unknown",
                "model_id_final": "gemini-3.1-pro-preview",
                "_execution_providers": ["unknown"],
                "_assigned_providers": ["unknown"],
            }
        ],
        mock=False,
        provider_allowlist=("google", "deepseek"),
        provider_denylist=(),
    )
    assert gate["status"] == "NO_GO"
    assert gate["execution_outside_allowlist"] == ["unknown"]


def test_provider_gate_pass_on_google_deepseek_only_execution() -> None:
    """B2 contract test 3: a clean google+deepseek-only execution under the
    same allowlist passes the gate."""
    gate = arm_d._provider_gate(
        [
            {
                "instance_id": "task-1",
                "provider_final": "google",
                "model_id_final": "gemini-3.1-pro-preview",
                "_execution_providers": ["google", "deepseek"],
                "_assigned_providers": ["google", "deepseek"],
            }
        ],
        mock=False,
        provider_allowlist=("google", "deepseek"),
        provider_denylist=(),
    )
    assert gate["status"] == "PASS"
    assert gate["execution_outside_allowlist"] == []


# ---------------------------------------------------------------------------
# B2_RERUN_UNBLOCKERS bug 2 — cost resolution (cgpro required tests 4-5)
# ---------------------------------------------------------------------------

def test_resolve_total_cost_recovers_hard_failure_from_event_audit() -> None:
    """B2 contract test 4: a hard CLI failure (no cli_complete payload) must
    surface the observed event cost with an explicit source + integrity
    warning — NOT report $0 (2026-05-12 canary tasks #2/#3 lost $0.16/$0.18)."""
    total, source, warning = arm_d._resolve_total_cost(
        cli_total_cost_usd=None,
        observed_event_cost_usd=0.158,
        had_llm_execution=True,
        cli_complete_expected=True,
    )
    assert total == pytest.approx(0.158)
    assert source == "event_audit_observed_event_cost_usd"
    assert warning is not None
    assert warning["reason_code"] == "cli_complete_cost_missing"


def test_resolve_total_cost_prefers_larger_observed_on_underreport() -> None:
    """B2 bug 2 success-path shape: cli_complete under-reported vs event audit
    (tutanota db90ac26: $0.134 reported vs $0.266 observed = ~50% loss)."""
    total, source, warning = arm_d._resolve_total_cost(
        cli_total_cost_usd=0.134,
        observed_event_cost_usd=0.266,
        had_llm_execution=True,
        cli_complete_expected=True,
    )
    assert total == pytest.approx(0.266)
    assert source == "event_audit_observed_event_cost_usd"
    assert warning is not None
    assert warning["reason_code"] == "cli_complete_cost_underreport"


def test_resolve_total_cost_uses_cli_complete_when_consistent() -> None:
    """B2 contract test 5: when cli_complete covers the observed cost, it is
    the authoritative source and no warning is emitted."""
    total, source, warning = arm_d._resolve_total_cost(
        cli_total_cost_usd=0.099,
        observed_event_cost_usd=0.099,
        had_llm_execution=True,
        cli_complete_expected=True,
    )
    assert total == pytest.approx(0.099)
    assert source == "cli_complete"
    assert warning is None


def test_resolve_total_cost_timeout_parity_no_missing_warning() -> None:
    """Timeout path parity: cli_complete is EXPECTED to be absent on timeout
    (eeb3a7fb behavior) — recovery happens without a missing-payload warning;
    the llm_execution_observed_zero_cost warning still fires on zero cost."""
    total, source, warning = arm_d._resolve_total_cost(
        cli_total_cost_usd=None,
        observed_event_cost_usd=0.05,
        had_llm_execution=True,
        cli_complete_expected=False,
    )
    assert total == pytest.approx(0.05)
    assert source == "event_audit_observed_event_cost_usd"
    assert warning is None

    total0, source0, warning0 = arm_d._resolve_total_cost(
        cli_total_cost_usd=None,
        observed_event_cost_usd=0.0,
        had_llm_execution=True,
        cli_complete_expected=False,
    )
    assert total0 == 0.0
    assert source0 == "no_cost_evidence"
    assert warning0 is not None
    assert warning0["reason_code"] == "llm_execution_observed_zero_cost"


# ---------------------------------------------------------------------------
# B2_RERUN_UNBLOCKERS bug 3 — diff verifier wiring (cgpro required tests 6-8)
# ---------------------------------------------------------------------------

def test_diff_verifier_env_propagates_to_subprocess() -> None:
    """B2 contract test 6: the per-task subprocess env must carry
    SAGE_DIFF_VERIFIER_MODE=observe (single source: the same constant the
    launcher-side annotation uses)."""
    env = arm_d._task_subprocess_env(tier="reasoner")
    assert env["SAGE_DIFF_VERIFIER_MODE"] == "observe"
    assert env["SAGE_LLM_TIER"] == "reasoner"


def test_patch_with_observe_mode_yields_non_null_verifier_outcome(tmp_path: Path) -> None:
    """B2 contract test 7: a non-empty patch under observe mode must produce
    a non-null _diff_verifier_outcome (2026-05-12 N=5: every task had None
    because the canary never invoked the verifier — it lives in
    SWEBenchBench, which the canary does not instantiate)."""
    target = tmp_path / "mod.py"
    target.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    patch = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    record = arm_d._annotate_diff_verifier(
        patch=patch, repo_dir=str(tmp_path), mode="observe"
    )
    assert record["_diff_verifier_outcome"] is not None
    assert record["_diff_verifier_outcome"] != ""
    assert isinstance(record["_diff_verifier_mismatches"], list)
    assert record["_diff_verifier_mismatches"] == []


def test_patch_with_context_mismatch_yields_mismatch_records(tmp_path: Path) -> None:
    """The wired verifier must surface real mismatches (compact dict shape,
    mirroring swebench_bench serialization — no expected/actual bodies)."""
    target = tmp_path / "mod.py"
    target.write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")
    patch = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    record = arm_d._annotate_diff_verifier(
        patch=patch, repo_dir=str(tmp_path), mode="observe"
    )
    assert record["_diff_verifier_outcome"] is not None
    assert record["_diff_verifier_mismatches"], "expected at least one mismatch record"
    first = record["_diff_verifier_mismatches"][0]
    assert set(first) == {"file", "hunk_index", "old_start", "old_count", "kind", "match_ratio"}


def test_no_patch_yields_explicit_skipped_no_patch() -> None:
    """B2 contract test 8: absence of patch must be an explicit outcome,
    NOT null (manifest stop condition #5 keys off non-null fields)."""
    record = arm_d._annotate_diff_verifier(patch="", repo_dir=None, mode="observe")
    assert record["_diff_verifier_outcome"] == "skipped_no_patch"
    assert record["_diff_verifier_mismatches"] is None


def test_mode_off_yields_explicit_skipped_mode_off(tmp_path: Path) -> None:
    record = arm_d._annotate_diff_verifier(
        patch="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        repo_dir=str(tmp_path),
        mode="off",
    )
    assert record["_diff_verifier_outcome"] == "skipped_mode_off"


def test_missing_repo_dir_yields_explicit_skipped_no_repo() -> None:
    """Repo clone failures must remain observable: patch present but no
    worktree to verify against is its own explicit outcome."""
    record = arm_d._annotate_diff_verifier(
        patch="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        repo_dir=None,
        mode="observe",
    )
    assert record["_diff_verifier_outcome"] == "skipped_no_repo_dir"


def test_event_audit_recognizes_real_provider_policy_violation_event(tmp_path: Path) -> None:
    """cgpro VERIFY EDIT_REQUIRED #2 (2026-06-10) — declared-vs-verified
    catch: the committed 2026-05-12 task #3 trace contains a real failure
    event with kind='provider_error' + error_type='ProviderPolicyViolation',
    but the summary said _provider_policy_failure_seen=false because the
    matcher only recognized kind=='provider_policy' or the snake_case
    error_type. The audit label MUST bind to the real event content."""
    events_path = tmp_path / "per_task" / "task.events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    # Exact payload shape from
    # docs/benchmarks/2026-05-12-b2-n5-graded/run/per_task/
    # instance_tutao__tutanota-219bc8f0...events.jsonl seq=101.
    real_event = {
        "schema_version": "1.0",
        "payload_schema_version": "v1_1",
        "event_type": "failure",
        "seq": 101,
        "parent_event_id": 30,
        "source_component": "topology_runner",
        "payload": {
            "kind": "provider_error",
            "error_type": "ProviderPolicyViolation",
            "message": (
                "provider policy violation: source=cli; "
                "model_id='gemini-3.1-pro-preview'; provider_id='unknown'; "
                "reason=outside_allowlist"
            ),
        },
    }
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(real_event, ensure_ascii=False) + "\n")

    audit = arm_d._event_audit_from_file(events_path)
    assert audit["_provider_policy_failure_seen"] is True


def test_event_audit_plain_provider_error_is_not_policy_failure(tmp_path: Path) -> None:
    """Tightness guard: a generic provider_error (network blip, rate limit)
    without policy content must NOT set the policy-failure flag."""
    events_path = tmp_path / "per_task" / "task.events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_type": "failure",
        "seq": 7,
        "payload": {
            "kind": "provider_error",
            "error_type": "RateLimitError",
            "message": "429 too many requests from provider google",
        },
    }
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    audit = arm_d._event_audit_from_file(events_path)
    assert audit["_provider_policy_failure_seen"] is False


# ---------------------------------------------------------------------------
# RESOLUTION_UNBLOCKERS criteria 3-4 — repair actually wired (cgpro 2026-06-10)
# ---------------------------------------------------------------------------

class _FakeCanaryRepairLLM:
    """Minimal LLMProvider double for the canary repair chain."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    async def generate(self, messages, **kwargs):
        self.calls += 1

        class _Resp:
            content = self._reply
            usage = {"input_tokens": 100, "output_tokens": 50}

        return _Resp()


def _write_target(tmp_path, body: str):
    target = tmp_path / "mod.py"
    target.write_text(body, encoding="utf-8")
    return target


def test_repair_chain_mechanical_counts_fix_no_llm(tmp_path) -> None:
    """cgpro trap #2 + research finding (RustAssistant/V4A class): a
    hunk_body_count_mismatch is fixed by MECHANICAL recount — the LLM must
    NOT be asked to do line arithmetic, and must not be constructed at all
    when the recount suffices."""
    _write_target(tmp_path, "a = 1\nb = 2\nc = 3\n")
    # Correct content, WRONG header counts (says 2 lines, body has 3+3).
    patch = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    factory_calls = {"n": 0}

    def _factory():
        factory_calls["n"] += 1
        raise AssertionError("LLM must not be built for a counts-only fix")

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=patch,
            repo_dir=str(tmp_path),
            problem_statement="fix b",
            instance_id="t-counts",
            repair_budget_usd=0.5,
            llm_factory=_factory,
        )
    )
    assert factory_calls["n"] == 0
    assert meta["_verifier_repair_stage"] == "mechanical_counts_fix"
    assert meta["_verifier_repair_budget_usd"] == 0.5
    assert "@@ -1,3 +1,3 @@" in final_patch
    assert annotation["_diff_verifier_mismatches"] == []
    assert annotation["_diff_verifier_outcome"] not in (None, "")


def test_repair_chain_llm_repairs_content_mismatch(tmp_path) -> None:
    """Criterion 3: the LLM repair is ACTUALLY invoked on content-class
    mismatches, and the repaired patch is re-verified before adoption
    (research trap: never count a patch repaired without re-verification)."""
    _write_target(tmp_path, "x = 10\ny = 20\nz = 30\n")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    repaired = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " x = 10\n"
        "-y = 20\n"
        "+y = 50\n"
        " z = 30\n"
    )
    llm = _FakeCanaryRepairLLM(reply=repaired)

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix y",
            instance_id="t-content",
            repair_budget_usd=0.5,
            llm_factory=lambda: (llm, "deepseek", "deepseek-v4-flash"),
        )
    )
    assert llm.calls == 1
    assert meta["_verifier_repair_stage"] == "verifier_repair"
    assert "y = 50" in final_patch
    assert annotation["_diff_verifier_mismatches"] == []


def test_repair_chain_non_repairable_is_explicit(tmp_path) -> None:
    """Criterion 4: when the LLM cannot produce a usable diff the outcome
    stays explicit (stage verifier_repair_empty, original patch kept,
    mismatches still reported)."""
    _write_target(tmp_path, "x = 10\ny = 20\nz = 30\n")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    llm = _FakeCanaryRepairLLM(reply="I cannot help with that.")

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix y",
            instance_id="t-bad",
            repair_budget_usd=0.5,
            llm_factory=lambda: (llm, "deepseek", "deepseek-v4-flash"),
        )
    )
    assert meta["_verifier_repair_stage"] == "verifier_repair_empty"
    assert final_patch == broken
    assert annotation["_diff_verifier_mismatches"]


def test_repair_chain_budget_zero_skips_llm(tmp_path) -> None:
    """Block D contract: zero repair budget means the LLM is never built;
    the skip is explicit."""
    _write_target(tmp_path, "x = 10\ny = 20\nz = 30\n")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )

    def _factory():
        raise AssertionError("LLM must not be built with zero budget")

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix y",
            instance_id="t-budget",
            repair_budget_usd=0.0,
            llm_factory=_factory,
        )
    )
    assert meta["_verifier_repair_stage"] == "repair_skipped_budget_exhausted"
    assert final_patch == broken


def test_repair_chain_clean_patch_not_needed(tmp_path) -> None:
    """A clean patch goes straight through: no recount, no LLM."""
    _write_target(tmp_path, "a = 1\nb = 2\nc = 3\n")
    clean = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )

    def _factory():
        raise AssertionError("LLM must not be built for a clean patch")

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=clean,
            repo_dir=str(tmp_path),
            problem_statement="fix b",
            instance_id="t-clean",
            repair_budget_usd=0.5,
            llm_factory=_factory,
        )
    )
    assert meta["_verifier_repair_stage"] == "repair_not_needed"
    assert final_patch == clean
    assert annotation["_diff_verifier_mismatches"] == []


def test_prediction_audit_record_carries_repo_unavailable_reason() -> None:
    """Review blocker (2026-06-10): the skip summary's patch_empty_reason
    must reach predictions.jsonl as the gate's pre-recorded _reason_code —
    summary-only fields never reach the pre-grader gate."""
    record = {"instance_id": "t", "patch": "", "prefix": "p"}
    summary = {
        "instance_id": "t",
        "mock": False,
        "timeout": False,
        "generation_skipped": True,
        "patch_empty_reason": "repo_unavailable",
    }
    audit = arm_d._prediction_audit_record(record, summary, task_index=0)
    assert audit["_reason_code"] == "repo_unavailable"

    plain_summary = {"instance_id": "t2", "mock": False, "timeout": False}
    audit2 = arm_d._prediction_audit_record(record, plain_summary, task_index=1)
    assert "_reason_code" not in audit2


def test_repair_adoption_rejects_structurally_broken_llm_patch(tmp_path) -> None:
    """Review MAJOR (2026-06-10): an LLM 'repair' that verifies with ZERO
    mismatches but a structural outcome (not_unified_diff) must NOT be
    adopted on the naive count comparison."""
    target = tmp_path / "mod.py"
    target.write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    # Extractable as a diff, but no @@ hunks at all -> not_unified_diff
    # with zero mismatches.
    structurally_broken_reply = "--- a/mod.py\n+++ b/mod.py\n+y = 50\n"
    llm = _FakeCanaryRepairLLM(reply=structurally_broken_reply)

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix y",
            instance_id="t-structural",
            repair_budget_usd=0.5,
            llm_factory=lambda: (llm, "deepseek", "deepseek-v4-flash"),
        )
    )
    assert final_patch == broken
    assert meta["_verifier_repair_stage"] == "verifier_repair_not_improved"
    assert annotation["_diff_verifier_mismatches"]


# ---------------------------------------------------------------------------
# cgpro VERIFY EDIT_REQUIRED (2026-06-11) — clean-strict adoption + audited
# repair LLM channel + tightened gate exemption
# ---------------------------------------------------------------------------

def test_repair_adoption_requires_clean_outcome(tmp_path) -> None:
    """cgpro Q1: adopt a repaired patch ONLY when post-repair verification
    is outcome=='clean'. A partial improvement (2 mismatches -> 1) keeps the
    ORIGINAL patch; the attempt stays visible as telemetry."""
    (tmp_path / "mod.py").write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("k = 1\nl = 2\nm = 3\n", encoding="utf-8")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
        "--- a/other.py\n"
        "+++ b/other.py\n"
        "@@ -1,3 +1,3 @@\n"
        " wrong = 0\n"
        "-also = 0\n"
        "+also = 9\n"
        " still = 0\n"
    )
    half_fixed = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " x = 10\n"
        "-y = 20\n"
        "+y = 50\n"
        " z = 30\n"
        "--- a/other.py\n"
        "+++ b/other.py\n"
        "@@ -1,3 +1,3 @@\n"
        " wrong = 0\n"
        "-also = 0\n"
        "+also = 9\n"
        " still = 0\n"
    )
    llm = _FakeCanaryRepairLLM(reply=half_fixed)
    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix",
            instance_id="t-partial",
            repair_budget_usd=0.5,
            llm_factory=lambda: (llm, "deepseek", "deepseek-v4-flash"),
        )
    )
    assert final_patch == broken
    assert meta["_verifier_repair_stage"] == "verifier_repair_not_improved"
    assert meta["_diff_verifier_outcome_post_repair"] is not None
    assert len(annotation["_diff_verifier_mismatches"]) == 2


def test_mechanical_fix_alone_not_clean_keeps_original(tmp_path) -> None:
    """cgpro Q1 'mecanique comme LLM': a counts recount that does not reach
    outcome=='clean' must not replace the patch sent to the grader."""
    (tmp_path / "mod.py").write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )

    def _no_llm():
        raise AssertionError("no LLM with zero budget")

    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix",
            instance_id="t-mech-dirty",
            repair_budget_usd=0.0,
            llm_factory=_no_llm,
        )
    )
    assert final_patch == broken
    assert meta["_verifier_repair_stage"] == "repair_skipped_budget_exhausted"
    assert meta["_diff_verifier_outcome_post_repair"] is not None


def test_repair_llm_provider_blocked_by_policy(tmp_path) -> None:
    """cgpro edit 2: the repair LLM is a provider spend — it must obey the
    canary allowlist/denylist. A blocked provider means NO call and an
    explicit stage."""
    (tmp_path / "mod.py").write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    llm = _FakeCanaryRepairLLM(reply="anything")
    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix",
            instance_id="t-blocked",
            repair_budget_usd=0.5,
            llm_factory=lambda: (llm, "openai", "gpt-5.4"),
            provider_allowlist=("google", "deepseek"),
            provider_denylist=("openai",),
        )
    )
    assert llm.calls == 0
    assert meta["_verifier_repair_stage"] == "repair_llm_provider_blocked"
    assert final_patch == broken


def test_repair_meta_records_provider_model_usage(tmp_path) -> None:
    """cgpro edit 2: the repair spend channel is auditable — provider,
    model and usage tokens land in the meta."""
    (tmp_path / "mod.py").write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")
    broken = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 5\n"
        " c = 3\n"
    )
    repaired = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " x = 10\n"
        "-y = 20\n"
        "+y = 50\n"
        " z = 30\n"
    )
    llm = _FakeCanaryRepairLLM(reply=repaired)
    final_patch, meta, annotation = asyncio.run(
        arm_d._repair_patch_with_feedback(
            patch=broken,
            repo_dir=str(tmp_path),
            problem_statement="fix",
            instance_id="t-audit",
            repair_budget_usd=0.5,
            llm_factory=lambda: (llm, "deepseek", "deepseek-v4-flash"),
            provider_allowlist=("google", "deepseek"),
        )
    )
    assert meta["_verifier_repair_provider"] == "deepseek"
    assert meta["_verifier_repair_model"] == "deepseek-v4-flash"
    assert meta["_verifier_repair_usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert meta["_verifier_repair_stage"] == "verifier_repair"
    assert "y = 50" in final_patch


def test_provider_gate_skip_exemption_requires_no_execution() -> None:
    """cgpro Q2: the generation_skipped exemption holds ONLY with zero LLM
    execution evidence and zero observed cost — a 'skipped' summary that
    somehow carries execution evidence still trips the missing check."""
    fake_skip_with_execution = {
        "instance_id": "weird",
        "generation_skipped": True,
        "provider_final": None,
        "model_id_final": None,
        "_execution_model_ids": ["deepseek-v4-flash"],
        "_observed_event_cost_usd": 0.01,
    }
    gate = arm_d._provider_gate(
        [fake_skip_with_execution],
        mock=False,
        provider_allowlist=("google", "deepseek"),
        provider_denylist=(),
    )
    assert gate["status"] == "NO_GO"
    assert gate["missing_provider_or_model"] == ["weird"]


def test_count_empty_patches_split_infra_vs_model() -> None:
    """cgpro Q3: keep repo-skips in patches_empty_total but report the
    infra/model split."""
    summaries = [
        {"extracted_patch_present": True},
        {"extracted_patch_present": False, "patch_empty_reason": "repo_unavailable"},
        {"extracted_patch_present": False},
        {"extracted_patch_present": False, "patch_empty_reason": "repo_unavailable"},
    ]
    total, infra, model = arm_d._count_empty_patches(summaries)
    assert total == 3
    assert infra == 2
    assert model == 1


def test_build_repair_llm_resolves_api_key_from_connector(monkeypatch) -> None:
    """Re-canary 2026-06-11 root cause: the repair client was built WITHOUT
    api_key (PydanticAIProvider does no self-lookup) -> 401 Authentication
    Fails on both repair attempts. The factory must resolve the key via the
    connector's api_key_env for the tier's provider."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-repair-key")
    llm, provider, model = arm_d._build_repair_llm()
    assert provider == "deepseek"
    assert model
    assert getattr(llm, "api_key", None) == "sk-test-repair-key"


def test_build_repair_llm_reasoner_tier(monkeypatch) -> None:
    """cgpro NEXT_BLOCK: repair tier -> reasoner, audit intact (provider
    identity resolved for the reasoner model, key from its api_key_env)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-test-google-key")
    llm, provider, model = arm_d._build_repair_llm(tier="reasoner")
    assert provider == "google"
    assert "gemini" in model
    assert getattr(llm, "api_key", None) == "sk-test-google-key"
