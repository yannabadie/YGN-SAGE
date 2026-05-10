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

    created: dict[str, _FakeProcess] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        proc = _FakeProcess()
        created["proc"] = proc
        return proc

    monkeypatch.setattr(arm_d.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(
        arm_d._run_sage_cli(
            "fix the bug",
            budget_usd=5.0,
            output_events_path=tmp_path / "events.jsonl",
            tier="budget",
        )
    )

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
    assert summary["acceptance_gate_results"]["provider_gate"]["observed_providers"] == [
        "openai"
    ]
    assert summary["canary_decision"] == "NO_GO"


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
