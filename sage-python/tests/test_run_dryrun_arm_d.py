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
