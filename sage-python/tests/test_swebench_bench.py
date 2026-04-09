from __future__ import annotations

from types import SimpleNamespace

import pytest

import sage.bench.swebench_bench as swebench_mod
from sage.bench.swebench_bench import SWEBenchBench


@pytest.mark.asyncio
async def test_generate_patches_uses_pipeline_context_metadata(monkeypatch):
    instance = {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "version": "1.0",
        "base_commit": "abc123",
        "problem_statement": "Fix the failing parser edge case.",
        "hints_text": "",
    }

    monkeypatch.setattr(
        swebench_mod,
        "load_swebench_dataset",
        lambda *args, **kwargs: [instance],
    )

    class _FakeSystem:
        def __init__(self) -> None:
            self.agent_loop = SimpleNamespace(
                _llm=SimpleNamespace(model_id="fake-model"),
                total_cost_usd=0.125,
            )
            self.pipeline = SimpleNamespace(last_context=None)
            self._last_execution_path = ""

        async def run(self, task: str) -> str:
            self.pipeline.last_context = SimpleNamespace(
                system=3,
                tool_call_count=2,
                tool_turn_count=1,
                executed_commands=["pwd", "git status"],
            )
            self._last_execution_path = "pipeline"
            return (
                "diff --git a/pkg/file.py b/pkg/file.py\n"
                "--- a/pkg/file.py\n"
                "+++ b/pkg/file.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            )

    bench = SWEBenchBench(system=_FakeSystem(), event_bus=None, dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _instance: None)

    predictions = await bench.generate_patches(limit=1)

    assert len(predictions) == 1
    assert bench.manifest is not None

    pred = predictions[0]
    trace = bench.manifest.traces[0]

    assert pred["instance_id"] == instance["instance_id"]
    assert pred["_system_used"] == 3
    assert pred["_tool_call_count"] == 2
    assert pred["_tool_turn_count"] == 1
    assert pred["_executed_commands"] == ["pwd", "git status"]
    assert pred["_execution_path"] == "pipeline"
    assert pred["model_patch"].startswith("diff --git")

    assert trace.routing == "S3"
    assert trace.meta["execution_path"] == "pipeline"
    assert trace.meta["tool_call_count"] == 2
    assert trace.meta["tool_turn_count"] == 1
    assert trace.meta["executed_commands"] == ["pwd", "git status"]
