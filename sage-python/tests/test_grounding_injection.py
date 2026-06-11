"""G1 injection contract — cgpro GROUNDING DESIGN_LOCKED (2026-06-11,
sequence 'G1 injection')."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sage.grounding import GROUNDING_MARKER, compose_grounded_task
from sage.topology.runner import TopologyRunner

ENVELOPE = (
    GROUNDING_MARKER + "\nbase_commit: abc\nfile_count: 1\n"
    "total_bytes: 10\n\nPatch only files shown below.\n\n"
    "### FILE: src/mod.py\n```\nx = 1\n```\n"
)


class FakeNode:
    def __init__(self, role: str):
        self.role = role
        self.model_id = ""
        self.system = 1
        self.required_capabilities: list[str] = []
        self.prompt = ""


class FakeGraph:
    def __init__(self, nodes):
        self._nodes = nodes

    def node_count(self):
        return len(self._nodes)

    def get_node(self, idx):
        return self._nodes[idx]


class FakeExecutor:
    def next_ready(self, g):
        return []

    def mark_completed(self, i):
        pass

    def is_done(self):
        return True


class _SpyLoop:
    """Stands in for the per-node AgentLoop: records run() input."""

    instances: list["_SpyLoop"] = []

    def __init__(self, **kwargs):
        self.factory_kwargs = kwargs
        self.run_arg: str | None = None
        self.total_cost_usd = 0.0
        self.last_exhaustion = None
        _SpyLoop.instances.append(self)

    async def run(self, task: str) -> str:
        self.run_arg = task
        return "node output"


def _spy_factory(**kwargs):
    return _SpyLoop(**kwargs)


def _runner(role: str) -> TopologyRunner:
    _SpyLoop.instances.clear()
    return TopologyRunner(
        FakeGraph([FakeNode(role)]),
        FakeExecutor(),
        llm_provider=AsyncMock(),
        agent_loop_factory=_spy_factory,
    )


@pytest.mark.asyncio
async def test_topology_emitter_receives_grounding_user_message(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    r = _runner("coder")

    async def _fake_ensure(task):
        r.last_grounding_telemetry = {"file_count": 1}
        return ENVELOPE

    monkeypatch.setattr(r, "_ensure_grounding_block", _fake_ensure)
    await r._execute_node_via_agent_loop(0, "fix the bug")
    loop = _SpyLoop.instances[-1]
    assert loop.run_arg is not None
    assert loop.run_arg.startswith(GROUNDING_MARKER)
    assert "## Task:\nfix the bug" in loop.run_arg


@pytest.mark.asyncio
async def test_grounding_not_in_system_prompt(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    r = _runner("coder")

    async def _fake_ensure(task):
        return ENVELOPE

    monkeypatch.setattr(r, "_ensure_grounding_block", _fake_ensure)
    await r._execute_node_via_agent_loop(0, "fix the bug")
    loop = _SpyLoop.instances[-1]
    assert GROUNDING_MARKER not in loop.factory_kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_non_emitter_nodes_do_not_receive_file_blocks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    for role in ("planner", "synthesizer", "verifier"):
        r = _runner(role)
        called = {"n": 0}

        async def _fake_ensure(task, _c=called):
            _c["n"] += 1
            return ENVELOPE

        monkeypatch.setattr(r, "_ensure_grounding_block", _fake_ensure)
        await r._execute_node_via_agent_loop(0, "fix the bug")
        loop = _SpyLoop.instances[-1]
        assert called["n"] == 0, role  # never even built
        assert GROUNDING_MARKER not in (loop.run_arg or ""), role


@pytest.mark.asyncio
async def test_grounding_exempt_from_smmu_truncation_and_similarity_dedup(
    monkeypatch,
) -> None:
    """The envelope joins full_task directly — a HUGE block survives
    intact (no predecessor 1000-char floor, no dedup pass)."""
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    huge = ENVELOPE + ("y = 2\n" * 20000)
    r = _runner("coder")

    async def _fake_ensure(task):
        return huge

    monkeypatch.setattr(r, "_ensure_grounding_block", _fake_ensure)
    await r._execute_node_via_agent_loop(0, "fix the bug")
    loop = _SpyLoop.instances[-1]
    assert huge in (loop.run_arg or "")  # byte-for-byte


def test_no_double_injection_when_bypass_and_runner_paths_overlap() -> None:
    once = compose_grounded_task(ENVELOPE, "fix the bug")
    twice = compose_grounded_task(ENVELOPE, once)
    assert twice == once
    assert twice.count(GROUNDING_MARKER) == 1


@pytest.mark.asyncio
async def test_ensure_grounding_block_skips_without_profile(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("SAGE_TASK_ARTIFACT_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path)  # no .git either
    r = _runner("coder")
    block = await r._ensure_grounding_block("fix")
    assert block == ""
