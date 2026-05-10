from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_v2.provider_policy import (
    ProviderPolicyViolation,
    configure_pipeline_provider_policy,
    evaluate_provider_policy,
    install_provider_call_guards,
)
from sage.pipeline_v2.runtime_events import runtime_provider_id_for_model
from sage.runtime.event_log import RuntimeEventLog, install_event_log


@dataclass
class _FakeNode:
    model_id: str = "gpt-5.5-pro"
    role: str = "worker"
    node_type: str = "llm"


class _FakeTopology:
    id = "provider-policy-topology"
    template_type = "unit"

    def node_count(self) -> int:
        return 1

    def get_node(self, idx: int) -> _FakeNode:
        if idx != 0:
            raise IndexError(idx)
        return _FakeNode()

    def get_edges(self) -> list[tuple[str, str, str]]:
        return []


def _provider_for(model_id: str) -> str:
    if model_id.startswith("gpt-"):
        return "openai"
    if model_id.startswith("gemini"):
        return "google"
    if model_id.startswith("deepseek"):
        return "deepseek"
    return ""


def _make_pipeline() -> CognitiveOrchestrationPipeline:
    provider_pool = MagicMock()
    provider_pool.infer_provider.side_effect = _provider_for
    return CognitiveOrchestrationPipeline(
        router=MagicMock(),
        engine=None,
        assigner=None,
        provider_pool=provider_pool,
        llm_config=SimpleNamespace(provider="google", model="gemini-2.5-flash"),
    )


def _install_policy(pipeline: CognitiveOrchestrationPipeline) -> None:
    configure_pipeline_provider_policy(
        pipeline,
        allowlist=("google", "deepseek"),
        denylist=("openai",),
        source="cli",
    )


class _MustNotBeCalledProvider:
    name = "openai"
    model_id = "gpt-5.5-pro"

    def __init__(self) -> None:
        self.called = False

    async def generate(self, *_args: Any, **_kwargs: Any) -> Any:
        self.called = True
        raise AssertionError("disallowed provider was called")


def test_provider_policy_ignores_spoofed_provider_hint() -> None:
    pipeline = _make_pipeline()
    _install_policy(pipeline)
    ctx = PipelineContext(task="x", budget=5.0)
    ctx.topology = _FakeTopology()
    ctx.assignments = {0: "gpt-5.5-pro"}
    ctx.provider_hints = {0: "google"}

    decision = evaluate_provider_policy(pipeline, ctx)

    assert len(decision.violations) == 1
    violation = decision.violations[0]
    assert violation.provider_id == "openai"
    assert violation.hint_provider_id == "google"
    assert violation.reason == "denylist"


def test_model_assigned_provider_id_ignores_spoofed_provider_hint() -> None:
    pipeline = _make_pipeline()
    ctx = PipelineContext(task="x", budget=5.0)
    ctx.assignments = {0: "gpt-5.5-pro"}
    ctx.provider_hints = {0: "google"}

    provider_id = runtime_provider_id_for_model(pipeline, "gpt-5.5-pro", ctx)

    assert provider_id == "openai"


@pytest.mark.asyncio
async def test_provider_call_guard_blocks_direct_default_provider() -> None:
    provider = _MustNotBeCalledProvider()
    pipeline = _make_pipeline()
    pipeline.llm_provider = provider
    _install_policy(pipeline)
    install_provider_call_guards(pipeline)

    with pytest.raises(ProviderPolicyViolation):
        await pipeline.llm_provider.generate(messages=[], config=None)

    assert provider.called is False


@pytest.mark.asyncio
async def test_provider_pool_health_check_skips_denied_provider() -> None:
    from sage.llm.base import LLMConfig
    from sage.llm.provider_pool import ProviderPool

    provider = _MustNotBeCalledProvider()
    pool = ProviderPool(
        default_provider=provider,
        registry=None,
        default_config=LLMConfig(provider="openai", model="gpt-5.5-pro"),
        providers={"openai": provider},
    )
    pool.set_provider_policy(
        allowlist=frozenset({"google", "deepseek"}),
        denylist=frozenset({"openai"}),
        source="cli",
    )

    result = await pool.health_check(timeout=0.01)

    assert result == {"openai": False}
    assert provider.called is False


def test_provider_pool_resolve_blocks_denied_provider() -> None:
    from sage.llm.base import LLMConfig
    from sage.llm.provider_pool import ProviderPool

    provider = MagicMock()
    provider.name = "openai"
    registry = MagicMock()
    registry.get.return_value = SimpleNamespace(
        provider="openai",
        context_window=128000,
    )
    pool = ProviderPool(
        default_provider=provider,
        registry=registry,
        default_config=LLMConfig(provider="openai", model="gpt-5.5-pro"),
        providers={"openai": provider},
    )
    pool.set_provider_policy(
        allowlist=frozenset({"google", "deepseek"}),
        denylist=frozenset({"openai"}),
        source="cli",
    )

    with pytest.raises(ProviderPolicyViolation):
        pool.resolve("gpt-5.5-pro")


@pytest.fixture(autouse=True)
def _runtime_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)


def _stub_pipeline_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    from sage.pipeline_v2 import assign_models as assign_models_mod
    from sage.pipeline_v2 import classify as classify_mod
    from sage.pipeline_v2 import decompose as decompose_mod
    from sage.pipeline_v2 import execute as execute_mod
    from sage.pipeline_v2 import learn as learn_mod
    from sage.pipeline_v2 import memory_gate as memory_gate_mod
    from sage.pipeline_v2 import select_topology as select_topology_mod

    calls = {
        "classify": 0,
        "decompose": 0,
        "select_topology": 0,
        "assign_models": 0,
        "execute": 0,
        "learn": 0,
        "record_to_memory": 0,
    }

    def fake_classify(_pipeline: Any, ctx: Any) -> Any:
        calls["classify"] += 1
        ctx.system = 1
        ctx.domain = "code"
        return ctx

    async def fake_decompose(_pipeline: Any, ctx: Any) -> Any:
        calls["decompose"] += 1
        return ctx

    def fake_select_topology(_pipeline: Any, ctx: Any) -> Any:
        calls["select_topology"] += 1
        ctx.topology = _FakeTopology()
        ctx.topology_id = "provider-policy-topology"
        return ctx

    def fake_assign_models(_pipeline: Any, ctx: Any) -> Any:
        calls["assign_models"] += 1
        ctx.assignments = {0: "gpt-5.5-pro"}
        ctx.provider_hints = {0: "google"}
        return ctx

    async def fake_execute(_pipeline: Any, ctx: Any, **_kwargs: Any) -> Any:
        calls["execute"] += 1
        ctx.result = "should not execute"
        return ctx

    async def fake_learn(_pipeline: Any, _ctx: Any) -> None:
        calls["learn"] += 1

    monkeypatch.setattr(memory_gate_mod, "build_write_gate", lambda _pipeline: object())
    monkeypatch.setattr(memory_gate_mod, "record_to_memory", lambda *_a, **_k: None)
    monkeypatch.setattr(classify_mod, "classify", fake_classify)
    monkeypatch.setattr(decompose_mod, "decompose", fake_decompose)
    monkeypatch.setattr(select_topology_mod, "select_topology", fake_select_topology)
    monkeypatch.setattr(assign_models_mod, "assign_models", fake_assign_models)
    monkeypatch.setattr(execute_mod, "execute", fake_execute)
    monkeypatch.setattr(learn_mod, "learn", fake_learn)
    return calls


@pytest.mark.asyncio
async def test_provider_policy_blocks_before_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_pipeline_stages(monkeypatch)
    pipeline = _make_pipeline()
    _install_policy(pipeline)
    event_log = RuntimeEventLog(run_id="provider-policy", trace_dir=tmp_path)
    token = install_event_log(event_log)
    try:
        with pytest.raises(ProviderPolicyViolation) as excinfo:
            await pipeline.run("use the denied model", budget_usd=0.1)
    finally:
        token.var.reset(token)

    assert "openai" in str(excinfo.value)
    assert calls["execute"] == 0
    assert calls["learn"] == 0
    assert event_log._path is not None  # noqa: SLF001 - tests inspect sink.
    events = [
        json.loads(line)
        for line in event_log._path.read_text(encoding="utf-8").splitlines()  # noqa: SLF001
        if line.strip()
    ]
    event_types = [event["event_type"] for event in events]
    failure = next(event for event in events if event["event_type"] == "failure")
    final_result = next(event for event in events if event["event_type"] == "final_result")
    assert "model_assigned" in event_types
    assert "node_started" not in event_types
    assert failure["kind"] == "provider_policy"
    assert failure["error_type"] == "provider_policy_violation"
    assert final_result["status"] == "failure"


@pytest.mark.asyncio
async def test_provider_policy_blocks_stage1_planner_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sage.pipeline_v2 import classify as classify_mod
    from sage.pipeline_v2 import execute as execute_mod
    from sage.pipeline_v2 import memory_gate as memory_gate_mod
    from sage.pipeline_v2 import select_topology as select_topology_mod

    provider = _MustNotBeCalledProvider()
    pipeline = _make_pipeline()
    pipeline.llm_provider = provider
    _install_policy(pipeline)
    calls = {"execute": 0}

    def fake_classify(_pipeline: Any, ctx: Any) -> Any:
        ctx.system = 2
        ctx.domain = "code"
        return ctx

    def fake_select_topology(_pipeline: Any, ctx: Any) -> Any:
        ctx.topology = _FakeTopology()
        return ctx

    async def fake_execute(_pipeline: Any, ctx: Any, **_kwargs: Any) -> Any:
        calls["execute"] += 1
        ctx.result = "should not execute"
        return ctx

    monkeypatch.setattr(memory_gate_mod, "build_write_gate", lambda _pipeline: object())
    monkeypatch.setattr(classify_mod, "classify", fake_classify)
    monkeypatch.setattr(select_topology_mod, "select_topology", fake_select_topology)
    monkeypatch.setattr(execute_mod, "execute", fake_execute)

    event_log = RuntimeEventLog(run_id="provider-policy-stage1", trace_dir=tmp_path)
    token = install_event_log(event_log)
    try:
        with pytest.raises(ProviderPolicyViolation):
            await pipeline.run("decompose with denied default provider", budget_usd=0.1)
    finally:
        token.var.reset(token)

    assert provider.called is False
    assert calls["execute"] == 0
