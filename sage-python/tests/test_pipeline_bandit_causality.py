"""A14 bandit causality regressions for the Python pipeline."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sage.llm.base import LLMConfig, LLMResponse
from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext
from sage.pipeline_v2.execute import execute
from sage.pipeline_v2.learn import learn
from sage.runtime.event_log import RuntimeEventLog, install_event_log


@dataclass
class _Decision:
    decision_id: str = "d-bandit"
    model_id: str = "bandit-model"
    template: str = "single_agent"
    context: list[float] | None = None


class _Router:
    def assess_complexity(self, task: str) -> Any:
        return SimpleNamespace(system=1)

    def route(self, profile: Any) -> Any:
        return SimpleNamespace(system=profile.system)


class _Provider:
    def __init__(self, name: str = "provider", content: str = "bandit output") -> None:
        self.name = name
        self.model_id = ""
        self.calls: list[LLMConfig | None] = []
        self._content = content

    async def generate(self, messages: Any, config: LLMConfig | None = None, **kwargs: Any) -> LLMResponse:
        self.calls.append(config)
        return LLMResponse(content=self._content, model=config.model if config else None)


class _ProviderPool:
    def __init__(self, provider: _Provider, model_id: str = "bandit-model") -> None:
        self.provider = provider
        self.model_id = model_id
        self.resolved: list[str] = []

    def is_model_available(self, model_id: str) -> bool:
        return model_id == self.model_id

    def infer_provider(self, model_id: str) -> str:
        return "fake"

    def resolve(self, model_id: str) -> tuple[_Provider, LLMConfig]:
        self.resolved.append(model_id)
        self.provider.model_id = model_id
        return self.provider, LLMConfig(provider="fake", model=model_id)


class _FakeRoutingConstraints:
    def __init__(
        self,
        max_cost_usd: float = 0.0,
        max_latency_ms: float = 0.0,
        min_quality: float = 0.0,
        required_capabilities: list[str] | None = None,
        security_label: str = "",
        exploration_budget: float = 0.0,
        domain_hint: str = "",
    ) -> None:
        self.max_cost_usd = max_cost_usd
        self.max_latency_ms = max_latency_ms
        self.min_quality = min_quality
        self.required_capabilities = required_capabilities or []
        self.security_label = security_label
        self.exploration_budget = exploration_budget
        self.domain_hint = domain_hint


class _RustRouter:
    def __init__(
        self,
        *,
        decision: _Decision | None = None,
        route_error: Exception | None = None,
        record_error: Exception | None = None,
        strict_pending: bool = False,
    ) -> None:
        self.decision = decision or _Decision()
        self.route_error = route_error
        self.record_error = record_error
        self.strict_pending = strict_pending
        self.pending_decisions: set[str] = {self.decision.decision_id}
        self.integrated_calls: list[tuple[str, Any, str]] = []
        self.checked_records: list[tuple[str, str, str, float, float, float]] = []
        self.cancelled_decisions: list[str] = []

    def route_integrated(self, task: str, constraints: Any, topology_id: str) -> Any:
        self.integrated_calls.append((task, constraints, topology_id))
        if self.route_error is not None:
            raise self.route_error
        self.pending_decisions.add(self.decision.decision_id)
        return SimpleNamespace(
            decision_id=self.decision.decision_id,
            system=1,
            model_id=self.decision.model_id,
            selected_template=self.decision.template,
            template=self.decision.template,
            confidence=0.91,
            estimated_cost=0.01,
            topology_id=topology_id,
        )

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> Any:
        self.checked_records.append(
            (decision_id, executed_model_id, executed_template, quality, cost, latency_ms)
        )
        if self.record_error is not None:
            raise self.record_error
        if self.strict_pending and decision_id not in self.pending_decisions:
            raise RuntimeError("decision_unknown")
        self.pending_decisions.discard(decision_id)
        return SimpleNamespace(status="recorded")

    def cancel_bandit_decision(self, decision_id: str) -> bool:
        self.cancelled_decisions.append(decision_id)
        was_pending = decision_id in self.pending_decisions
        self.pending_decisions.discard(decision_id)
        return was_pending


class _QualityEstimator:
    def __init__(self, quality: float = 0.8) -> None:
        self.quality = quality

    def estimate(self, task: str, result: str, latency_s: float = 0.0) -> float:
        return self.quality


class _Bandit:
    def __init__(self, decision: _Decision | None = None) -> None:
        self.decision = decision or _Decision(context=[1.0, 0.0, 0.0])
        self.select_calls: list[tuple[float, str, list[float]]] = []
        self.checked_records: list[tuple[str, str, str, float, float, float]] = []
        self.cancelled: list[str] = []
        self.unchecked_record_calls = 0

    def select_with_context_for_template(
        self,
        exploration_budget: float,
        template: str,
        context: list[float],
    ) -> _Decision:
        self.select_calls.append((exploration_budget, template, context))
        self.decision.context = context
        return self.decision

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> None:
        self.checked_records.append(
            (decision_id, executed_model_id, executed_template, quality, cost, latency_ms)
        )

    def cancel_decision(self, decision_id: str) -> bool:
        self.cancelled.append(decision_id)
        return True

    def record(self, *args: Any, **kwargs: Any) -> None:
        self.unchecked_record_calls += 1
        raise AssertionError("unchecked record() must not be called")

    def record_outcome(self, *args: Any, **kwargs: Any) -> None:
        self.unchecked_record_calls += 1
        raise AssertionError("unchecked record_outcome() must not be called")


class _ExplodingBandit:
    def __init__(self) -> None:
        self.select_calls = 0
        self.record_calls = 0

    def select_with_context_for_template(self, *args: Any, **kwargs: Any) -> Any:
        self.select_calls += 1
        raise AssertionError("multi-agent path must not select standalone bandit arms")

    def select_with_context(self, *args: Any, **kwargs: Any) -> Any:
        self.select_calls += 1
        raise AssertionError("multi-agent path must not use legacy bandit selection")

    def select(self, *args: Any, **kwargs: Any) -> Any:
        self.select_calls += 1
        raise AssertionError("multi-agent path must not use legacy bandit selection")

    def record_outcome_checked(self, *args: Any, **kwargs: Any) -> None:
        self.record_calls += 1
        raise AssertionError("multi-agent path must not record standalone bandit outcomes")

    def record(self, *args: Any, **kwargs: Any) -> None:
        self.record_calls += 1
        raise AssertionError("unchecked record() must not be called")

    def record_outcome(self, *args: Any, **kwargs: Any) -> None:
        self.record_calls += 1
        raise AssertionError("unchecked record_outcome() must not be called")


class _Topology:
    template_type = "sequential"
    id = "topology-id"

    def __init__(self, n_nodes: int) -> None:
        self._nodes = [
            SimpleNamespace(model_id=f"model-{idx}", max_cost_usd=0.0)
            for idx in range(n_nodes)
        ]

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> Any:
        return self._nodes[idx]


def _pipeline(
    *,
    bandit: Any,
    provider_pool: Any | None = None,
    llm_provider: Any | None = None,
) -> CognitiveOrchestrationPipeline:
    default_provider = llm_provider or _Provider("default", "default output")
    return CognitiveOrchestrationPipeline(
        router=_Router(),
        engine=None,
        assigner=None,
        provider_pool=provider_pool,
        bandit=bandit,
        quality_estimator=_QualityEstimator(0.8),
        event_bus=None,
        llm_provider=default_provider,
        llm_config=LLMConfig(provider="default", model="default-model"),
    )


def _install_fake_sage_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sage_core", SimpleNamespace(RoutingConstraints=_FakeRoutingConstraints))


def _runtime_events(trace_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return events


def _mismatch_payloads(trace_dir: Path) -> list[dict[str, Any]]:
    return [
        event.get("payload", {})
        for event in _runtime_events(trace_dir)
        if event.get("event_type") == "bandit_attribution_mismatch"
    ]


async def _learn_with_trace(
    pipeline: CognitiveOrchestrationPipeline,
    ctx: PipelineContext,
    trace_dir: Path,
) -> None:
    log = RuntimeEventLog(run_id="01BANDITATTRIBUTION000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        log.emit_task_started(ctx.task)
        await learn(pipeline, ctx)
    finally:
        token.var.reset(token)
        log.close()


@pytest.mark.asyncio
async def test_single_agent_bandit_decision_is_executed_and_recorded_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    _install_fake_sage_core(monkeypatch)
    bandit = _Bandit()
    bandit_provider = _Provider("bandit-provider", "bandit output")
    provider_pool = _ProviderPool(bandit_provider)
    rust_router = _RustRouter()
    pipeline = _pipeline(bandit=bandit, provider_pool=provider_pool)
    pipeline._rust_router = rust_router

    result = await pipeline.run("single agent task", budget_usd=3.0)

    ctx = pipeline.last_context
    assert result == "bandit output"
    assert ctx.bandit_decision_id == "d-bandit"
    assert ctx.bandit_model_id == "bandit-model"
    assert ctx.bandit_template == "single_agent"
    assert ctx.executed_model_id == "bandit-model"
    assert ctx.executed_template == "single_agent"
    assert provider_pool.resolved == ["bandit-model"]
    assert bandit.select_calls == []
    assert bandit.checked_records == []
    assert len(rust_router.checked_records) == 1
    decision_id, model_id, template, quality, cost, latency = rust_router.checked_records[0]
    assert (decision_id, model_id, template) == ("d-bandit", "bandit-model", "single_agent")
    assert quality == pytest.approx(0.8)
    assert cost == pytest.approx(ctx.cost)
    assert latency == pytest.approx(ctx.latency_ms)
    assert ctx.bandit_attribution_state == "verified"
    assert bandit.unchecked_record_calls == 0


@pytest.mark.asyncio
async def test_route_integrated_fallback_marks_attribution_degraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Threat 1: Rust route_integrated raises -> fallback is auditable and unrecorded."""
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    _install_fake_sage_core(monkeypatch)
    rust_router = _RustRouter(route_error=RuntimeError("route_integrated unavailable"))
    pipeline = _pipeline(bandit=None)
    pipeline._rust_router = rust_router

    result = await pipeline.run("fallback task", budget_usd=3.0, system_hint=1)

    assert result == "default output"
    assert len(rust_router.integrated_calls) == 1
    _task, constraints, topology_id = rust_router.integrated_calls[0]
    assert isinstance(constraints, _FakeRoutingConstraints)
    assert constraints.max_cost_usd == pytest.approx(3.0)
    assert topology_id == ""
    assert rust_router.checked_records == []
    assert rust_router.cancelled_decisions == [""]
    assert _mismatch_payloads(tmp_path) == [
        {
            "decision_id": "",
            "selected_model_id": "",
            "selected_template": "",
            "executed_model_id": "",
            "executed_template": "",
            "reason_code": "router_fallback_degraded",
        }
    ]


@pytest.mark.asyncio
async def test_bandit_records_executed_model_after_controller_upgrade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Threat 2: controller upgrade changes the model; off-policy update is blocked."""
    monkeypatch.setenv("SAGE_ORACLE", "0")
    rust_router = _RustRouter(record_error=RuntimeError("model_mismatch"))
    pipeline = _pipeline(bandit=None)
    pipeline._rust_router = rust_router
    ctx = PipelineContext(
        task="controller upgrade task",
        bandit_decision_id="d-upgrade",
        bandit_model_id="model-a",
        bandit_template="single_agent",
        executed_model_id="model-b",
        executed_template="single_agent",
        result="non-empty",
        cost=0.03,
        latency_ms=42.0,
    )

    await _learn_with_trace(pipeline, ctx, tmp_path)

    assert rust_router.checked_records == [
        ("d-upgrade", "model-b", "single_agent", 0.8, 0.03, 42.0)
    ]
    payloads = _mismatch_payloads(tmp_path)
    assert [payload["reason_code"] for payload in payloads] == ["model_mismatch"]
    assert payloads[0]["selected_model_id"] == "model-a"
    assert payloads[0]["executed_model_id"] == "model-b"
    assert ctx.bandit_attribution_state == "mismatch"


@pytest.mark.asyncio
async def test_parallel_topology_emits_multi_node_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Threat 3: parallel/debate multi-model runs are skipped for round-1 attribution."""
    monkeypatch.setenv("SAGE_ORACLE", "0")
    rust_router = _RustRouter(
        decision=_Decision(decision_id="d-parallel", model_id="model-a", template="parallel"),
        strict_pending=True,
    )
    pipeline = _pipeline(bandit=None)
    pipeline._rust_router = rust_router
    ctx = PipelineContext(
        task="parallel task",
        bandit_decision_id="d-parallel",
        bandit_model_id="model-a",
        bandit_template="parallel",
        executed_model_id="",
        executed_template="parallel",
        executed_model_ids=["model-a", "model-b"],
        result="non-empty",
        cost=0.04,
        latency_ms=99.0,
    )

    await _learn_with_trace(pipeline, ctx, tmp_path)

    assert rust_router.checked_records == []
    assert rust_router.cancelled_decisions == ["d-parallel"]
    assert "d-parallel" not in rust_router.pending_decisions
    with pytest.raises(RuntimeError, match="decision_unknown"):
        rust_router.record_outcome_checked(
            "d-parallel",
            "model-a",
            "parallel",
            0.8,
            0.04,
            99.0,
        )
    payloads = _mismatch_payloads(tmp_path)
    assert [payload["reason_code"] for payload in payloads] == ["multi_node_ambiguous"]
    assert payloads[0]["decision_id"] == "d-parallel"
    assert ctx.bandit_attribution_state == "skipped"


@pytest.mark.asyncio
async def test_quality_abstain_cancels_pending_bandit_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    rust_router = _RustRouter(
        decision=_Decision(decision_id="d-abstain", model_id="model-a"),
        strict_pending=True,
    )
    pipeline = _pipeline(bandit=None)
    pipeline._rust_router = rust_router
    pipeline.quality_estimator = None
    ctx = PipelineContext(
        task="abstain task",
        bandit_decision_id="d-abstain",
        bandit_model_id="model-a",
        bandit_template="single_agent",
        executed_model_id="model-a",
        executed_template="single_agent",
        result="non-empty but unscored",
        cost=0.01,
        latency_ms=10.0,
    )

    await _learn_with_trace(pipeline, ctx, tmp_path)

    assert rust_router.checked_records == []
    assert rust_router.cancelled_decisions == ["d-abstain"]
    assert "d-abstain" not in rust_router.pending_decisions
    assert _mismatch_payloads(tmp_path) == []
    assert ctx.bandit_attribution_state == "skipped"


@pytest.mark.asyncio
async def test_record_outcome_uses_same_bandit_as_issuer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Threat 4: Stage-5 records through SystemRouter's internal issuer bandit."""
    monkeypatch.setenv("SAGE_ORACLE", "0")
    rust_router = _RustRouter()
    wrong_bandit = _ExplodingBandit()
    pipeline = _pipeline(bandit=wrong_bandit)
    pipeline._rust_router = rust_router
    ctx = PipelineContext(
        task="same issuer recorder task",
        bandit_decision_id="d-issued-by-router",
        bandit_model_id="model-a",
        bandit_template="single_agent",
        executed_model_id="model-a",
        executed_template="single_agent",
        result="non-empty",
        cost=0.05,
        latency_ms=77.0,
    )

    await _learn_with_trace(pipeline, ctx, tmp_path)

    assert rust_router.checked_records == [
        ("d-issued-by-router", "model-a", "single_agent", 0.8, 0.05, 77.0)
    ]
    assert wrong_bandit.record_calls == 0
    assert wrong_bandit.select_calls == 0
    assert _mismatch_payloads(tmp_path) == []
    assert ctx.bandit_attribution_state == "verified"


@pytest.mark.asyncio
async def test_stage_learn_refuses_off_policy_bandit_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    bandit = _Bandit(_Decision(decision_id="d-offpolicy", model_id="selected"))
    rust_router = _RustRouter(record_error=RuntimeError("model_mismatch"))
    pipeline = _pipeline(bandit=bandit)
    pipeline._rust_router = rust_router
    ctx = PipelineContext(
        task="learn task",
        bandit_decision_id="d-offpolicy",
        bandit_model_id="selected",
        bandit_template="single_agent",
        executed_model_id="other",
        executed_template="single_agent",
        result="non-empty",
        cost=0.02,
        latency_ms=123.0,
    )

    await _learn_with_trace(pipeline, ctx, tmp_path)

    assert rust_router.checked_records == [
        ("d-offpolicy", "other", "single_agent", 0.8, 0.02, 123.0)
    ]
    assert bandit.checked_records == []
    assert bandit.cancelled == []
    assert bandit.unchecked_record_calls == 0
    assert rust_router.cancelled_decisions == ["d-offpolicy"]
    assert [payload["reason_code"] for payload in _mismatch_payloads(tmp_path)] == [
        "model_mismatch"
    ]
    assert ctx.bandit_attribution_state == "mismatch"


@pytest.mark.asyncio
async def test_multi_agent_path_does_not_select_or_record_standalone_bandit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeExecutor:
        def __init__(self, graph: Any) -> None:
            self.graph = graph

    class _FakeRunner:
        tool_call_count = 0
        tool_turn_count = 0
        executed_commands: list[str] = []
        total_cost_usd = 0.03

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(self, task: str) -> str:
            return "multi-agent output"

    import sage.topology.runner as runner_mod

    monkeypatch.setitem(sys.modules, "sage_core", SimpleNamespace(TopologyExecutor=_FakeExecutor))
    monkeypatch.setattr(runner_mod, "TopologyRunner", _FakeRunner)

    bandit = _ExplodingBandit()
    pipeline = _pipeline(bandit=bandit)
    ctx = PipelineContext(
        task="multi-agent task",
        topology=_Topology(2),
        assignments={0: "model-0", 1: "model-1"},
        system=2,
        result="",
    )

    ctx = await execute(pipeline, ctx)
    await learn(pipeline, ctx)

    assert ctx.result == "multi-agent output"
    assert ctx.bandit_decision_id == ""
    assert ctx.executed_model_ids == ["model-0", "model-1"]
    assert ctx.executed_template == "sequential"
    assert bandit.select_calls == 0
    assert bandit.record_calls == 0


@pytest.mark.asyncio
async def test_pipeline_never_uses_unchecked_bandit_record_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    _install_fake_sage_core(monkeypatch)
    bandit = _Bandit(_Decision(decision_id="d-checked", model_id="bandit-model"))
    bandit_provider = _Provider("bandit-provider", "checked output")
    provider_pool = _ProviderPool(bandit_provider)
    rust_router = _RustRouter(decision=_Decision(decision_id="d-checked", model_id="bandit-model"))
    pipeline = _pipeline(bandit=bandit, provider_pool=provider_pool)
    pipeline._rust_router = rust_router

    await pipeline.run("single agent checked-only task", budget_usd=3.0)

    assert bandit.select_calls == []
    assert bandit.checked_records == []
    assert [record[0] for record in rust_router.checked_records] == ["d-checked"]
    assert bandit.unchecked_record_calls == 0
