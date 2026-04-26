"""A14 bandit causality regressions for the Python pipeline."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from sage.llm.base import LLMConfig, LLMResponse
from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


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


@pytest.mark.asyncio
async def test_single_agent_bandit_decision_is_executed_and_recorded_checked() -> None:
    bandit = _Bandit()
    bandit_provider = _Provider("bandit-provider", "bandit output")
    provider_pool = _ProviderPool(bandit_provider)
    pipeline = _pipeline(bandit=bandit, provider_pool=provider_pool)

    result = await pipeline.run("single agent task", budget_usd=3.0)

    ctx = pipeline.last_context
    assert result == "bandit output"
    assert ctx.bandit_decision_id == "d-bandit"
    assert ctx.bandit_model_id == "bandit-model"
    assert ctx.bandit_template == "single_agent"
    assert ctx.executed_model_id == "bandit-model"
    assert ctx.executed_template == "single_agent"
    assert provider_pool.resolved == ["bandit-model"]
    assert len(bandit.checked_records) == 1
    decision_id, model_id, template, quality, cost, latency = bandit.checked_records[0]
    assert (decision_id, model_id, template) == ("d-bandit", "bandit-model", "single_agent")
    assert quality == pytest.approx(0.8)
    assert cost == pytest.approx(ctx.cost)
    assert latency == pytest.approx(ctx.latency_ms)
    assert bandit.unchecked_record_calls == 0


@pytest.mark.asyncio
async def test_stage_learn_refuses_off_policy_bandit_outcome() -> None:
    bandit = _Bandit(_Decision(decision_id="d-offpolicy", model_id="selected"))
    pipeline = _pipeline(bandit=bandit)
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

    await pipeline._stage_learn(ctx)

    assert bandit.checked_records == []
    assert bandit.cancelled == ["d-offpolicy"]
    assert bandit.unchecked_record_calls == 0


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

    ctx = await pipeline._stage_execute(ctx)
    await pipeline._stage_learn(ctx)

    assert ctx.result == "multi-agent output"
    assert ctx.bandit_decision_id is None
    assert ctx.executed_model_ids == {0: "model-0", 1: "model-1"}
    assert ctx.executed_template == "sequential"
    assert bandit.select_calls == 0
    assert bandit.record_calls == 0


@pytest.mark.asyncio
async def test_pipeline_never_uses_unchecked_bandit_record_methods() -> None:
    bandit = _Bandit(_Decision(decision_id="d-checked", model_id="bandit-model"))
    bandit_provider = _Provider("bandit-provider", "checked output")
    provider_pool = _ProviderPool(bandit_provider)
    pipeline = _pipeline(bandit=bandit, provider_pool=provider_pool)

    await pipeline.run("single agent checked-only task", budget_usd=3.0)

    assert [record[0] for record in bandit.checked_records] == ["d-checked"]
    assert bandit.unchecked_record_calls == 0
