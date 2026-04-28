from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.llm.model_assigner import ModelAssigner
from sage.llm.model_registry import ModelCardCatalog
from sage.topology.runner import TopologyRunner


_CAPABILITY_CATALOG_TOML = """
[[models]]
id = "no-tools-cheap"
provider = "test"
family = "test"
code_score = 0.8
reasoning_score = 0.6
tool_use_score = 0.1
math_score = 0.2
formal_z3_strength = 0.1
cost_input_per_m = 0.01
cost_output_per_m = 0.01
latency_ttft_ms = 50.0
tokens_per_sec = 500.0
s1_affinity = 0.95
s2_affinity = 0.95
s3_affinity = 0.4
supports_tools = false
supports_json_mode = false
context_window = 32000

[models.domain_scores]
code = 0.95
general = 0.95

[[models]]
id = "tools-cap"
provider = "test"
family = "test"
code_score = 0.85
reasoning_score = 0.7
tool_use_score = 0.95
math_score = 0.3
formal_z3_strength = 0.2
cost_input_per_m = 0.1
cost_output_per_m = 0.1
latency_ttft_ms = 120.0
tokens_per_sec = 300.0
s1_affinity = 0.7
s2_affinity = 0.8
s3_affinity = 0.5
supports_tools = true
supports_json_mode = false
context_window = 128000

[models.domain_scores]
code = 0.8
general = 0.8

[[models]]
id = "json-cap"
provider = "test"
family = "test"
code_score = 0.8
reasoning_score = 0.8
tool_use_score = 0.4
math_score = 0.4
formal_z3_strength = 0.3
cost_input_per_m = 0.1
cost_output_per_m = 0.1
latency_ttft_ms = 150.0
tokens_per_sec = 250.0
s1_affinity = 0.7
s2_affinity = 0.8
s3_affinity = 0.6
supports_tools = false
supports_json_mode = true
context_window = 128000

[models.domain_scores]
code = 0.82
general = 0.82
"""


class FakeNode:
    def __init__(
        self,
        role: str,
        model_id: str,
        system: int,
        required_capabilities: list[str] | None = None,
    ) -> None:
        self.role = role
        self.model_id = model_id
        self.system = system
        self.required_capabilities = required_capabilities or []


class FakeGraph:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = list(nodes)
        self.set_calls: list[tuple[int, str]] = []

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, idx: int) -> FakeNode:
        return self._nodes[idx]

    def get_predecessors(self, idx: int) -> list[int]:
        return list(range(idx))

    def set_node_model_id(self, idx: int, model_id: str) -> None:
        self.set_calls.append((idx, model_id))
        self._nodes[idx].model_id = model_id


class FakeExecutor:
    def __init__(self, ready_sequence: list[list[int]]) -> None:
        self._batches = list(ready_sequence)
        self._batch_idx = 0

    def next_ready(self, graph: FakeGraph) -> list[int]:
        del graph
        if self._batch_idx >= len(self._batches):
            return []
        batch = self._batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def mark_completed(self, idx: int) -> None:
        del idx

    def is_done(self) -> bool:
        return self._batch_idx >= len(self._batches)


class FakeProviderPool:
    """ProviderPool subset used by TopologyRunner fallback tests."""

    def __init__(
        self,
        model_to_provider: dict[str, str],
        dead_models: set[str] | None = None,
    ) -> None:
        self._model_to_provider = model_to_provider
        self._dead = dead_models or set()
        self._model_to_provider_instance: dict[str, Any] = {}
        self.failures: list[tuple[str, str]] = []
        self.successes: list[str] = []
        self.resolved_model_ids: list[str] = []

    def record_failure(self, provider_name: str, exc: BaseException) -> None:
        self.failures.append((provider_name, str(exc)))

    def record_success(self, provider_name: str) -> None:
        self.successes.append(provider_name)

    def is_model_available(self, model_id: str) -> bool:
        return model_id not in self._dead

    def resolve(self, model_id: str) -> tuple[Any, Any]:
        provider_name = self._model_to_provider.get(model_id)
        if provider_name is None:
            raise ValueError(f"Unknown model {model_id}")
        self.resolved_model_ids.append(model_id)
        provider = self._model_to_provider_instance.get(model_id)
        if provider is None:
            provider = MagicMock()
            provider.generate = AsyncMock(
                return_value=SimpleNamespace(content=f"reply from {model_id}"),
            )
        config = SimpleNamespace(provider=provider_name, model=model_id)
        return provider, config


class CapturingAssigner:
    """Records assign_single_node call args and returns queued model IDs."""

    def __init__(self, returns: list[str]) -> None:
        self._returns = list(returns)
        self.calls: list[dict[str, Any]] = []

    def assign_single_node(
        self,
        graph: FakeGraph,
        node_idx: int,
        task_domain: str = "",
        budget_usd: float = 0.0,
        exclude_model_ids: list[str] | None = None,
        task_system: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "node_idx": node_idx,
                "task_domain": task_domain,
                "budget_usd": budget_usd,
                "exclude_model_ids": list(exclude_model_ids or []),
                "task_system": task_system,
            },
        )
        if not self._returns:
            raise ValueError("no more candidates")
        chosen = self._returns.pop(0)
        graph.set_node_model_id(node_idx, chosen)
        return chosen


def _real_assigner() -> ModelAssigner:
    return ModelAssigner(ModelCardCatalog.from_toml_str(_CAPABILITY_CATALOG_TOML))


def _make_runner_with_fallback(
    *,
    graph_nodes: list[FakeNode],
    primary_fail: bool = True,
    assigner: Any = None,
    pool: Any = None,
    task_domain: str = "code",
    budget_usd: float = 1.0,
) -> tuple[TopologyRunner, FakeGraph, FakeExecutor, Any]:
    graph = FakeGraph(graph_nodes)
    executor = FakeExecutor([[i] for i in range(len(graph_nodes))])

    primary_provider = MagicMock()
    if primary_fail:
        primary_provider.generate = AsyncMock(side_effect=ConnectionError("primary down"))
    else:
        primary_provider.generate = AsyncMock(
            return_value=SimpleNamespace(content="primary OK"),
        )
    default_provider = MagicMock()
    default_provider.generate = AsyncMock(
        return_value=SimpleNamespace(content="default should not be used"),
    )

    config = SimpleNamespace(provider="primary", model=graph_nodes[0].model_id)
    if isinstance(pool, FakeProviderPool):
        pool._model_to_provider[graph_nodes[0].model_id] = "primary"
        pool._model_to_provider_instance[graph_nodes[0].model_id] = primary_provider

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=default_provider,
        llm_config=config,
        provider_pool=pool,
        controller=None,
        agent_loop_factory=None,
        assigner=assigner,
        task_domain=task_domain,
        budget_usd=budget_usd,
    )
    return runner, graph, executor, primary_provider


@pytest.mark.asyncio
async def test_fallback_never_selects_no_tools_model_for_tools_node() -> None:
    pool = FakeProviderPool(
        model_to_provider={"no-tools-cheap": "cheap", "tools-cap": "openai"},
    )
    graph_nodes = [
        FakeNode(
            role="coder",
            model_id="failed-primary",
            system=2,
            required_capabilities=["tools"],
        ),
    ]

    runner, graph, _, primary = _make_runner_with_fallback(
        graph_nodes=graph_nodes,
        assigner=_real_assigner(),
        pool=pool,
    )
    result = await runner.run("test task")

    primary.generate.assert_awaited_once()
    assert "no-tools-cheap" not in pool.resolved_model_ids
    assert any(set_call == (0, "tools-cap") for set_call in graph.set_calls)
    assert "tools-cap" in result


@pytest.mark.asyncio
async def test_fallback_never_selects_no_json_model_for_json_node() -> None:
    pool = FakeProviderPool(
        model_to_provider={"no-tools-cheap": "cheap", "json-cap": "openai"},
    )
    graph_nodes = [
        FakeNode(
            role="parser",
            model_id="failed-primary",
            system=2,
            required_capabilities=["json"],
        ),
    ]

    runner, graph, _, primary = _make_runner_with_fallback(
        graph_nodes=graph_nodes,
        assigner=_real_assigner(),
        pool=pool,
    )
    result = await runner.run("parse this")

    primary.generate.assert_awaited_once()
    assert "no-tools-cheap" not in pool.resolved_model_ids
    assert any(set_call == (0, "json-cap") for set_call in graph.set_calls)
    assert "json-cap" in result


@pytest.mark.asyncio
async def test_dead_provider_excluded_from_runtime_fallback() -> None:
    pool = FakeProviderPool(
        model_to_provider={"dead-1": "deepseek", "alive-1": "openai"},
        dead_models={"dead-1"},
    )
    assigner = CapturingAssigner(returns=["dead-1", "alive-1"])
    graph_nodes = [
        FakeNode(
            role="actor",
            model_id="failed-primary",
            system=1,
            required_capabilities=[],
        ),
    ]

    runner, _, _, primary = _make_runner_with_fallback(
        graph_nodes=graph_nodes,
        assigner=assigner,
        pool=pool,
    )
    result = await runner.run("any")

    primary.generate.assert_awaited_once()
    assert len(assigner.calls) == 2
    assert "dead-1" in assigner.calls[1]["exclude_model_ids"]
    assert "alive-1" in result


@pytest.mark.asyncio
async def test_budget_respected_after_failed_primary_model() -> None:
    pool = FakeProviderPool(model_to_provider={"alive": "openai"})
    assigner = CapturingAssigner(returns=["alive"])
    graph_nodes = [
        FakeNode(
            role="actor",
            model_id="failed-primary",
            system=1,
            required_capabilities=[],
        ),
    ]

    runner, _, _, _ = _make_runner_with_fallback(
        graph_nodes=graph_nodes,
        assigner=assigner,
        pool=pool,
        budget_usd=2.50,
    )
    await runner.run("any")

    assert len(assigner.calls) == 1
    forwarded_budget = assigner.calls[0]["budget_usd"]
    assert forwarded_budget > 0.0
    assert abs(forwarded_budget - 2.50) < 0.01


@pytest.mark.asyncio
async def test_budget_exhausted_cost_tracker_passes_zero_and_fails_closed() -> None:
    """cgpro 2026-04-28 R3 verify nudge: cost_tracker.remaining == 0.0 is the
    fail-closed signal (not unknown/unlimited). Runner must forward 0.0 to
    assign_single_node, which rejects all paid models, raising ValueError.
    Helper restores original_model_id and returns None; caller re-raises the
    original ConnectionError.
    """
    pool = FakeProviderPool(model_to_provider={"alive": "openai"})
    # Assigner is a real ModelAssigner — it will refuse paid models on budget=0
    assigner = _real_assigner()
    graph_nodes = [
        FakeNode(
            role="actor",
            model_id="failed-primary",
            system=1,
            required_capabilities=[],
        ),
    ]

    class ExhaustedCostTracker:
        """Mimics CostTracker with a fully-spent budget."""
        @property
        def remaining(self) -> float:
            return 0.0

        @property
        def is_over_budget(self) -> bool:
            # Don't trip the runner's pre-execution budget gate; the bug we
            # exercise is the assigner-side budget filter, fired AFTER the
            # primary fails. Pre-execution gate is orthogonal.
            return False

        def record_spend(self, amount: float) -> None:
            pass

    runner, graph, _, _ = _make_runner_with_fallback(
        graph_nodes=graph_nodes,
        assigner=assigner,
        pool=pool,
        budget_usd=10.0,  # would otherwise allow paid models
    )
    runner._cost_tracker = ExhaustedCostTracker()  # type: ignore[assignment]

    # _remaining_budget_usd should return min(tracker=0.0, internal=10.0) = 0.0
    assert runner._remaining_budget_usd() == 0.0

    # Run must propagate ConnectionError (fail-closed) since assigner refuses
    # all candidates at budget=0.0.
    with pytest.raises(ConnectionError):
        await runner.run("any")

    # Original model_id restored on the graph after exhaustion.
    assert graph._nodes[0].model_id == "failed-primary"


@pytest.mark.asyncio
async def test_fallback_path_compatible_with_existing_topology_runner() -> None:
    pool = FakeProviderPool(model_to_provider={})
    assigner = CapturingAssigner(returns=[])
    graph_nodes = [
        FakeNode(
            role="actor",
            model_id="primary-good",
            system=1,
            required_capabilities=[],
        ),
    ]

    runner, _, _, primary = _make_runner_with_fallback(
        graph_nodes=graph_nodes,
        primary_fail=False,
        assigner=assigner,
        pool=pool,
    )
    result = await runner.run("any")

    primary.generate.assert_awaited_once()
    assert len(assigner.calls) == 0
    assert "primary OK" in result
