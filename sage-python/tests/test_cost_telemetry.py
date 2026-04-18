"""Regression guards for bench cost telemetry.

Until 2026-04-18, ``sage.phases.think`` computed per-step cost from
``_estimate_tokens(content)`` × ``_COST_PER_1K.get(model, DEFAULT)`` and
never read ``LLMResponse.usage["cost_usd"]``. That estimate silently
floored to $0 whenever the model id was missing from the local cost
table, which is exactly why every v5* SWE-bench smoke reported
``_cost_usd=0.0`` even when LiteLLM was metering cost correctly.

These tests pin the fix: the provider's own ``cost_usd`` (populated by
``LiteLLMProvider._finalize_response`` from LiteLLM's
``response._hidden_params["response_cost"]``) MUST win over the local
estimate, and the fallback path MUST still work for abstaining providers
and tests that don't bother setting a cost.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sage.phases.think import _extract_step_cost


def _fake_response(**usage: object) -> object:
    return SimpleNamespace(usage=dict(usage) if usage else None)


def test_provider_cost_used_when_present() -> None:
    response = _fake_response(cost_usd=0.0123, total_tokens=500)
    cost = _extract_step_cost(response, model_name="irrelevant-model", content="ignored")
    assert cost == pytest.approx(0.0123)


def test_provider_cost_takes_precedence_over_tokens() -> None:
    # A giant content + present cost must not inflate the bill.
    response = _fake_response(cost_usd=0.0001, total_tokens=100_000)
    cost = _extract_step_cost(response, "any-model", content="x" * 50_000)
    assert cost == pytest.approx(0.0001)


def test_zero_provider_cost_falls_back_to_estimate() -> None:
    # provider_cost == 0.0 is treated as "no cost reported" — otherwise a
    # free-tier model with a flaky meter would silently zero-out. The
    # fallback uses the local table and the content length.
    response = _fake_response(cost_usd=0.0, total_tokens=500)
    cost = _extract_step_cost(response, "gpt-5.4", content="hello world")
    assert cost > 0.0


def test_missing_usage_uses_content_estimate() -> None:
    response = _fake_response()
    cost = _extract_step_cost(response, "gpt-5.4", content="a" * 4_000)
    assert cost > 0.0


def test_missing_usage_and_unknown_model_returns_default_band() -> None:
    response = _fake_response()
    # Unknown model id falls through to DEFAULT_COST_PER_1K — still > 0 so
    # tests that used to silently floor to $0 now emit a visible number.
    cost = _extract_step_cost(response, "definitely-not-in-cost-table", content="x" * 2_000)
    assert cost > 0.0


# --- TopologyRunner aggregation (multi-agent path) -----------------------


@pytest.mark.asyncio
async def test_topology_runner_aggregates_per_node_cost() -> None:
    """TopologyRunner.total_cost_usd = sum of per-node AgentLoop costs.

    Before 2026-04-18 the per-node loops each held their own
    total_cost_usd but no one summed them up, so pipeline.ctx.cost read
    from system.agent_loop.total_cost_usd (never ran in multi-agent mode)
    and returned 0 for every multi-node bench.
    """
    from unittest.mock import AsyncMock, MagicMock

    from sage.topology.runner import TopologyRunner

    # Two nodes, each reports a different cost via its AgentLoop stub
    node_costs = [0.0045, 0.0125]
    loops = []
    for cost in node_costs:
        loop = MagicMock()
        loop.run = AsyncMock(return_value=f"output-cost-{cost}")
        loop.total_cost_usd = cost
        loop.tool_call_count = 0
        loop.tool_turn_count = 0
        loop.executed_commands = []
        loops.append(loop)

    call_count = [0]

    def _factory(**kwargs: object) -> MagicMock:
        loop = loops[call_count[0]]
        call_count[0] += 1
        return loop

    # Minimal topology graph: 2 LLM nodes, no predecessors
    graph = MagicMock()
    graph.node_count.return_value = 2
    nodes = []
    for i in range(2):
        n = MagicMock()
        n.role = "actor"
        n.model_id = ""
        n.prompt = ""
        n.node_type = "llm"
        n.required_capabilities = []
        nodes.append(n)
    graph.get_node = lambda idx: nodes[idx]
    graph.get_predecessors = lambda idx: []

    executor = MagicMock()
    ready_sequence = [[0], [1]]
    state = [0]

    def _next_ready(g: object) -> list[int]:
        if state[0] < len(ready_sequence):
            out = ready_sequence[state[0]]
            state[0] += 1
            return out
        return []

    executor.next_ready = _next_ready
    executor.is_done = lambda: state[0] >= len(ready_sequence)

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=_factory,
    )

    await runner.run("any task")

    expected = sum(node_costs)
    assert runner.total_cost_usd == pytest.approx(expected), (
        f"TopologyRunner aggregated {runner.total_cost_usd!r}; expected {expected!r}. "
        "Per-node costs were not summed — pipeline.ctx.cost will report 0 on any "
        "multi-agent bench run. Re-add the `self.total_cost_usd += ...` line in "
        "sage/topology/runner.py next to the tool_call_count aggregation."
    )
