"""F7 wiring: FrugalGPT cascade upgrade forwards `task_system` to the
Rust ModelAssigner via `assign_single_node`.

Pre-fix path: cascade picked the next-best per-node-tier model only,
ignoring overall task complexity. On an S3 SWE-bench task with a coder
node at template tier S2, the upgrade-after-low-quality cascade would
re-pick within S2 candidates instead of promoting to S3 — the very
escalation that motivated the cascade.

Paired with the Rust-side tests in sage-core/src/routing/model_assigner.rs
(test_effective_system_*) and the batch-call test in
test_f7_task_system_forward.py.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from unittest.mock import MagicMock

from sage.topology_controller import TopologyController


class _SingleNodeSpy:
    """Captures kwargs forwarded to assigner.assign_single_node."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self.call_count = 0

    def assign_single_node(
        self,
        graph: Any,
        node_idx: int,
        task_domain: str,
        budget_usd: float,
        exclude_model_ids: list[str] | None = None,
        task_system: int | None = None,
    ) -> str:
        self.call_count += 1
        self.last_kwargs = {
            "node_idx": node_idx,
            "task_domain": task_domain,
            "budget_usd": budget_usd,
            "exclude_model_ids": exclude_model_ids,
            "task_system": task_system,
        }
        return "stub-model-id"


def _mk_topology(node_idx_count: int = 1) -> Any:
    topology = MagicMock()
    topology.set_node_model_id = MagicMock()
    node = MagicMock()
    node.model_id = "old-model"
    node.max_cost_usd = 1.0
    node.max_retries = 3
    topology.get_node = MagicMock(return_value=node)
    topology.node_count = MagicMock(return_value=node_idx_count)
    return topology


def _mk_ctx(system: int | None) -> Any:
    # SimpleNamespace, not MagicMock: TopologyController._ctx_value tries
    # `ctx.get(key, default)` first, and MagicMock auto-supplies a .get
    # that returns Mock objects. SimpleNamespace lacks .get, so the
    # helper falls through to getattr(ctx, key) and reads real ints.
    return SimpleNamespace(
        system=system,
        budget_usd=5.0,
        budget=5.0,
        domain="code",
        task="fix bug",
    )


def test_cascade_forwards_task_system_when_ctx_s3():
    """ctx.system=3 must reach assign_single_node(task_system=3) so the
    Rust effective_system() can floor producer nodes at S2 (or S3 for
    math/formal — handled by the domain-aware floor).
    """
    spy = _SingleNodeSpy()
    controller = TopologyController(assigner=spy)

    topology = _mk_topology()
    ctx = _mk_ctx(system=3)

    result = controller._resolve_upgrade_model(0, "fix bug", topology, ctx)
    assert result == "stub-model-id"
    assert spy.call_count == 1
    assert spy.last_kwargs["task_system"] == 3
    # Verify excluded_model_ids was forwarded too — not lost in the rewrite.
    assert spy.last_kwargs["exclude_model_ids"] == ["old-model"]


def test_cascade_forwards_task_system_when_ctx_s2():
    spy = _SingleNodeSpy()
    controller = TopologyController(assigner=spy)
    result = controller._resolve_upgrade_model(0, "x", _mk_topology(), _mk_ctx(system=2))
    assert result == "stub-model-id"
    assert spy.last_kwargs["task_system"] == 2


def test_cascade_forwards_none_when_ctx_system_unset():
    """ctx.system=0 (default unset) is out of (1,2,3) → forward None,
    matching the batch-call contract enforced in test_f7_task_system_forward.
    """
    spy = _SingleNodeSpy()
    controller = TopologyController(assigner=spy)
    result = controller._resolve_upgrade_model(0, "x", _mk_topology(), _mk_ctx(system=0))
    assert result == "stub-model-id"
    assert spy.last_kwargs["task_system"] is None


def test_cascade_falls_back_when_assigner_lacks_task_system():
    """Older Rust .pyd or pure-Python fallback may not accept task_system.
    The TypeError fallback must retry without it (degraded but functional).
    """

    class _OldSpy:
        def __init__(self):
            self.calls = []

        def assign_single_node(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            if "task_system" in kwargs:
                raise TypeError("unexpected keyword 'task_system'")
            return "fallback-id"

    spy = _OldSpy()
    controller = TopologyController(assigner=spy)
    result = controller._resolve_upgrade_model(0, "x", _mk_topology(), _mk_ctx(system=3))
    assert result == "fallback-id"
    assert len(spy.calls) == 2
    # First call attempted task_system; second dropped it.
    assert "task_system" in spy.calls[0][1]
    assert "task_system" not in spy.calls[1][1]
