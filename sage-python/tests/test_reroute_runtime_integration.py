"""Deterministic REROUTE_REBUILD runtime integration test.

cgpro DESIGN_LOCKED 2026-05-12
FUTURE_BLOCK_ID=DETERMINISTIC_REROUTE_REBUILD_CONTROLLER_FIXTURE on
conv ``cgpro_i11_design_20260511``. Closes the empirical reachability
gap that the paid Option B canary exposed: cheap-tier production
canaries can't reach `execute.py:364` reroute branch because the
controller never decides reroute (budget tier disabled via Fix C,
fast tier agent gives up empty).

This test forces the branch entry deterministically by mocking only
the heavy execution/Rust boundaries:
- `sage.topology.runner.TopologyRunner` (fake — first call emits
  controller_decision + returns "__REROUTE__"; second call must be
  unreachable)
- `sage_core.TopologyExecutor` (fake constructor)
- `sage.pipeline_v2.select_topology.select_topology` (pass-through)
- `sage.pipeline_v2.assign_models.assign_models` (pass-through)

REAL code paths exercised end-to-end:
- `sage.pipeline_v2.execute.execute` — the function under test
- `sage.pipeline_v2.runtime_events.runtime_emit_provider_execution_witness`
- `sage.pipeline_v2.provider_policy.enforce_provider_policy`
  → `_maybe_emit_i11_assertion`
- `RuntimeEventLog.emit_*` + `validate_invariants()`

Sentinel regression contract (cgpro lock):
- Fake first runner returns EXACTLY ``"__REROUTE__"``.
- The branch entry is tested via ``pipeline._emit`` capture catching
  ``"REROUTE_REBUILD"`` (the pipeline-level stage signal).
- The controller_decision event uses ``action="reroute_topology"``
  (the controller-event schema's allowed value, NOT
  "REROUTE_REBUILD" which would coerce to "continue").
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from sage.pipeline_v2 import bandit_attribution as bandit_attr_mod
from sage.pipeline_v2 import execute as execute_mod
from sage.pipeline_v2 import assign_models as assign_models_mod
from sage.pipeline_v2 import select_topology as select_topology_mod
from sage.pipeline_v2.provider_policy import ProviderPolicyViolation
from sage.runtime.event_log import RuntimeEventLog


@pytest.fixture
def tmp_trace_dir() -> Path:
    path = Path(".tmp") / "pytest-reroute-integration" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _read_events(log_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_topology(node_count: int, template: str = "sequential") -> Any:
    """Fake topology supporting the attrs `execute()` reads."""
    nodes = [
        SimpleNamespace(
            role=f"node-{i}",
            model_id="",
            required_capabilities=(),
            node_type="",
        )
        for i in range(node_count)
    ]
    topo = SimpleNamespace()
    topo.id = f"topo-{node_count}-{template}"
    topo.template_type = template
    topo._nodes = nodes
    topo.get_node = lambda idx, _t=topo: _t._nodes[idx]
    topo.node_count = lambda _t=topo: len(_t._nodes)
    return topo


def _make_pipeline_for_reroute() -> Any:
    """Minimal pipeline that satisfies `execute()` reroute branch.

    Per cgpro Q2 lock: pass-through select_topology / assign_models,
    only mock heavy execution/Rust boundaries. `engine` must be
    truthy for the `result == "__REROUTE__" and self.engine` branch
    entry. `_llm_tier != "budget"` so Fix C doesn't disable the
    controller. Provider policy is active with `openai` denylisted —
    the routing candidate (`gpt-5.4-pro`) will resolve to openai and
    the witness will record `decision=blocked`.
    """
    pool = SimpleNamespace()
    pool.infer_provider = lambda mid: (
        "openai" if mid == "gpt-5.4-pro" else
        "deepseek" if mid == "deepseek-v4-pro" else ""
    )
    captured_emits: list[tuple[str, dict[str, Any]]] = []

    def _capture_emit(stage: str, data: dict[str, Any]) -> None:
        captured_emits.append((stage, data))

    pipeline = SimpleNamespace()
    pipeline.engine = object()  # truthy — satisfies reroute branch gate
    pipeline.controller = object()  # truthy — fake runner asserts this
    pipeline.assigner = None
    pipeline._agent_loop = None  # skip create_node_agent_loop factory
    pipeline.tool_registry = None
    pipeline._llm_tier = "fast"  # NOT "budget" — Fix C bypass
    pipeline.llm_provider = None
    pipeline.llm_config = None
    pipeline.provider_pool = pool
    pipeline.write_gate = None
    pipeline.episodic_memory = None
    pipeline.semantic_memory = None
    pipeline.memory_agent = None
    pipeline.causal_memory = None
    pipeline.event_bus = None
    pipeline.quality_estimator = None
    pipeline._provider_allowlist = ("deepseek",)
    pipeline._provider_denylist = ("openai",)
    pipeline._provider_policy_source = "cli"
    pipeline._last_runtime_routing_model_id = "gpt-5.4-pro"
    pipeline._emit = _capture_emit
    pipeline._captured_emits = captured_emits
    return pipeline


def _make_ctx_for_reroute(topology: Any) -> Any:
    ctx = SimpleNamespace()
    ctx.task = "deterministic reroute integration smoke"
    ctx.cost_tracker = None
    ctx.verification_passed = True
    ctx.topology = topology
    # The routing candidate is gpt-5.4-pro (openai, denied). After the
    # fake reroute, assign_models pass-through leaves the assignments
    # as-is, so per-node will violate the policy → enforce_provider
    # _policy emits failure + raises.
    ctx.assignments = {0: "gpt-5.4-pro", 1: "gpt-5.4-pro"}
    ctx.provider_hints = {}
    ctx.axis_hint = None
    ctx.system = 3
    ctx.domain = "code"
    ctx.budget = 5.0
    ctx.executed_model_ids = []
    ctx.executed_template = ""
    ctx.topology_id = topology.id
    ctx.bandit_template = ""
    ctx.dag_features = None
    ctx.confidence = 0.85
    ctx.routing_source = "rust_system_router"
    ctx.tool_call_count = 0
    ctx.tool_turn_count = 0
    ctx.executed_commands = []
    ctx.cost = 0.0
    ctx.result = None
    return ctx


def test_execute_reroute_rebuild_blocked_candidate_emits_i11_chain_and_blocks_dispatch(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro DESIGN_LOCKED 2026-05-12 deterministic fixture.

    Drives the REAL `execute()` reroute branch (`execute.py:364`)
    via minimal mocks of the heavy execution boundaries.

    Acceptance criteria (cgpro lock):
    1. controller_decision event with `action="reroute_topology"`
       emitted by the fake first runner before returning "__REROUTE__"
    2. execute.py enters real reroute branch (pipeline._emit captured
       event_type="REROUTE_REBUILD")
    3. provider_execution_witness phase=reroute, decision=blocked
    4. runtime_integrity_assertion I-11 phase=reroute verdict=pass
       (declared=blocked, verified=blocked, witness_seq link)
    5. failure(provider_policy_violation) with
       correlation_witness_seq → reroute witness seq
    6. NO node_started after the reroute witness AND second
       TopologyRunner construction/run never happens
    7. ProviderPolicyViolation raised (with FAIL_CLOSED disabled)
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_trace_dir))

    run_id = "01REROUTEDETERMINISTIC001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("reroute integration")
    log.emit_task_started("reroute integration")

    # Force multi-agent execution path (Fix C bypass + skip single-
    # agent branch) by patching is_single_agent_execution.
    monkeypatch.setattr(
        bandit_attr_mod,
        "is_single_agent_execution",
        lambda pipeline, ctx: False,
    )

    # ── Mock heavy boundaries (cgpro Q2=B lock) ─────────────────────

    # Fake first TopologyRunner: emits controller_decision +
    # returns "__REROUTE__". Stores reference so we can prove it
    # was the first instance and no second was constructed.
    construction_count = {"n": 0}

    class _FakeFirstRunner:
        def __init__(self, **kwargs: Any) -> None:
            construction_count["n"] += 1
            # cgpro lock: assert controller is non-None on first run
            assert kwargs.get("controller") is not None, (
                "first TopologyRunner MUST receive a non-None controller "
                "— Fix C should NOT have disabled it for _llm_tier=fast"
            )
            self._event_log = kwargs.get("event_log")
            self.tool_call_count = 0
            self.tool_turn_count = 0
            self.executed_commands: list[str] = []
            self.total_cost_usd = 0.0

        async def run(self, task: str) -> str:
            # Emit a real controller_decision event before returning
            # the reroute sentinel — proves the controller drove the
            # branch entry (per cgpro Q3 §1).
            if self._event_log is not None:
                self._event_log.emit_controller_decision(
                    node_id="0",
                    action="reroute_topology",  # NOT "REROUTE_REBUILD"
                    reason_code="fixture_forced_reroute",
                    quality_source="abstain",
                    threshold_band="critical",
                )
            return "__REROUTE__"  # EXACT sentinel — cgpro regression contract

    class _UnreachableSecondRunner:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError(
                "Second TopologyRunner construction MUST be unreachable — "
                "ProviderPolicyViolation should abort enforce_provider_policy "
                "before runner2 is built (cgpro acceptance #6)"
            )

        async def run(self, task: str) -> str:
            raise AssertionError("Second TopologyRunner.run unreachable")

    runner_box = {"call_count": 0}

    def _runner_factory(**kwargs: Any) -> Any:
        runner_box["call_count"] += 1
        if runner_box["call_count"] == 1:
            return _FakeFirstRunner(**kwargs)
        return _UnreachableSecondRunner(**kwargs)

    # Patch where execute.py imports TopologyRunner (local import at
    # line 286). The patch must hit the source module since execute
    # does `from sage.topology.runner import TopologyRunner` at call
    # time.
    import sage.topology.runner as _runner_module
    monkeypatch.setattr(_runner_module, "TopologyRunner", _runner_factory)

    # Patch sage_core.TopologyExecutor — execute() imports it at line
    # 290 AND line 388 (reroute path). Make it a no-op stub.
    class _FakeTopologyExecutor:
        def __init__(self, topo: Any) -> None:
            self._topo = topo

    import sage_core as _sage_core
    monkeypatch.setattr(
        _sage_core,
        "TopologyExecutor",
        _FakeTopologyExecutor,
    )

    # Pass-through select_topology + assign_models (no Rust engine
    # needed; preserve ctx state).
    monkeypatch.setattr(
        select_topology_mod,
        "select_topology",
        lambda pipeline, ctx: ctx,
    )
    monkeypatch.setattr(
        assign_models_mod,
        "assign_models",
        lambda pipeline, ctx: ctx,
    )

    # ── Build the pipeline + ctx ────────────────────────────────────

    pipeline = _make_pipeline_for_reroute()
    topology = _make_topology(node_count=2)
    ctx = _make_ctx_for_reroute(topology)

    # ── Drive real execute() ────────────────────────────────────────

    with pytest.raises(ProviderPolicyViolation):
        asyncio.run(execute_mod.execute(pipeline, ctx, event_log=log))

    log.emit_final_result(
        status="failure",
        output="",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )
    log.close()

    # ── Assertions per cgpro lock §Q3 ───────────────────────────────

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    event_types = [e["event_type"] for e in events]

    # (1) controller_decision with action="reroute_topology"
    controller_events = [e for e in events if e["event_type"] == "controller_decision"]
    assert len(controller_events) == 1, (
        f"exactly one controller_decision expected; got {len(controller_events)}"
    )
    cd_action = (controller_events[0].get("payload") or {}).get("action") \
        or controller_events[0].get("action")
    assert cd_action == "reroute_topology", (
        f"controller_decision.action MUST be 'reroute_topology' "
        f"(not coerced to 'continue'); got {cd_action!r}"
    )

    # (2) pipeline._emit captured "REROUTE_REBUILD" stage signal
    reroute_emits = [
        (stage, data) for stage, data in pipeline._captured_emits
        if stage == "REROUTE_REBUILD"
    ]
    assert len(reroute_emits) == 1, (
        f"expected pipeline._emit('REROUTE_REBUILD', ...) once; "
        f"got {len(reroute_emits)} (all emits: {pipeline._captured_emits})"
    )

    # (3) reroute witness emitted
    witnesses = [
        e for e in events
        if e["event_type"] == "provider_execution_witness"
        and e["payload"]["assignment_phase"] == "reroute"
    ]
    assert len(witnesses) == 1, (
        f"exactly one reroute witness expected; got {len(witnesses)}"
    )
    reroute_witness = witnesses[0]
    pol = reroute_witness["payload"]["policy"]
    assert pol["active"] is True
    assert pol["routing_candidate_decision"] == "blocked"

    # (4) runtime_integrity_assertion phase=reroute verdict=pass
    assertions = [
        e for e in events
        if e["event_type"] == "runtime_integrity_assertion"
        and e["payload"]["phase"] == "reroute"
    ]
    assert len(assertions) == 1
    a_payload = assertions[0]["payload"]
    assert a_payload["invariant_id"] == "I-11"
    assert a_payload["verdict"] == "pass"
    assert a_payload["declared_decision"] == "blocked"
    assert a_payload["verified_decision"] == "blocked"
    assert a_payload["witness_seq"] == reroute_witness["seq"]

    # (5) failure with correlation_witness_seq → reroute witness
    failures = [
        e for e in events
        if e["event_type"] == "failure"
        and e.get("error_type") == "provider_policy_violation"
    ]
    assert len(failures) == 1
    f_payload = failures[0].get("payload", {})
    assert f_payload.get("correlation_witness_seq") == reroute_witness["seq"], (
        "failure event correlation_witness_seq MUST link to the reroute "
        "witness seq — cgpro DESIGN_LOCKED Q3 §5"
    )

    # (6) No node_started anywhere + second runner unreachable
    assert "node_started" not in event_types, (
        "blocked reroute candidate MUST NOT reach node_started — "
        "provider dispatch was supposed to be blocked"
    )
    assert runner_box["call_count"] == 1, (
        f"TopologyRunner MUST have been constructed exactly once; "
        f"got {runner_box['call_count']} constructions (second is unreachable)"
    )

    # (7) Already proven by pytest.raises(ProviderPolicyViolation) above

    # Bonus: close-time audit accepts the trace (witness paired with
    # failure via correlation; no orphaned blocked witness)
    violations = log.validate_invariants()
    assert violations == [], (
        f"close-time audit MUST accept the reroute witness ↔ failure "
        f"pairing; got violations: {violations}"
    )
