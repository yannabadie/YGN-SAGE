"""P9 cycle-11 test #1 (ADR-015 acceptance gate): pipeline.run() byte-identical.

Locks ADR-015 §"Required golden tests" item #1 — the **flagship**
characterization test for the cycle-12 phase 2 move of ``pipeline.py``
into ``pipeline_v2/``. After cycle-12 ships ``pipeline_v2/__init__.py``
as the thin façade rebuilding ``CognitiveOrchestrationPipeline``, this
test must pass byte-identically against the new implementation —
proving the public contract surface (``run()`` final string + ctx
fields + event ledger sequence + bandit decision_id) is preserved.

Cycle-11 phase 1 scope (this file)
==================================
**One fixture only**: a single S1-tier task running through the
single-agent bypass path. This is the simplest end-to-end execution
because:

- Stage 0: kNN/SystemRouter routes via the stub, sets system=1.
- Stage 1: ``_stage_decompose`` short-circuits to trivial
  ``DAGFeatures(omega=1, delta=1, gamma=0.0)`` without LLM call.
- Stage 2: ``_stage_select_topology`` S1 fast path
  (``ctx.topology=None``) — no Rust engine, no template lookup.
- Stage 3: ``_stage_assign_models`` is skipped (no topology).
- Stage 4: bypass branch via ``llm_provider.generate()`` — single
  call, deterministic stub.
- Stage 5: bandit recorder fires on a settled decision_id.

Cycle-12 phase 2 will extend this file with additional fixtures
(multi-agent S2 sequential, S2 parallel cascade, S3 reasoner) as
risk during the move warrants. Each fixture lives behind its own
test function with explicit golden literals.

Why "byte-identical" is interpretable, not literal
==================================================
Across the public surface, byte-identical means:

- ``result``: string-equal. The user-visible final output must be
  the same character sequence for identical inputs + stubs.
- ``ctx`` **semantic** fields: dataclass fields listed below have
  specific literal expected values for fields that are deterministic
  given the stubs (``system``, ``domain``, ``topology_id``,
  ``bandit_template`` / ``executed_template``, ``bandit_decision_id``,
  ``executed_model_id``, ``bandit_attribution_state``,
  ``dag_features.{omega,delta,gamma}``).
- ``ctx.cost`` / ``ctx.latency_ms``: scoped to "non-negative finite",
  NOT a specific numeric value. They come from ``time.monotonic()``
  differences and stub interactions, so a hardware-variance run
  would not produce byte-identical floats. Per cgpro VERIFY follow-up
  2026-05-05: the "byte-identical" claim applies to semantic fields
  + result + event sequence + decision_id, NOT timing/cost numbers.
  If cycle-12 phase 2 wants byte-identical timing, it would need to
  freeze ``time.monotonic()`` — out of scope for this test.
- Event ledger: ``event_type`` list AND order. New event types
  added during cycle-12 are an ADR-015 contract change requiring
  this test to be updated alongside the change.
- Bandit decision_id: present in ``ctx.bandit_decision_id`` and
  consumed by Stage 5 via ``record_outcome_checked``.

Internal-only state (cache shapes, internal counters,
``_last_runtime_routing_*`` debug strings) is **not** locked here.
The cycle-12 decomposition is allowed to refactor those freely as
long as the public surface above stays stable.

Stub determinism
================
All time-dependent code paths are pinned:

- Provider responses are pre-built ``LLMResponse`` objects.
- Quality estimator returns a fixed 0.8.
- Rust router stub returns a fixed Decision with
  ``decision_id="d-byte-id-test-001"``.
- ``SAGE_ORACLE=0`` → quality_estimator path (no oracle dependency).
- ``SAGE_TRACE_JSONL_DIR`` set so events land in tmp_path; the
  test reads them back and asserts the type sequence.

If a future commit makes any of these less deterministic, this
test will start flaking — the right fix is to repair the
non-determinism at the source, not to soften the assertion.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sage.llm.base import LLMConfig, LLMResponse
from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline


# Fixtures: structurally identical to test_pipeline_bandit_causality.py.
# Inlined here (not imported) because pytest fixtures are scope-local
# and importing across test files couples this characterization gate
# to the bandit-causality test's stability — they must stay
# independently maintained.


@dataclass
class _Decision:
    decision_id: str = "d-byte-id-test-001"
    model_id: str = "stub-model"
    template: str = "single_agent"
    context: list[float] | None = None


class _Router:
    def assess_complexity(self, task: str) -> Any:
        return SimpleNamespace(system=1)

    def route(self, profile: Any) -> Any:
        return SimpleNamespace(system=profile.system)


class _Provider:
    def __init__(self, name: str = "stub-provider", content: str = "stub output") -> None:
        self.name = name
        self.model_id = ""
        self.calls: list[LLMConfig | None] = []
        self._content = content

    async def generate(
        self, messages: Any, config: LLMConfig | None = None, **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(config)
        return LLMResponse(content=self._content, model=config.model if config else None)


class _ProviderPool:
    def __init__(self, provider: _Provider, model_id: str = "stub-model") -> None:
        self.provider = provider
        self.model_id = model_id
        self.resolved: list[str] = []

    def is_model_available(self, model_id: str) -> bool:
        return model_id == self.model_id

    def infer_provider(self, model_id: str) -> str:
        return "stub"

    def resolve(self, model_id: str) -> tuple[_Provider, LLMConfig]:
        self.resolved.append(model_id)
        self.provider.model_id = model_id
        return self.provider, LLMConfig(provider="stub", model=model_id)


class _FakeRoutingConstraints:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _RustRouter:
    """Stub SystemRouter with deterministic decision + verifying recorder."""

    def __init__(self) -> None:
        self.decision = _Decision()
        self.pending_decisions: set[str] = {self.decision.decision_id}
        self.integrated_calls: list[tuple[str, Any, str]] = []
        self.checked_records: list[tuple[str, str, str, float, float, float]] = []
        self.cancelled_decisions: list[str] = []

    def route_integrated(self, task: str, constraints: Any, topology_id: str) -> Any:
        self.integrated_calls.append((task, constraints, topology_id))
        self.pending_decisions.add(self.decision.decision_id)
        return SimpleNamespace(
            decision_id=self.decision.decision_id,
            system=1,
            model_id=self.decision.model_id,
            selected_template=self.decision.template,
            template=self.decision.template,
            confidence=0.91,
            estimated_cost=0.001,
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
            (decision_id, executed_model_id, executed_template,
             quality, cost, latency_ms)
        )
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


def _install_fake_sage_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sage_core",
        SimpleNamespace(RoutingConstraints=_FakeRoutingConstraints),
    )


def _make_pipeline(*, provider_pool: _ProviderPool) -> Pipeline:
    """Construct Pipeline via real ``__init__`` (not __new__).

    The byte-identical test must exercise the real construction path
    because cycle-12 phase 2 may refactor the constructor signature
    via the thin façade in ``pipeline_v2/__init__.py``. If __init__
    semantics drift (e.g. a new required arg appears), this test
    will catch it at construction time.
    """
    return Pipeline(
        router=_Router(),
        engine=None,
        assigner=None,
        provider_pool=provider_pool,
        bandit=None,
        quality_estimator=_QualityEstimator(0.8),
        event_bus=None,
        llm_provider=_Provider("default-provider", "default output"),
        llm_config=LLMConfig(provider="default", model="default-model"),
    )


def _read_event_types(trace_dir: Path) -> list[str]:
    """Read RuntimeEventLog .jsonl files and return event_type list in order."""
    events: list[str] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        if path.name == "learning_side_effects.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line)["event_type"])
    return events


# ─────────────────────────────────────────────────────────────────
# Test 1: full result + ctx field surface locked
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_byte_identical_s1_bypass_locks_result_and_ctx_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run a single S1 task end-to-end and lock the full contract surface.

    Asserts the literal expected values for every ADR-015 #1 contract
    field. Any drift in cycle-12 phase 2 where the field-set
    semantically changes will fail this test and force the change to
    be reflected as an ADR-015 contract update.
    """
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    _install_fake_sage_core(monkeypatch)

    bandit_provider = _Provider("bandit-provider", "bandit output")
    provider_pool = _ProviderPool(bandit_provider)
    rust_router = _RustRouter()
    pipeline = _make_pipeline(provider_pool=provider_pool)
    pipeline._rust_router = rust_router

    result = await pipeline.run("characterization S1 task", budget_usd=3.0)

    ctx = pipeline.last_context

    # 1. Final result string-equal.
    assert result == "bandit output", (
        f"Final result drift: got {result!r}, expected 'bandit output'. "
        f"Cycle-12 phase 2 must preserve the user-visible output "
        f"byte-identically."
    )
    assert ctx is not None

    # 2. PipelineContext contract fields. Per ADR-015 #1 fixture list:
    #    system / domain / topology_id / selected_template / executed_template /
    #    node_count / dag_features / assignments / cost / latency_ms.
    assert ctx.system == 1, f"ctx.system drift: {ctx.system!r}"
    # _infer_domain reads the task text — "characterization S1 task"
    # has no code keyword, so it lands on "general". Locking the
    # actual deterministic output, NOT what we'd want — cycle-12
    # phase 2 must preserve _infer_domain's behavior byte-identically.
    assert ctx.domain == "general", f"ctx.domain drift: {ctx.domain!r}"
    assert ctx.topology_id == "", (
        f"ctx.topology_id drift: {ctx.topology_id!r}. S1 bypass path "
        f"sets topology=None and never assigns topology_id."
    )
    assert ctx.bandit_template == "single_agent", (
        f"ctx.bandit_template (selected_template) drift: "
        f"{ctx.bandit_template!r}"
    )
    assert ctx.executed_template == "single_agent"
    assert ctx.topology is None, "S1 bypass must leave ctx.topology=None"
    assert ctx.assignments == {}, (
        f"ctx.assignments drift: {ctx.assignments!r}. Bypass path "
        f"never populates assignments."
    )
    assert ctx.dag_features is not None, "Stage 1 must populate dag_features"
    assert ctx.dag_features.omega == 1
    assert ctx.dag_features.delta == 1
    assert ctx.dag_features.gamma == 0.0

    # 3. Bandit decision_id from Stage 0 → recorded in Stage 5.
    assert ctx.bandit_decision_id == "d-byte-id-test-001"
    assert ctx.bandit_model_id == "stub-model"
    assert ctx.executed_model_id == "stub-model"

    # 4. Bandit attribution lifecycle: ONE record, ZERO cancellations.
    assert len(rust_router.checked_records) == 1, (
        f"Expected exactly 1 record_outcome_checked call, got "
        f"{len(rust_router.checked_records)}."
    )
    decision_id, model_id, template, quality, cost, latency = (
        rust_router.checked_records[0]
    )
    assert decision_id == "d-byte-id-test-001"
    assert model_id == "stub-model"
    assert template == "single_agent"
    assert quality == pytest.approx(0.8)
    assert rust_router.cancelled_decisions == [], (
        f"Expected zero cancellations on the happy path, got "
        f"{rust_router.cancelled_decisions!r}."
    )

    # 5. Attribution state transitioned to 'verified' (recorder success).
    assert ctx.bandit_attribution_state == "verified"

    # 6. Telemetry numbers exist (deterministic stubs report cost/latency
    # via record_outcome_checked args). They're floats; we don't assert
    # exact values because they come from time.monotonic() differences
    # and provider stubs. The contract is "non-negative finite", not
    # "specific number".
    assert isinstance(ctx.cost, float) and ctx.cost >= 0.0
    assert isinstance(ctx.latency_ms, float) and ctx.latency_ms >= 0.0


# ─────────────────────────────────────────────────────────────────
# Test 2: event ledger sequence locked
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_byte_identical_s1_bypass_locks_event_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lock the event ledger ``event_type`` sequence for an S1 bypass run.

    Per ADR-015 #1: "Event ledger sequence (event_type list +
    ordering)". Cycle-12 phase 2 must preserve the order. Adding new
    event types is an ADR contract change requiring this test to be
    updated alongside the change — that's intentional, the goal is
    to make event-emission changes visible in code review.

    The expected sequence for an S1 bypass run with SAGE_ORACLE=0:

      task_started
      routing_decision
      final_result

    No topology_selected / model_assigned (bypass has neither).
    No oracle_verdict (oracle off). No run_frame_summary (
    SAGE_RUN_FRAME unset).
    """
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)
    _install_fake_sage_core(monkeypatch)

    bandit_provider = _Provider("bandit-provider", "bandit output")
    provider_pool = _ProviderPool(bandit_provider)
    rust_router = _RustRouter()
    pipeline = _make_pipeline(provider_pool=provider_pool)
    pipeline._rust_router = rust_router

    await pipeline.run("characterization S1 task", budget_usd=3.0)

    event_types = _read_event_types(tmp_path)
    # Slice 10D Route A (2026-05-11): provider_execution_witness is
    # emitted by the orchestrator AFTER model_assigned and BEFORE
    # enforce_provider_policy — applies to both bypass and multi-
    # agent paths. ADR-015 contract change captured here.
    expected = [
        "task_started",
        "routing_decision",
        "provider_execution_witness",
        "final_result",
    ]
    assert event_types == expected, (
        f"Event ledger sequence drift: got {event_types!r}, expected "
        f"{expected!r}. Cycle-12 phase 2 must preserve ordering. "
        f"Adding a new event is an ADR-015 contract change — update "
        f"this test alongside the change."
    )


# ─────────────────────────────────────────────────────────────────
# Test 3: determinism — two runs produce identical contract surface
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_byte_identical_two_runs_same_inputs_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two independent runs with identical inputs produce identical contract.

    This is the **byte-identical** litmus: same task, same stubs,
    same env → same final result + same ctx fields. If a non-
    determinism source slips in (random init, time-based hash,
    unset env defaulting differently), this test will catch it on
    the second run.

    Without this, a test that checks "run 1 produces expected
    values" but happens to be non-deterministic would pass on luck.
    Two runs is the minimum to surface the issue.
    """
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path / "run1"))
    monkeypatch.delenv("SAGE_RUN_FRAME", raising=False)
    _install_fake_sage_core(monkeypatch)

    def _build_pipeline(trace_subdir: Path) -> tuple[Pipeline, _RustRouter]:
        bandit_provider = _Provider("bandit-provider", "bandit output")
        provider_pool = _ProviderPool(bandit_provider)
        rust_router = _RustRouter()
        pipeline = _make_pipeline(provider_pool=provider_pool)
        pipeline._rust_router = rust_router
        # Re-pin trace dir per run.
        monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(trace_subdir))
        trace_subdir.mkdir(parents=True, exist_ok=True)
        return pipeline, rust_router

    # Run 1
    p1, r1 = _build_pipeline(tmp_path / "run1")
    result1 = await p1.run("characterization S1 task", budget_usd=3.0)
    ctx1 = p1.last_context

    # Run 2
    p2, r2 = _build_pipeline(tmp_path / "run2")
    result2 = await p2.run("characterization S1 task", budget_usd=3.0)
    ctx2 = p2.last_context

    # Same final string.
    assert result1 == result2, (
        f"Two runs produced different results: {result1!r} vs "
        f"{result2!r}. A non-deterministic input slipped in; the "
        f"byte-identical contract requires identical inputs to "
        f"produce identical outputs."
    )
    assert ctx1 is not None and ctx2 is not None

    # Same contract fields.
    assert ctx1.system == ctx2.system
    assert ctx1.domain == ctx2.domain
    assert ctx1.bandit_template == ctx2.bandit_template
    assert ctx1.executed_template == ctx2.executed_template
    assert ctx1.bandit_decision_id == ctx2.bandit_decision_id
    assert ctx1.executed_model_id == ctx2.executed_model_id
    assert ctx1.bandit_attribution_state == ctx2.bandit_attribution_state

    # Same dag_features shape.
    assert ctx1.dag_features is not None
    assert ctx2.dag_features is not None
    assert (ctx1.dag_features.omega, ctx1.dag_features.delta,
            ctx1.dag_features.gamma) == (ctx2.dag_features.omega,
                                         ctx2.dag_features.delta,
                                         ctx2.dag_features.gamma)

    # Same bandit lifecycle: one record, no cancel, on each run.
    assert len(r1.checked_records) == 1
    assert len(r2.checked_records) == 1
    assert r1.cancelled_decisions == r2.cancelled_decisions == []
