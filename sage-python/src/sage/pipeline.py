"""CognitiveOrchestrationPipeline — 6-stage cognitive orchestration façade.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.

The 6 stage bodies + the orchestrator + the helper modules + the
`PipelineContext` dataclass live in `sage.pipeline_v2`. What remains
here:

  - public entry points (`run`, `run_with_frame`, `run_with_bench_evaluator`)
  - the `CognitiveOrchestrationPipeline.__init__` constructor
  - module-level helpers consumed by the orchestrator
    (`_new_runtime_run_id`, `_resolve_task_budget_usd`,
    `_is_strict_governance`, `BUDGET_EXCEEDED_RESULT`,
    `_BANDIT_ATTRIBUTION_REASON_CODES`)
  - `_run_internal` private façade method (subclass override seam)
    + `_emit` event-bus seam
  - helper instance methods that touch self-state and are retained
    until cycle-13 K Phase 2.2 Stage D
  - `PipelineContext` re-export from `sage.pipeline_v2.context`
    (`from sage.pipeline import PipelineContext` continues to resolve).

Cycle-13 K Phase 2.2 Stage C (cgpro `cgpro_phase22_test_rewrite_20260506`,
2026-05-06): the 6 `_stage_*` methods that previously lived on this
class were deleted. Stage entry points are now module functions in
`sage.pipeline_v2.<stage>`, called by the orchestrator with the
pipeline instance as first positional argument.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Awaitable, Callable, Literal, Mapping

from sage.events import (
    EXECUTE_BUDGET_EXCEEDED,  # noqa: F401 - re-exported for tests/test_pipeline_budget.py + pipeline_v2.memory_gate uses sage.events directly
    EXECUTE_HALTED_UNVERIFIED,  # noqa: F401 - imported by pipeline_v2.execute
    EXECUTE_UNVERIFIED,  # noqa: F401 - imported by pipeline_v2.execute
)

from sage.runtime.oracle import OracleConfig
from sage.runtime.run_frame import RunFrame, RunStatus

# OxiZ formal verification — imported lazily to allow graceful fallback.
# Annotated as `Any` so mypy does not infer the real Callable / type and
# then complain about the `None` sentinels in the ImportError branch.
verify_provider_assignment: Any = None
ProviderSpec: Any = None
_Z3_VERIFY_AVAILABLE = False
try:
    from sage.contracts import z3_verify as _z3_verify_mod
    verify_provider_assignment = _z3_verify_mod.verify_provider_assignment
    ProviderSpec = _z3_verify_mod.ProviderSpec
    _Z3_VERIFY_AVAILABLE = True
except ImportError:
    pass

log = logging.getLogger(__name__)

BUDGET_EXCEEDED_RESULT = "[sage: budget exceeded]"
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
BanditAttributionState = Literal["pending", "verified", "mismatch", "skipped"]
BanditAttributionReasonCode = Literal[
    "router_fallback_degraded",
    "model_mismatch",
    "template_mismatch",
    "multi_node_ambiguous",
    "decision_unknown",
    "recorder_instance_mismatch",
]
_BANDIT_ATTRIBUTION_REASON_CODES: tuple[BanditAttributionReasonCode, ...] = (
    "router_fallback_degraded",
    "model_mismatch",
    "template_mismatch",
    "multi_node_ambiguous",
    "decision_unknown",
    "recorder_instance_mismatch",
)

def _new_runtime_run_id() -> str:
    """Return a canonical 26-char uppercase ULID, with a local fallback."""
    try:
        import ulid

        return str(ulid.new()).upper()
    except Exception:  # noqa: BLE001 - tracing must not depend on ulid availability
        timestamp_ms = (time.time_ns() // 1_000_000) & ((1 << 48) - 1)
        value = (timestamp_ms << 80) | secrets.randbits(80)
        chars: list[str] = []
        for _ in range(26):
            chars.append(_ULID_ALPHABET[value & 0x1F])
            value >>= 5
        return "".join(reversed(chars))


def _is_strict_governance() -> bool:
    """Read the SAGE_STRICT_GOVERNANCE env var (A0b, 2026-04-23).

    When truthy, governance failures (write-gate init failure,
    verification-failed provider assignment) abort the pipeline
    instead of logging-and-continuing. Default off — the existing
    dev-friendly fail-open behaviour is preserved unless an operator
    explicitly opts in. Accepts ``1`` / ``true`` / ``yes`` / ``on``
    (case-insensitive) as truthy; everything else is off.
    """
    v = os.environ.get("SAGE_STRICT_GOVERNANCE", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _resolve_task_budget_usd(budget_usd: float | None) -> float:
    """Resolve task-level spend cap; 0 means unlimited."""
    raw_budget: float | str | None = budget_usd
    if raw_budget is None:
        env_budget = os.environ.get("SAGE_TASK_BUDGET_USD")
        if env_budget is None or not env_budget.strip():
            return 0.0
        raw_budget = env_budget
    try:
        return float(raw_budget)
    except (TypeError, ValueError):
        log.warning("Invalid SAGE_TASK_BUDGET_USD=%r; task budget disabled", raw_budget)
        return 0.0


# Cycle-13 K Phase 2.1 Step E1 (2026-05-06): the canonical
# PipelineContext dataclass body lives in
# `sage.pipeline_v2.context`. The re-export below preserves
# `from sage.pipeline import PipelineContext` for backward
# compatibility (cgpro round-3 Q4 backward-compat lock) AND
# `PipelineContext.__module__ == "sage.pipeline"` (cgpro round-3
# garde-fou: tests / bench / dashboards depend on this literal
# module-path).
from sage.pipeline_v2.context import PipelineContext  # noqa: E402


class CognitiveOrchestrationPipeline:
    """6-stage pipeline: Classify -> Decompose -> Select Topology -> Assign Models -> Execute -> Learn.

    Parameters
    ----------
    router : AdaptiveRouter
        For Stage 0 (classify).
    engine : TopologyEngine (Rust) or None
        For Stage 2 (select topology). If None, uses sequential template.
    assigner : ModelAssigner (Rust or Python)
        For Stage 3 (assign models per node).
    provider_pool : ProviderPool
        For Stage 4 (resolve model_id -> provider at execution).
    bandit : ContextualBandit or None
        For Stage 5 (learn from outcome).
    quality_estimator : QualityEstimator or None
        For Stage 5 (quality scoring).
    event_bus : EventBus or None
        For observability (emit events at each stage transition).
    llm_provider : LLMProvider
        Default provider for AgentLoop / TopologyRunner.
    llm_config : LLMConfig or None
        Default config.
    """

    # Class-level attribute declarations for transient runtime state.
    # Cycle-13 K Phase 2.1 Step D (2026-05-06): now that `_run_internal`
    # body lives in `pipeline_v2/orchestrator.py`, mypy needs these
    # declared on the class so the orchestrator's `pipeline.<attr> = X`
    # assignments are type-clean.
    _model_catalog: Any = None
    _last_routing_decision: Any = None
    _last_runtime_routing_source: str = "default"
    _last_runtime_routing_confidence: float | None = None
    _last_runtime_routing_model_id: str = ""
    last_context: "PipelineContext | None" = None

    def __init__(
        self,
        router: Any,
        engine: Any,
        assigner: Any,
        provider_pool: Any,
        bandit: Any = None,
        quality_estimator: Any = None,
        event_bus: Any = None,
        llm_provider: Any = None,
        llm_config: Any = None,
        prm: Any = None,
        controller: Any = None,
        smmu: Any = None,
        consolidator: Any = None,
        working_memory: Any = None,
        episodic_memory: Any = None,
        semantic_memory: Any = None,
        memory_agent: Any = None,
        causal_memory: Any = None,
        tool_forge: Any = None,
        tool_registry: Any = None,
        harness_config: Any = None,
        agent_loop: Any = None,
        budget_usd: float | None = None,
        oracle_config: OracleConfig | None = None,
        llm_tier: str = "",
    ) -> None:
        self.router = router
        self.engine = engine
        self.assigner = assigner
        self.provider_pool = provider_pool
        self.bandit = bandit
        self.quality_estimator = quality_estimator
        self.event_bus = event_bus
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        self.prm = prm
        self.controller = controller
        self.tool_registry = tool_registry
        self._rust_registry = None  # Set by boot if Rust ModelRegistry available
        self._rust_router = None    # Set by boot if Rust SystemRouter available
        self._smmu = smmu
        self.consolidator = consolidator
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        # T2 phase 0/1 (cgpro 2026-04-29): forward the other 3 memory
        # backends to per-node agent loops so write-gate skips can target
        # real backends instead of "memory_backend_unwired".
        self.semantic_memory = semantic_memory
        self.memory_agent = memory_agent
        self.causal_memory = causal_memory
        self.tool_forge = tool_forge
        self.harness_config = harness_config  # Meta-Harness: loaded from config/harness.json at boot
        self._harness_patcher = None
        if harness_config:
            try:
                from sage.meta_harness.patcher import HarnessPatcher
                self._harness_patcher = HarnessPatcher(harness_config)
                log.info("Meta-Harness config '%s' loaded: %s",
                         harness_config.id, harness_config.description)
            except ImportError:
                log.debug("meta_harness module not available, skipping harness config")
        self._agent_loop = agent_loop
        self._task_count = 0
        self.budget_usd = _resolve_task_budget_usd(budget_usd)
        self._llm_tier = llm_tier
        self._oracle_config = oracle_config or OracleConfig()

        # G-series audit fix (2026-04-19 docs/audits/2026-04-18-astropy-14995-*):
        # RustCompositeWriteGate was built, exported, but never called at
        # runtime (investigation confirmed 0 runtime call sites). Memory
        # writes in phases/act.py and _record_to_memory here all skipped
        # the 5-signal salience check.
        #
        # Weights: w_confidence=0.0 because AgentLoop has no per-turn
        # confidence signal — redistributing that 0.25 to novelty (+0.10)
        # and relevance (+0.15) keeps the composite summing to 1.0 and
        # leans on signals that ARE available (task text + content text).
        # Not a heuristic tweak: an honest statement that this engine cannot
        # produce the "confidence" input the research paper assumed.
        #
        # Gate is REBUILT per-task in `run()` (not reset in-place) so the
        # Rust class — which has no `reset_task()` method yet — doesn't need
        # an ABI bump. `_gate_config` holds the construction args; `write_gate`
        # is swapped out per task.
        self._gate_config = dict(
            threshold=0.35,
            w_confidence=0.0,
            w_novelty=0.40,
            w_reliability=0.20,
            w_recency=0.10,
            w_relevance=0.30,
        )
        from sage.pipeline_v2.memory_gate import build_write_gate
        self.write_gate = build_write_gate(self)

    # ── EventBus seam (Q3a lock: stays as stateful pipeline._emit hook) ────

    def _emit(self, stage: str, data: dict) -> None:  # type: ignore[type-arg]
        """Emit a PIPELINE event on EventBus if available.

        Body lives in `sage.pipeline_v2.runtime_events.emit`. The method
        form is preserved on the class because pipeline_v2/execute.py +
        carved-out runtime helpers and many tests depend on
        `pipeline._emit(stage, data)` being a stateful seam (Q3a lock).
        LOCAL import per cgpro DESIGN trap on circular-import risk.
        """
        from sage.pipeline_v2.runtime_events import emit as _v2_emit
        _v2_emit(self, stage, data)

    async def run(
        self,
        task: str,
        budget_usd: float | None = None,
        system_hint: int | None = None,
    ) -> str:
        """Execute the full 6-stage pipeline and return only the output string."""
        result, _frame = await self._run_internal(
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=os.environ.get("SAGE_RUN_FRAME") == "1",
        )
        return result

    async def run_with_frame(
        self,
        task: str,
        budget_usd: float | None = None,
        system_hint: int | None = None,
    ) -> tuple[str, RunFrame]:
        """Like run() but returns (output, frozen RunFrame).

        Signature mirrors run() so bench/traced adapters can use either
        entry point without parameter loss (cgpro 2026-04-29 cycle 4
        reassess R7.0.2).
        """
        return await self._run_internal(
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=os.environ.get("SAGE_RUN_FRAME") == "1",
        )

    async def run_with_bench_evaluator(
        self,
        task: str,
        evaluator: "Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]",
        *,
        budget_usd: float | None = None,
        system_hint: int | None = None,
    ) -> tuple[str, RunFrame]:
        """Run the pipeline with a synchronous-eval bench evaluator wired in.

        cgpro 2026-04-29 R6.1a verify Path E: synchronous-eval benches
        (BigCodeBench, EvalPlus, HumanEval) need their pass/fail to be
        available to the OracleStack BEFORE final_result + oracle_verdict
        + Stage 5 learning fire. Without this seam, those adapters call
        ``system.run()``, get the output, and only then evaluate — but by
        then the live oracle has already abstained because ``ctx.bench_result``
        was never populated.

        Locked event order::

            Stage 0-4 execute  →  evaluator(final_output)  →
            final_result  →  oracle_verdict  →  Stage 5 learn  →
            run_frame_summary

        The evaluator MUST return a Mapping with at least ``{"passed": bool}``;
        ``score``, ``reason``, ``output_sha256``, ``tool_call_id``,
        ``verifier_id`` are accepted by ``_exact_oracle``. If the evaluator
        raises or returns an invalid shape, ``ctx.bench_result`` stays None
        and the oracle abstains as if no evaluator were attached
        (fail-closed by design via ``_exact_oracle`` itself, which returns
        None on missing/malformed input).

        Sync and async evaluators are both supported: if ``evaluator(...)``
        returns an awaitable, it is awaited.
        """
        return await self._run_internal(
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=os.environ.get("SAGE_RUN_FRAME") == "1",
            bench_evaluator=evaluator,
        )

    async def _run_internal(
        self,
        task: str,
        budget_usd: float | None = None,
        system_hint: int | None = None,
        *,
        emit_run_frame_summary: bool = False,
        bench_evaluator: (
            "Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]] | None"
        ) = None,
    ) -> tuple[str, RunFrame]:
        """Execute the full 6-stage pipeline.

        Body lives in `sage.pipeline_v2.orchestrator.run_internal`. The
        method is preserved as a 1-line LOCAL-import wrapper so subclass
        overrides and any direct method patches keep working.
        """
        from sage.pipeline_v2.orchestrator import run_internal
        return await run_internal(
            self,
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=emit_run_frame_summary,
            bench_evaluator=bench_evaluator,
        )

