"""CognitiveOrchestrationPipeline — 6-stage cognitive orchestration façade.

Stage bodies, orchestrator, helpers, and `PipelineContext` live in
`sage.pipeline_v2`. This module retains: the public class
`CognitiveOrchestrationPipeline` with its constructor + public
entry points (`run`, `run_with_frame`, `run_with_bench_evaluator`)
+ `_run_internal` private façade + `_emit` EventBus seam; module-level
helpers consumed by the orchestrator (`_new_runtime_run_id`,
`_resolve_task_budget_usd`, `_is_strict_governance`,
`BUDGET_EXCEEDED_RESULT`, `_BANDIT_ATTRIBUTION_REASON_CODES`); and the
`PipelineContext` re-export so legacy `from sage.pipeline import
PipelineContext` keeps resolving.
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
from sage.runtime.run_frame import RunFrame

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


# PipelineContext dataclass body lives in `sage.pipeline_v2.context`.
# This re-export keeps `from sage.pipeline import PipelineContext`
# resolving and pins `PipelineContext.__module__ == "sage.pipeline"`
# (tests / bench / dashboards depend on the literal module-path).
from sage.pipeline_v2.context import PipelineContext  # noqa: E402


class CognitiveOrchestrationPipeline:
    """6-stage pipeline: classify -> decompose -> select_topology -> assign_models -> execute -> learn."""

    # Class-level attribute declarations for transient runtime state.
    # `_run_internal` body lives in `pipeline_v2/orchestrator.py`; mypy
    # needs these declared on the class so the orchestrator's
    # `pipeline.<attr> = X` assignments are type-clean.
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
        from sage.pipeline_v2.constructor import initialize_pipeline
        initialize_pipeline(
            self,
            router=router,
            engine=engine,
            assigner=assigner,
            provider_pool=provider_pool,
            bandit=bandit,
            quality_estimator=quality_estimator,
            event_bus=event_bus,
            llm_provider=llm_provider,
            llm_config=llm_config,
            prm=prm,
            controller=controller,
            smmu=smmu,
            consolidator=consolidator,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            memory_agent=memory_agent,
            causal_memory=causal_memory,
            tool_forge=tool_forge,
            tool_registry=tool_registry,
            harness_config=harness_config,
            agent_loop=agent_loop,
            budget_usd=budget_usd,
            oracle_config=oracle_config,
            llm_tier=llm_tier,
        )

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

        Synchronous-eval benches (BigCodeBench, EvalPlus, HumanEval) need
        their pass/fail available to the OracleStack BEFORE final_result
        + oracle_verdict + Stage 5 learning fire. Locked event order:
        execute → evaluator(final_output) → final_result → oracle_verdict
        → learn → run_frame_summary. The evaluator returns a Mapping with
        ``{"passed": bool}`` and optional ``score`` / ``reason`` / etc.
        Sync and async evaluators are both supported.
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

        Body lives in `sage.pipeline_v2.orchestrator.run_internal`; this
        method is the subclass-override seam that calls it.
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

