"""Cycle-13 K Phase 2.2 RED/GREEN deletion contract tests.

Per cgpro `cgpro_phase22_test_rewrite_20260506` DESIGN_LOCK
(2026-05-06):

> "Add contract tests before deletion: [...] Do not make these pass
>  yet. They are the red/green deletion contract."

Stage A (this commit): every contract is marked
``pytest.mark.xfail(strict=True)``. The seams are still present on
``CognitiveOrchestrationPipeline``; the contracts therefore *must
fail*. ``strict=True`` ensures the suite breaks if a contract
accidentally passes early (which would mean the seam disappeared
before Stage B's test rewrites had a chance to retarget it).

Stage C (orchestrator rewrite + 6 ``_stage_*`` deletion): the
``xfail`` markers on the stage contracts are removed. Stage D
(helper purge): the markers on helper contracts are removed in the
same batch as the corresponding deletion.

Trap 5 from the DESIGN_LOCK: the assertions check **class-level**
absence (``hasattr(CognitiveOrchestrationPipeline, ...)``), not
instance-dict absence. Instance-level absence alone misses inherited
or class-attribute seams.

Reference: ``project_phase22_design_lock`` memory file +
``.tmp/cgpro_phase22_design_lock_finaltext.md`` for full lock text.
"""
from __future__ import annotations

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline


# ────────────────────────────────────────────────────────────────────
# 1. Stage methods to delete in Stage C (atomic with orchestrator
#    rewrite). Six methods, each is a pure 1-line LOCAL-import
#    delegator to a callable in ``sage.pipeline_v2.<stage>``.
# ────────────────────────────────────────────────────────────────────

STAGE_METHODS_TO_DELETE: tuple[str, ...] = (
    "_stage_classify",
    "_stage_decompose",
    "_stage_select_topology",
    "_stage_assign_models",
    "_stage_execute",
    "_stage_learn",
)


@pytest.mark.parametrize("method_name", STAGE_METHODS_TO_DELETE)
def test_stage_method_deleted_from_class(method_name: str) -> None:
    """Stage method must be absent at class level after Stage C."""
    assert not hasattr(CognitiveOrchestrationPipeline, method_name), (
        f"Phase 2.2 Stage C contract: "
        f"CognitiveOrchestrationPipeline.{method_name} should be deleted. "
        f"All callsites in pipeline_v2/orchestrator.py and the 27 test "
        f"files in scope must call sage.pipeline_v2.<stage>.<fn>(pipeline, ctx) "
        f"directly (Pattern A) or monkeypatch the module function "
        f"(Pattern B). Per cgpro DESIGN_LOCK 2026-05-06, no instance-level "
        f"compatibility shim is permitted."
    )


# ────────────────────────────────────────────────────────────────────
# 2. Helper delegator methods to purge in Stage D. Cgpro Q3 purge
#    list — 36 helpers (3 memory_gate + 13 runtime_events excl _emit
#    + 10 topology_helpers + 2 costing + 2 model assignment + 5 bandit
#    attribution + 1 provider fallback). ``_emit`` and ``_run_internal``
#    are NOT in this list — they survive Phase 2.2 by explicit lock
#    (Q3a + Trap 6). Total deletion candidates Stage C + D = 6 + 36 = 42,
#    matching the xfailed count.
# ────────────────────────────────────────────────────────────────────

HELPER_METHODS_TO_DELETE: tuple[str, ...] = (
    # memory_gate (3)
    "_emit_budget_exceeded",
    "_build_write_gate",
    "_record_to_memory",
    # runtime_events (13 — _emit excluded per Q3a)
    "_emit_bandit_attribution_mismatch",
    "_bandit_reason_from_exception",
    "_runtime_node_count",
    "_runtime_edge_type",
    "_runtime_edge_summary",
    "_runtime_node_summary",
    "_runtime_provider_id_for_model",
    "_runtime_node_capabilities",
    "_runtime_graph_digest",
    "_runtime_emit_topology_selected",
    "_runtime_emit_model_assigned",
    "_runtime_final_status",
    "_runtime_final_node_count",
    # topology_helpers (10)
    "_build_topology_from_hint",
    "_topology_candidate_items",
    "_log_topology_candidates",
    "_candidate_text_attr",
    "_candidate_float_attr",
    "_candidate_node_count",
    "_log_topology_structure",
    "_apply_topology_budget_and_cache",
    "_make_single_node_topology",
    "_load_model_catalog",
    # costing (2)
    "_check_topology_budget",
    "_estimate_topology_cost",
    # model assignment (2)
    "_log_model_assigner_chosen_fallback",
    "_verify_assignment_formal",
    # bandit attribution (5)
    "_bandit_task_context",
    "_is_single_agent_execution",
    "_clear_bandit_decision",
    "_cancel_bandit_decision",
    "_record_bandit_outcome_checked",
    # provider fallback (1)
    "_pick_fallback_provider",
)


@pytest.mark.parametrize("method_name", HELPER_METHODS_TO_DELETE)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2.2 Stage A — helper delegators still present. "
        "Contracts go green at Stage D, after pipeline_v2 internal "
        "self._foo callsites are rewritten (Q7 audit list) and the "
        "36 helpers are deleted from the class. Drop the xfail in the "
        "same Stage D commit that deletes the methods."
    ),
)
def test_helper_method_deleted_from_class(method_name: str) -> None:
    """Helper delegator must be absent at class level after Stage D."""
    assert not hasattr(CognitiveOrchestrationPipeline, method_name), (
        f"Phase 2.2 Stage D contract: "
        f"CognitiveOrchestrationPipeline.{method_name} should be deleted. "
        f"All callsites in sage.pipeline_v2.* and any tests must call the "
        f"backing pipeline_v2 module function directly. Per cgpro DESIGN_LOCK "
        f"2026-05-06 Q3 purge list."
    )


# ────────────────────────────────────────────────────────────────────
# 3. Methods that explicitly STAY on CognitiveOrchestrationPipeline.
#    Locked by cgpro Q3 / Q3a / Trap 6. These tests are NOT xfail —
#    they must pass at every commit in Phase 2.2.
# ────────────────────────────────────────────────────────────────────

METHODS_THAT_STAY: tuple[str, ...] = (
    "__init__",
    "run",
    "run_with_frame",
    "run_with_bench_evaluator",
    "_run_internal",
    "_emit",
)


@pytest.mark.parametrize("method_name", METHODS_THAT_STAY)
def test_pipeline_method_survives_phase_22(method_name: str) -> None:
    """Public API + _run_internal + _emit must stay on the class."""
    assert hasattr(CognitiveOrchestrationPipeline, method_name), (
        f"Phase 2.2 contract violation: "
        f"CognitiveOrchestrationPipeline.{method_name} must NOT be deleted. "
        f"Per cgpro DESIGN_LOCK 2026-05-06: public API methods, "
        f"_run_internal (façade delegator, Trap 6), and _emit (stateful "
        f"EventBus seam, Q3a) all stay."
    )


# ────────────────────────────────────────────────────────────────────
# 4. Module-level symbols on sage.pipeline that MUST stay (orchestrator
#    pipeline_mod indirection contract — Q6).
# ────────────────────────────────────────────────────────────────────

MODULE_LEVEL_SYMBOLS_THAT_STAY: tuple[str, ...] = (
    "_new_runtime_run_id",
    "_is_strict_governance",
    "_resolve_task_budget_usd",
    "_BANDIT_ATTRIBUTION_REASON_CODES",
    "BUDGET_EXCEEDED_RESULT",
    "EXECUTE_HALTED_UNVERIFIED",
    "EXECUTE_UNVERIFIED",
    "PipelineContext",
    "CognitiveOrchestrationPipeline",
    "time",  # accessed via sage.pipeline.time.monotonic monkeypatch
)


@pytest.mark.parametrize("symbol", MODULE_LEVEL_SYMBOLS_THAT_STAY)
def test_pipeline_module_symbol_survives_phase_22(symbol: str) -> None:
    """sage.pipeline module-level symbols must stay for orchestrator/test contracts."""
    import sage.pipeline as pipeline_mod

    assert hasattr(pipeline_mod, symbol), (
        f"Phase 2.2 contract violation: sage.pipeline.{symbol} must stay "
        f"at module level. Per cgpro DESIGN_LOCK 2026-05-06 Q6: orchestrator "
        f"resolves _new_runtime_run_id and time.monotonic via "
        f"`from sage import pipeline as pipeline_mod` to preserve "
        f"monkeypatch contracts in test_run_frame.py and test_oracle_stack.py."
    )
