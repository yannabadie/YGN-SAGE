"""Tests for TopologyController — runtime adaptation decisions."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from sage.topology_controller import TopologyController, AdaptationDecision


@pytest.fixture
def controller():
    qe = MagicMock()
    qe.estimate.return_value = 0.5  # default medium quality
    return TopologyController(
        assigner=MagicMock(),
        quality_estimator=qe,
        prm=None,
        embedder=None,
    )


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.latency_ms = 100.0
    return ctx


def test_continue_on_good_quality(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.8
    d = controller.evaluate_and_decide(0, "good result", "task", MagicMock(), mock_ctx)
    assert d.action == "continue"


def test_upgrade_model_on_critical_quality(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.1
    topo = MagicMock()
    topo.get_node.return_value = MagicMock(system=2)
    controller._assigner.assign_single_node.return_value = "reasoner-v2"
    d = controller.evaluate_and_decide(0, "bad result", "task", topo, mock_ctx)
    assert d.action == "upgrade_model"
    assert d.target_node == 0
    assert d.new_model_id == "reasoner-v2"
    controller._assigner.assign_single_node.assert_called_once()


def test_upgrade_respects_max_retries(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.1
    controller._seed_for_tests(retries={0: 2})  # already at max
    d = controller.evaluate_and_decide(0, "bad", "task", MagicMock(), mock_ctx)
    assert d.action == "continue"  # no more retries -> accept


def test_reroute_on_inconsistency(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5  # medium quality
    topo = MagicMock()
    topo.get_predecessors.return_value = []  # no predecessors -> skip open_gate
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.2):
        d = controller.evaluate_and_decide(
            0, "result", "task", topo, mock_ctx,
            parallel_outputs=["output1", "output2"],
        )
        assert d.action == "reroute_topology"


def test_max_reroute_forces_continue(controller, mock_ctx):
    controller._seed_for_tests(reroute=1)  # at max
    controller._qe.estimate.return_value = 0.5
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.2):
        d = controller.evaluate_and_decide(
            0, "result", "task", MagicMock(), mock_ctx,
            parallel_outputs=["a", "b"],
        )
        assert d.action != "reroute_topology"


def test_prune_on_low_importance(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5
    topo = MagicMock()
    topo.get_predecessors.return_value = []  # no predecessors -> skip open_gate
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.8):
        with patch('sage.topology_controller.TopologyController.compute_importance_score', return_value=0.1):
            d = controller.evaluate_and_decide(
                0, "redundant", "task", topo, mock_ctx,
                parallel_outputs=["same", "content"],
            )
            assert d.action == "prune_node"


def test_empty_output_reroutes_immediately(controller, mock_ctx):
    d = controller.evaluate_and_decide(0, "", "task", MagicMock(), mock_ctx)
    assert d.action == "reroute_topology"
    assert "empty" in d.reason


def test_error_output_reroutes_immediately(controller, mock_ctx):
    d = controller.evaluate_and_decide(0, "ERROR: provider timeout", "task", MagicMock(), mock_ctx)
    assert d.action == "reroute_topology"
    assert "error-like" in d.reason


def test_heuristic_quality_used_when_estimator_abstains(mock_ctx):
    qe = MagicMock()
    qe.estimate.return_value = None
    ctrl = TopologyController(quality_estimator=qe, prm=None)

    quality = ctrl._compute_quality(
        0,
        "1. Inspect the input.\n2. Return a valid Python function with tests.",
        "Write a Python function",
        mock_ctx,
    )

    assert quality > 0.3  # short output scores low in the new weighted heuristic
    assert ctrl.abstain_count == 1


def test_debate_disagreement_opens_gate(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5
    topo = MagicMock()
    topo.template_type = "debate"
    topo.get_predecessors.return_value = [0]
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.2):
        d = controller.evaluate_and_decide(
            1,
            "Counterargument with unresolved disagreement.",
            "task",
            topo,
            mock_ctx,
            parallel_outputs=["Answer A", "Answer B"],
        )
    assert d.action == "open_gate"
    assert d.gate_source == 0
    assert d.gate_target == 1


def test_debate_agreement_stops_additional_round(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5
    topo = MagicMock()
    topo.template_type = "debate"
    topo.get_predecessors.return_value = [0]
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.9):
        d = controller.evaluate_and_decide(
            1,
            "Matching argument with peer output.",
            "task",
            topo,
            mock_ctx,
            parallel_outputs=["Answer A", "Answer A"],
        )
    assert d.action == "continue"


def test_quality_blends_prm(mock_ctx):
    qe = MagicMock()
    qe.estimate.return_value = 0.6
    prm = MagicMock()
    prm.calculate_r_path.return_value = (0.9, {})

    ctrl = TopologyController(quality_estimator=qe, prm=prm)
    quality = ctrl._compute_quality(0, "<think>step 1</think>", "task", mock_ctx)
    # 0.8 * 0.6 + 0.2 * 0.9 = 0.48 + 0.18 = 0.66
    assert abs(quality - 0.66) < 0.01


def test_no_prm_on_plain_text(mock_ctx):
    qe = MagicMock()
    qe.estimate.return_value = 0.6
    prm = MagicMock()

    ctrl = TopologyController(quality_estimator=qe, prm=prm)
    quality = ctrl._compute_quality(0, "just plain text no think tags", "task", mock_ctx)
    prm.calculate_r_path.assert_not_called()
    assert quality == 0.6


# --- D3 audit fix (2026-04-18): sentinel output handling ---


def test_is_empty_or_error_detects_sentinel_string():
    """D3 audit (docs/audits/2026-04-18-astropy-14995-decision-path.md):
    the AgentLoop sentinel "[sage: agent exited after N steps with no
    content]" must be flagged as structural failure. Before the fix,
    a 51-char sentinel passed the empty-or-error check, fell through
    to quality scoring (heuristic ~0.5 neutral), and returned `continue`
    — wasting the chance to reroute_topology / upgrade_model."""
    assert TopologyController._is_empty_or_error("[sage: agent exited after 5 steps with no content]")
    assert TopologyController._is_empty_or_error("[sage: agent exited after 20 steps with no content]")
    # Leading whitespace must still strip before prefix check
    assert TopologyController._is_empty_or_error(
        "  [sage: agent exited after 3 steps with no content]  "
    )
    # Legitimate content that merely MENTIONS the sentinel text is NOT a sentinel
    assert not TopologyController._is_empty_or_error(
        "I just read [sage: agent exited after 5 steps with no content] in the logs"
    )


def test_sentinel_triggers_reroute_not_continue(controller, mock_ctx):
    """End-to-end: when a node returns the sentinel, controller must
    propose reroute_topology (for the first failure) instead of silently
    continuing. This is the fix that would have broken the astropy-14995
    cascade: coder sentinel → reroute → pipeline rebuilds → retry."""
    topo = MagicMock()
    sentinel = "[sage: agent exited after 20 steps with no content]"
    d = controller.evaluate_and_decide(0, sentinel, "task", topo, mock_ctx)
    assert d.action == "reroute_topology", (
        f"Expected reroute_topology for first sentinel, got {d.action}. "
        "The fix in _is_empty_or_error must flag sentinel prefix."
    )


def test_sentinel_respects_max_reroute_budget(controller, mock_ctx):
    """After MAX_REROUTES=1 reroute, further sentinels → continue
    (prevents infinite reroute loops)."""
    topo = MagicMock()
    sentinel = "[sage: agent exited after 20 steps with no content]"
    # First sentinel → reroute
    d1 = controller.evaluate_and_decide(0, sentinel, "task", topo, mock_ctx)
    assert d1.action == "reroute_topology"
    # Second sentinel (any node) → continue, budget exhausted
    d2 = controller.evaluate_and_decide(1, sentinel, "task", topo, mock_ctx)
    assert d2.action == "continue"
    assert "budget" in d2.reason.lower()


# --- D4 audit fix: AgentLoopBudgetExhausted exception ---


def test_agent_loop_exhaustion_dataclass_exposes_fields():
    """D4 audit: AgentLoopExhaustion carries structured metadata.
    Controllers read ``loop.last_exhaustion`` to decide whether the
    failure was a hard budget exhaustion or a soft stall."""
    from sage.agent_loop import AgentLoopExhaustion
    e = AgentLoopExhaustion(
        reason="stalled",
        step_count=10,
        consecutive_tool_steps=10,
        last_tool_name="execute_bash",
        last_assistant_snippet="I need to explore more files...",
    )
    assert e.reason == "stalled"
    assert e.step_count == 10
    assert e.consecutive_tool_steps == 10
    assert e.last_tool_name == "execute_bash"
    assert e.last_assistant_snippet.startswith("I need to")


# --- B.1 / B.2 / B.4 Task-B façade contract tests (2026-04-20) ---


def test_missing_sage_core_raises_importerror(monkeypatch):
    """B.1: TopologyController.__init__ must raise ImportError immediately
    when sage_core is unavailable. Before Task B it would silently fall
    back to a legacy Python path; after Task B that path is deleted and
    the ImportError is the only honest signal."""
    import sage.topology_controller as tc_mod
    monkeypatch.setattr(tc_mod, "_HAS_RUST_CTRL", False)
    monkeypatch.setattr(tc_mod, "_RustTopologyControllerImpl", None)
    with pytest.raises(ImportError, match="sage_core"):
        tc_mod.TopologyController()


def test_python_facade_reads_rust_state():
    """B.2: @property getters on TopologyController must proxy Rust state.
    After _seed_for_tests() plants values in Rust, the properties must
    return those exact values — confirming they are not reading stale
    Python shadow fields."""
    ctrl = TopologyController(assigner=MagicMock(), quality_estimator=None)
    ctrl._seed_for_tests(reroute=1, spawn=2, retries={3: 4}, abstain=5)
    assert ctrl.reroute_count == 1
    assert ctrl.spawn_count == 2
    assert ctrl.node_retries[3] == 4
    assert ctrl.abstain_count == 5


def test_python_facade_rejects_direct_mutation():
    """B.2: Setting a façade property directly must raise AttributeError.
    The old shadow-field pattern allowed ``ctrl._reroute_count = N`` to
    route around Rust state; that pathway is closed — any attempt to
    bypass _seed_for_tests() must fail loudly."""
    ctrl = TopologyController(assigner=MagicMock(), quality_estimator=None)
    with pytest.raises(AttributeError):
        ctrl.reroute_count = 99  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ctrl.spawn_count = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ctrl.abstain_count = 1  # type: ignore[misc]


def test_state_equivalence_after_cascade_scenario():
    """B.4: Façade properties must reflect Rust state after a multi-step cascade.
    Three mutating paths are exercised:
      1. QE returns None → record_abstain() fires → abstain_count becomes 1.
      2. QE returns critical (0.1) on a DIFFERENT node → check_quality_cascade
         increments node_retries[1] to 1.
    Façade properties read Rust state; shadow fields no longer exist."""
    qe = MagicMock()
    # Call 1 (node 0): QE abstains → heuristic used, abstain_count → 1.
    # Call 2 (node 1): QE returns 0.1 (critical) → upgrade_model, retries[1] → 1.
    qe.estimate.side_effect = [None, 0.1]
    topo = MagicMock()
    node = MagicMock()
    node.max_retries = 2
    node.system = 1
    node.model_id = ""
    node.required_capabilities = []
    node.role = "agent"
    node.max_cost_usd = 1.0
    topo.get_node.return_value = node
    topo.get_predecessors.return_value = []
    ctrl = TopologyController(assigner=MagicMock(), quality_estimator=qe)
    ctrl._assigner.assign_single_node.return_value = "reasoner-v2"

    ctx = MagicMock()
    ctx.latency_ms = 100.0

    # Call 1 (node 0): QE abstains → heuristic scores → abstain_count becomes 1.
    ctrl.evaluate_and_decide(0, "some output for the first node", "task", topo, ctx)
    assert ctrl.abstain_count == 1, (
        f"After QE abstain, abstain_count should be 1, got {ctrl.abstain_count}"
    )

    # Call 2 (node 1, fresh): QE returns 0.1 (critical) → upgrade_model.
    # node_retries[1] must be 1; abstain_count stays at 1.
    d2 = ctrl.evaluate_and_decide(1, "poor quality answer", "task", topo, ctx)
    assert d2.action == "upgrade_model"
    assert ctrl.node_retries.get(1, 0) == 1, (
        f"After one critical-quality upgrade on node 1, node_retries[1] should be 1, "
        f"got {ctrl.node_retries.get(1, 0)}"
    )
    assert ctrl.abstain_count == 1, (
        "abstain_count should still be 1 — QE returned a value on call 2"
    )


def test_is_cross_provider_same_provider():
    """Same-provider pair is NOT cross-provider."""
    assert not TopologyController._is_cross_provider("deepseek-v4-flash", "deepseek-v4-pro")


def test_is_cross_provider_different_providers():
    """gemini→deepseek is cross-provider — root cause of BCB/89 HTTP 400."""
    assert TopologyController._is_cross_provider(
        "deepseek-v4-flash", "gemini-3.1-flash-lite-preview"
    )


def test_is_cross_provider_unknown_model_does_not_block():
    """Unknown model IDs must NOT block the upgrade (fail-open to avoid silently
    disabling upgrades for future provider additions)."""
    assert not TopologyController._is_cross_provider("unknown-model-xyz", "deepseek-v4-flash")
    assert not TopologyController._is_cross_provider("deepseek-v4-flash", "unknown-model-xyz")


def test_resolve_upgrade_model_skips_cross_provider():
    """_resolve_upgrade_model returns None when assign_single_node would cross providers."""
    from unittest.mock import MagicMock, patch

    assigner = MagicMock()
    assigner.assign_single_node.return_value = "gemini-3.1-flash-lite-preview"
    ctrl = TopologyController(assigner=assigner)

    node = MagicMock()
    node.model_id = "deepseek-v4-flash"
    node.max_cost_usd = 1.0
    node.required_capabilities = []
    node.role = "coder"

    topo = MagicMock()
    topo.get_node.return_value = node
    topo.set_node_model_id = MagicMock()

    result = ctrl._resolve_upgrade_model(0, "code task", topo, MagicMock())
    assert result is None, "cross-provider upgrade must be blocked (gemini→deepseek endpoint)"
    # Revert: topology model_id must be restored to the original (deepseek-v4-flash)
    # because assign_single_node may have mutated the topology node as a side effect.
    topo.set_node_model_id.assert_called_once_with(0, "deepseek-v4-flash")


def test_agent_loop_budget_exhausted_wraps_detail():
    """AgentLoopBudgetExhausted is a RuntimeError subclass that carries
    AgentLoopExhaustion metadata as .detail — callers can catch and
    inspect structurally instead of parsing the sentinel string."""
    from sage.agent_loop import AgentLoopBudgetExhausted, AgentLoopExhaustion
    detail = AgentLoopExhaustion(
        reason="budget_exhausted", step_count=20, consecutive_tool_steps=20,
    )
    exc = AgentLoopBudgetExhausted(detail)
    assert isinstance(exc, RuntimeError)
    assert exc.detail is detail
    assert "budget_exhausted" in str(exc)
    assert "20" in str(exc)
