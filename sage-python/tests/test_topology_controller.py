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
    controller._node_retries[0] = 2  # already at max
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
    controller._reroute_count = 1  # at max
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


def test_spawn_on_emergent_subtask(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5
    topo = MagicMock()
    topo.get_predecessors.return_value = []  # no predecessors -> skip open_gate
    result = "The analysis is done. We need to also verify the edge cases for negative inputs."
    d = controller.evaluate_and_decide(0, result, "task", topo, mock_ctx)
    assert d.action == "spawn_subagent"


def test_max_spawns_respected(controller, mock_ctx):
    controller._spawn_count = 3  # at max
    controller._qe.estimate.return_value = 0.5
    topo = MagicMock()
    topo.get_predecessors.return_value = []  # no predecessors -> skip open_gate
    result = "Need to also check the boundary conditions."
    d = controller.evaluate_and_decide(0, result, "task", topo, mock_ctx)
    assert d.action == "continue"  # spawn blocked


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
    assert ctrl._abstain_count == 1


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
