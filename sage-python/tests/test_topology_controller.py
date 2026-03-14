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
    d = controller.evaluate_and_decide(0, "bad result", "task", topo, mock_ctx)
    assert d.action == "upgrade_model"
    assert d.target_node == 0


def test_upgrade_respects_max_retries(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.1
    controller._node_retries[0] = 2  # already at max
    d = controller.evaluate_and_decide(0, "bad", "task", MagicMock(), mock_ctx)
    assert d.action == "continue"  # no more retries -> accept


def test_reroute_on_inconsistency(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5  # medium quality
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.2):
        d = controller.evaluate_and_decide(
            0, "result", "task", MagicMock(), mock_ctx,
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
    with patch('sage.topology_controller.TopologyController.compute_consistency_score', return_value=0.8):
        with patch('sage.topology_controller.TopologyController.compute_importance_score', return_value=0.1):
            d = controller.evaluate_and_decide(
                0, "redundant", "task", MagicMock(), mock_ctx,
                parallel_outputs=["same", "content"],
            )
            assert d.action == "prune_node"


def test_spawn_on_emergent_subtask(controller, mock_ctx):
    controller._qe.estimate.return_value = 0.5
    result = "The analysis is done. We need to also verify the edge cases for negative inputs."
    d = controller.evaluate_and_decide(0, result, "task", MagicMock(), mock_ctx)
    assert d.action == "spawn_subagent"


def test_max_spawns_respected(controller, mock_ctx):
    controller._spawn_count = 3  # at max
    controller._qe.estimate.return_value = 0.5
    result = "Need to also check the boundary conditions."
    d = controller.evaluate_and_decide(0, result, "task", MagicMock(), mock_ctx)
    assert d.action == "continue"  # spawn blocked


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
