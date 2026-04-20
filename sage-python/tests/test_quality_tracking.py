"""Tests for quality estimation tracking.

Issue D audit fix: Controller must track when QualityEstimator abstains
(returns None) and uses the 0.5 default.
"""
from sage.topology_controller import TopologyController
from sage.quality_estimator import QualityEstimator


class TestQualityAbstainTracking:
    def test_abstain_count_starts_at_zero(self):
        ctrl = TopologyController()
        assert ctrl.abstain_count == 0

    def test_quality_stats_structure(self):
        ctrl = TopologyController()
        stats = ctrl.quality_stats()
        assert "abstain_count" in stats
        assert "node_qualities" in stats
        assert "reroute_count" in stats
        assert "spawn_count" in stats
        assert stats["abstain_count"] == 0

    def test_abstain_count_increments_on_none_quality(self):
        """When QualityEstimator returns None, controller should count it."""
        ctrl = TopologyController(quality_estimator=None)
        # _compute_quality returns None when no estimator → triggers abstain
        # We test via evaluate_and_decide with a mock topology
        class _FakeNode:
            role = "agent"
            model_id = "test"
        class _FakeTopo:
            def node_count(self): return 1
            def get_node(self, i): return _FakeNode()

        class _FakeCtx:
            task = "test"
            system = 1
            topology = _FakeTopo()

        decision = ctrl.evaluate_and_decide(
            node_idx=0, result="hello world", task="test",
            topology=_FakeTopo(), ctx=_FakeCtx(),
        )
        assert ctrl.abstain_count >= 1


class TestQualityEstimatorBackend:
    def test_backend_name_no_backends(self):
        qe = QualityEstimator()
        # In test env without compiled sage_core, both backends are None
        assert qe.backend_name in ("onnx", "z3_labeler", "none")

    def test_backend_name_is_string(self):
        qe = QualityEstimator()
        assert isinstance(qe.backend_name, str)
        assert len(qe.backend_name) > 0
