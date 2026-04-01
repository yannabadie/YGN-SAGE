"""Tests for MASBENCH 5-axis improvements.

Validates axis detection, topology construction, prompt engineering,
answer checking, and arithmetic verification — all without LLM calls.
"""
from __future__ import annotations

import pytest


# ── Axis Detection Tests ──────────────────────────────────────────────────


class TestTopologySelection:
    """Test select_macro_topology() DAG-driven template selection.

    The pipeline selects templates based on structural DAG features
    (omega=parallelism, delta=depth, gamma=coupling), NOT regex on task text.
    """

    def test_deep_sequential_selects_horizon_pipeline(self):
        """High delta + low omega → horizon_pipeline for chained reasoning."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=1, delta=8, gamma=0.1)
        assert select_macro_topology(features) == "horizon_pipeline"

    def test_wide_parallel_selects_parallel_fanout(self):
        """High omega + low gamma → parallel_fanout for independent sub-tasks."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=4, delta=2, gamma=0.1)
        assert select_macro_topology(features) == "parallel_fanout"

    def test_coupled_parallel_selects_robust(self):
        """Moderate omega + high gamma → robust for cross-validation."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=3, delta=3, gamma=0.7)
        assert select_macro_topology(features) == "robust"

    def test_s3_parallel_selects_robust(self):
        """S3 formal reasoning + parallel → robust for redundancy."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=2, delta=3, gamma=0.3)
        assert select_macro_topology(features, system=3) == "robust"

    def test_simple_sequential_unchanged(self):
        """Simple DAG → sequential (original behavior preserved)."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=1, delta=3, gamma=0.0)
        assert select_macro_topology(features) == "sequential"

    def test_moderate_parallel_unchanged(self):
        """Moderate parallel + low coupling → parallel (original)."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=2, delta=3, gamma=0.3)
        assert select_macro_topology(features) == "parallel"

    def test_high_coupling_with_parallel_selects_robust(self):
        """High coupling + parallel workers → robust (cross-validation)."""
        from sage.pipeline_stages import DAGFeatures, select_macro_topology

        features = DAGFeatures(omega=2, delta=3, gamma=0.8)
        assert select_macro_topology(features) == "robust"


# ── Prompt Engineering Tests ──────────────────────────────────────────────


class TestPromptEngineering:
    """Test _parse_task() axis-specific prompt injection."""

    def _make_item(self, question: str, ground_truth: str = "42") -> dict:
        import json

        return {
            "prompt_json": json.dumps([{"role": "user", "content": question}]),
            "reward_model_json": json.dumps({"ground_truth": ground_truth}),
        }

    def test_robustness_prefix(self):
        from sage.bench.masbench import _parse_task

        item = self._make_item("Solve this math problem with noise.")
        q, gt = _parse_task(item, axis="robustness")
        assert q.startswith("Ignore all irrelevant information")
        assert "number only" in q
        assert gt == "42"

    def test_horizon_prefix(self):
        from sage.bench.masbench import _parse_task

        item = self._make_item("Part 1 and Part 2")
        q, gt = _parse_task(item, axis="horizon")
        assert "<<horizon>>" in q
        assert "step by step" in q.lower()

    def test_depth_prefix(self):
        from sage.bench.masbench import _parse_task

        item = self._make_item("Chain reasoning problem")
        q, gt = _parse_task(item, axis="depth")
        assert "Think step by step carefully" in q
        assert "Verify each intermediate result" in q

    def test_default_suffix(self):
        from sage.bench.masbench import _parse_task

        item = self._make_item("Simple question")
        q, gt = _parse_task(item, axis="")
        assert "number only" in q.lower()

    def test_no_axis(self):
        from sage.bench.masbench import _parse_task

        item = self._make_item("Simple question")
        q, gt = _parse_task(item)  # no axis parameter
        assert "number only" in q.lower()


# ── Answer Checking Tests ─────────────────────────────────────────────────


class TestAnswerChecking:
    """Test _check_answer() with axis-specific logic."""

    def test_exact_match(self):
        from sage.bench.masbench import _check_answer

        assert _check_answer("The answer is 42", "42")

    def test_numeric_match(self):
        from sage.bench.masbench import _check_answer

        assert _check_answer("I calculated 42.0 as the result", "42")

    def test_horizon_multipart(self):
        from sage.bench.masbench import _check_answer

        gt = "10<<horizon>>20<<horizon>>30"
        response = "First answer is 10, second is 20, third is 30"
        assert _check_answer(response, gt)

    def test_horizon_multipart_missing(self):
        from sage.bench.masbench import _check_answer

        gt = "10<<horizon>>20<<horizon>>30"
        response = "First answer is 10, second is 20"
        assert not _check_answer(response, gt)

    def test_robustness_multi_number(self):
        from sage.bench.masbench import _check_answer

        gt = "42 and 17 and 99"
        response = "The answers are 42, 17, 99"
        assert _check_answer(response, gt, axis="robustness")

    def test_robustness_missing_number(self):
        from sage.bench.masbench import _check_answer

        gt = "42 and 17 and 99"
        response = "The answers are 42, 17"
        assert not _check_answer(response, gt, axis="robustness")

    def test_empty_ground_truth(self):
        from sage.bench.masbench import _check_answer

        assert not _check_answer("some response", "")

    def test_case_insensitive(self):
        from sage.bench.masbench import _check_answer

        assert _check_answer("YES", "yes")


# ── Arithmetic Verification Tests ─────────────────────────────────────────


class TestArithmeticVerification:
    """Test _verify_arithmetic() in TopologyController."""

    @pytest.fixture
    def controller(self):
        from sage.topology_controller import TopologyController

        return TopologyController()

    def test_correct_addition(self, controller):
        result = "Step 1: 5 + 3 = 8"
        ok, msg = controller._verify_arithmetic(result)
        assert ok
        assert msg == ""

    def test_incorrect_addition(self, controller):
        result = "Step 1: 5 + 3 = 9"
        ok, msg = controller._verify_arithmetic(result)
        assert not ok
        assert "should be 8" in msg

    def test_correct_multiplication(self, controller):
        result = "We compute 7 * 6 = 42"
        ok, msg = controller._verify_arithmetic(result)
        assert ok

    def test_incorrect_multiplication(self, controller):
        result = "We compute 7 * 6 = 43"
        ok, msg = controller._verify_arithmetic(result)
        assert not ok
        assert "should be 42" in msg

    def test_no_equations(self, controller):
        result = "The answer is forty-two."
        ok, msg = controller._verify_arithmetic(result)
        assert ok
        assert msg == ""

    def test_multiple_equations_one_wrong(self, controller):
        result = "Step 1: 10 + 5 = 15. Step 2: 15 * 2 = 31"
        ok, msg = controller._verify_arithmetic(result)
        assert not ok
        assert "should be 30" in msg


# ── Topology Construction Tests ───────────────────────────────────────────


class TestTopologyConstruction:
    """Test Rust template construction via PyO3."""

    def test_robust_template_exists(self):
        try:
            from sage_core import PyTemplateStore

            store = PyTemplateStore()
            topo = store.create("robust", "")
            assert topo.node_count() == 5
            assert topo.is_acyclic()
        except ImportError:
            pytest.skip("sage_core not available")

    def test_horizon_pipeline_template_exists(self):
        try:
            from sage_core import PyTemplateStore

            store = PyTemplateStore()
            topo = store.create("horizon_pipeline", "")
            assert topo.node_count() == 5  # splitter + 3 stages + aggregator
            assert topo.is_acyclic()
        except ImportError:
            pytest.skip("sage_core not available")

    def test_parallel_fanout_template_exists(self):
        try:
            from sage_core import PyTemplateStore

            store = PyTemplateStore()
            topo = store.create("parallel_fanout", "")
            assert topo.node_count() == 7  # dispatcher + 5 workers + aggregator
            assert topo.is_acyclic()
        except ImportError:
            pytest.skip("sage_core not available")

    def test_template_store_has_11(self):
        try:
            from sage_core import PyTemplateStore

            store = PyTemplateStore()
            names = store.available()
            assert len(names) == 11
            assert "robust" in names
            assert "horizon_pipeline" in names
            assert "parallel_fanout" in names
        except ImportError:
            pytest.skip("sage_core not available")

    def test_robust_node_roles(self):
        try:
            from sage_core import PyTemplateStore

            store = PyTemplateStore()
            topo = store.create("robust", "")
            roles = [topo.get_node(i).role for i in range(topo.node_count())]
            assert "preprocessor" in roles
            assert "verifier" in roles
            assert sum(1 for r in roles if r.startswith("worker_")) == 3
        except ImportError:
            pytest.skip("sage_core not available")

    def test_parallel_fanout_diverse_tiers(self):
        """Workers must have diverse system tiers (not all the same)."""
        try:
            from sage_core import PyTemplateStore

            store = PyTemplateStore()
            topo = store.create("parallel_fanout", "")
            worker_systems = set()
            for i in range(topo.node_count()):
                node = topo.get_node(i)
                if node.role.startswith("worker_"):
                    worker_systems.add(node.system)
            # Must have at least 2 different tiers (S1, S2, S3)
            assert len(worker_systems) >= 2
        except ImportError:
            pytest.skip("sage_core not available")


# ── Prune Bug Fix Test ────────────────────────────────────────────────────


class TestPruneBugFix:
    """Test that prune path properly computes importance before using it."""

    def test_prune_calls_compute_importance(self):
        """The prune branch must call compute_importance_score, not use undefined var."""
        from sage.topology_controller import TopologyController

        controller = TopologyController()
        # Simulate a scenario where quality is known and below threshold
        # but importance is high (should NOT prune)
        decision = controller.evaluate_and_decide(
            node_idx=0,
            result="A decent result with some content here to analyze",
            task="test task",
            topology=None,
            ctx={"node_idx": 0, "latency_ms": 100},
            parallel_outputs=["output1", "output2", "output3"],
        )
        # Should not crash with NameError on 'importance'
        assert decision.action in (
            "continue", "upgrade_model", "prune_node",
            "reroute_topology", "spawn_subagent",
        )
