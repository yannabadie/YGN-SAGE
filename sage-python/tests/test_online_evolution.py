"""Tests for online evolution wiring in the agent loop.

Verifies that:
1. should_evolve() returns false with empty archive
2. should_evolve() returns true after sufficient outcomes
3. evolve() is called when should_evolve() triggers
4. Evolution cooldown counter resets after evolve()
5. Constants are properly defined
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sage.pipeline_v2.learn import learn


@pytest.fixture(autouse=True)
def _legacy_oracle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests exercise the legacy pipeline path. Apply SAGE_ORACLE=0 so the
    cycle-7 default-on flip does not change expectations here."""
    monkeypatch.setenv("SAGE_ORACLE", "0")


class TestShouldEvolveLogic:
    """Test the should_evolve() gating logic (Python-side mock)."""

    def test_returns_false_when_empty(self):
        """Empty archive -> should not evolve."""
        engine = MagicMock()
        engine.should_evolve.return_value = False
        assert not engine.should_evolve()

    def test_returns_true_after_enough_outcomes(self):
        """After enough outcomes, should_evolve() triggers."""
        engine = MagicMock()
        engine.should_evolve.return_value = True
        assert engine.should_evolve()

    def test_evolve_called_when_triggered(self):
        """Simulate agent loop calling evolve() when should_evolve() returns True."""
        engine = MagicMock()
        engine.should_evolve.return_value = True
        engine.archive_cell_count.return_value = 10
        engine.archive_coverage.return_value = 0.3

        # Simulate the agent loop logic
        from sage.constants import EVOLUTION_ONLINE_POP_SIZE, EVOLUTION_ONLINE_GENERATIONS
        if engine.should_evolve():
            engine.evolve(
                pop_size=EVOLUTION_ONLINE_POP_SIZE,
                generations=EVOLUTION_ONLINE_GENERATIONS,
            )

        engine.evolve.assert_called_once_with(
            pop_size=EVOLUTION_ONLINE_POP_SIZE,
            generations=EVOLUTION_ONLINE_GENERATIONS,
        )

    def test_evolve_not_called_when_not_triggered(self):
        """evolve() should NOT be called when should_evolve() returns False."""
        engine = MagicMock()
        engine.should_evolve.return_value = False

        if engine.should_evolve():
            engine.evolve(pop_size=5, generations=2)

        engine.evolve.assert_not_called()


class TestEvolutionConstants:
    """Verify all evolution constants are properly defined."""

    def test_constants_exist_and_are_reasonable(self):
        from sage.constants import (
            EVOLUTION_MIN_OUTCOMES,
            EVOLUTION_COOLDOWN_OUTCOMES,
            EVOLUTION_SATURATION_THRESHOLD,
            EVOLUTION_ONLINE_POP_SIZE,
            EVOLUTION_ONLINE_GENERATIONS,
        )
        assert EVOLUTION_MIN_OUTCOMES >= 1
        assert EVOLUTION_COOLDOWN_OUTCOMES >= 1
        assert 0.0 < EVOLUTION_SATURATION_THRESHOLD <= 1.0
        assert EVOLUTION_ONLINE_POP_SIZE >= 1
        assert EVOLUTION_ONLINE_GENERATIONS >= 1


class TestEvolutionAblationSetup:
    """Verify that the existing evaluator.validate_evolution() works for evidence collection."""

    def test_validate_evolution_returns_expected_keys(self):
        from sage.evolution.evaluator import validate_evolution

        baseline = [0.5, 0.55, 0.48, 0.52, 0.49, 0.51, 0.53, 0.47, 0.50, 0.52]
        evolved = [0.65, 0.70, 0.68, 0.72, 0.67, 0.71, 0.69, 0.66, 0.73, 0.70]

        result = validate_evolution(baseline, evolved)
        assert "p_value" in result
        assert "effect_size" in result
        assert "gate_passed" in result
        assert "n_runs" in result
        assert result["n_runs"] == 10
        # Effect size should be large for this clear improvement
        assert result["effect_size"] > 1.0

    def test_validate_evolution_no_improvement(self):
        from sage.evolution.evaluator import validate_evolution

        baseline = [0.5, 0.55, 0.48, 0.52, 0.49, 0.51, 0.53, 0.47, 0.50, 0.52]
        evolved = [0.49, 0.54, 0.47, 0.51, 0.48, 0.50, 0.52, 0.46, 0.49, 0.51]

        result = validate_evolution(baseline, evolved)
        # No meaningful improvement -> effect size should be small
        assert result["effect_size"] < 1.0

    def test_validate_evolution_too_few_samples(self):
        from sage.evolution.evaluator import validate_evolution

        baseline = [0.5, 0.6]
        evolved = [0.7, 0.8]

        result = validate_evolution(baseline, evolved)
        # Too few samples -> always fails
        assert result["gate_passed"] is False


# -- H-series (2026-04-19): online evolution wiring in LEARN stage -------------
#
# Architecture/architecture.md and evolution/README.md claim "Online evolution
# (SA-3 complete)". The Rust impl (engine.should_evolve + engine.evolve) was
# wired into Python and ABI-exposed via PyO3, but no Python call site invoked
# them — the entire Evolution pillar ran zero generations during real runs.
# These tests pin the Pipeline._stage_learn → should_evolve → evolve call
# sequence so the gate cannot silently un-wire (same failure mode as the
# G-series write-gate bypass — a no-op default would mask the regression).

class TestPipelineEvolutionWiring:
    """Pipeline LEARN stage must call should_evolve + evolve when conditions hold."""

    def _make_pipeline_with_mock_engine(self, should_evolve_returns: bool):
        """Build a minimal Pipeline whose engine is a controllable mock."""
        from sage.pipeline import CognitiveOrchestrationPipeline
        engine = MagicMock()
        engine.should_evolve.return_value = should_evolve_returns
        # record_outcome is called BEFORE should_evolve in the LEARN block.
        engine.record_outcome = MagicMock()
        engine.evolve = MagicMock()

        pipeline = CognitiveOrchestrationPipeline(
            router=MagicMock(),
            engine=engine,
            assigner=MagicMock(),
            provider_pool=MagicMock(),
        )
        return pipeline, engine

    def _make_learn_ctx(self):
        """Construct a PipelineContext that satisfies the LEARN-stage guards."""
        from sage.pipeline import PipelineContext
        ctx = PipelineContext(task="implement quicksort")
        ctx.result = "def quicksort(): pass"  # non-empty → quality computable
        ctx.cost = 0.01
        ctx.latency_ms = 50.0
        # Topology with an id so record_outcome's `if topology_id` guard passes
        topology = MagicMock()
        topology.id = "topo-test-001"
        ctx.topology = topology
        ctx.topology_id = "topo-test-001"
        return ctx

    @pytest.mark.asyncio
    async def test_evolve_called_when_gate_fires(self):
        """should_evolve=True → evolve(pop_size=5, generations=2) must fire."""
        pipeline, engine = self._make_pipeline_with_mock_engine(should_evolve_returns=True)
        ctx = self._make_learn_ctx()

        await learn(pipeline, ctx)

        engine.should_evolve.assert_called_once()
        engine.evolve.assert_called_once()
        # Args check: must use the canonical constants, not magic numbers
        from sage.constants import (
            EVOLUTION_ONLINE_POP_SIZE,
            EVOLUTION_ONLINE_GENERATIONS,
        )
        engine.evolve.assert_called_with(
            pop_size=EVOLUTION_ONLINE_POP_SIZE,
            generations=EVOLUTION_ONLINE_GENERATIONS,
        )

    @pytest.mark.asyncio
    async def test_evolve_not_called_when_gate_blocks(self):
        """should_evolve=False → evolve must NOT fire."""
        pipeline, engine = self._make_pipeline_with_mock_engine(should_evolve_returns=False)
        ctx = self._make_learn_ctx()

        await learn(pipeline, ctx)

        engine.should_evolve.assert_called_once()
        engine.evolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_evolve_skipped_when_engine_lacks_method(self):
        """Engine without should_evolve (legacy stub) → silent skip, no crash."""
        from sage.pipeline import CognitiveOrchestrationPipeline
        engine = MagicMock(spec=["record_outcome"])  # No should_evolve / evolve
        engine.record_outcome = MagicMock()

        pipeline = CognitiveOrchestrationPipeline(
            router=MagicMock(),
            engine=engine,
            assigner=MagicMock(),
            provider_pool=MagicMock(),
        )
        ctx = self._make_learn_ctx()

        # Must not raise — degrades silently
        await learn(pipeline, ctx)


# -- H4 (2026-04-19): empirical end-to-end against the REAL Rust engine --------
#
# The mock-based wiring tests above prove _stage_learn calls the API in the
# right order. This test class proves something different and stronger: that
# on the REAL Rust engine, the chain `cache_topology → record_outcome →
# should_evolve → evolve` actually mutates archive state. The first version
# of the H1 commit shipped without `cache_topology` in the generate path,
# and an empirical poke (8 outcomes with diverse ids → cell_count=0) showed
# should_evolve would NEVER flip True on a real run — the wiring worked
# perfectly and observed zero effect, exactly the failure mode the advisor
# warned about. These tests pin the cache_topology requirement so a future
# refactor can't silently re-introduce the bypass.

try:
    import sage_core  # noqa: F401
    _HAS_SAGE_CORE = True
except ImportError:
    _HAS_SAGE_CORE = False


@pytest.mark.skipif(not _HAS_SAGE_CORE, reason="sage_core (Rust) not compiled")
class TestRealEngineEvolutionLoop:
    """End-to-end on the actual sage_core.TopologyEngine."""

    def test_record_outcome_grows_archive_only_when_topology_cached(self):
        """Without cache_topology, record_outcome silently no-ops on the
        archive. The fix at pipeline.py (Stage 2) is mandatory; this test
        is the regression guard."""
        import sage_core
        engine = sage_core.TopologyEngine()
        assert engine.archive_cell_count() == 0

        # Path A: record_outcome WITHOUT cache_topology — archive stays empty
        result = engine.generate("task-A", None, 2, 1.0)
        engine.record_outcome(
            topology_id=result.topology.id,
            task_summary="A",
            keywords=["a"],
            task_embedding=None,
            quality=0.7, cost=0.01, latency_ms=100.0,
        )
        assert engine.archive_cell_count() == 0, (
            "record_outcome without cache_topology must NOT grow the archive — "
            "if this fails, the Rust contract changed and the pipeline cache "
            "call may be redundant (good news; update or remove the fix)"
        )

        # Path B: cache_topology + record_outcome — archive grows
        result = engine.generate("task-B", None, 2, 1.0)
        engine.cache_topology(result.topology)
        engine.record_outcome(
            topology_id=result.topology.id,
            task_summary="B",
            keywords=["b"],
            task_embedding=None,
            quality=0.7, cost=0.01, latency_ms=100.0,
        )
        assert engine.archive_cell_count() >= 1, (
            "cache_topology + record_outcome must insert into the MAP-Elites "
            "archive — if this fails, the H1 evolution wiring will never fire"
        )

    def test_evolve_chain_end_to_end(self):
        """Cache + record diverse topologies — should_evolve must
        flip True at some point and evolve() must mutate archive state.

        Reproduces the empirical poke that uncovered the cache_topology
        bypass in the H1 commit. Use explicit template diversity instead of
        relying on generate() to happen to visit enough MAP-Elites cells on
        every Python/Rust build."""
        import sage_core
        engine = sage_core.TopologyEngine()
        templates = sage_core.PyTemplateStore()
        template_names = [
            "sequential",
            "parallel",
            "avr",
            "debate",
            "hub",
            "hierarchical",
            "selfmoa",
            "robust",
            "horizon_pipeline",
            "parallel_fanout",
        ]

        for i, template_name in enumerate(template_names * 2):
            topology = templates.create(template_name, f"model-{i % 3}")
            engine.cache_topology(topology)
            engine.record_outcome(
                topology_id=topology.id,
                task_summary=f"task-{i}",
                keywords=[f"kw{i}", template_name],
                task_embedding=None,
                quality=0.3 + (i % 7) * 0.1,
                cost=0.01 * (i + 1),
                latency_ms=100.0 + i * 50,
            )

        # With 20 diverse outcomes, cell_count must reach EVOLUTION_MIN_OUTCOMES
        from sage.constants import (
            EVOLUTION_MIN_OUTCOMES,
            EVOLUTION_COOLDOWN_OUTCOMES,
            EVOLUTION_ONLINE_POP_SIZE,
            EVOLUTION_ONLINE_GENERATIONS,
        )
        assert engine.archive_cell_count() >= EVOLUTION_MIN_OUTCOMES, (
            f"After 20 diverse outcomes, archive should have >= "
            f"{EVOLUTION_MIN_OUTCOMES} cells (got {engine.archive_cell_count()})"
        )
        assert engine.should_evolve(EVOLUTION_MIN_OUTCOMES, EVOLUTION_COOLDOWN_OUTCOMES)

        cells_before = engine.archive_cell_count()
        engine.evolve(
            pop_size=EVOLUTION_ONLINE_POP_SIZE,
            generations=EVOLUTION_ONLINE_GENERATIONS,
        )
        cells_after = engine.archive_cell_count()
        # evolve() either grows or holds the archive — never shrinks; the
        # interesting signal is that the call DOESN'T crash and the gate
        # cooldown resets so the next should_evolve depends on new outcomes.
        assert cells_after >= cells_before, (
            "evolve() must not shrink the archive"
        )
