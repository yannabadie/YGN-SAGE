"""Tests for RL-guided topology evolution engine."""
import pytest
from sage.topology.rl_evolution import TopologyEvolutionEngine, TopologyRecord
from sage.topology.z3_topology import TopologySpec


def test_record_creation():
    rec = TopologyRecord(
        spec=TopologySpec(agents=["a", "b"], edges=[("a", "b")], topology_type="sequential"),
        score=0.85,
        task_type="code",
    )
    assert rec.score == 0.85


def test_engine_stores_record():
    engine = TopologyEvolutionEngine()
    spec = TopologySpec(agents=["a"], edges=[], topology_type="parallel")
    engine.record(spec, score=0.9, task_type="code")
    assert engine.count() == 1


def test_engine_recommends_best():
    engine = TopologyEvolutionEngine()
    spec1 = TopologySpec(agents=["a"], edges=[], topology_type="parallel")
    spec2 = TopologySpec(agents=["a", "b"], edges=[("a", "b")], topology_type="sequential")
    engine.record(spec1, score=0.6, task_type="code")
    engine.record(spec2, score=0.9, task_type="code")
    best = engine.recommend(task_type="code")
    assert best is not None
    assert best.topology_type == "sequential"


def test_engine_returns_none_for_unknown_type():
    engine = TopologyEvolutionEngine()
    best = engine.recommend(task_type="unknown")
    assert best is None


def test_engine_tracks_task_types():
    engine = TopologyEvolutionEngine()
    engine.record(TopologySpec(agents=["a"], edges=[], topology_type="parallel"), score=0.5, task_type="code")
    engine.record(TopologySpec(agents=["a"], edges=[], topology_type="parallel"), score=0.7, task_type="reasoning")
    assert "code" in engine.task_types()
    assert "reasoning" in engine.task_types()
