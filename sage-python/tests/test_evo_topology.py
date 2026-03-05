import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.topology.evo_topology import (
    TopologyGenome, TopologyPopulation, TopologyEvolver
)

def test_topology_genome():
    g = TopologyGenome(
        nodes=["planner", "coder", "reviewer"],
        edges=[("planner", "coder"), ("coder", "reviewer")],
        pattern="vertical",
    )
    assert len(g.nodes) == 3
    assert g.pattern == "vertical"

def test_population_add_and_best():
    pop = TopologyPopulation(bins_per_dim=5)
    g1 = TopologyGenome(nodes=["a"], edges=[], pattern="single", features=(1, 1))
    pop.add(g1, score=0.5)
    g2 = TopologyGenome(nodes=["a", "b"], edges=[("a","b")], pattern="vertical", features=(1, 1))
    pop.add(g2, score=0.8)
    best = pop.best()
    assert best is not None
    assert best[1] == 0.8

def test_population_sample():
    pop = TopologyPopulation(bins_per_dim=5)
    for i in range(5):
        g = TopologyGenome(nodes=[f"n{i}"], edges=[], pattern="single", features=(i, i))
        pop.add(g, score=float(i) / 5)
    samples = pop.sample(3)
    assert len(samples) == 3
    assert all(isinstance(s, TopologyGenome) for s in samples)

def test_evolver_mutate_genome():
    evolver = TopologyEvolver()
    g = TopologyGenome(
        nodes=["planner", "coder"],
        edges=[("planner", "coder")],
        pattern="vertical",
        features=(3, 5),
    )
    mutated = evolver.mutate_genome(g)
    assert isinstance(mutated, TopologyGenome)

def test_evolver_crossover():
    evolver = TopologyEvolver()
    a = TopologyGenome(nodes=["planner", "coder"], edges=[("planner", "coder")], pattern="vertical", features=(2, 3))
    b = TopologyGenome(nodes=["researcher", "analyst"], edges=[("researcher", "analyst")], pattern="horizontal", features=(4, 1))
    child = evolver.crossover(a, b)
    assert isinstance(child, TopologyGenome)
    assert len(child.nodes) >= 2
