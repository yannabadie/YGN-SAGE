# sage-python/tests/test_topology_verifier.py
import pytest
from sage.topology.topology_verifier import TopologyVerifier, TopologySpec, VerificationResult


def test_sequential_topology_terminates():
    spec = TopologySpec(
        agents=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c")],
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.terminates is True
    assert result.is_dag is True


def test_cyclic_topology_detected():
    spec = TopologySpec(
        agents=["a", "b"],
        edges=[("a", "b"), ("b", "a")],  # Cycle!
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.terminates is False
    assert result.is_dag is False


def test_parallel_topology_safe():
    spec = TopologySpec(
        agents=["a", "b", "c"],
        edges=[],  # No dependencies = all parallel
        topology_type="parallel",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.terminates is True
    assert result.no_deadlock is True


def test_verify_returns_proof():
    spec = TopologySpec(
        agents=["analyzer", "coder", "reviewer"],
        edges=[("analyzer", "coder"), ("coder", "reviewer")],
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert result.proof is not None
    assert "verified" in result.proof.lower()
    assert "kahn" in result.proof.lower()


def test_disconnected_agents_warning():
    spec = TopologySpec(
        agents=["a", "b", "orphan"],
        edges=[("a", "b")],
        topology_type="sequential",
    )
    verifier = TopologyVerifier()
    result = verifier.verify(spec)
    assert len(result.warnings) > 0
    assert "orphan" in str(result.warnings)


def test_max_depth_exceeded():
    # Very deep chain
    agents = [f"a{i}" for i in range(50)]
    edges = [(agents[i], agents[i+1]) for i in range(49)]
    spec = TopologySpec(agents=agents, edges=edges, topology_type="sequential")
    verifier = TopologyVerifier(max_depth=20)
    result = verifier.verify(spec)
    assert len(result.warnings) > 0  # Depth warning
