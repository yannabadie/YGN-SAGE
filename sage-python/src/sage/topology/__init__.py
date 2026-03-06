"""Topology engine: dynamic agent orchestration patterns."""
from sage.topology.engine import TopologyEngine, Topology, AgentNode
from sage.topology.patterns import vertical, horizontal, mesh

try:
    from sage.topology.planner import TopologyPlanner, StochasticDTS
except ImportError:
    TopologyPlanner = None  # type: ignore[assignment,misc]
    StochasticDTS = None  # type: ignore[assignment,misc]

from sage.topology.kg_rlvr import ProcessRewardModel, FormalKnowledgeGraph
from sage.topology.z3_topology import TopologyVerifier, TopologySpec, VerificationResult
from sage.topology.topology_archive import TopologyArchive, TopologyRecord

__all__ = ["TopologyEngine", "Topology", "AgentNode", "vertical", "horizontal", "mesh", "TopologyPlanner", "StochasticDTS", "ProcessRewardModel", "FormalKnowledgeGraph", "TopologyVerifier", "TopologySpec", "VerificationResult", "TopologyArchive", "TopologyRecord"]
