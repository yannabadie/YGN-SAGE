"""Topology engine: dynamic agent orchestration patterns."""
from typing import Any

from sage.topology.engine import TopologyEngine, Topology, AgentNode
from sage.topology.patterns import vertical, horizontal, mesh

TopologyPlanner: Any
StochasticDTS: Any
try:
    from sage.topology.planner import TopologyPlanner, StochasticDTS
except ImportError:
    TopologyPlanner = None
    StochasticDTS = None

from sage.topology.kg_rlvr import ProcessRewardModel, FormalKnowledgeGraph
from sage.topology.topology_verifier import TopologyVerifier, TopologySpec, VerificationResult
from sage.topology.topology_archive import TopologyArchive, TopologyRecord
from sage.topology.ltl_bridge import verify_topology_ltl, check_reachability

__all__ = ["TopologyEngine", "Topology", "AgentNode", "vertical", "horizontal", "mesh", "TopologyPlanner", "StochasticDTS", "ProcessRewardModel", "FormalKnowledgeGraph", "TopologyVerifier", "TopologySpec", "VerificationResult", "TopologyArchive", "TopologyRecord", "verify_topology_ltl", "check_reachability"]
