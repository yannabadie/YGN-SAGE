"""Topology engine: manages dynamic agent graphs.

Supports three core patterns:
- Vertical: parent delegates sequential sub-tasks to child agents
- Horizontal: parallel agents explore independently, results merged
- Mesh: fully connected agents communicate via message board
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TopologyType(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    MESH = "mesh"


@dataclass
class AgentNode:
    """A node in the agent topology graph."""
    id: str
    name: str
    role: str  # e.g. "coordinator", "researcher", "coder"
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    peers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Directed edge between two agent nodes."""
    source: str
    target: str
    edge_type: str  # "delegates_to", "reports_to", "peers_with"


class Topology:
    """A topology defines the structure of an agent ensemble."""

    def __init__(self, topology_type: TopologyType, root_id: str | None = None):
        self.type = topology_type
        self.root_id = root_id
        self._nodes: dict[str, AgentNode] = {}
        self._edges: list[Edge] = []

    def add_node(self, node: AgentNode) -> None:
        """Add a node to the topology."""
        self._nodes[node.id] = node
        if self.root_id is None:
            self.root_id = node.id

    def add_edge(self, source: str, target: str, edge_type: str) -> None:
        """Add a directed edge."""
        self._edges.append(Edge(source=source, target=target, edge_type=edge_type))
        # Update node relationships
        src = self._nodes.get(source)
        tgt = self._nodes.get(target)
        if src and tgt:
            if edge_type == "delegates_to":
                if target not in src.children:
                    src.children.append(target)
                tgt.parent_id = source
            elif edge_type == "peers_with":
                if target not in src.peers:
                    src.peers.append(target)
                if source not in tgt.peers:
                    tgt.peers.append(source)

    def get_node(self, node_id: str) -> AgentNode | None:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[AgentNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children if cid in self._nodes]

    def get_peers(self, node_id: str) -> list[AgentNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[pid] for pid in node.peers if pid in self._nodes]

    def node_count(self) -> int:
        return len(self._nodes)

    def all_nodes(self) -> list[AgentNode]:
        return list(self._nodes.values())

    def all_edges(self) -> list[Edge]:
        return list(self._edges)


class TopologyEngine:
    """Engine that creates and manages agent topologies."""

    def __init__(self):
        self._topologies: dict[str, Topology] = {}

    def create_topology(self, topology_type: TopologyType) -> str:
        """Create a new topology and return its ID."""
        topo_id = str(uuid.uuid4())[:8]
        self._topologies[topo_id] = Topology(topology_type=topology_type)
        return topo_id

    def get_topology(self, topo_id: str) -> Topology | None:
        return self._topologies.get(topo_id)

    def add_node(self, topo_id: str, name: str, role: str, **kwargs: Any) -> str:
        """Add a node to a topology. Returns node ID."""
        topo = self._topologies.get(topo_id)
        if topo is None:
            raise ValueError(f"Topology {topo_id} not found")

        node_id = str(uuid.uuid4())[:8]
        node = AgentNode(id=node_id, name=name, role=role, **kwargs)
        topo.add_node(node)
        return node_id

    def connect(self, topo_id: str, source: str, target: str, edge_type: str) -> None:
        """Connect two nodes."""
        topo = self._topologies.get(topo_id)
        if topo is None:
            raise ValueError(f"Topology {topo_id} not found")
        topo.add_edge(source, target, edge_type)

    def list_topologies(self) -> list[str]:
        return list(self._topologies.keys())
