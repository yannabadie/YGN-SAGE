"""Pure-Python TopologyGraph for environments without sage_core.

Duck-type compatible with sage_core.TopologyGraph for basic operations.
Uses adjacency list internally. No networkx dependency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class PyTopologyNode:
    """Node in a pure-Python topology graph."""
    role: str = "worker"
    model_id: str = ""
    system: int = 1
    required_capabilities: list[str] = field(default_factory=list)


class PyTopologyGraph:
    """Pure-Python topology graph, duck-type compatible with sage_core.TopologyGraph.

    Uses adjacency list internally. No networkx dependency.
    """

    def __init__(self) -> None:
        self._nodes: list[PyTopologyNode] = []
        self._edges: list[tuple[int, int, str]] = []  # (from, to, edge_type)

    def add_node(self, role: str = "worker", model_id: str = "", system: int = 1,
                 required_capabilities: list[str] | None = None) -> int:
        """Add a node, return its index."""
        node = PyTopologyNode(
            role=role, model_id=model_id, system=system,
            required_capabilities=required_capabilities or [],
        )
        idx = len(self._nodes)
        self._nodes.append(node)
        return idx

    def add_edge(self, from_idx: int, to_idx: int, edge_type: str = "control") -> None:
        self._edges.append((from_idx, to_idx, edge_type))

    def get_node(self, idx: int) -> PyTopologyNode:
        return self._nodes[idx]

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def node_ids(self) -> list[int]:
        return list(range(len(self._nodes)))

    def is_acyclic(self) -> bool:
        """Check acyclicity via DFS."""
        n = len(self._nodes)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for f, t, _ in self._edges:
            adj[f].append(t)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n

        def dfs(u: int) -> bool:
            color[u] = GRAY
            for v in adj[u]:
                if color[v] == GRAY:
                    return False  # back edge = cycle
                if color[v] == WHITE and not dfs(v):
                    return False
            color[u] = BLACK
            return True

        return all(dfs(i) if color[i] == WHITE else True for i in range(n))

    def topological_sort(self) -> list[int]:
        """Kahn's algorithm. Returns empty list if cyclic."""
        n = len(self._nodes)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        in_degree = [0] * n
        for f, t, _ in self._edges:
            adj[f].append(t)
            in_degree[t] += 1

        queue = [i for i in range(n) if in_degree[i] == 0]
        result = []
        while queue:
            u = queue.pop(0)
            result.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return result if len(result) == n else []
