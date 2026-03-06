"""TaskDAG — directed acyclic graph of TaskNodes with topological scheduling.

Provides: add_node, add_edge, topological_sort (Kahn's algorithm),
cycle detection, I/O compatibility validation, and ready_nodes scheduler.
"""
from __future__ import annotations

from collections import defaultdict, deque

from sage.contracts.task_node import TaskNode


class CycleError(Exception):
    """Raised when the DAG contains a cycle."""


class TaskDAG:
    """A DAG of TaskNodes with dependency edges."""

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}
        self._successors: dict[str, list[str]] = defaultdict(list)
        self._predecessors: dict[str, list[str]] = defaultdict(list)

    # -- Construction -------------------------------------------------------

    def add_node(self, node: TaskNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists")
        self._nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        for nid in (from_id, to_id):
            if nid not in self._nodes:
                raise KeyError(f"Unknown node: '{nid}'")
        self._successors[from_id].append(to_id)
        self._predecessors[to_id].append(from_id)

    # -- Queries ------------------------------------------------------------

    def get_node(self, node_id: str) -> TaskNode:
        return self._nodes[node_id]

    def successors(self, node_id: str) -> list[str]:
        return list(self._successors[node_id])

    def predecessors(self, node_id: str) -> list[str]:
        return list(self._predecessors[node_id])

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes)

    # -- Topological sort (Kahn's algorithm) --------------------------------

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order. Raises CycleError if cycle exists."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for succ in self._successors[nid]:
                in_degree[succ] += 1

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._successors[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self._nodes):
            raise CycleError("DAG contains a cycle")

        return order

    # -- I/O compatibility validation ---------------------------------------

    def validate_io_compatibility(self) -> list[str]:
        """Check that for each edge A->B, B's input fields are in A's output."""
        issues: list[str] = []
        for from_id, succs in self._successors.items():
            src = self._nodes[from_id]
            for to_id in succs:
                dst = self._nodes[to_id]
                for field_name in dst.input_schema.fields:
                    if field_name not in src.output_schema.fields:
                        issues.append(
                            f"Edge {from_id}->{to_id}: "
                            f"'{field_name}' required by {to_id} "
                            f"but not produced by {from_id}"
                        )
        return issues

    # -- Scheduling ---------------------------------------------------------

    def ready_nodes(self, completed: set[str]) -> list[str]:
        """Return node IDs whose predecessors are all completed."""
        ready: list[str] = []
        for nid in self._nodes:
            if nid in completed:
                continue
            preds = self._predecessors[nid]
            if all(p in completed for p in preds):
                ready.append(nid)
        return ready
