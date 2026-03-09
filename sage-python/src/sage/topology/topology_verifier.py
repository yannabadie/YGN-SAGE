"""Graph-analysis topology verification -- prove DAG properties before execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger(__name__)


@dataclass
class TopologySpec:
    agents: list[str]
    edges: list[tuple[str, str]]  # (from, to) directed edges
    topology_type: str = "sequential"  # sequential, parallel, hierarchical, hybrid


@dataclass
class VerificationResult:
    terminates: bool = False
    is_dag: bool = False
    no_deadlock: bool = False
    proof: str = ""
    warnings: list[str] = field(default_factory=list)


class TopologyVerifier:
    """Verify topology properties using graph analysis (Kahn's algorithm)."""

    def __init__(self, max_depth: int = 30):
        self.max_depth = max_depth

    def verify(self, spec: TopologySpec) -> VerificationResult:
        result = VerificationResult()

        # 1. Check DAG (no cycles) via topological sort
        result.is_dag = self._is_dag(spec)
        result.terminates = result.is_dag  # DAGs always terminate

        # 2. Check for deadlocks (parallel: no shared dependencies)
        if spec.topology_type == "parallel":
            result.no_deadlock = self._check_no_deadlock(spec)
        else:
            result.no_deadlock = result.is_dag

        # 3. Check for disconnected agents
        connected: set[str] = set()
        for src, dst in spec.edges:
            connected.add(src)
            connected.add(dst)
        orphans = [a for a in spec.agents if a not in connected and len(spec.edges) > 0]
        if orphans:
            result.warnings.append(f"Disconnected agents: {orphans}")

        # 4. Check depth
        depth = self._max_chain_depth(spec)
        if depth > self.max_depth:
            result.warnings.append(f"Chain depth {depth} exceeds max {self.max_depth}")

        # 5. Generate proof string
        if result.is_dag and result.terminates:
            result.proof = (
                f"VERIFIED (graph analysis): DAG with {len(spec.agents)} agents, "
                f"{len(spec.edges)} edges, max depth {depth}. "
                f"Acyclic: yes. Terminates: yes. Method: Kahn's algorithm."
            )
        else:
            result.proof = "FAILED: Cycle detected, topology may not terminate."

        return result

    def _is_dag(self, spec: TopologySpec) -> bool:
        """Kahn's algorithm for cycle detection."""
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)
        for a in spec.agents:
            in_degree[a] = 0
        for src, dst in spec.edges:
            adj[src].append(dst)
            in_degree[dst] += 1

        queue = [a for a in spec.agents if in_degree[a] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(spec.agents)

    def _check_no_deadlock(self, spec: TopologySpec) -> bool:
        """Parallel topology: no agent waits on another."""
        return len(spec.edges) == 0 or self._is_dag(spec)

    def _max_chain_depth(self, spec: TopologySpec) -> int:
        """Longest path in DAG."""
        if not spec.edges:
            return 1
        adj: dict[str, list[str]] = defaultdict(list)
        for src, dst in spec.edges:
            adj[src].append(dst)

        memo: dict[str, int] = {}

        def dfs(node: str) -> int:
            if node in memo:
                return memo[node]
            children = adj.get(node, [])
            if not children:
                memo[node] = 1
                return 1
            memo[node] = 1 + max(dfs(c) for c in children)
            return memo[node]

        roots = [a for a in spec.agents if all(dst != a for _, dst in spec.edges)]
        if not roots:
            return len(spec.agents)
        return max(dfs(r) for r in roots)
