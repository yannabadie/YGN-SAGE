"""Evolutionary Topology Search: MAP-Elites on Agent DAG configurations."""
from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PATTERNS = ["vertical", "horizontal", "mesh", "coordinator", "pipeline"]
ROLES = ["planner", "coder", "reviewer", "researcher", "debugger", "tester", "analyst"]


@dataclass
class TopologyGenome:
    """Genome encoding an agent DAG topology."""
    nodes: list[str]
    edges: list[tuple[str, str]]
    pattern: str
    features: tuple[int, int] = (0, 0)  # Behavioral features for MAP-Elites
    metadata: dict[str, Any] = field(default_factory=dict)


class TopologyPopulation:
    """MAP-Elites grid for topology diversity."""

    def __init__(self, bins_per_dim: int = 10):
        self.bins = bins_per_dim
        self._grid: dict[tuple[int, int], tuple[TopologyGenome, float]] = {}

    def add(self, genome: TopologyGenome, score: float) -> bool:
        key = (
            min(genome.features[0], self.bins - 1),
            min(genome.features[1], self.bins - 1),
        )
        if key not in self._grid or self._grid[key][1] < score:
            self._grid[key] = (genome, score)
            return True
        return False

    def best(self) -> tuple[TopologyGenome, float] | None:
        if not self._grid:
            return None
        return max(self._grid.values(), key=lambda x: x[1])

    def sample(self, n: int = 1) -> list[TopologyGenome]:
        items = list(self._grid.values())
        if not items:
            return []
        chosen = random.choices(items, k=min(n, len(items)))
        return [g for g, _ in chosen]

    def size(self) -> int:
        return len(self._grid)


class TopologyEvolver:
    """Evolves agent DAG topologies using mutation operators."""

    def mutate_genome(self, genome: TopologyGenome) -> TopologyGenome:
        """Apply a random mutation to the topology genome."""
        nodes = list(genome.nodes)
        edges = list(genome.edges)
        pattern = genome.pattern

        op = random.choice(["add_node", "remove_node", "rewire", "change_pattern"])

        if op == "add_node" and len(nodes) < 8:
            available = [r for r in ROLES if r not in nodes]
            if available:
                new_role = random.choice(available)
                nodes.append(new_role)
                # Connect to a random existing node
                if nodes:
                    parent = random.choice(nodes[:-1]) if len(nodes) > 1 else nodes[0]
                    edges.append((parent, new_role))

        elif op == "remove_node" and len(nodes) > 1:
            victim = random.choice(nodes[1:])  # Never remove first node
            nodes.remove(victim)
            edges = [(a, b) for a, b in edges if a != victim and b != victim]

        elif op == "rewire" and edges:
            idx = random.randrange(len(edges))
            src, _ = edges[idx]
            targets = [n for n in nodes if n != src]
            if targets:
                tgt = random.choice(targets)
                edges[idx] = (src, tgt)

        elif op == "change_pattern":
            pattern = random.choice([p for p in PATTERNS if p != pattern])

        # Compute behavioral features: (node_count_bin, edge_density_bin)
        nc = min(9, len(nodes))
        max_edges = max(1, len(nodes) * (len(nodes) - 1))
        ed = min(9, int(10 * len(edges) / max_edges))

        return TopologyGenome(
            nodes=nodes, edges=edges, pattern=pattern,
            features=(nc, ed),
        )

    def crossover(self, a: TopologyGenome, b: TopologyGenome) -> TopologyGenome:
        """Combine two topologies."""
        # Take nodes from both, edges from the one with better diversity
        all_nodes = list(set(a.nodes + b.nodes))[:8]
        all_edges = list(set(a.edges + b.edges))
        # Filter edges to valid nodes
        valid_edges = [(s, t) for s, t in all_edges if s in all_nodes and t in all_nodes]
        pattern = random.choice([a.pattern, b.pattern])

        nc = min(9, len(all_nodes))
        max_e = max(1, len(all_nodes) * (len(all_nodes) - 1))
        ed = min(9, int(10 * len(valid_edges) / max_e))

        return TopologyGenome(
            nodes=all_nodes, edges=valid_edges, pattern=pattern,
            features=(nc, ed),
        )
