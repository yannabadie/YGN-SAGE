"""Causal Memory — entity-relation graph with directed causal edges.

Extends the basic entity graph with:
- Causal edges (A caused/enabled B) with BFS chain traversal
- Temporal ordering (insertion order preserved)
- Ancestor/descendant queries for provenance tracking
- Task-context generation from entity mentions

Inspired by AMA-Bench (2602.22769) causal memory baselines.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalEdge:
    """A directed causal edge between two entities."""

    source: str
    target: str
    cause_type: str = "caused"  # caused, enabled, triggered, inhibited
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalMemory:
    """Entity-relation graph with causal edges and temporal ordering.

    Parameters
    ----------
    max_entities:
        Maximum number of entities before oldest are evicted. 0 = unlimited.
    max_context_lines:
        Maximum lines returned by get_context_for(). 0 = unlimited.
    """

    def __init__(
        self,
        max_entities: int = 0,
        max_context_lines: int = 50,
    ) -> None:
        self.max_entities = max_entities
        self.max_context_lines = max_context_lines
        self._entities: dict[str, dict[str, Any]] = {}  # name -> metadata
        self._entity_order: list[str] = []  # insertion order
        self._relations: list[tuple[str, str, str]] = []
        self._adj: dict[str, list[int]] = defaultdict(list)

        # Causal graph (separate from semantic relations)
        self._causal_edges: list[CausalEdge] = []
        self._causal_fwd: dict[str, list[int]] = defaultdict(list)  # source -> edge indices
        self._causal_bwd: dict[str, list[int]] = defaultdict(list)  # target -> edge indices

    # -- Entities -----------------------------------------------------------

    def add_entity(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        if name not in self._entities:
            self._entities[name] = metadata or {}
            self._entity_order.append(name)
            # Evict oldest entity if over capacity
            if self.max_entities > 0 and len(self._entities) > self.max_entities:
                oldest = self._entity_order.pop(0)
                self._entities.pop(oldest, None)

    def has_entity(self, name: str) -> bool:
        return name in self._entities

    def entity_count(self) -> int:
        return len(self._entities)

    # -- Relations ----------------------------------------------------------

    def add_relation(self, subject: str, predicate: str, obj: str) -> None:
        idx = len(self._relations)
        self._relations.append((subject, predicate, obj))
        self._adj[subject].append(idx)
        self._adj[obj].append(idx)

    def get_relations(self, entity: str) -> list[tuple[str, str, str]]:
        return [self._relations[i] for i in self._adj.get(entity, [])]

    # -- Causal edges -------------------------------------------------------

    def add_causal_edge(
        self,
        source: str,
        target: str,
        cause_type: str = "caused",
        **metadata: Any,
    ) -> CausalEdge:
        edge = CausalEdge(source=source, target=target, cause_type=cause_type, metadata=metadata)
        idx = len(self._causal_edges)
        self._causal_edges.append(edge)
        self._causal_fwd[source].append(idx)
        self._causal_bwd[target].append(idx)
        return edge

    def get_causal_chain(self, entity: str) -> list[str]:
        """BFS forward from entity through causal edges."""
        visited: list[str] = []
        seen: set[str] = set()
        queue: deque[str] = deque([entity])

        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            visited.append(current)

            for idx in self._causal_fwd.get(current, []):
                edge = self._causal_edges[idx]
                if edge.target not in seen:
                    queue.append(edge.target)

        return visited

    def get_causal_ancestors(self, entity: str) -> list[str]:
        """BFS backward from entity through causal edges (excludes self)."""
        ancestors: list[str] = []
        seen: set[str] = {entity}
        queue: deque[str] = deque([entity])

        while queue:
            current = queue.popleft()
            for idx in self._causal_bwd.get(current, []):
                edge = self._causal_edges[idx]
                if edge.source not in seen:
                    seen.add(edge.source)
                    ancestors.append(edge.source)
                    queue.append(edge.source)

        return ancestors

    # -- Temporal ordering --------------------------------------------------

    def temporal_order(self) -> list[str]:
        """Return entities in insertion order."""
        return list(self._entity_order)

    # -- Context generation -------------------------------------------------

    def get_context_for(self, task: str) -> str:
        """Find entities mentioned in task and return relations + causal info."""
        if not self._entities:
            return ""

        mentioned = [ent for ent in self._entities if ent in task]
        if not mentioned:
            return ""

        lines: list[str] = []
        seen_rels: set[int] = set()

        for ent in mentioned:
            # Semantic relations
            for idx in self._adj.get(ent, []):
                if idx not in seen_rels:
                    seen_rels.add(idx)
                    s, p, o = self._relations[idx]
                    lines.append(f"{s} --[{p}]--> {o}")

            # Causal edges (forward)
            for idx in self._causal_fwd.get(ent, []):
                edge = self._causal_edges[idx]
                lines.append(f"{edge.source} ==[{edge.cause_type}]==> {edge.target}")

            # Causal edges (backward)
            for idx in self._causal_bwd.get(ent, []):
                edge = self._causal_edges[idx]
                lines.append(f"{edge.source} ==[{edge.cause_type}]==> {edge.target}")

        if not lines:
            return ""

        # Deduplicate while preserving order
        seen_lines: set[str] = set()
        unique: list[str] = []
        for line in lines:
            if line not in seen_lines:
                seen_lines.add(line)
                unique.append(line)

        if self.max_context_lines > 0:
            unique = unique[:self.max_context_lines]

        return "\n".join(unique)
