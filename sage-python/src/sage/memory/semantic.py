"""Semantic memory - entity graph built from extraction results.

Stores entities (as a set) and relationships (as a list of triples)
extracted by :class:`~sage.memory.memory_agent.MemoryAgent`. Provides
multi-hop neighbourhood queries and task-context generation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sage.memory.memory_agent import ExtractionResult


class SemanticMemory:
    """In-memory entity graph with neighbourhood search.

    Entities are stored as a set (deduplicated).  Relationships are stored
    as an append-only list of ``(subject, predicate, object)`` triples.
    An adjacency index accelerates neighbourhood queries.
    """

    def __init__(self) -> None:
        self._entities: set[str] = set()
        self._relations: list[tuple[str, str, str]] = []
        # Adjacency index: entity -> list of relation indices
        self._adj: dict[str, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_extraction(self, result: ExtractionResult) -> None:
        """Ingest entities and relationships from an extraction result."""
        for ent in result.entities:
            self._entities.add(ent)

        for rel in result.relationships:
            if len(rel) != 3:
                continue
            subj, pred, obj = rel[0], rel[1], rel[2]
            idx = len(self._relations)
            self._relations.append((subj, pred, obj))
            self._adj[subj].append(idx)
            self._adj[obj].append(idx)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def entity_count(self) -> int:
        """Return the number of unique entities."""
        return len(self._entities)

    def query_entities(
        self, entity: str, hops: int = 1
    ) -> list[tuple[str, str, str]]:
        """Return relations reachable within *hops* of *entity*.

        A BFS expands from the seed entity.  At each hop the direct
        neighbours (subjects/objects sharing a relation) are explored.
        """
        visited_entities: set[str] = set()
        frontier: set[str] = {entity}
        collected_indices: set[int] = set()

        for _ in range(hops):
            next_frontier: set[str] = set()
            for ent in frontier:
                if ent in visited_entities:
                    continue
                visited_entities.add(ent)
                for idx in self._adj.get(ent, []):
                    if idx not in collected_indices:
                        collected_indices.add(idx)
                        subj, _, obj = self._relations[idx]
                        # Add the *other* end to the next frontier
                        if subj not in visited_entities:
                            next_frontier.add(subj)
                        if obj not in visited_entities:
                            next_frontier.add(obj)
            frontier = next_frontier

        return [self._relations[i] for i in sorted(collected_indices)]

    def get_context_for(self, task: str) -> str:
        """Find entities mentioned in *task* and return their relations as text.

        Returns an empty string when no entities are matched.
        """
        if not self._entities:
            return ""

        # Find entities present in the task text
        mentioned = [ent for ent in self._entities if ent in task]
        if not mentioned:
            return ""

        # Gather 1-hop relations for each mentioned entity
        all_rels: list[tuple[str, str, str]] = []
        seen: set[int] = set()
        for ent in mentioned:
            for idx in self._adj.get(ent, []):
                if idx not in seen:
                    seen.add(idx)
                    all_rels.append(self._relations[idx])

        if not all_rels:
            return ""

        lines = [f"{s} --[{p}]--> {o}" for s, p, o in all_rels]
        return "\n".join(lines)
