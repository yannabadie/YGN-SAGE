"""Semantic memory - entity graph built from extraction results.

Stores entities (as a set) and relationships (as a list of triples)
extracted by :class:`~sage.memory.memory_agent.MemoryAgent`. Provides
multi-hop neighbourhood queries and task-context generation.
"""
from __future__ import annotations

from collections import defaultdict

from sage.memory.memory_agent import ExtractionResult


class SemanticMemory:
    """In-memory entity graph with neighbourhood search.

    Entities are stored as a set (deduplicated).  Relationships are stored
    as an append-only list of ``(subject, predicate, object)`` triples.
    An adjacency index accelerates neighbourhood queries.
    """

    def __init__(self, max_relations: int = 10_000, max_context_lines: int = 50) -> None:
        self.max_relations = max_relations
        self.max_context_lines = max_context_lines
        self._entities: set[str] = set()
        self._relations: list[tuple[str, str, str]] = []
        self._relations_set: set[tuple[str, str, str]] = set()
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
            triple = (subj, pred, obj)
            if triple in self._relations_set:
                continue  # Dedup
            self._relations_set.add(triple)
            idx = len(self._relations)
            self._relations.append(triple)
            self._adj[subj].append(idx)
            self._adj[obj].append(idx)
            # Evict oldest if over capacity
            if self.max_relations > 0 and len(self._relations) > self.max_relations:
                oldest = self._relations.pop(0)
                self._relations_set.discard(oldest)
                # Shift adjacency indices down by 1 (oldest was at index 0)
                new_adj: dict[str, list[int]] = defaultdict(list)
                for key, indices in self._adj.items():
                    new_adj[key] = [i - 1 for i in indices if i > 0]
                self._adj = new_adj

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
        if self.max_context_lines > 0:
            lines = lines[:self.max_context_lines]
        return "\n".join(lines)
