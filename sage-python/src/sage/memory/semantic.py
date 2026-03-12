"""Semantic memory - entity graph built from extraction results.

Stores entities (as a set) and relationships (as a list of triples)
extracted by :class:`~sage.memory.memory_agent.MemoryAgent`. Provides
multi-hop neighbourhood queries and task-context generation.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict, deque

from sage.memory.memory_agent import ExtractionResult


class SemanticMemory:
    """In-memory entity graph with neighbourhood search.

    Entities are stored as a set (deduplicated).  Relationships are stored
    as an append-only list of ``(subject, predicate, object)`` triples.
    An adjacency index accelerates neighbourhood queries.
    """

    def __init__(
        self,
        max_relations: int = 10_000,
        max_context_lines: int = 50,
        db_path: str | None = None,
    ) -> None:
        self.max_relations = max_relations
        self.max_context_lines = max_context_lines
        self.db_path = db_path
        self._entities: set[str] = set()
        self._relations: deque[tuple[str, str, str]] = deque()
        self._relations_set: set[tuple[str, str, str]] = set()
        # Adjacency index: entity -> list of relation indices (absolute)
        self._adj: dict[str, list[int]] = defaultdict(list)
        # Offset tracking: absolute index of the first element in _relations
        self._evicted_count: int = 0

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
            idx = self._evicted_count + len(self._relations)
            self._relations.append(triple)
            self._adj[subj].append(idx)
            self._adj[obj].append(idx)
            # Evict oldest if over capacity — O(k) lazy cleanup
            if self.max_relations > 0 and len(self._relations) > self.max_relations:
                oldest = self._relations.popleft()  # O(1)
                self._relations_set.discard(oldest)
                evicted_idx = self._evicted_count
                self._evicted_count += 1
                # Lazy adjacency cleanup: only touch the evicted triple's entities
                for entity in (oldest[0], oldest[2]):  # subject and object
                    if entity in self._adj:
                        self._adj[entity] = [
                            i for i in self._adj[entity] if i != evicted_idx
                        ]
                        if not self._adj[entity]:
                            del self._adj[entity]

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
                    rel_idx = idx - self._evicted_count
                    if rel_idx < 0 or rel_idx >= len(self._relations):
                        continue  # stale index
                    if idx not in collected_indices:
                        collected_indices.add(idx)
                        subj, _, obj = self._relations[rel_idx]
                        # Add the *other* end to the next frontier
                        if subj not in visited_entities:
                            next_frontier.add(subj)
                        if obj not in visited_entities:
                            next_frontier.add(obj)
            frontier = next_frontier

        return [
            self._relations[i - self._evicted_count]
            for i in sorted(collected_indices)
            if 0 <= i - self._evicted_count < len(self._relations)
        ]

    def get_context_for(self, task: str) -> str:
        """Find entities mentioned in *task* and return their relations as text.

        Returns an empty string when no entities are matched.
        """
        if not self._entities:
            return ""

        # Find entities present in the task text (word-boundary match)
        mentioned = [ent for ent in self._entities if re.search(r'\b' + re.escape(ent) + r'\b', task, re.IGNORECASE)]
        if not mentioned:
            return ""

        # Gather 1-hop relations for each mentioned entity
        all_rels: list[tuple[str, str, str]] = []
        seen: set[int] = set()
        for ent in mentioned:
            for idx in self._adj.get(ent, []):
                rel_idx = idx - self._evicted_count
                if rel_idx < 0 or rel_idx >= len(self._relations):
                    continue  # stale index
                if idx not in seen:
                    seen.add(idx)
                    all_rels.append(self._relations[rel_idx])

        if not all_rels:
            return ""

        lines = [f"{s} --[{p}]--> {o}" for s, p, o in all_rels]
        if self.max_context_lines > 0:
            lines = lines[:self.max_context_lines]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence (SQLite)
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist entities and relations to SQLite."""
        if not self.db_path:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entities (name TEXT PRIMARY KEY)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS relations "
                "(subj TEXT, pred TEXT, obj TEXT, PRIMARY KEY (subj, pred, obj))"
            )
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM relations")
            conn.executemany(
                "INSERT OR IGNORE INTO entities VALUES (?)",
                [(e,) for e in self._entities],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO relations VALUES (?, ?, ?)",
                self._relations,
            )
            conn.commit()
        finally:
            conn.close()

    def load(self) -> None:
        """Load entities and relations from SQLite."""
        if not self.db_path:
            return
        import os

        if not os.path.exists(self.db_path):
            return
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if tables exist
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "entities" not in tables:
                return
            for (name,) in conn.execute("SELECT name FROM entities"):
                self._entities.add(name)
            for subj, pred, obj in conn.execute(
                "SELECT subj, pred, obj FROM relations"
            ):
                triple = (subj, pred, obj)
                if triple not in self._relations_set:
                    self._relations_set.add(triple)
                    idx = self._evicted_count + len(self._relations)
                    self._relations.append(triple)
                    self._adj[subj].append(idx)
                    self._adj[obj].append(idx)
        finally:
            conn.close()
