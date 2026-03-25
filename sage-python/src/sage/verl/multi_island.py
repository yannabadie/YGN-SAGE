"""Multi-island evolutionary topology search — HyEvo-inspired.

Maintains K separate island populations, each exploring different
regions of the topology space. Periodic ring migration transfers
elite solutions between islands to spread superior patterns.

Behavior descriptors (per-island archiving):
  - node_count: total nodes in topology
  - llm_ratio: fraction of LLM nodes
  - code_ratio: fraction of code nodes
  - provider_diversity: distinct providers used

Reference: HyEvo (arXiv 2603.19639) Section 3.3 — Multi-Island Evolution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("multi_island")


@dataclass
class IslandEntry:
    """An entry in an island's archive cell."""
    yaml: str
    score: float
    cost: float = 0.0
    latency_ms: float = 0.0
    descriptor: tuple = ()
    metadata: dict = field(default_factory=dict)


@dataclass
class Island:
    """A single island in the multi-island system."""
    island_id: int
    archive: dict[tuple, IslandEntry] = field(default_factory=dict)
    generation: int = 0

    def insert(self, descriptor: tuple, entry: IslandEntry) -> bool:
        """Insert entry if cell is empty or new entry is better."""
        key = self._discretize(descriptor)
        existing = self.archive.get(key)
        if existing is None or entry.score > existing.score:
            entry.descriptor = key
            self.archive[key] = entry
            return True
        return False

    def best_entry(self) -> IslandEntry | None:
        """Return the highest-scoring entry across all cells."""
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda e: e.score)

    def random_entry(self) -> IslandEntry | None:
        """Return a random entry for parent selection."""
        import random
        if not self.archive:
            return None
        return random.choice(list(self.archive.values()))

    def cell_count(self) -> int:
        return len(self.archive)

    @staticmethod
    def _discretize(descriptor: tuple) -> tuple:
        """Discretize continuous behavior descriptor into grid cells.

        descriptor = (node_count, llm_ratio, code_ratio, provider_diversity)
        """
        if len(descriptor) < 4:
            return descriptor

        node_count, llm_ratio, code_ratio, provider_div = descriptor
        return (
            min(node_count, 10),           # cap at 10 nodes
            round(llm_ratio * 4) / 4,      # 0.0, 0.25, 0.5, 0.75, 1.0
            round(code_ratio * 4) / 4,     # 0.0, 0.25, 0.5, 0.75, 1.0
            min(provider_div, 5),           # cap at 5 providers
        )


class MultiIslandEvolver:
    """HyEvo multi-island evolutionary topology search.

    K=2 islands by default (matching HyEvo paper), with ring migration
    every Δ_mig=15 generations.

    Usage:
        evolver = MultiIslandEvolver(k=2, migration_interval=15)
        for generation in range(n_iter):
            for island_id in range(evolver.k):
                parent = evolver.select_parent(island_id)
                # ... reflect-then-generate new candidate ...
                evolver.insert(island_id, descriptor, entry)
            evolver.maybe_migrate()  # ring migration if interval reached
    """

    def __init__(self, k: int = 2, migration_interval: int = 15):
        self.k = k
        self.migration_interval = migration_interval
        self.islands = [Island(island_id=i) for i in range(k)]
        self._global_generation = 0

    def select_parent(
        self,
        island_id: int,
        exploration_ratio: float = 0.3,
    ) -> IslandEntry | None:
        """Select a parent from the island.

        HyEvo parent selection: ρ_exp=0.3 explore (random), ρ_ploit=0.5 exploit (best).
        """
        import random

        island = self.islands[island_id]
        if not island.archive:
            return None

        r = random.random()
        if r < exploration_ratio:
            # Explore: random entry
            return island.random_entry()
        else:
            # Exploit: best entry
            return island.best_entry()

    def insert(
        self,
        island_id: int,
        descriptor: tuple,
        entry: IslandEntry,
    ) -> bool:
        """Insert a new entry into the specified island."""
        return self.islands[island_id].insert(descriptor, entry)

    def maybe_migrate(self) -> int:
        """Ring migration: transfer top elites between neighboring islands.

        Called after each generation. Only performs migration at intervals.
        Returns number of entries migrated.
        """
        self._global_generation += 1
        if self._global_generation % self.migration_interval != 0:
            return 0

        migrated = 0
        for i in range(self.k):
            src = self.islands[i]
            dst = self.islands[(i + 1) % self.k]
            best = src.best_entry()
            if best is not None:
                if dst.insert(best.descriptor, IslandEntry(
                    yaml=best.yaml,
                    score=best.score,
                    cost=best.cost,
                    latency_ms=best.latency_ms,
                    metadata={**best.metadata, "migrated_from": i},
                )):
                    migrated += 1

        if migrated > 0:
            log.info(
                "Multi-island migration (gen=%d): %d entries migrated across %d islands",
                self._global_generation, migrated, self.k,
            )
        return migrated

    def global_best(self) -> IslandEntry | None:
        """Return the best entry across all islands."""
        best = None
        for island in self.islands:
            entry = island.best_entry()
            if entry and (best is None or entry.score > best.score):
                best = entry
        return best

    def total_cells(self) -> int:
        return sum(island.cell_count() for island in self.islands)

    def stats(self) -> dict:
        """Return multi-island statistics."""
        return {
            "k": self.k,
            "generation": self._global_generation,
            "total_cells": self.total_cells(),
            "per_island": [
                {
                    "island_id": island.island_id,
                    "cells": island.cell_count(),
                    "best_score": (island.best_entry().score if island.best_entry() else 0.0),
                }
                for island in self.islands
            ],
        }
