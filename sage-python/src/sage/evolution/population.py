"""MAP-Elites population for managing diverse solution candidates.

Inspired by AlphaEvolve's population database: maintains a grid of
solutions binned by behavioral characteristics (features), keeping
the best-scoring individual per bin.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    """A single candidate solution in the population."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    code: str = ""
    score: float = 0.0
    features: tuple[int, ...] = ()  # Behavioral descriptor for MAP-Elites bin
    metadata: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_id: str | None = None


class Population:
    """MAP-Elites population: diversity-preserving archive of solutions.

    Each individual is placed in a grid cell defined by its feature
    descriptor. Only the highest-scoring individual per cell is kept.
    """

    def __init__(self, feature_dims: int = 2, bins_per_dim: int = 10):
        self.feature_dims = feature_dims
        self.bins_per_dim = bins_per_dim
        self._archive: dict[tuple[int, ...], Individual] = {}

    def add(self, individual: Individual) -> bool:
        """Add an individual to the archive. Returns True if it was inserted."""
        features = self._clamp_features(individual.features)
        individual.features = features

        existing = self._archive.get(features)
        if existing is None or individual.score > existing.score:
            self._archive[features] = individual
            return True
        return False

    def get(self, features: tuple[int, ...]) -> Individual | None:
        """Get the individual in a specific cell."""
        return self._archive.get(self._clamp_features(features))

    def best(self, n: int = 1) -> list[Individual]:
        """Get the top-N individuals by score."""
        sorted_inds = sorted(self._archive.values(), key=lambda x: x.score, reverse=True)
        return sorted_inds[:n]

    def sample(self, n: int = 1) -> list[Individual]:
        """Sample N individuals from the archive (uniform over occupied cells)."""
        import random
        inds = list(self._archive.values())
        if not inds:
            return []
        return random.choices(inds, k=min(n, len(inds)))

    def size(self) -> int:
        """Number of occupied cells in the archive."""
        return len(self._archive)

    def coverage(self) -> float:
        """Fraction of cells occupied."""
        total_cells = self.bins_per_dim ** self.feature_dims
        return self.size() / total_cells if total_cells > 0 else 0.0

    def all_individuals(self) -> list[Individual]:
        """Get all individuals in the archive."""
        return list(self._archive.values())

    def _clamp_features(self, features: tuple[int, ...]) -> tuple[int, ...]:
        """Clamp features to valid bin indices."""
        return tuple(
            max(0, min(self.bins_per_dim - 1, f))
            for f in features
        )
