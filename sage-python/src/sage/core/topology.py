"""Rust TopologyEngine Python wrapper (Phase 8 boundary)."""

from __future__ import annotations

from typing import Any

import sage_core


class RustTopologyEngine:
    """Thin façade over ``sage_core.TopologyEngine``."""

    def __init__(self) -> None:
        self._inner = sage_core.TopologyEngine()

    def generate(
        self,
        task: str,
        system: int,
        exploration_budget: float = 0.5,
    ) -> Any:
        return self._inner.generate(task, None, system, exploration_budget)

    def generate_with_options(
        self,
        task: str,
        system: int,
        *,
        allow_smmu: bool = True,
        allow_archive: bool = True,
        allow_mutation: bool = True,
        allow_mcts: bool = True,
        allow_template: bool = True,
    ) -> Any:
        return self._inner.generate_with_options(
            task, None, system, 0.5,
            allow_smmu=allow_smmu,
            allow_archive=allow_archive,
            allow_mutation=allow_mutation,
            allow_mcts=allow_mcts,
            allow_template=allow_template,
        )

    def save_state(self, path: str) -> None:
        self._inner.save_state(path)
