"""Evolution ablation: measure value of evolutionary topology search.

3 configurations:
  1. no_evolution — fixed template topologies only (no MAP-Elites, no CMA-ME)
  2. random_mutation — random mutations without fitness-guided selection
  3. full_evolution — MAP-Elites + CMA-ME + MCTS (full 6-path engine)

Measures: topology diversity, task pass rate, quality delta.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


@dataclass
class EvolutionAblationResult:
    config: str
    tasks_run: int = 0
    tasks_passed: int = 0
    unique_topologies: int = 0
    avg_quality: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.tasks_passed / self.tasks_run if self.tasks_run > 0 else 0.0
