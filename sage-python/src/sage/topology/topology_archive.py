"""Experience-based Quality-Diversity topology archive -- learns best topologies per task type."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from sage.topology.z3_topology import TopologySpec

log = logging.getLogger(__name__)


@dataclass
class TopologyRecord:
    spec: TopologySpec
    score: float
    task_type: str
    uses: int = 0


class TopologyArchive:
    """Learns and evolves optimal topologies per task type."""

    def __init__(self, max_records_per_type: int = 50):
        self._records: dict[str, list[TopologyRecord]] = defaultdict(list)
        self._max = max_records_per_type

    def record(self, spec: TopologySpec, score: float, task_type: str) -> None:
        records = self._records[task_type]
        records.append(TopologyRecord(spec=spec, score=score, task_type=task_type))
        records.sort(key=lambda r: r.score, reverse=True)
        if len(records) > self._max:
            self._records[task_type] = records[:self._max]

    def recommend(self, task_type: str) -> TopologySpec | None:
        records = self._records.get(task_type, [])
        if not records:
            return None
        best = records[0]
        best.uses += 1
        return best.spec

    def count(self) -> int:
        return sum(len(v) for v in self._records.values())

    def task_types(self) -> list[str]:
        return list(self._records.keys())

    def stats(self) -> dict:
        return {
            task_type: {
                "count": len(records),
                "best_score": records[0].score if records else 0.0,
                "avg_score": sum(r.score for r in records) / len(records) if records else 0.0,
            }
            for task_type, records in self._records.items()
        }
