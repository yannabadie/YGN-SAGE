"""ModelRegistry — manages ModelCards with telemetry calibration.

Python reimplementation of sage-core/src/routing/model_registry.rs (438 LOC).

NOTE: This is sage.llm.model_registry.ModelRegistry (TOML-based catalog with
telemetry). NOT sage.providers.registry.ModelRegistry (runtime API discovery).
"""
from __future__ import annotations

import logging
from collections import deque

from sage.llm.model_card import CognitiveSystem, ModelCard

log = logging.getLogger(__name__)

_MAX_LATENCY_SAMPLES = 100


class TelemetryRecord:
    def __init__(self) -> None:
        self.quality_sum: float = 0.0
        self.cost_sum: float = 0.0
        self.count: int = 0
        self._latencies: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)

    def record(self, quality: float, cost: float) -> None:
        self.record_full(quality, cost, 0.0)

    def record_full(self, quality: float, cost: float, latency_ms: float) -> None:
        self.quality_sum += quality
        self.cost_sum += cost
        self.count += 1
        if latency_ms > 0.0:
            self._latencies.append(latency_ms)

    def avg_quality(self) -> float:
        return self.quality_sum / self.count if self.count > 0 else 0.0

    def latency_p95(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lats = sorted(self._latencies)
        # int() truncates toward zero; equivalent to Rust floor() for non-negative values
        idx = int((len(sorted_lats) - 1) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]


class ModelRegistry:
    def __init__(self) -> None:
        self._cards: dict[str, ModelCard] = {}
        self._telemetry: dict[str, TelemetryRecord] = {}

    def __len__(self) -> int:
        return len(self._cards)

    def is_empty(self) -> bool:
        return len(self._cards) == 0

    def register(self, card: ModelCard) -> None:
        self._cards[card.id] = card

    def unregister(self, id: str) -> None:
        self._cards.pop(id, None)

    def get(self, id: str) -> ModelCard | None:
        return self._cards.get(id)

    def list_ids(self) -> list[str]:
        return list(self._cards.keys())

    def all_models(self) -> list[ModelCard]:
        return list(self._cards.values())

    def select_for_system(self, system: CognitiveSystem | int) -> list[ModelCard]:
        s = CognitiveSystem(int(system))
        candidates = list(self._cards.values())
        candidates.sort(key=lambda c: c.affinity_for(s), reverse=True)
        return candidates

    def select_best_for_domain(self, domain: str, max_cost_usd: float = 0.0) -> ModelCard | None:
        candidates = list(self._cards.values())
        if max_cost_usd > 0:
            candidates = [c for c in candidates if c.estimate_cost(1000, 500) <= max_cost_usd]
        if not candidates:
            return None

        max_cost = max((c.estimate_cost(1000, 500) for c in candidates), default=0.001)
        max_cost = max(max_cost, 0.001)

        system = self._system_for_domain(domain)

        def score(c: ModelCard) -> float:
            ds = c.domain_score(domain)
            aff = self.calibrated_affinity(c.id, system)
            cost_norm = c.estimate_cost(1000, 500) / max_cost
            return ds * 0.6 + aff * 0.3 + (1.0 - cost_norm) * 0.1

        return max(candidates, key=score)

    @staticmethod
    def _system_for_domain(domain: str) -> CognitiveSystem:
        if domain in ("math", "formal"):
            return CognitiveSystem.S3
        if domain in ("code", "reasoning", "tool_use"):
            return CognitiveSystem.S2
        return CognitiveSystem.S1

    def record_telemetry(self, model_id: str, quality: float, cost: float) -> None:
        self.record_telemetry_full(model_id, quality, cost, 0.0)

    def record_telemetry_full(self, model_id: str, quality: float, cost: float, latency_ms: float) -> None:
        if model_id not in self._telemetry:
            self._telemetry[model_id] = TelemetryRecord()
        self._telemetry[model_id].record_full(quality, cost, latency_ms)
        log.debug("telemetry_recorded model=%s count=%d", model_id, self._telemetry[model_id].count)

    def observed_latency_p95(self, model_id: str) -> float:
        tr = self._telemetry.get(model_id)
        return tr.latency_p95() if tr else 0.0

    def calibrated_affinity(self, model_id: str, system: CognitiveSystem | int) -> float:
        s = CognitiveSystem(int(system))
        card = self._cards.get(model_id)
        card_affinity = card.affinity_for(s) if card else 0.0

        tr = self._telemetry.get(model_id)
        if not tr or tr.count == 0:
            return card_affinity

        w = min(tr.count / 50.0, 0.8)
        observed = tr.avg_quality()
        return (1.0 - w) * card_affinity + w * observed

    @classmethod
    def from_toml_file(cls, path: str) -> ModelRegistry:
        cards = ModelCard.load_from_file(path)
        reg = cls()
        for card in cards:
            reg.register(card)
        return reg

    def __repr__(self) -> str:
        return f"ModelRegistry(models={len(self._cards)})"
