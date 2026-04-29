"""ModelAssigner — Python fallback for per-node model assignment.

Same algorithm as Rust sage_core.ModelAssigner. Used when sage_core
is not compiled. See spec for weight rationale (0.4/0.4/0.2).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

WEIGHT_AFFINITY = 0.4
WEIGHT_DOMAIN = 0.4
WEIGHT_COST = 0.2
BUDGET_EPSILON = 0.01
_LOG_TOP3_ENV = "SAGE_ASSIGNER_LOG_TOP3"


@dataclass(frozen=True, slots=True)
class _CandidateScore:
    model_id: str
    score: float
    affinity: float
    domain: float
    cost_norm: float
    hint_bonus: float
    diversity_penalty: float
    est_cost: float
    source: str
    reason_code: str


class ModelAssigner:
    """Python fallback ModelAssigner — field-for-field compatible with Rust version."""

    def __init__(self, catalog: Any) -> None:
        """catalog: ModelRegistry (sage.llm.model_registry.ModelRegistry)."""
        self._catalog = catalog

    def assign_models(
        self,
        graph: Any,
        task_domain: str,
        budget_usd: float,
        hints: Any = None,
        task_system: int | None = None,
    ) -> int:
        """Assign model_id to every node. Modifies graph in-place. Returns count assigned."""
        del hints, task_system
        node_count = graph.node_count()
        remaining = budget_usd
        assigned = 0
        cards = self._catalog.all_models()
        if not cards:
            log.warning("ModelAssigner: no models in catalog")
            return 0
        max_cost = max((c.estimate_cost(1000, 500) for c in cards), default=0.001)

        for idx in range(node_count):
            if remaining < BUDGET_EPSILON:
                log.warning("budget_exhausted_node_%d: %d nodes remaining", idx, node_count - idx)
                break
            node = graph.get_node(idx) if hasattr(graph, 'get_node') else None
            if node is None:
                continue
            node_budget = min(getattr(node, "max_cost_usd", remaining), remaining)

            candidates = self._score_candidates(
                cards=cards,
                node=node,
                task_domain=task_domain,
                node_budget=node_budget,
                max_cost=max_cost,
            )

            if candidates:
                best = candidates[0]
                graph.set_node_model_id(idx, best.model_id)
                self._log_top_candidates(idx, candidates)
                remaining -= best.est_cost
                assigned += 1
            else:
                log.warning("node %d (%s): no candidate, keeping existing model_id",
                           idx, getattr(node, "role", "?"))
        return assigned

    def assign_single_node(
        self,
        graph: Any,
        node_idx: int,
        task_domain: str,
        budget_usd: float,
        exclude_model_ids: list[str] | None = None,
        task_system: int | None = None,
    ) -> str:
        """Assign a single node. Returns model_id or raises ValueError.

        Parameters
        ----------
        exclude_model_ids : list[str], optional
            Model IDs to exclude from selection (e.g. for FrugalGPT cascade
            retry — exclude the model that produced low-quality output).
        task_system : int, optional
            Overall cognitive tier of the task (1/2/3). Accepted for
            signature compatibility with the Rust ModelAssigner (F7
            role-aware tier promotion). The pure-Python fallback does not
            implement the floor logic — it scores per-node only — but
            taking the param here lets callers forward ctx.system without
            a TypeError dance. When this fallback is on the hot path
            (Rust unavailable), the system already runs in degraded mode.
        """
        del task_system  # Documented above: param accepted, not used.
        node = graph.get_node(node_idx) if hasattr(graph, 'get_node') else None
        if node is None:
            raise ValueError(f"Node index {node_idx} out of range")
        cards = self._catalog.all_models()
        if not cards:
            raise ValueError("No models in catalog")
        max_cost = max((c.estimate_cost(1000, 500) for c in cards), default=0.001)
        _excluded = set(exclude_model_ids) if exclude_model_ids else set()

        candidates = self._score_candidates(
            cards=cards,
            node=node,
            task_domain=task_domain,
            node_budget=budget_usd,
            max_cost=max_cost,
            exclude_model_ids=_excluded,
        )
        if not candidates:
            raise ValueError(f"No candidate for node {node_idx}")
        best = candidates[0]
        graph.set_node_model_id(node_idx, best.model_id)
        self._log_top_candidates(node_idx, candidates)
        return best.model_id

    def _score_candidates(
        self,
        *,
        cards: list[Any],
        node: Any,
        task_domain: str,
        node_budget: float,
        max_cost: float,
        exclude_model_ids: set[str] | None = None,
    ) -> list[_CandidateScore]:
        caps = getattr(node, "required_capabilities", [])
        needs_tools = "tools" in caps
        needs_json = "json" in caps
        system = getattr(node, "system", 1)
        excluded = exclude_model_ids or set()
        candidates: list[_CandidateScore] = []

        for card in cards:
            if card.id in excluded:
                continue
            if needs_tools and not card.supports_tools:
                continue
            if needs_json and not card.supports_json_mode:
                continue
            est = card.estimate_cost(1000, 500)
            if est > node_budget:
                continue
            affinity_raw = _float_or_nan(
                self._catalog.calibrated_affinity(card.id, system)
            )
            domain_raw = _float_or_nan(card.domain_score(task_domain))
            cost_norm_raw = _float_or_nan(est / max(max_cost, 0.001))
            affinity = _finite_or_zero(affinity_raw)
            domain = _finite_or_zero(domain_raw)
            cost_norm = _finite_or_zero(cost_norm_raw)
            hint_bonus = 0.0
            diversity_penalty = 0.0
            score_raw = (
                WEIGHT_AFFINITY * affinity_raw
                + WEIGHT_DOMAIN * domain_raw
                + WEIGHT_COST * (1.0 - cost_norm_raw)
                + hint_bonus
                - diversity_penalty
            )
            reason_code = "ok"
            if not math.isfinite(score_raw):
                score = 0.0
                reason_code = "non_finite_score"
            else:
                score = score_raw
            candidates.append(
                _CandidateScore(
                    model_id=card.id,
                    score=score,
                    affinity=affinity,
                    domain=domain,
                    cost_norm=cost_norm,
                    hint_bonus=hint_bonus,
                    diversity_penalty=diversity_penalty,
                    est_cost=est,
                    source="python_fallback",
                    reason_code=reason_code,
                )
            )

        candidates.sort(key=lambda item: (-item.score, item.model_id))
        return candidates

    def _log_top_candidates(
        self,
        node_idx: int,
        candidates: list[_CandidateScore],
    ) -> None:
        if os.environ.get(_LOG_TOP3_ENV) != "1":
            return
        for rank, candidate in enumerate(candidates[:3], start=1):
            log.info(
                "model_assigner.candidates node_id=%d rank=%d model=%s "
                "source=%s reason_code=%s "
                "score=%.6f affinity=%.6f domain=%.6f cost_norm=%.6f "
                "hint_bonus=%.6f diversity_penalty=%.6f",
                node_idx,
                rank,
                candidate.model_id,
                candidate.source,
                candidate.reason_code,
                candidate.score,
                candidate.affinity,
                candidate.domain,
                candidate.cost_norm,
                candidate.hint_bonus,
                candidate.diversity_penalty,
            )


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_zero(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return value
