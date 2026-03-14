"""ModelAssigner — Python fallback for per-node model assignment.

Same algorithm as Rust sage_core.ModelAssigner. Used when sage_core
is not compiled. See spec for weight rationale (0.4/0.4/0.2).
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

WEIGHT_AFFINITY = 0.4
WEIGHT_DOMAIN = 0.4
WEIGHT_COST = 0.2
BUDGET_EPSILON = 0.01


class ModelAssigner:
    """Python fallback ModelAssigner — field-for-field compatible with Rust version."""

    def __init__(self, catalog: Any) -> None:
        """catalog: ModelRegistry (sage.llm.model_registry.ModelRegistry)."""
        self._catalog = catalog

    def assign_models(self, graph: Any, task_domain: str, budget_usd: float) -> int:
        """Assign model_id to every node. Modifies graph in-place. Returns count assigned."""
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
            caps = getattr(node, "required_capabilities", [])
            needs_tools = "tools" in caps
            needs_json = "json" in caps
            node_budget = min(getattr(node, "max_cost_usd", remaining), remaining)
            system = getattr(node, "system", 1)

            best_id, best_score = None, float("-inf")
            for card in cards:
                if needs_tools and not card.supports_tools:
                    continue
                if needs_json and not card.supports_json_mode:
                    continue
                est = card.estimate_cost(1000, 500)
                if est > node_budget:
                    continue
                aff = self._catalog.calibrated_affinity(card.id, system)
                dom = card.domain_score(task_domain)
                cost_n = est / max_cost
                score = WEIGHT_AFFINITY * aff + WEIGHT_DOMAIN * dom + WEIGHT_COST * (1.0 - cost_n)
                if score > best_score:
                    best_score = score
                    best_id = card.id

            if best_id is not None:
                graph.set_node_model_id(idx, best_id)
                est = next((c.estimate_cost(1000, 500) for c in cards if c.id == best_id), 0)
                remaining -= est
                assigned += 1
            else:
                log.warning("node %d (%s): no candidate, keeping existing model_id",
                           idx, getattr(node, "role", "?"))
        return assigned

    def assign_single_node(self, graph: Any, node_idx: int, task_domain: str, budget_usd: float) -> str:
        """Assign a single node. Returns model_id or raises ValueError."""
        node = graph.get_node(node_idx) if hasattr(graph, 'get_node') else None
        if node is None:
            raise ValueError(f"Node index {node_idx} out of range")
        cards = self._catalog.all_models()
        if not cards:
            raise ValueError("No models in catalog")
        max_cost = max((c.estimate_cost(1000, 500) for c in cards), default=0.001)
        caps = getattr(node, "required_capabilities", [])
        needs_tools = "tools" in caps
        needs_json = "json" in caps
        system = getattr(node, "system", 1)

        best_id, best_score = None, float("-inf")
        for card in cards:
            if needs_tools and not card.supports_tools:
                continue
            if needs_json and not card.supports_json_mode:
                continue
            if card.estimate_cost(1000, 500) > budget_usd:
                continue
            aff = self._catalog.calibrated_affinity(card.id, system)
            dom = card.domain_score(task_domain)
            cost_n = card.estimate_cost(1000, 500) / max_cost
            score = WEIGHT_AFFINITY * aff + WEIGHT_DOMAIN * dom + WEIGHT_COST * (1.0 - cost_n)
            if score > best_score:
                best_score = score
                best_id = card.id

        if best_id is None:
            raise ValueError(f"No candidate for node {node_idx}")
        graph.set_node_model_id(node_idx, best_id)
        return best_id
