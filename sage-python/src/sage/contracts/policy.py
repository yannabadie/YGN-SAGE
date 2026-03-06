"""PolicyVerifier — checks info-flow, budget, and structural constraints on a TaskDAG.

Rules:
1. Info-flow: data cannot flow from higher security label to lower (no HIGH→LOW)
2. Budget: sum of per-node max_cost_usd must not exceed total budget
3. Fan-in/fan-out: no single node should have excessive in/out degree
"""
from __future__ import annotations

from dataclasses import dataclass

from sage.contracts.dag import TaskDAG


@dataclass
class PolicyViolation:
    """A single policy violation."""

    rule: str
    node_id: str
    message: str


class PolicyVerifier:
    """Verifies structural and security policies on a TaskDAG."""

    def __init__(
        self,
        dag: TaskDAG,
        *,
        total_budget_usd: float = 0.0,
        max_fan_in: int = 0,
        max_fan_out: int = 0,
    ) -> None:
        self.dag = dag
        self.total_budget_usd = total_budget_usd
        self.max_fan_in = max_fan_in
        self.max_fan_out = max_fan_out

    # -- Info-flow ----------------------------------------------------------

    def check_info_flow(self) -> list[PolicyViolation]:
        """No data flow from higher security label to lower."""
        violations: list[PolicyViolation] = []
        for nid in self.dag.node_ids:
            src = self.dag.get_node(nid)
            for succ_id in self.dag.successors(nid):
                dst = self.dag.get_node(succ_id)
                if src.security_label > dst.security_label:
                    violations.append(PolicyViolation(
                        rule="info_flow",
                        node_id=nid,
                        message=(
                            f"Info-flow violation: {nid} "
                            f"({src.security_label.name}) → {succ_id} "
                            f"({dst.security_label.name})"
                        ),
                    ))
        return violations

    # -- Budget -------------------------------------------------------------

    def check_budget(self) -> list[PolicyViolation]:
        """Sum of per-node max_cost_usd must not exceed total budget."""
        if self.total_budget_usd <= 0:
            return []

        total = sum(
            self.dag.get_node(nid).budget.max_cost_usd
            for nid in self.dag.node_ids
            if self.dag.get_node(nid).budget.max_cost_usd > 0
        )

        if total > self.total_budget_usd:
            return [PolicyViolation(
                rule="budget",
                node_id="*",
                message=(
                    f"Budget exceeded: ${total:.4f} > "
                    f"${self.total_budget_usd:.4f}"
                ),
            )]
        return []

    # -- Fan-in / fan-out ---------------------------------------------------

    def check_fan_limits(self) -> list[PolicyViolation]:
        """Check fan-in and fan-out limits."""
        violations: list[PolicyViolation] = []

        for nid in self.dag.node_ids:
            if self.max_fan_out > 0:
                out_degree = len(self.dag.successors(nid))
                if out_degree > self.max_fan_out:
                    violations.append(PolicyViolation(
                        rule="fan_out",
                        node_id=nid,
                        message=(
                            f"Fan-out exceeded: {nid} has {out_degree} "
                            f"successors (limit {self.max_fan_out})"
                        ),
                    ))

            if self.max_fan_in > 0:
                in_degree = len(self.dag.predecessors(nid))
                if in_degree > self.max_fan_in:
                    violations.append(PolicyViolation(
                        rule="fan_in",
                        node_id=nid,
                        message=(
                            f"Fan-in exceeded: {nid} has {in_degree} "
                            f"predecessors (limit {self.max_fan_in})"
                        ),
                    ))

        return violations

    # -- Combined -----------------------------------------------------------

    def verify_all(self) -> list[PolicyViolation]:
        """Run all policy checks and return combined violations."""
        violations: list[PolicyViolation] = []
        violations.extend(self.check_info_flow())
        violations.extend(self.check_budget())
        violations.extend(self.check_fan_limits())
        return violations
