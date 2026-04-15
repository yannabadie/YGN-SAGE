"""Policy verification for DAG execution — stub after training code removal.

The full PolicyVerifier validated topology constraints (fan-in, fan-out, budget).
This stub preserves the interface for DAGExecutor compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyViolation:
    """A policy violation found during verification."""
    rule: str
    message: str
    severity: str = "warning"


class PolicyVerifier:
    """Verify DAG execution policies (budget, fan-in, fan-out)."""

    def __init__(
        self,
        dag: Any,
        total_budget_usd: float = 10.0,
        max_fan_in: int = 10,
        max_fan_out: int = 10,
    ) -> None:
        self.dag = dag
        self.total_budget_usd = total_budget_usd
        self.max_fan_in = max_fan_in
        self.max_fan_out = max_fan_out

    def check_info_flow(self) -> list[PolicyViolation]:
        """Check that data does not flow from higher to lower security label."""
        violations: list[PolicyViolation] = []
        for nid in self.dag.node_ids:
            src = self.dag.get_node(nid)
            for succ_id in self.dag.successors(nid):
                dst = self.dag.get_node(succ_id)
                if src.security_label > dst.security_label:
                    violations.append(PolicyViolation(
                        rule="info_flow",
                        message=(
                            f"Edge {nid}->{succ_id}: security label downgrade "
                            f"({src.security_label.name}->{dst.security_label.name})"
                        ),
                        severity="error",
                    ))
        return violations

    def check_fan_limits(self) -> list[PolicyViolation]:
        """Check fan-in and fan-out limits for each node."""
        violations: list[PolicyViolation] = []
        for nid in self.dag.node_ids:
            if self.max_fan_out > 0:
                fan_out = len(self.dag.successors(nid))
                if fan_out > self.max_fan_out:
                    violations.append(PolicyViolation(
                        rule="fan_out",
                        message=f"Node {nid}: fan-out {fan_out} exceeds limit {self.max_fan_out}",
                        severity="error",
                    ))
            if self.max_fan_in > 0:
                fan_in = len(self.dag.predecessors(nid))
                if fan_in > self.max_fan_in:
                    violations.append(PolicyViolation(
                        rule="fan_in",
                        message=f"Node {nid}: fan-in {fan_in} exceeds limit {self.max_fan_in}",
                        severity="error",
                    ))
        return violations

    def verify_all(self) -> list[PolicyViolation]:
        """Verify all policies: info-flow, fan-in, fan-out."""
        return self.check_info_flow() + self.check_fan_limits()
