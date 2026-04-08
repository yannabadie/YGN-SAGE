"""Execution Trace — structured per-run observability for every pipeline decision.

Captures what happened, why, and at what cost — from routing to final output.
Designed to be the single source of truth for debugging, benchmarking, and evolution.

Usage:
    trace = ExecutionTrace(task_id="bench_42")
    trace.record_routing(system=2, confidence=0.98, method="knn")
    trace.record_node(idx=0, role="actor", model="gpt-5.4", ...)
    trace.finalize()
    print(trace.to_dict())  # Full structured trace
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeTrace:
    """Per-node execution record."""
    idx: int = 0
    role: str = ""
    model_id: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    output_len: int = 0
    error: str = ""
    quality_score: float | None = None
    controller_action: str = ""  # continue/upgrade/prune/reroute/gate


@dataclass
class ExecutionTrace:
    """Full pipeline execution trace for one task."""

    task_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Stage 0: Classify
    system: int = 0
    domain: str = ""
    routing_method: str = ""  # "knn", "system_router", "heuristic"
    routing_confidence: float = 0.0

    # Stage 1: Decompose
    dag_omega: float = 0.0
    dag_delta: float = 0.0
    dag_gamma: float = 0.0
    decompose_node_count: int = 0

    # Stage 2: Select Topology
    topology_hint: str = ""  # "sequential", "avr", "parallel_fanout", etc.
    topology_id: str = ""
    topology_node_count: int = 0
    topology_bypassed: bool = False
    topology_source: str = ""  # "template", "engine", "path6", "bypass"

    # Stage 3: Assign Models
    assignments: dict[int, str] = field(default_factory=dict)
    reassignments: list[dict[str, str]] = field(default_factory=list)  # [{node, from, to, reason}]

    # Stage 4: Execute
    nodes: list[NodeTrace] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    reroute_count: int = 0
    multi_turn_rounds: int = 0

    # Stage 4b: Quality + FrugalGPT
    quality_score: float | None = None
    frugalgpt_triggered: bool = False
    frugalgpt_succeeded: bool = False

    # Stage 5: Learn
    bandit_updated: bool = False
    archive_recorded: bool = False
    evolution_triggered: bool = False
    consolidation_ran: bool = False

    # Repair / Escalation (bench-level, not pipeline)
    avr_repair_attempted: bool = False
    avr_repair_succeeded: bool = False
    topology_escalation_attempted: bool = False
    topology_escalation_succeeded: bool = False

    # Final
    result_len: int = 0
    passed: bool | None = None  # Set by bench after eval
    error: str = ""

    def record_routing(self, system: int, confidence: float = 0.0,
                       method: str = "", domain: str = "") -> None:
        self.system = system
        self.routing_confidence = confidence
        self.routing_method = method
        self.domain = domain

    def record_decompose(self, omega: float, delta: float, gamma: float,
                         node_count: int = 0) -> None:
        self.dag_omega = omega
        self.dag_delta = delta
        self.dag_gamma = gamma
        self.decompose_node_count = node_count

    def record_topology(self, hint: str = "", topology_id: str = "",
                        node_count: int = 0, bypassed: bool = False,
                        source: str = "") -> None:
        self.topology_hint = hint
        self.topology_id = topology_id
        self.topology_node_count = node_count
        self.topology_bypassed = bypassed
        self.topology_source = source

    def record_node(self, idx: int, role: str = "", model_id: str = "",
                    provider: str = "", input_tokens: int = 0,
                    output_tokens: int = 0, latency_ms: float = 0.0,
                    cost_usd: float = 0.0, output_len: int = 0,
                    error: str = "", quality_score: float | None = None,
                    controller_action: str = "") -> None:
        self.nodes.append(NodeTrace(
            idx=idx, role=role, model_id=model_id, provider=provider,
            input_tokens=input_tokens, output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms, cost_usd=cost_usd,
            output_len=output_len, error=error,
            quality_score=quality_score,
            controller_action=controller_action,
        ))

    def record_reassignment(self, node_idx: int, from_model: str,
                            to_model: str, reason: str) -> None:
        self.reassignments.append({
            "node": node_idx, "from": from_model,
            "to": to_model, "reason": reason,
        })

    def finalize(self) -> None:
        """Compute aggregates from per-node traces."""
        self.total_input_tokens = sum(n.input_tokens for n in self.nodes)
        self.total_output_tokens = sum(n.output_tokens for n in self.nodes)
        self.total_tokens = sum(n.total_tokens for n in self.nodes)
        self.total_cost_usd = sum(n.cost_usd for n in self.nodes)
        if not self.total_latency_ms:
            self.total_latency_ms = sum(n.latency_ms for n in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        d = {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "routing": {
                "system": self.system,
                "domain": self.domain,
                "method": self.routing_method,
                "confidence": self.routing_confidence,
            },
            "decompose": {
                "omega": self.dag_omega,
                "delta": self.dag_delta,
                "gamma": self.dag_gamma,
            },
            "topology": {
                "hint": self.topology_hint,
                "id": self.topology_id,
                "node_count": self.topology_node_count,
                "bypassed": self.topology_bypassed,
                "source": self.topology_source,
            },
            "assignments": self.assignments,
            "reassignments": self.reassignments,
            "nodes": [
                {
                    "idx": n.idx, "role": n.role, "model": n.model_id,
                    "provider": n.provider,
                    "tokens": {"in": n.input_tokens, "out": n.output_tokens, "total": n.total_tokens},
                    "latency_ms": round(n.latency_ms, 1),
                    "cost_usd": round(n.cost_usd, 6),
                    "output_len": n.output_len,
                    "error": n.error,
                    "quality": n.quality_score,
                    "action": n.controller_action,
                }
                for n in self.nodes
            ],
            "totals": {
                "tokens": self.total_tokens,
                "tokens_in": self.total_input_tokens,
                "tokens_out": self.total_output_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
                "latency_ms": round(self.total_latency_ms, 1),
            },
            "quality": {
                "score": self.quality_score,
                "frugalgpt_triggered": self.frugalgpt_triggered,
                "frugalgpt_succeeded": self.frugalgpt_succeeded,
            },
            "repair": {
                "avr_attempted": self.avr_repair_attempted,
                "avr_succeeded": self.avr_repair_succeeded,
                "escalation_attempted": self.topology_escalation_attempted,
                "escalation_succeeded": self.topology_escalation_succeeded,
            },
            "learn": {
                "bandit_updated": self.bandit_updated,
                "archive_recorded": self.archive_recorded,
                "evolution_triggered": self.evolution_triggered,
            },
            "result_len": self.result_len,
            "passed": self.passed,
            "error": self.error,
        }
        return d
