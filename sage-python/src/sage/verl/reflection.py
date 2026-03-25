"""Reflect-then-generate mechanism — HyEvo-inspired topology refinement.

Given a parent topology, its execution logs, and exemplars from the archive,
produces a structured diagnostic and generates an improved candidate.

The reflection cycle:
  1. Collect: parent workflow + error logs + cost/latency metrics
  2. Diagnose: identify structural failures, provider issues, cost overruns
  3. Generate: synthesize improved candidate addressing diagnosed issues
  4. Evaluate: cascaded evaluation (stage 1-4)
  5. Archive: if better, replace in MAP-Elites cell

Reference: HyEvo (arXiv 2603.19639) Section 3.2 — Reflect-Then-Generate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("reflection")


@dataclass
class ReflectionDiagnostic:
    """Structured diagnostic from analyzing execution traces.

    HyEvo: "h_reflect = comparative analysis to diagnose shortcomings"
    """
    parent_score: float = 0.0
    top_score: float = 0.0
    failure_types: list[str] = field(default_factory=list)
    cost_ratio: float = 0.0       # parent_cost / budget
    latency_ratio: float = 0.0    # parent_latency / target
    structural_issues: list[str] = field(default_factory=list)
    provider_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One-line summary for logging."""
        issues = len(self.failure_types) + len(self.structural_issues) + len(self.provider_issues)
        return (
            f"score={self.parent_score:.2f} (top={self.top_score:.2f}), "
            f"{issues} issues, {len(self.recommendations)} recommendations"
        )


def diagnose(
    parent_yaml: str,
    parent_score: float,
    parent_traces: list[dict],
    top_yaml: str = "",
    top_score: float = 0.0,
    budget: float = 5.0,
    target_latency_ms: float = 30000.0,
) -> ReflectionDiagnostic:
    """Analyze execution traces and produce a structured diagnostic.

    Args:
        parent_yaml: The parent topology YAML
        parent_score: Reward score of the parent
        parent_traces: List of per-node execution trace dicts
        top_yaml: Best-performing topology in archive (exemplar)
        top_score: Score of the top exemplar
        budget: Budget constraint (USD)
        target_latency_ms: Target latency constraint

    Returns:
        ReflectionDiagnostic with identified issues and recommendations
    """
    from sage.verl.topology_schema import TopologySchema

    diag = ReflectionDiagnostic(parent_score=parent_score, top_score=top_score)

    # Parse parent topology
    parent = TopologySchema.from_yaml(parent_yaml)
    if parent is None:
        diag.failure_types.append("YAML_PARSE_FAILURE")
        diag.recommendations.append("Generate valid YAML with nodes key")
        return diag

    # Structural analysis
    errors = parent.validate()
    if errors:
        diag.structural_issues.extend(errors)
        diag.recommendations.append("Fix structural validation errors")

    # Trace analysis
    total_cost = 0.0
    total_latency = 0.0
    for trace in parent_traces:
        status = trace.get("status", "")
        if status in ("ERROR", "TIMEOUT", "CRASH"):
            diag.failure_types.append(f"node_{trace.get('node_idx', '?')}:{status}")
        total_cost += trace.get("cost", 0.0)
        total_latency += trace.get("latency", 0.0)

        # Provider-specific failures
        provider = trace.get("provider", "")
        if status == "ERROR" and provider:
            diag.provider_issues.append(f"{provider}: {trace.get('error', 'unknown')}")

    diag.cost_ratio = total_cost / max(budget, 0.001)
    diag.latency_ratio = total_latency / max(target_latency_ms, 1.0)

    # Generate recommendations based on diagnosis
    if diag.cost_ratio > 0.8:
        diag.recommendations.append(
            "Cost too high — replace reasoner nodes with fast/budget for non-critical roles"
        )
    if diag.latency_ratio > 0.8:
        diag.recommendations.append(
            "Latency too high — add code nodes for deterministic sub-tasks"
        )
    if not parent.has_code_nodes and len(parent.nodes) >= 3:
        diag.recommendations.append(
            "Consider adding code nodes for format validation or computation (HyEvo hybrid)"
        )
    if parent.provider_diversity == 0 and len(parent.nodes) >= 2:
        diag.recommendations.append(
            "Add provider_hint diversity — spread load across multiple providers"
        )
    if "TIMEOUT" in str(diag.failure_types):
        diag.recommendations.append(
            "Timeout detected — reduce node count or use faster model tiers"
        )

    # Compare with top exemplar
    if top_yaml:
        top = TopologySchema.from_yaml(top_yaml)
        if top:
            if top.has_code_nodes and not parent.has_code_nodes:
                diag.recommendations.append(
                    f"Top exemplar uses code nodes (code_ratio={top.code_ratio:.1%}) — consider adding"
                )
            if top.provider_diversity > parent.provider_diversity:
                diag.recommendations.append(
                    f"Top exemplar has {top.provider_diversity} providers vs {parent.provider_diversity}"
                )

    log.info("Reflection diagnostic: %s", diag.summary)
    return diag


def format_reflection_prompt(
    diag: ReflectionDiagnostic,
    parent_yaml: str,
    task: str,
) -> str:
    """Format a reflection diagnostic as a prompt for the policy model.

    The model will use this to generate an improved topology candidate
    (reflect-then-generate cycle).
    """
    parts = [
        f"Task: {task[:500]}",
        "",
        "Previous topology had these issues:",
    ]

    if diag.structural_issues:
        parts.append(f"  Structural: {', '.join(diag.structural_issues[:3])}")
    if diag.failure_types:
        parts.append(f"  Failures: {', '.join(diag.failure_types[:3])}")
    if diag.provider_issues:
        parts.append(f"  Provider: {', '.join(diag.provider_issues[:3])}")
    if diag.cost_ratio > 0.5:
        parts.append(f"  Cost: {diag.cost_ratio:.0%} of budget used")
    if diag.latency_ratio > 0.5:
        parts.append(f"  Latency: {diag.latency_ratio:.0%} of target")

    parts.append("")
    parts.append("Recommendations:")
    for rec in diag.recommendations[:5]:
        parts.append(f"  - {rec}")

    parts.append("")
    parts.append(f"Previous score: {diag.parent_score:.3f}")
    parts.append(f"Best known score: {diag.top_score:.3f}")
    parts.append("")
    parts.append("Generate an improved YAML topology addressing these issues.")

    return "\n".join(parts)
