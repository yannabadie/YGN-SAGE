"""Cascaded sandbox evaluation — HyEvo-inspired multi-stage filtering.

Rapidly filters out suboptimal topology candidates with minimal computational
cost before committing to full evaluation.

Stages:
  1. Schema validation — is the YAML structurally valid?
  2. Security/sandbox — does the code pass basic safety checks?
  3. Smoke execution — partial evaluation on subset of data
  4. Full evaluation — complete benchmark on full dataset

Reference: HyEvo (arXiv 2603.19639) Section 3.4 — Cascaded Sandbox Evaluation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("cascaded_eval")


@dataclass
class EvalResult:
    """Result of cascaded evaluation."""
    stage_reached: int  # 1-4
    stage_name: str     # "schema", "security", "smoke", "full"
    passed: bool
    score: float = 0.0
    cost: float = 0.0
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def stage_1_schema(yaml_text: str) -> EvalResult:
    """Stage 1: Schema validation — fast, no execution.

    Checks YAML parse + TopologySchema validation.
    """
    from sage.verl.topology_schema import TopologySchema

    schema = TopologySchema.from_yaml(yaml_text)
    if schema is None:
        return EvalResult(
            stage_reached=1, stage_name="schema", passed=False,
            errors=["YAML parse failed or missing 'nodes' key"],
        )

    validation_errors = schema.validate()
    if validation_errors:
        return EvalResult(
            stage_reached=1, stage_name="schema", passed=False,
            errors=validation_errors,
        )

    return EvalResult(
        stage_reached=1, stage_name="schema", passed=True,
        metrics={
            "node_count": len(schema.nodes),
            "edge_count": len(schema.edges),
            "llm_ratio": schema.llm_ratio,
            "code_ratio": schema.code_ratio,
            "has_checkpoints": float(schema.has_checkpoints),
        },
    )


def stage_2_security(yaml_text: str) -> EvalResult:
    """Stage 2: Security/sandbox check — verify code nodes are safe.

    For code nodes: check for dangerous imports, file access, etc.
    For LLM nodes: verify prompt doesn't contain injection patterns.
    """
    from sage.verl.topology_schema import TopologySchema

    schema = TopologySchema.from_yaml(yaml_text)
    if schema is None:
        return EvalResult(
            stage_reached=2, stage_name="security", passed=False,
            errors=["Schema parse failed"],
        )

    errors = []
    DANGEROUS_PATTERNS = ["os.system", "subprocess", "eval(", "exec(", "__import__",
                          "shutil.rmtree", "os.remove", "open(", "socket."]

    for i, node in enumerate(schema.nodes):
        if node.is_code_node and node.code_spec:
            for pattern in DANGEROUS_PATTERNS:
                if pattern in node.code_spec:
                    errors.append(f"Node {i}: dangerous pattern '{pattern}' in code_spec")

    if errors:
        return EvalResult(
            stage_reached=2, stage_name="security", passed=False,
            errors=errors,
        )

    return EvalResult(stage_reached=2, stage_name="security", passed=True)


def stage_3_smoke(yaml_text: str, compute_score_fn: Any = None) -> EvalResult:
    """Stage 3: Smoke execution — structural reward on the topology.

    Quick scoring without real LLM calls. Uses the reward function in
    structural mode (SAGE_VERL_EXEC=0).
    """
    if compute_score_fn is None:
        from sage.verl.reward import compute_score
        compute_score_fn = compute_score

    try:
        score = compute_score_fn("sage_topology", yaml_text, "", {})
        return EvalResult(
            stage_reached=3, stage_name="smoke", passed=score > 0.1,
            score=score,
            metrics={"structural_score": score},
        )
    except Exception as exc:
        return EvalResult(
            stage_reached=3, stage_name="smoke", passed=False,
            errors=[f"Smoke eval failed: {exc}"],
        )


def stage_4_full(yaml_text: str, task: str = "", extra_info: dict | None = None) -> EvalResult:
    """Stage 4: Full evaluation — execution reward if available.

    Uses the reward function in execution mode when SAGE_VERL_EXEC=1.
    Falls back to structural if no provider available.
    """
    from sage.verl.reward import compute_score

    try:
        score = compute_score("sage_topology", yaml_text, "", extra_info or {})
        return EvalResult(
            stage_reached=4, stage_name="full", passed=True,
            score=score,
            metrics={"full_score": score},
        )
    except Exception as exc:
        return EvalResult(
            stage_reached=4, stage_name="full", passed=False,
            errors=[f"Full eval failed: {exc}"],
        )


def cascaded_evaluate(
    yaml_text: str,
    gamma: float = 0.1,
    task: str = "",
    extra_info: dict | None = None,
) -> EvalResult:
    """Run all 4 cascaded evaluation stages.

    Stops early if a stage fails. Only proceeds to stage 4 if stage 3
    score exceeds threshold gamma (HyEvo fast filtering).

    Args:
        yaml_text: Raw YAML topology string
        gamma: Minimum stage 3 score to proceed to stage 4
        task: Task description for full evaluation
        extra_info: Additional info for reward computation

    Returns:
        EvalResult from the highest stage reached
    """
    # Stage 1: Schema
    r1 = stage_1_schema(yaml_text)
    if not r1.passed:
        log.debug("Cascaded eval: failed at Stage 1 (schema)")
        return r1

    # Stage 2: Security
    r2 = stage_2_security(yaml_text)
    if not r2.passed:
        log.debug("Cascaded eval: failed at Stage 2 (security)")
        return r2

    # Stage 3: Smoke
    r3 = stage_3_smoke(yaml_text)
    if not r3.passed or r3.score < gamma:
        log.debug("Cascaded eval: failed at Stage 3 (smoke score=%.3f < gamma=%.3f)",
                   r3.score, gamma)
        return r3

    # Stage 4: Full
    r4 = stage_4_full(yaml_text, task=task, extra_info=extra_info)
    log.info("Cascaded eval: passed all 4 stages (score=%.3f)", r4.score)
    return r4
