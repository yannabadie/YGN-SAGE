"""RuntimeEventLog schema constants. Public API."""
from typing import Literal

SCHEMA_VERSION: Literal["1.0"] = "1.0"

EVENT_TYPES = (
    "task_started",
    "routing_decision",
    "bandit_attribution_mismatch",
    "topology_selected",
    "model_assigned",
    "node_started",
    "node_completed",
    "controller_decision",
    "state_applied",
    "failure",
    "budget",
    "final_result",
    "oracle_verdict",
    "run_frame_summary",
    "prompt_injection_detected",
    # Slice 10D (cgpro DESIGN_LOCK 2026-05-11 Route A, v0):
    # `provider_execution_witness` makes the chain
    # routing_chosen_model → policy_decision → per-node assignments
    # explicit. v0 witness only — NOT an invariant yet. Cf.
    # docs/superpowers/plans/2026-05-10-handoff-recovery-plan.md.
    "provider_execution_witness",
)

REDACTION_STATES = (
    "redacted",
    "raw",
    "partial",
    "none_applicable",
)

SOURCE_COMPONENTS = (
    "pipeline",
    "topology_runner",
    "controller",
    "model_assigner",
    "provider_pool",
)

CONTROLLER_ACTIONS = (
    "continue",
    "upgrade_model",
    "prune_node",
    "reroute_topology",
    "spawn_subagent",
    "open_gate",
)

FINAL_RESULT_STATUSES = ("success", "failure", "budget_exceeded")
