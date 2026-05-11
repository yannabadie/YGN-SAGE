"""Slice 10D — reproduce a `provider_execution_witness` JSONL event.

Recreates the slice 9 scenario through the LIVE writer
(`emit_provider_execution_witness`) into a real JSONL file. The output
is used as the canonical worked example in `summary.md`.

Run from the repo root:
    python docs/benchmarks/2026-05-11-provider-execution-witness-chain/reproduce.py

Writes `events.jsonl` next to this script.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the in-repo sage package importable without an editable install
HERE = Path(__file__).resolve()
SAGE_SRC = HERE.parents[3] / "sage-python" / "src"
sys.path.insert(0, str(SAGE_SRC))

from sage.runtime.event_log import RuntimeEventLog


def main() -> int:
    out_dir = HERE.parent
    run_id = "01WITNESSREPRO0000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=out_dir)
    log.set_task_text("slice 10D reproduction — slice 9 scenario witness")
    log.emit_task_started("slice 10D reproduction — slice 9 scenario witness")

    # The slice 9 chain: routing picked gpt-5.4-pro (openai),
    # CLI provider policy says allow={deepseek,google}, deny={openai},
    # ModelAssigner substituted deepseek-v4-pro / gemini-3-flash-preview.
    seq = log.emit_provider_execution_witness(
        witness_schema_version="v0",
        assignment_phase="initial",
        routing={
            "routing_model_id": "gpt-5.4-pro",
            "routing_provider_id": "openai",
            "routing_source": "rust_system_router",
            "system": 3,
            "domain": "code",
            "confidence": 0.8788,
        },
        policy={
            "active": True,
            "allowlist": ["deepseek", "google"],
            "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_index": 0,
                "node_role": "coder",
                "required_capabilities": ["code_generation", "reasoning", "tools"],
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            },
            {
                "node_index": 1,
                "node_role": "synthesizer",
                "required_capabilities": ["text_processing"],
                "assigned_model_id": "gemini-3-flash-preview",
                "assigned_provider_id": "google",
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            },
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": True,
            "routing_candidate_blocked_by_policy": True,
            "assignment_count": 2,
            "allowed_assignment_count": 2,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )
    assert seq is not None, "witness emit should not fail in this controlled run"

    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=2,
    )
    log.close()

    out_path = out_dir / f"{run_id}.jsonl"
    final = out_dir / "events.jsonl"
    if final.exists():
        final.unlink()
    out_path.rename(final)
    print(f"Wrote {final}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
