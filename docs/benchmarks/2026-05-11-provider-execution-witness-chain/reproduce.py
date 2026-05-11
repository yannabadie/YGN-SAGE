"""Slice 10D — reproduce a `provider_execution_witness` JSONL event
through the ACTUAL helper (not a hand-built payload).

cgpro VERIFY 2026-05-11 EDIT_REQUIRED #1: the previous version of this
script hand-built the payload, which masked the real
``per_node_assignments`` shape contract (node_id vs node_index). This
version calls ``runtime_emit_provider_execution_witness`` against a
SimpleNamespace pipeline + ctx that mimics the slice 9 scenario, so
``events.jsonl`` reflects exactly what production runs will write.

Run from the repo root:
    python docs/benchmarks/2026-05-11-provider-execution-witness-chain/reproduce.py

Writes ``events.jsonl`` next to this script.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Make the in-repo sage package importable without an editable install
HERE = Path(__file__).resolve()
SAGE_SRC = HERE.parents[3] / "sage-python" / "src"
sys.path.insert(0, str(SAGE_SRC))

from sage.pipeline_v2.runtime_events import (  # noqa: E402
    runtime_emit_provider_execution_witness,
)
from sage.runtime.event_log import RuntimeEventLog  # noqa: E402


def _make_node(role: str, model_id: str, caps: tuple[str, ...]) -> Any:
    return SimpleNamespace(
        role=role,
        model_id=model_id,
        required_capabilities=caps,
    )


def _make_pipeline(provider_inferred: dict[str, str]) -> Any:
    pool = SimpleNamespace()
    pool.infer_provider = (
        lambda model_id, _m=provider_inferred: _m.get(model_id, "")
    )
    return SimpleNamespace(provider_pool=pool, llm_config=None)


def _make_ctx(nodes: list[Any], *, assignments: dict[int, str]) -> Any:
    topo = SimpleNamespace()
    topo._nodes = nodes
    topo.id = "slice10d-repro-topo"
    topo.template_type = "sequential"
    topo.get_node = lambda idx, _t=topo: _t._nodes[idx]
    topo.node_count = lambda _t=topo: len(_t._nodes)
    ctx = SimpleNamespace()
    ctx.topology = topo
    ctx.assignments = assignments
    ctx.provider_hints = {}
    ctx.provider_allowlist = ("deepseek", "google")
    ctx.provider_denylist = ("openai",)
    ctx.routing_source = "rust_system_router"
    ctx.system = 3
    ctx.domain = "code"
    ctx.confidence = 0.8788
    return ctx


def main() -> int:
    out_dir = HERE.parent
    run_id = "01WITNESSREPRO0000000001"

    log = RuntimeEventLog(run_id=run_id, trace_dir=out_dir)
    log.set_task_text("slice 10D reproduction — slice 9 scenario witness")
    log.emit_task_started("slice 10D reproduction — slice 9 scenario witness")

    # Slice 9 scenario: routing picked gpt-5.4-pro (openai),
    # CLI provider policy says allow={deepseek,google}, deny={openai},
    # ModelAssigner substituted deepseek-v4-pro / gemini-3-flash-preview.
    pipeline = _make_pipeline({
        "gpt-5.4-pro": "openai",
        "deepseek-v4-pro": "deepseek",
        "gemini-3-flash-preview": "google",
    })
    nodes = [
        _make_node(
            "coder",
            "deepseek-v4-pro",
            ("code_generation", "reasoning", "tools"),
        ),
        _make_node(
            "synthesizer",
            "gemini-3-flash-preview",
            ("text_processing",),
        ),
    ]
    ctx = _make_ctx(
        nodes,
        assignments={
            0: "deepseek-v4-pro",
            1: "gemini-3-flash-preview",
        },
    )

    seq = runtime_emit_provider_execution_witness(
        pipeline, ctx, log,
        routing_model_id="gpt-5.4-pro",
    )
    assert seq is not None, "helper emit must succeed in this controlled run"

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
