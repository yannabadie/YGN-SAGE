"""Benchmark runner for the Meta-Harness × YGN-SAGE reference example.

Evaluates a candidate SageCandidate against SWE-bench Lite (val set)
IN-PROCESS so the candidate's monkey-patched AgentSystem actually
drives the run. An earlier iteration of this file shelled out to
`python -m sage.bench` which booted a vanilla AgentSystem and silently
discarded every candidate override — a correctness bug caught before
any real iteration ran.

Flow:
  1. Import the candidate module (validates the contract)
  2. Call candidate.build_system() → get AgentSystem instance
  3. Wrap it in SWEBenchBench(system=..., ...)
  4. await bench.run_generate_only(limit=..., offset=...)
  5. Parse the resulting predictions_meta.json
  6. Return {val_score, per_task, cost, latency_s}
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure external/meta-harness is on sys.path so candidates can be
# imported as "reference_examples.ygn_sage.agents.<id>". Idempotent —
# safe to import this module multiple times.
_EXTERNAL_ROOT = Path(__file__).resolve().parents[3]  # external/meta-harness/
if str(_EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ROOT))


# ── Score computation ────────────────────────────────────────────

_SENTINEL_PREFIX = "[sage: agent exited after"


def classify_prediction(patch: str | None) -> str:
    """real | sentinel | empty — matches bench/swebench_bench.py classifier."""
    if not patch:
        return "empty"
    if _SENTINEL_PREFIX in patch:
        return "sentinel"
    return "real"


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate score + per-task breakdown from a predictions.jsonl list."""
    counts = {"real": 0, "sentinel": 0, "empty": 0}
    per_task: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    total_tool_calls = 0
    for pred in predictions:
        klass = classify_prediction(pred.get("model_patch"))
        counts[klass] += 1
        iid = pred.get("instance_id", "unknown")
        per_task[iid] = {
            "class": klass,
            "patch_len": len(pred.get("model_patch") or ""),
            "tool_calls": int(pred.get("_tool_call_count", 0) or 0),
            "latency_ms": float(pred.get("_latency_ms", 0.0) or 0.0),
            "error": pred.get("_error", "") or "",
        }
        total_cost += float(pred.get("_cost_usd", 0.0) or 0.0)
        total_tool_calls += int(pred.get("_tool_call_count", 0) or 0)

    total = max(len(predictions), 1)
    # val_score = fraction of real patches (primary metric).
    # Sentinels count as 0.25 (tools were called but no diff) so the
    # proposer gets gradient even when no candidate crosses the real-patch
    # threshold yet.
    val_score = (counts["real"] + 0.25 * counts["sentinel"]) / total

    return {
        "val_score": val_score,
        "real": counts["real"],
        "sentinel": counts["sentinel"],
        "empty": counts["empty"],
        "total": total,
        "cost_usd": total_cost,
        "tool_calls_total": total_tool_calls,
        "per_task": per_task,
    }


# ── Bench invocation ─────────────────────────────────────────────

def _import_sage_candidate():
    """Import SageCandidate via the canonical absolute path so this module
    works whether benchmark.py is called as a script or imported as part
    of the reference_examples.ygn_sage package."""
    from reference_examples.ygn_sage.sage_candidate import SageCandidate  # type: ignore
    return SageCandidate


def validate_candidate(module_name: str) -> str:
    """Import-check a candidate. Returns 'ok' or error message."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return f"ImportError: {type(exc).__name__}: {exc}"
    SageCandidate = _import_sage_candidate()
    candidate = getattr(mod, "CANDIDATE", None)
    if not isinstance(candidate, SageCandidate):
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, SageCandidate) and obj is not SageCandidate:
                candidate = obj()
                break
    if not isinstance(candidate, SageCandidate):
        return "no CANDIDATE symbol and no SageCandidate subclass"
    return "ok"


def _load_candidate_instance(module_name: str):
    """Return a concrete SageCandidate from a module import path."""
    mod = importlib.import_module(module_name)
    SageCandidate = _import_sage_candidate()
    candidate = getattr(mod, "CANDIDATE", None)
    if isinstance(candidate, SageCandidate):
        return candidate
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, SageCandidate) and obj is not SageCandidate:
            return obj()
    raise ImportError(f"no CANDIDATE in {module_name!r}")


async def _run_sage_bench_in_process(
    system: Any,
    dataset: str,
    limit: int,
    offset: int,
    timeout_per_task_s: int,
) -> tuple[Path, Path]:
    """Drive SWEBenchBench directly with the candidate's AgentSystem.

    Returns (predictions_path, meta_path). Raises on fatal error.
    """
    from sage.bench.swebench_bench import SWEBenchBench  # type: ignore[import-not-found]
    bench = SWEBenchBench(
        system=system,
        event_bus=None,
        dataset=dataset,
        timeout_per_task=timeout_per_task_s,
    )
    preds_path = await bench.run_generate_only(limit=limit, offset=offset)
    preds_path = Path(preds_path)
    meta_path = preds_path.parent / "predictions_meta.json"
    return preds_path, meta_path


def evaluate(
    candidate_module: str,
    *,
    dataset: str = "lite",
    limit: int = 5,
    offset: int = 3,
    timeout_per_task_s: int = 300,
) -> dict[str, Any]:
    """Full pipeline: validate → build system → run bench → score.

    `candidate_module` must resolve to a SageCandidate via
    `reference_examples.ygn_sage.agents.<id>` (sys.path is prepped by
    the import of this module).
    """
    t0 = time.time()
    validation = validate_candidate(candidate_module)
    if validation != "ok":
        return {
            "val_score": 0.0,
            "error": validation,
            "latency_s": time.time() - t0,
        }

    try:
        candidate = _load_candidate_instance(candidate_module)
        system = candidate.build_system({"dataset": dataset, "limit": limit, "offset": offset})
    except Exception as exc:  # noqa: BLE001
        return {
            "val_score": 0.0,
            "error": f"build_system failed: {type(exc).__name__}: {exc}",
            "latency_s": time.time() - t0,
        }

    import asyncio
    try:
        preds_path, meta_path = asyncio.run(
            _run_sage_bench_in_process(
                system=system,
                dataset=dataset,
                limit=limit,
                offset=offset,
                timeout_per_task_s=timeout_per_task_s,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "val_score": 0.0,
            "error": f"bench run failed: {type(exc).__name__}: {exc}",
            "latency_s": time.time() - t0,
        }

    if meta_path.exists():
        preds = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        with preds_path.open("r", encoding="utf-8") as f:
            preds = [json.loads(line) for line in f if line.strip()]

    scored = score_predictions(preds)
    scored["latency_s"] = time.time() - t0
    scored["predictions_path"] = str(preds_path)
    scored["meta_path"] = str(meta_path)
    return scored


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Meta-Harness × YGN-SAGE benchmark runner")
    p.add_argument("candidate", help="import path e.g. reference_examples.ygn_sage.agents.baseline")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--offset", type=int, default=3)
    args = p.parse_args()
    result = evaluate(args.candidate, limit=args.limit, offset=args.offset)
    print(json.dumps(result, indent=2, default=str))
