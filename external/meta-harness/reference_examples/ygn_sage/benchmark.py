"""Benchmark runner for the Meta-Harness × YGN-SAGE reference example.

Evaluates a candidate SageCandidate against SWE-bench Lite (val set) by
shelling out to `python -m sage.bench`, then parsing the predictions
manifest that the bench writes.

Kept tiny on purpose: the real eval logic lives in sage-python
(`sage/bench/swebench_bench.py`). This file just:
  1. Imports the candidate module (validates the contract)
  2. Invokes `python -m sage.bench ... --generate-only`
  3. Parses the resulting predictions_meta.json
  4. Returns {val_score, per_task, cost, latency_s}

Design note: we intentionally DON'T instantiate the candidate here —
`sage.bench` boots its own AgentSystem via boot_agent_system(). A more
integrated version (where the candidate controls the AgentSystem passed
to SWEBenchBench) is future work; for v1 we evaluate candidates that
affect SAGE through its config files / env vars / monkey-patched modules.
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

def validate_candidate(module_name: str) -> str:
    """Import-check a candidate. Returns 'ok' or error message."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return f"ImportError: {type(exc).__name__}: {exc}"
    from .sage_candidate import SageCandidate
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


def run_sage_bench(
    dataset: str,
    limit: int,
    offset: int,
    timeout_per_task_s: int,
    env_overrides: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Invoke `python -m sage.bench --type swebench --generate-only` and
    return (predictions_path, meta_path) once it finishes.

    Raises RuntimeError on non-zero exit.
    """
    cmd = [
        sys.executable, "-m", "sage.bench",
        "--type", "swebench",
        "--dataset", dataset,
        "--limit", str(limit),
        "--offset", str(offset),
        "--generate-only",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    if env_overrides:
        env.update(env_overrides)

    log_path = Path(tempfile.mkdtemp(prefix="mh_sage_")) / "bench.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_per_task_s * limit + 120,
            check=False,
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sage.bench exited {proc.returncode}; see {log_path}"
        )

    # sage.bench writes to a tempdir named sage_swebench_* and prints the
    # path near the end. Parse the log to find it.
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    import re
    m = re.search(r"Predictions saved to:\s+(\S+)", log_text)
    if not m:
        raise RuntimeError(f"could not locate predictions path in {log_path}")
    preds_path = Path(m.group(1))
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
    """Full pipeline: validate → invoke bench → score → return.

    `candidate_module` should be an import path reachable from Python, e.g.
    `reference_examples.ygn_sage.agents.baseline`.
    """
    t0 = time.time()
    validation = validate_candidate(candidate_module)
    if validation != "ok":
        return {
            "val_score": 0.0,
            "error": validation,
            "latency_s": time.time() - t0,
        }

    preds_path, meta_path = run_sage_bench(
        dataset=dataset,
        limit=limit,
        offset=offset,
        timeout_per_task_s=timeout_per_task_s,
        # Candidate-specific env vars could be forwarded here (e.g. to
        # activate feature flags defined by the candidate module). For v1
        # the baseline doesn't need any.
        env_overrides=None,
    )

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
