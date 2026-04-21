"""Classify BCB Hard Instruct failures from a predictions JSONL + report JSON.

Usage:
    python scripts/audits/classify_bcb_failures.py \
        --predictions sage-python/docs/benchmarks/2026-04-08-predictions-hard-instruct.jsonl \
        --report docs/benchmarks/2026-04-08-bigcodebench-hard-instruct.json

Outputs a JSON summary to stdout. Read-only, no data is regenerated.

Categories (cascaded, NOT OR'd — first match wins):
    1. api_error              — report.error non-empty (provider 4xx/5xx/SSL)
    2. timeout_generation     — _trace.generation_error == "TIMEOUT"
    3. generation_exception   — _trace.generation_error non-empty, non-TIMEOUT
    4. empty_or_sentinel      — solution is empty / contains sage sentinel
    5. test_env_error         — eval_error_snippet matches BCB fixture patterns
                                 (missing HF snowflake-arctic model cache, etc.)
    6. timeout_eval           — "TimeoutExpired" or "TIMEOUT:" in eval_error_snippet
    7. syntax_error           — "SyntaxError" in eval_error_snippet
    8. import_error           — "ModuleNotFoundError" or "ImportError" in snippet
    9. assertion_failure_logical — "AssertionError" in snippet
    10. other_runtime_exception  — non-empty snippet, none of the above
    11. silent_fail           — passed=False AND no error anywhere (plumbing gap)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            out[entry["task_id"]] = entry
    return out


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


SENTINEL_RE = re.compile(r"\[sage:\s*agent exited", re.IGNORECASE)
TEST_ENV_PATTERNS = [
    "snowflake-arctic-embed",
    "no file named pytorch_model.bin",
    "models--Snowflake",
    "huggingface/hub/models--",
    "ConnectionError: HTTPSConnectionPool",
    "Couldn't connect to 'https://huggingface.co'",
]


def classify(
    task_id: str,
    report_result: dict[str, Any],
    pred_entry: dict[str, Any],
) -> tuple[str, str]:
    """Return (category, reason_example). Cascaded, first match wins.

    Note: eval_error_snippet is capped at 200 chars in bigcodebench_bench.py,
    so key evidence is often truncated. We have to rely on both keyword
    matches AND positional heuristics (e.g. Traceback at start of snippet
    with an `import`/`from` line early on = almost certainly an import error
    with ModuleNotFoundError cut off).
    """
    report_error: str = report_result.get("error") or ""
    trace = pred_entry.get("_trace", {})
    gen_error: str = trace.get("generation_error") or ""
    eval_snippet: str = trace.get("eval_error_snippet") or ""
    solution: str = pred_entry.get("solution") or ""

    # 1. Provider-side error before eval ran
    if report_error:
        return "api_error", report_error[:160]

    # 2. Generation-side timeouts / exceptions
    if gen_error:
        if "TIMEOUT" in gen_error.upper():
            return "timeout_generation", gen_error[:160]
        return "generation_exception", gen_error[:160]

    # 3. Empty or sentinel output — nothing to eval
    stripped = solution.strip()
    if not stripped or SENTINEL_RE.search(stripped):
        return "empty_or_sentinel", f"solution_len={len(stripped)}"

    # 4. Eval snippet based classification (order matters)
    if eval_snippet:
        # 4a. Syntax/indentation errors in generated code top-level
        if "IndentationError" in eval_snippet or "SyntaxError" in eval_snippet:
            return "syntax_error", eval_snippet[:160]
        # 4b. Explicit import failures
        if "ModuleNotFoundError" in eval_snippet or "ImportError" in eval_snippet:
            return "import_error", eval_snippet[:160]
        # 4c. Truncated traceback with import-looking preamble
        #     (snippet starts with `Traceback`, has `import`/`from` in first 400
        #     chars of the snippet — keyword for ModuleNotFoundError got cut)
        if eval_snippet.startswith("Traceback") and (
            " import " in eval_snippet[:400] or "from " in eval_snippet[:400]
        ):
            return "import_error_truncated", eval_snippet[:160]
        # 4d. BCB test fixture failures (NOT SAGE fault)
        for pat in TEST_ENV_PATTERNS:
            if pat in eval_snippet:
                return "test_env_error", eval_snippet[:160]
        # 4e. Eval-time timeouts (30s default)
        if "TimeoutExpired" in eval_snippet or eval_snippet.startswith("TIMEOUT:"):
            return "timeout_eval", eval_snippet[:160]
        # 4f. unittest FAIL: = assertion failure (logical wrong output)
        if "FAIL:" in eval_snippet or "AssertionError" in eval_snippet:
            return "assertion_failure_logical", eval_snippet[:160]
        # 4g. unittest ERROR: = uncaught exception during test
        #     (wrong return type, KeyError, TypeError, etc.)
        if "ERROR:" in eval_snippet:
            return "runtime_exception_in_test", eval_snippet[:160]
        # 4h. Warning-only prefix: real failure got truncated past 200 chars
        if "Warning" in eval_snippet and "Error" not in eval_snippet:
            return "truncated_warning_prefix", eval_snippet[:160]
        return "other", eval_snippet[:160]

    # 5. passed=False with no eval snippet → plumbing gap
    return "silent_fail", f"solution_len={len(stripped)}, avr_attempted={trace.get('avr_attempted')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="Optional output JSON path")
    args = ap.parse_args()

    preds = load_predictions(args.predictions)
    report = load_report(args.report)

    report_by_id = {r["task_id"]: r for r in report["results"]}
    # Sanity: alignment
    missing_preds = [tid for tid in report_by_id if tid not in preds]
    missing_report = [tid for tid in preds if tid not in report_by_id]

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Split discriminators for top-lever reasoning
    avr_bucket = Counter()        # (category, avr_attempted, avr_repaired)
    bypass_bucket = Counter()     # (category, was_bypassed)
    omega_bucket = defaultdict(list)  # category -> [omega values]

    failures_total = 0
    passes_total = 0
    for tid, rep in report_by_id.items():
        if rep.get("passed"):
            passes_total += 1
            continue
        failures_total += 1
        pred = preds.get(tid, {"_trace": {}, "solution": ""})
        category, example = classify(tid, rep, pred)
        trace = pred.get("_trace", {})
        categories[category].append(
            {
                "task_id": tid,
                "example_reason": example,
                "avr_attempted": trace.get("avr_attempted"),
                "avr_repaired": trace.get("avr_repaired"),
                "topology_nodes": trace.get("topology_nodes"),
                "omega": (trace.get("dag_features") or {}).get("omega"),
                "solution_len": len(pred.get("solution", "") or ""),
            }
        )
        avr_bucket[(category, bool(trace.get("avr_attempted")), bool(trace.get("avr_repaired")))] += 1
        bypass_bucket[(category, (trace.get("topology_nodes") or 0) == 0)] += 1
        omega = (trace.get("dag_features") or {}).get("omega")
        if omega is not None:
            omega_bucket[category].append(omega)

    # Assemble summary
    summary = {
        "source": {
            "report": str(args.report),
            "predictions": str(args.predictions),
        },
        "report_header": {
            "total": report["total"],
            "passed": report["passed"],
            "failed": report["failed"],
            "errors_marked": report["errors"],
            "pass_rate": report["pass_rate"],
            "avg_latency_ms": report["avg_latency_ms"],
            "routing_breakdown": report["routing_breakdown"],
        },
        "alignment": {
            "tasks_in_report": len(report_by_id),
            "tasks_in_predictions": len(preds),
            "missing_in_preds": missing_preds,
            "missing_in_report": missing_report,
        },
        "totals": {
            "passes": passes_total,
            "failures": failures_total,
        },
        "categories": {},
        "discriminators": {
            "avr": {
                f"{cat}|avr_attempted={att}|avr_repaired={rep}": n
                for (cat, att, rep), n in sorted(avr_bucket.items(), key=lambda kv: -kv[1])
            },
            "bypass": {
                f"{cat}|bypassed={byp}": n
                for (cat, byp), n in sorted(bypass_bucket.items(), key=lambda kv: -kv[1])
            },
        },
    }

    for cat, entries in sorted(categories.items(), key=lambda kv: -len(kv[1])):
        omegas = omega_bucket.get(cat, [])
        summary["categories"][cat] = {
            "count": len(entries),
            "fraction_of_failures": round(len(entries) / failures_total, 4) if failures_total else 0.0,
            "example_task_id": entries[0]["task_id"] if entries else None,
            "example_reason": entries[0]["example_reason"] if entries else None,
            "avr_attempted_count": sum(1 for e in entries if e["avr_attempted"]),
            "bypass_count": sum(1 for e in entries if (e["topology_nodes"] or 0) == 0),
            "mean_omega": round(sum(omegas) / len(omegas), 3) if omegas else None,
            "task_ids": [e["task_id"] for e in entries],
            "sample_reasons": list({e["example_reason"] for e in entries})[:5],
        }

    output_json = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_json, encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
