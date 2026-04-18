"""Autonomous evolution loop for YGN-SAGE harness candidates.

Val-only during evolution (test never exposed). Uses codex_wrapper +
the in-repo meta-harness skill to propose new candidate modules under
`agents/`.

    uv run python meta_harness.py --iterations 10 --fresh
    uv run python meta_harness.py --iterations 5 --run-name nightly-sweep
    uv run python meta_harness.py --dry-run         # propose only, don't evaluate
    uv run python meta_harness.py --evaluate-only   # evaluate existing agents/, skip proposer

Adapted from upstream text_classification/meta_harness.py (stanford-iris-lab).
Key changes for SAGE + Codex:
  - Proposer: Codex CLI (gpt-5.4 reasoning_effort=high) instead of Claude CLI
  - Candidate contract: SageCandidate (agents/*.py) instead of MemorySystem
  - Inner loop: python -m sage.bench (subprocess) instead of direct call
  - Val/test: SWE-bench Lite offset=3 limit=5 / MASBENCH breadth
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml  # type: ignore[import-not-found]

# Ensure our reference_example sibling modules and the parent external/
# meta-harness tree are on sys.path so `reference_examples.ygn_sage.
# agents.<id>` is importable (used by benchmark.py).
_HERE = Path(__file__).resolve().parent
_EXTERNAL_ROOT = _HERE.parents[2]  # external/meta-harness/
for _p in (str(_HERE), str(_EXTERNAL_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Local imports (siblings in reference_examples/ygn_sage/)
import codex_wrapper  # type: ignore[import-not-found]
import benchmark  # type: ignore[import-not-found]


EVOLVE_DIR = Path(__file__).parent
CONFIG_PATH = EVOLVE_DIR / "config.yaml"
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {"__init__.py", "baseline.py"}

# Updated per-run if --run-name is set
LOGS_DIR = EVOLVE_DIR / "logs" / "default"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
CLAUDE_SESSIONS = LOGS_DIR / "codex_sessions"

_interrupted = False


def _sigint_handler(signum, frame) -> None:  # noqa: ARG001
    global _interrupted
    _interrupted = True
    print("\n[meta_harness] Interrupted; finishing current iteration…", flush=True)


# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


# ── Logging helpers ──────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _update_paths_for_run(run_name: str) -> None:
    """Point LOGS_DIR and friends at logs/<run_name>/."""
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY, CLAUDE_SESSIONS
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    CLAUDE_SESSIONS = LOGS_DIR / "codex_sessions"


def _ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CLAUDE_SESSIONS.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "reports").mkdir(parents=True, exist_ok=True)


def _append_summary(row: dict) -> None:
    with EVOLUTION_SUMMARY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Proposer ─────────────────────────────────────────────────


PROPOSER_PROMPT_TEMPLATE = """You are the proposer agent for a Meta-Harness evolution loop
applied to YGN-SAGE, a Rust+Python self-adaptive multi-agent engine.

**ACT NOW. This is not a design discussion — you will use tools to
  read source + write a new Python file + write a pending_eval.json.**
  At the end of this turn, both files MUST exist on disk.

## Your task

Propose ONE new candidate harness that may improve the aggregate
`val_score` on SWE-bench Lite (offset=3 limit=5, measured by
`val_score = (real + 0.25 * sentinel) / total`).

## What you CAN change (mutable axes)

Anything reachable from the returned `AgentSystem`:
- Pipeline stages, TopologyEngine configuration, MAP-Elites archive
- TopologyRunner hooks (e.g. `_gather_predecessor_context`,
  `_execute_llm_node`), agent_loop_factory
- Memory tiers (episodic / semantic / ExoCortex retrieval policy)
- Provider routing (ModelAssigner penalty weights, per-node hints)
- Tool registry (add / remove / reorder tools)
- Agent prompts per role (F6 prompt overrides)
- Validation levels, step budgets, cost budgets

## What you CANNOT change (fixed by the harness)

- The base models exposed via providers (gpt-5.4, gemini-3.1, etc.)
- The benchmark dataset or scoring metric
- The sage-core Rust binary (you can parameterise through its exposed
  Python bindings but not recompile)
- The evaluation workflow (`python -m sage.bench --type swebench`)

## Filesystem layout (read FIRST with Bash/Read)

Repository root: {repo_root}
YGN-SAGE source:
  {repo_root}/sage-core/src/           (Rust)
  {repo_root}/sage-python/src/sage/    (Python)

This harness:
  {evolve_dir}/agents/            (existing candidates you can read + compare)
  {evolve_dir}/logs/{run_name}/   (prior iteration results — READ THESE)

Relevant files you should read to understand SAGE:
  sage-python/src/sage/boot.py                       boot_agent_system entrypoint
  sage-python/src/sage/pipeline.py                   CognitiveOrchestrationPipeline
  sage-python/src/sage/topology/runner.py            TopologyRunner
  sage-python/src/sage/agent_loop.py                 Per-node agent loop
  sage-python/src/sage/agent_loop_factory.py         Per-node agent factory
  external/meta-harness/reference_examples/ygn_sage/sage_candidate.py  (base class)
  external/meta-harness/reference_examples/ygn_sage/agents/baseline.py (reference)

## Contract for your new candidate

Write a NEW file at `{evolve_dir}/agents/<short_descriptive_id>.py`
that:

1. Inherits from `SageCandidate` (see `sage_candidate.py`).
2. Sets `name`, `hypothesis` (1-2 sentences on what you changed + why),
   and `axis` (one of: topology, memory, routing, prompts, tools, budget).
3. Overrides `build_system(hints)` to return a customised AgentSystem.
   Typical pattern: call `boot_agent_system()` for the baseline, then
   monkey-patch the returned object (or its pipeline / topology runner)
   with your changes.
4. Exposes `CANDIDATE = YourClass()` at module bottom.

## Proposal output

After writing the file, write `{pending_eval}`:

```json
{{
  "iteration": {iteration},
  "timestamp": "<ISO-8601>",
  "candidates": [
    {{
      "name": "<id>",
      "module": "reference_examples.ygn_sage.agents.<id>",
      "hypothesis": "<why you think this helps>",
      "axis": "topology | memory | routing | prompts | tools | budget",
      "file_path": "{agents_dir}/<id>.py",
      "changes_from_baseline": "<short diff summary>"
    }}
  ]
}}
```

## Prior history

Read `{evolution_summary}` and `{frontier_val}` if they exist to see
what has been tried. Do NOT propose a candidate that duplicates a
prior one; build on what worked (or what didn't).

## Rules

- One single candidate per proposal iteration.
- Change ONE axis per candidate (no multi-axis bundles — we can't
  attribute impact otherwise).
- Respect the base model and evaluation harness as fixed.
- Never touch files outside `{agents_dir}` and `{pending_eval}`.
"""


def _build_proposer_prompt(iteration: int, run_name: str) -> str:
    repo_root = EVOLVE_DIR.resolve().parents[3]  # external/meta-harness/reference_examples/ygn_sage → up 4 = repo root
    return PROPOSER_PROMPT_TEMPLATE.format(
        repo_root=str(repo_root),
        evolve_dir=str(EVOLVE_DIR),
        run_name=run_name,
        iteration=iteration,
        pending_eval=str(PENDING_EVAL),
        agents_dir=str(AGENTS_DIR),
        evolution_summary=str(EVOLUTION_SUMMARY),
        frontier_val=str(FRONTIER_VAL),
    )


def propose(config: dict, iteration: int, run_name: str) -> list[dict]:
    """Invoke Codex CLI to write a new candidate file + pending_eval.json.

    Returns the list of candidate dicts the proposer wrote. Empty if
    proposer failed or wrote nothing.
    """
    proposer_cfg = config.get("proposer", {})
    prompt = _build_proposer_prompt(iteration=iteration, run_name=run_name)

    print(_bold(f"\n[iteration {iteration}] Proposing via Codex CLI…"))
    # Clear the pending file so we can detect if the proposer wrote a new one
    if PENDING_EVAL.exists():
        PENDING_EVAL.unlink()

    result = codex_wrapper.run(
        prompt=prompt,
        cwd=EVOLVE_DIR.resolve().parents[3],  # repo root — proposer reads source there
        model=proposer_cfg.get("model", "gpt-5.4"),
        reasoning_effort=proposer_cfg.get("reasoning_effort", "high"),
        sandbox=proposer_cfg.get("sandbox", "workspace-write"),
        timeout=float(proposer_cfg.get("timeout_s", 600)),
        skip_git_repo_check=bool(proposer_cfg.get("skip_git_repo_check", True)),
        ephemeral=bool(proposer_cfg.get("ephemeral", True)),
        log_dir=CLAUDE_SESSIONS,
        name=f"propose-iter{iteration}",
    )
    print(_dim(f"  [codex] exit={result.exit_code} duration={result.duration_seconds:.1f}s "
               f"text={len(result.text)}ch tools={len(result.tool_calls)}"))

    if result.exit_code != 0:
        print(_dim(f"  proposer failed; stderr tail: {result.stderr[-300:]}"))
        return []

    if not PENDING_EVAL.exists():
        print(_dim("  proposer did not write pending_eval.json"))
        return []

    try:
        data = json.loads(PENDING_EVAL.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(_dim(f"  pending_eval.json invalid JSON: {exc}"))
        return []

    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        print(_dim("  pending_eval.json has no 'candidates' list"))
        return []

    print(_bold(f"  → proposed {len(candidates)} candidate(s): "
                + ", ".join(c.get("name", "?") for c in candidates)))
    return candidates


# ── Evaluation ───────────────────────────────────────────────


def evaluate(config: dict, candidate: dict, iteration: int) -> dict:
    """Run benchmark.evaluate() on a candidate; write report; return row."""
    bench_cfg = config.get("benchmark", {})
    module = candidate.get("module") or f"reference_examples.ygn_sage.agents.{candidate['name']}"

    print(_bold(f"\n[iteration {iteration}] Evaluating {candidate.get('name')}"))

    try:
        score = benchmark.evaluate(
            module,
            dataset=bench_cfg.get("dataset", "lite"),
            limit=int(bench_cfg.get("limit", 5)),
            offset=int(bench_cfg.get("offset", 3)),
            timeout_per_task_s=int(bench_cfg.get("timeout_per_task_s", 300)),
        )
    except Exception as exc:  # noqa: BLE001
        print(_dim(f"  evaluation raised {type(exc).__name__}: {exc}"))
        score = {
            "val_score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_s": 0.0,
        }

    val_score = float(score.get("val_score", 0.0) or 0.0)
    print(_bold(f"  val_score = {val_score:.3f}") + _dim(
        f"   (real={score.get('real',0)} sentinel={score.get('sentinel',0)} empty={score.get('empty',0)})"
    ))

    # Persist report
    report_path = LOGS_DIR / "reports" / f"{candidate.get('name','unknown')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"candidate": candidate, "score": score, "iteration": iteration},
                   indent=2, default=str),
        encoding="utf-8",
    )

    row = {
        "iteration": iteration,
        "timestamp": _now_iso(),
        "candidate_name": candidate.get("name"),
        "candidate_module": module,
        "hypothesis": candidate.get("hypothesis"),
        "axis": candidate.get("axis"),
        "val_score": val_score,
        "real": score.get("real"),
        "sentinel": score.get("sentinel"),
        "empty": score.get("empty"),
        "cost_usd": score.get("cost_usd"),
        "error": score.get("error", ""),
        "report_path": str(report_path),
    }
    _append_summary(row)
    return row


def update_frontier(rows: list[dict]) -> None:
    """Simple pareto: just track the single top val_score for now.

    Upstream uses multi-axis pareto (val_score × cost × latency). For v1
    we keep it flat; multi-axis is follow-up work (see ADR-010 "Alternatives").
    """
    if not rows:
        return
    best = max(rows, key=lambda r: r.get("val_score", 0.0))
    frontier = {"best": best, "updated_at": _now_iso()}
    if FRONTIER_VAL.exists():
        try:
            prior = json.loads(FRONTIER_VAL.read_text(encoding="utf-8"))
            if prior.get("best", {}).get("val_score", 0.0) >= best["val_score"]:
                frontier = prior  # keep the better prior
        except (json.JSONDecodeError, OSError):
            pass
    FRONTIER_VAL.write_text(json.dumps(frontier, indent=2), encoding="utf-8")


# ── Main loop ────────────────────────────────────────────────


def main() -> int:
    # DO NOT REMOVE: Windows consoles default to cp1252 and crash on the
    # arrows/box-drawing/ellipses sprinkled through our prints. The v2
    # dry-run (2026-04-18) died on exactly this after Codex succeeded.
    # Regression guard — if you strip this, restore the log test below.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(description="Meta-Harness × YGN-SAGE evolution loop")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe logs/<run-name>/ before starting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only propose; don't evaluate")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip proposer; evaluate existing agents/ instead")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sigint_handler)

    _update_paths_for_run(args.run_name)

    if args.fresh and LOGS_DIR.exists():
        import shutil
        shutil.rmtree(LOGS_DIR)
        print(_dim(f"[fresh] wiped {LOGS_DIR}"))

    _ensure_dirs()
    config = _load_config()
    rows: list[dict] = []

    if args.evaluate_only:
        # Evaluate every agents/*.py that is not a baseline file
        for py_file in sorted(AGENTS_DIR.glob("*.py")):
            if py_file.name in BASELINE_FILES:
                continue
            name = py_file.stem
            module = f"reference_examples.ygn_sage.agents.{name}"
            candidate = {"name": name, "module": module, "hypothesis": "(evaluate-only)"}
            row = evaluate(config, candidate, iteration=0)
            rows.append(row)
        # Always also evaluate baseline for the reference row
        baseline_row = evaluate(
            config,
            {"name": "baseline", "module": "reference_examples.ygn_sage.agents.baseline",
             "hypothesis": "reference"},
            iteration=0,
        )
        rows.append(baseline_row)
        update_frontier(rows)
        return 0

    for iteration in range(1, args.iterations + 1):
        if _interrupted:
            print(_dim("[main] exiting due to SIGINT"))
            break
        candidates = propose(config, iteration=iteration, run_name=args.run_name)
        if not candidates:
            print(_dim(f"[iteration {iteration}] skipping evaluation (no candidates)"))
            continue
        if args.dry_run:
            print(_dim("[--dry-run] not evaluating"))
            continue
        for cand in candidates:
            row = evaluate(config, cand, iteration=iteration)
            rows.append(row)
        update_frontier(rows)

    if rows:
        best = max(rows, key=lambda r: r.get("val_score", 0.0))
        print(_bold(f"\n[final] best val_score = {best['val_score']:.3f}  "
                    f"candidate={best['candidate_name']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
