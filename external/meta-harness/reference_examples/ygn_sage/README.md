# Meta-Harness × YGN-SAGE reference example

Scaffold for running harness search against YGN-SAGE's self-adaptive
multi-agent engine. Follows the same pattern as the upstream
`text_classification` and `terminal_bench_2` examples:

```
propose (Claude CLI subprocess) → write agents/<id>.py → validate import
  → benchmark via python -m sage.bench → parse JSON report → update frontier
  → append evolution_summary.jsonl → repeat
```

## Candidate contract

Each candidate is a Python module under `agents/` exposing a single
top-level symbol named `build_system()`:

```python
# agents/<candidate_id>.py
from sage.boot import boot_agent_system

def build_system(hints: dict | None = None) -> "AgentSystem":
    """Return a configured AgentSystem (the SAGE entry point) on which
    `system.run(task)` can be called. The proposer may:

    - Override agent_loop_factory to use a custom runner
    - Replace memory tiers (semantic cache, ExoCortex hook)
    - Rewire topology templates
    - Swap provider-selection heuristics
    - Introduce new tool scaffolding

    The base model, bench harness, and evaluation dataset are fixed.
    """
    return boot_agent_system()  # baseline: return the default SAGE system
```

Hooks the proposer can override: anything reachable from the returned
`AgentSystem` — pipeline stages, TopologyRunner methods, phases,
memory, tool registry, provider pool.

## Running

From repo root:

```bash
cd external/meta-harness/reference_examples/ygn_sage
uv sync                                          # install deps
uv run python meta_harness.py --iterations 1     # one evolve step
uv run python meta_harness.py --iterations 10 --fresh   # 10 candidates
uv run python meta_harness.py --run-name nightly-sweep --iterations 20
```

## Directory layout per run

```
logs/<run_name>/
  pending_eval.json         # candidates proposed this iteration
  frontier_val.json         # pareto frontier (val set)
  evolution_summary.jsonl   # one row per (candidate × iteration)
  reports/
    <candidate_id>.json     # raw bench output
  claude_sessions/
    <timestamp>.log         # proposer Claude session
```

## Eval set

By default: SWE-bench Lite offset=3 limit=5 (the 5-task smoke window
we've been iterating on). Configurable via `config.yaml`.

Held-out test: MASBENCH breadth (only `breadth` axis, p=0.015 significant).
Never exposed to the proposer — used at end of sweep for final score.

## Baseline candidates (`agents/`)

- `baseline.py` — `build_system()` returns the default SAGE boot, no
  customization. Sets the floor.
- (future) `sequential_only.py` — always sequential topology (remove
  MAP-Elites, bandit).
- (future) `minimax_only.py` — pin coder to MiniMax (matches what routing
  converges to in practice; baseline for "dumb" routing).

## Proposer

`claude_wrapper.py` is vendored from the upstream framework. Invokes the
Claude Code CLI as subprocess with a restricted tool set (Read, Glob,
Grep, Agent, Write, Edit, Bash) and the `meta-harness` skill. The
proposer reads `logs/<run>/` (source + traces + scores of all prior
candidates) then writes a new `agents/<id>.py`.

## State

**2026-04-18**: scaffold initialized. Not yet runnable end-to-end —
`meta_harness.py`, `benchmark.py`, `llm.py` still need to be ported from
upstream text_classification + adapted for SAGE. This README documents
the intended shape so the next session (or proposer) has the spec.

## References

- Paper: Meta-Harness: End-to-End Optimization of Model Harnesses, Lee et al.,
  arXiv 2603.28052 (2026-03-30).
- Upstream: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness)
- ADR: `YGN-SAGE/Decisions/ADR-010-Meta-Harness-Divergence.md`
