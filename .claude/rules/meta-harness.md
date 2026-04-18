---
paths:
  - "external/meta-harness/**"
---

# Meta-Harness: Harness Optimization Rules

## What is Meta-Harness?

An outer-loop that searches over harness code (Lee et al., arXiv 2603.28052).
Reference framework: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness),
cloned to `external/meta-harness/`. SAGE harness search lives as a
`reference_example` under that tree, not as an in-tree Python module.

Removed 2026-04-18: our home-made `sage.meta_harness` (ADR-010 explains
the divergence — it was a dataclass hyperparameter tuner, not true
structural-evolution search). The official framework allows candidates
to be full Python modules that override arbitrary agent behaviour.

## Two layers — don't confuse them

**YGN-SAGE runtime** (`sage-python/src/sage/`) — the product. Uses API
providers only (DeepSeek, Google, OpenAI, xAI, MiniMax, Kimi, OpenRouter).
Codex CLI was removed as a provider on 2026-04-07 and is NOT re-introduced
anywhere in `sage-python/`.

**Meta-Harness proposer** (`external/meta-harness/reference_examples/ygn_sage/meta_harness.py`)
— the OPTIMIZER for SAGE's scaffold. Calls `codex exec` via subprocess
(see `codex_wrapper.py`). Codex CLI is the CORRECT proposer here per the
paper's "10M tokens/iteration filesystem access" argument — the agent
reads prior candidates' source + traces + scores from disk instead of
being fed aggregates. User confirmed 2026-04-18: keep Codex CLI as the
Meta-Harness proposer; API providers only apply to the SAGE runtime.

If you need to change the proposer, EDIT `codex_wrapper.py` — don't
sprinkle a Codex call anywhere in `sage-python/`.

## Workflow (when you own the proposer role)

1. **DIAGNOSE** — Read filesystem FIRST at
   `external/meta-harness/reference_examples/ygn_sage/logs/<run>/`:
   ```bash
   cat logs/<run>/evolution_summary.jsonl | tail
   cat logs/<run>/frontier_val.json
   cat logs/<run>/reports/<best_candidate_id>.json
   ```

2. **HYPOTHESIZE** — Identify which scaffold axis hurt. The search is
   over Python modules in `agents/` that satisfy the candidate contract —
   NOT over numeric knobs.

3. **PROPOSE** — Write a new Python candidate to `agents/<name>.py`
   inheriting the SAGE candidate base class. Override hooks such as
   `build_system`, `topology`, memory retrieval, tool scaffolding.

4. **EVALUATE** — `uv run python meta_harness.py --iterations 1`
   (outer loop imports, validates, benchmarks, logs).

5. **ITERATE** — Change one axis at a time. Multi-axis changes = no
   clean attribution of what moved the score.

## Key Insight

The proposer's advantage over text optimizers is filesystem access to
prior candidates' source + traces + scores ("10M tokens per step" per
the paper). Don't propose from aggregate scores alone — read the
individual traces, find the root cause.
