---
paths:
  - "sage-python/src/sage/meta_harness/**"
  - ".sage-meta-harness/**"
---

# Meta-Harness: Harness Optimization Rules

## What is Meta-Harness?

An outer-loop that searches over SAGE's harness code — the code that determines
what context the LLM sees at each pipeline stage (Lee et al., arXiv 2603.28052).

Harness surface = TopologyRunner context aggregation + system prompts +
topology selection + quality thresholds + context budgets + similarity gates.

## When Acting as Proposer

When the user asks to run or advance a Meta-Harness search:

1. **DIAGNOSE** — Read filesystem FIRST:
   ```bash
   cat ~/.sage-meta-harness/leaderboard.json
   grep -r '"passed": false' ~/.sage-meta-harness/candidates/*/traces.jsonl | wc -l
   cat ~/.sage-meta-harness/candidates/<best>/traces.jsonl | head -30
   ```

2. **HYPOTHESIZE** — Identify which harness parameter caused failures

3. **PROPOSE** — `python -m sage.meta_harness propose` then edit config.json

4. **EVALUATE** — `python -m sage.meta_harness evaluate <id>`

5. **ITERATE** — Change ONE parameter at a time for clean attribution

## Key Insight

The proposer's advantage over text optimizers is access to raw execution traces.
Don't propose from scores alone — read the traces, find the root cause.
