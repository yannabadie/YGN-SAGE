# SWE-bench Lite Smoke — Debug Report

**Date:** 2026-04-17
**Run:** `sage-python/logs/swebench-smoke-20260417T111306Z/`
**Configs:** full / no_sage_recurse / no_toolforge / bare
**Dataset:** princeton-nlp/SWE-bench_Lite, N=5 per config, `--tier reasoner`
**Status:** smoke run in progress; pipeline wiring validated, **zero real patches produced**.

## TL;DR

Every task in every config so far produces the same 25-char "patch":

```
Agent finished at step 5
```

Two compounding root causes:

1. **`max_steps=5` per topology node is too tight for SWE-bench.**
   `sage/agent_loop_factory.py:74` sets `max_steps=5` for every topology
   node (comment: "1-3 steps typical, 5 max to prevent timeouts"). After
   the CORAL cherry-pick removed the S2+sequential bypass (commit
   `30ee004`), every SWE-bench task is now routed through the
   `sequential` template (planner → coder → synthesizer). Each node
   burns its 5 steps on `execute_bash` exploration and never reaches
   the "emit final answer" state.

2. **`phases/learn.py:91` fallback swallows the problem.**
   When the loop exits without a final text (only tool calls), learn's
   last line `return result_text or f"Agent finished at step {loop.step_count}"`
   returns the placeholder. This masks the real failure mode —
   downstream sees a non-empty string and treats it as a valid response.

## Evidence chain

From `sage-python/logs/swebench-smoke-20260417T111306Z/master.log`:

```
[TopologyRunner] node 0 (planner) completed via agent_loop, output 24 chars
Node 0 model upgraded to gemini-2.5-flash        # TopologyController CORAL
[TopologyRunner] node 0 (planner) completed via agent_loop, output 24 chars  # retry
[TopologyRunner] node 1 (coder) completed via agent_loop, output 24 chars
Node 1 model upgraded to minimax-m2.7
[TopologyRunner] node 2 (synthesizer) completed via agent_loop, output 24 chars
Node 2 model upgraded to gemini-2.5-flash
  [1/5] astropy__astropy-12907: PATCH (70333ms, 25 chars)
```

Prediction JSONL contents (from
`$TEMP/sage_swebench_*/predictions.jsonl`):

```jsonl
{"instance_id": "astropy__astropy-12907", "model_patch": "Agent finished at step 5\n"}
{"instance_id": "astropy__astropy-14182", "model_patch": "Agent finished at step 5\n"}
{"instance_id": "astropy__astropy-14365", "model_patch": "Agent finished at step 5\n"}
{"instance_id": "astropy__astropy-14995", "model_patch": "Agent finished at step 5\n"}
{"instance_id": "astropy__astropy-6938",  "model_patch": "Agent finished at step 5\n"}
```

24 characters = `"Agent finished at step 5"` minus the trailing newline.

## Positive signals — the new wiring works

- **TopologyController (CORAL `d3af215`/`47784c7`) actively fires**: it
  observes the low quality (24-char outputs), calls `_resolve_upgrade_model`,
  and retries with `gemini-2.5-flash` then `minimax-m2.7`. Telemetry is
  correct; only the downstream nodes can't finish.
- **Sequential topology (planner/coder/synthesizer) engages** on every
  task — CORAL's bypass removal (`30ee004`) is in effect.
- **system_hint=3 override applies** — pipeline log shows S3 context
  throughout.
- **kNN routing** loads 60 exemplars with Rust acceleration, no errors.
- **Per-task latency** is 46–70 s — acceptable once real patches appear.

## Secondary issue — model assignment for SWE-bench

Each topology node is assigned its model via Stage 3 (`ModelAssigner`).
Despite `--tier reasoner` at CLI, the assigner picks
`gemini-3.1-flash-lite-preview` for the planner (S2 cheap tier). Low
temperature on Gemini 3 additionally triggers LiteLLM warnings:

> Setting temperature < 1.0 for Gemini 3 models can cause infinite loops,
> degraded reasoning performance, and failure on complex tasks.

`--tier` only affects the boot default for **bypass** execution; per-node
models bypass the tier and go through affinity scoring. Documented
limitation from Sprint 2 ("system_hint does NOT re-pick a model") now
concretely biting.

## Recommended fixes (in priority order)

### F1 — Scale `max_steps` by `system_level` in `agent_loop_factory.py`

```python
# current: max_steps=5 everywhere
# proposed:
if system_level >= 3:
    max_steps = 20
elif system_level >= 2:
    max_steps = 10
else:
    max_steps = 5
```

Gives SWE-bench tasks (S3 via system_hint) enough room to explore + patch.

### F2 — Improve `phases/learn.py` fallback

Instead of `f"Agent finished at step {n}"`, emit the last ASSISTANT
message content (or concat all tool outputs) so downstream gets
**something** from the partial execution. Alternative: surface an
explicit error in the predictions JSONL so the harness records a miss
rather than a fake patch.

### F3 — Force reasoner-tier models for SWE-bench topology nodes

Options:
- Add a `node_model_override: dict[role, model_id]` param to
  `TopologyRunner` that bench adapters can populate (e.g. SWEBenchBench
  passes `{"planner": "gpt-5.4", "coder": "gpt-5.4", "synthesizer": "gpt-5.4"}`).
- Or: boost domain_score for SWE-bench tasks in `cards.toml` so the
  assigner picks reasoner-tier affinities first.
- Or: pass temperature=1.0 for Gemini 3 models globally (independent fix).

### F4 — Raise Gemini 3 temperature

Honour the LiteLLM warning: temperature < 1.0 on Gemini 3 degrades
reasoning. Should be handled in `providers/connector.py` or cards.toml.

## What the Sprint 6 decision gate will show

After completion, `decide_next_phase.py` will almost certainly return
**Gate B** (`full` < 20%) — but this would be the *wrong* conclusion.
The architecture isn't broken; the leaf-level agent config is. The real
next step is to apply F1–F4, re-run the smoke, and then interpret the
ablation.

## Action items

1. Let the current smoke complete so we have a full ablation baseline
   for the "agent-config-is-broken" world.
2. Apply F1 (max_steps by system_level) — highest-impact, smallest
   change.
3. Re-run smoke at `--limit 5`.
4. If still 0%, apply F2 and F3.
5. Only then interpret Sprint 6 gates.
