---
name: April 23 — Track 3 semantic-miss investigation + close-out
description: Track 3 (SWE-bench semantic-miss bucket) close-out. Sub-tasks 3.1/3.2 invalidated, 3.5 deferred, 4 hygiene commits landed. Three distinct failure modes identified across 3 tracers (context hallucination, over-scope, wrong layer).
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
Track 3 investigation on SWE-bench Lite semantic-miss bucket, 2026-04-23.

**Four ship items landed** (main branch commits):
- `cb03773` — F3: `_extraction_method` persisted in predictions.jsonl
- `29987bc` — prompt hygiene: `search_exocortex` reframed + SR template
  carryover fixed (opening line + step 7)
- `2793a74` — F2 instrumentation: `SAGE_PERSIST_SR_MISSING=1` env flag
  writes raw response + parsed SR blocks sidecar at
  `<out_dir>/sr_missing/<instance_id>.json` when match fails (default OFF)
- `9ec3dfd` — gen-log-by-default for `--type swebench`
  (`<args.output.stem>-gen.log` sibling; opt-out via
  `SAGE_BENCH_LOG_FILE=0`)

**Sub-task outcomes:**
- 3.1 INVALIDATED (both tracers read test files — `_executed_commands`
  field proves it)
- 3.1b found 3 distinct failure modes across 3 tracers: context
  hallucination (astropy-14182 Arm B: diff emitted `class RST(Table):`
  where base_commit has `class RST(FixedWidth):`), over-aggressive scope
  (astropy-14182 Arm A: deleted a helper mid-rewrite), right-intent
  wrong-layer (django-10924 Arm A: coerced callable `path` in `__init__`
  losing Django's migration autodetector reference)
- 3.2 DELETED (dead hypothesis — agents DO read test files)
- 3.3 prompt-gap → shipped 29987bc
- 3.4 max_steps is NOT the lever (4/6 Arm B Empties have tool_call_count=0,
  i.e., early-abort not budget-exhaustion)
- 3.5 N=50 paired smoke DEFERRED (SE ~2pp at N=50; no remaining prompt-
  level change has a ≥4pp lift hypothesis)

**Why:** advisor-driven decision to avoid "measurement theater" — no
smoke without a concrete lift hypothesis worth measuring.

**How to apply:** when Track 3 resumes, two breadcrumbs are pinned in
`docs/benchmarks/2026-04-23-track3-closeout.md`:
1. pre-emission diff-context verifier (addresses astropy-14182 Arm B
   hallucination)
2. django-10924 wrong-layer: open question on whether domain-knowledge
   injection is addressable at the prompt level or is an irreducible
   failure mode.

**Related F2 pattern:** "topology-starved synthesizer" — when both
planner and coder stall, `TopologyRunner` drops the sentinel and the
synthesizer runs with 0 tool calls and fabricates content. Correlates
perfectly with SR-missing at N=4 (2/2 miss vs 2/2 hit on the hit-tasks
where the planner did NOT stall). Two candidate mechanisms still
indistinguishable (verbose-fabrication vs working-memory-starvation) —
the SR-missing sidecar unlocks future disambiguation.
