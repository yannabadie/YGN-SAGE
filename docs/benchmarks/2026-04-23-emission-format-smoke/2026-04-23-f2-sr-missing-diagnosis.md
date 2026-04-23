# F2 — diagnosis of SR `missing` on astropy-14995 + astropy-7746

**Date:** 2026-04-23
**Follow-up from:** `2026-04-23-emission-format-summary.md` finding #3 ("2/4 SR blocks missed file match")
**Method:** log-only, zero `$`, zero code change. Read `2026-04-23-arm-b-search-replace.log` around the `SR fallback:` lines for each instance, plus the preceding TopologyRunner node sequence.

## TL;DR

The 2/4 `search-replace-missing` rate is **not** a Context7 threshold / fuzzy
tuning problem. It is a **topology-starved-synthesizer** problem.

Pattern across all 4 SR-using tasks in Arm B:

| Instance | Planner | Coder | Synthesizer (output, tool_calls) | Outcome |
|---|---|---|---|---|
| astropy__astropy-14365 | ok | stalled 19/20 → sentinel | **301 chars**, 0 calls | SR-**exact** (363-char patch) |
| astropy__astropy-6938 | ok | stalled 19/20 → sentinel | **635 chars**, 0 calls | SR-**exact** (727-char patch) |
| astropy__astropy-14995 | stalled 9/10 → sentinel | stalled 19/20 → sentinel | **2781 chars**, 0 calls | SR-**missing** |
| astropy__astropy-7746 | stalled 9/10 → sentinel | stalled 19/20 → sentinel | **2099 chars**, 0 calls | SR-**missing** |

The coder always stalls; the planner stalls on the two misses but finishes on the
two hits. The synthesizer (Gemini 2.5 flash, `tier=budget`) always runs with 0
tool calls. The distinction is output length: **short synthesizer output = SR
match, long synthesizer output = SR missing.**

## Why this shape

The synthesizer's job per the template is to take the coder's output and emit
the final patch. When the coder stalls, the `TopologyRunner` drops the sentinel
(`topology.runner: dropped sentinel output from predecessor 1 (role=coder)`),
so the synthesizer receives **no predecessor content** — just the original
problem statement plus whatever is still in the agent_loop's working memory
from tool calls two hops earlier. With 0 tool calls of its own, the
synthesizer cannot verify what text is actually in any file.

In that regime, the model has two stable behaviours:

- **Short**: emit the minimum SR block shape and copy search text it happens
  to have in its immediate context (from a tool call output that survived).
  301 / 635 chars. Matches.
- **Long**: fabricate a plausible SR block from what the model *thinks* the
  repo looks like (training-data-plus-common-sense). 2099 / 2781 chars. Does
  not match, because the fabricated code isn't what's actually at those
  line positions in the post-`base_commit` checkout.

The fuzzy threshold at 0.95 doesn't rescue the long-mode fabrications because
the divergence is not whitespace drift — it's different code.

## What the gold patches say

Both misses correspond to real, localised edits in real files:

- **astropy-14995** gold: 2-line edit in `astropy/nddata/mixins/ndarithmetic.py`
  around L520–523. Changes `operand is None` → `operand.mask is None` and fixes
  `lets` → `let's`.
- **astropy-7746** gold: 2 insertions in `astropy/wcs/wcs.py` L1212–1238,
  early-return on empty input arrays.

Both files exist at the base_commit. The problem statements are short and
actionable. An agent that actually reads the file around the target lines and
copies the real surrounding text into the SEARCH block will match on the first
try. The 0-tool-call synthesizer can't do that.

## What this rules out

- **Whitespace drift**: a whitespace-only divergence would have scored ≥ 0.95
  under `SequenceMatcher`, not `missing`. The fuzzy threshold is not the
  problem here. Do not tune it based on this case.
- **Wrong file path**: even if the model emitted the wrong `## File:` header,
  the extractor's `_scan_repo_for_unique_match` fallback scans the entire
  repo tree for a unique exact/fuzzy match. If the SEARCH text were a real
  repo substring, we'd have caught it. The fact that we didn't says the
  text is not in the repo.
- **Context7 / doc-lookup gap**: the misses are on stdlib-only astropy code,
  not third-party API boundaries. No amount of library docs would help.

## What this implicates

The decisive lever is **not the emission format**. It is what the synthesizer
node receives after predecessors stall. Options (not yet scoped here; this is
a diagnosis, not a plan):

1. **Surface tool-call history to the synthesizer.** When the predecessor
   produced a sentinel, inject the last N tool-call outputs (`read_file`
   results especially) into the synthesizer prompt instead of dropping the
   node entirely.
2. **Let the synthesizer make its own tool calls** before emitting. Currently
   it's `tier=budget` and runs 0 tool calls — that's fine for a true
   synthesis step but wrong when the upstream was empty.
3. **Refuse to emit SEARCH text for files that weren't read this turn.**
   Bench-side guard: reject SR blocks whose file path didn't appear in any
   `tool.call name=read_file args_keys=...path` during the run. Drop the
   patch to EMPTY rather than ship a fabricated SEARCH. Correct behaviour
   for Arm B but also worth auditing against Arm A (unified) where the
   same fabrication mode can produce malformed diffs.

## Instrumentation gap (blocker for future F2-class analyses)

`meta.json` captures `_extraction_method` but not the raw LLM response or the
parsed SR blocks. To diagnose missing-match cases post-hoc we have to:

- re-run the instance locally (costs `$` + 3-5 min + non-determinism risk), OR
- read the log and infer from node output lengths (this note's method — works
  for broad pattern, doesn't reveal the exact SEARCH text).

**Proposed minimal fix** (~10 LOC inside `generate_patches`, bench-side): when
`extraction_method == "search-replace-missing"`, persist the raw response and
the parsed block(s) to a sidecar file alongside `predictions.jsonl`, keyed by
`instance_id`. Gated by an env var so normal runs don't bloat disk.

## Not proposed in this note

- No fuzzy-threshold tuning. The advisor's directive-#2 guardrail applies:
  don't tune a magic number when the fault is elsewhere in the pipeline.
- No emission-format default change. Arm A (unified) ships as default;
  Track 2's gate stays operator-opt-in.
- No Track 3 re-ordering. This note is input to Track 3.1, which is the
  right next step (check whether astropy-14182's agent reads the test file
  — a different-shape-but-related question about what the agent actually
  sees before emitting).

## Artefacts

- Log: `docs/benchmarks/2026-04-23-emission-format-smoke/2026-04-23-arm-b-search-replace.log`
  lines 450–608 (astropy-14995) and 700–838 (astropy-7746).
- Meta: `..-arm-b-search-replace-meta.json`, both entries
  `_extraction_method: "search-replace-missing"`.
- Gold patches: pulled from the cached HF dataset
  `princeton-nlp/SWE-bench_Lite` (cf. `datasets` cache at
  `C:\Users\yann.abadie\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite`).
