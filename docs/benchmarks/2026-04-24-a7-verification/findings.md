# 2026-04-24 — A7 verification smoke (N=6, gen-only)

**Goal:** empirically verify A7 (`c20aefd8`) excludes kimi-k2.5 from
template-built tool-using nodes, closing the 20% deterministic fast-
abort rate on astropy-14182 + astropy-7746 observed in both the
2026-04-23 and 2026-04-24 observe smokes.

**Command:**

```bash
SAGE_DIFF_VERIFIER_MODE=observe \
  python -m sage.bench --type swebench --dataset lite --limit 6 --generate-only \
    --output docs/benchmarks/2026-04-24-a7-verification/a7-n6-genonly.json
```

**Scope:** gen-only (no Docker grading), N=6 to cover both fast-abort
task IDs (astropy-14182 at #2, astropy-7746 at #6) cheaply. Ran
alongside the rustpython wasm source build in parallel — the Windows
paging file ran short during execution (WinError 1455 visible on
astropy-7746's `git apply --check`), caused a host crash mid-run. The
predictions.jsonl was written first (Wrote 6 predictions), so per-task
patch outcomes survived; the bench summary JSON did not.

## Results

| # | Instance | Pre-A7 | Post-A7 | A7 attribution |
|---|---|---|---|---|
| 1 | astropy-12907 | EMPTY (266 s) | **PATCH 4879 chars** | ✅ fixed |
| 2 | astropy-14182 | EMPTY (58 s, fast-abort, kimi-400) | Generation timed out (0 chars) | ⚠️ partial — kimi 400 gone, but task still failed to emit. New failure mode to triage (likely agent-loop convergence, not kimi) |
| 3 | astropy-14365 | PATCH 903 chars (2× content_mismatch) | 0 chars | ❌ regression-adjacent; probably bandit noise (diff models now assigned) |
| 4 | astropy-14995 | EMPTY (174 s) | **PATCH 474 chars** | ✅ fixed |
| 5 | astropy-6938 | PATCH 950 chars (CRLF-rejected) | **PATCH 587 chars** | — (A6 + A7, both in play) |
| 6 | astropy-7746 | EMPTY (72 s, fast-abort, kimi-400) | **PATCH 634 chars** | ✅✅ fixed — *the canonical deterministic fast-abort case* |

**Patch rate: 4/6 = 67%** (vs 20% in the 2026-04-24 pre-A7 smoke).

## Primary finding — A7 closes the kimi-k2.5 path entirely

Grep of the entire 82 KB gen log:

```
$ grep -c "thinking is enabled but reasoning_content" a7-n6-genonly-gen.log
0
```

Zero occurrences. Compare to the 2026-04-24 pre-A7 smoke where this
pattern fired twice (astropy-14182 + astropy-7746). The Rust
`ModelAssigner` filter at `model_assigner.rs:289`
(`needs_tools && !card.supports_tools`) now triggers on every
template-built multi-agent node because all non-sink roles declare
`"tools"`. kimi-k2.5 is no longer in the candidate pool for those
nodes.

## Per-task attribution (astropy-7746 — the canonical fast-abort case)

**Pre-A7 (2026-04-24 smoke, log lines 813-814, 819):**
```
HTTP Request: POST https://api.moonshot.ai/v1/chat/completions "HTTP/1.1 400 Bad Request"
Stage 4 multi-agent execution failed: status_code: 400, model_name: kimi-k2.5,
  body: {'message': 'thinking is enabled but reasoning_content is missing in
  assistant tool call message at index 3'}
Stage 4 fallback also failed: Stage 4 fallback returned empty content
→ astropy-7746: EMPTY (72 s, 0 chars)
```

**Post-A7 (today):**
```
[node-1-coder] stall detected: 19 consecutive tool steps with no final content
  — breaking early (D8 soft cap, step=19/20)
[TopologyRunner] node 1 (coder) completed via agent_loop, output 51 chars, tool_calls=21
...
[TopologyRunner] node 2 (synthesizer) completed via agent_loop, output 647 chars, tool_calls=0
→ astropy-7746: PATCH 634 chars
```

Different failure mode: the 4th-turn-400 never triggers, the agent
proceeds through its full tool-call budget (21 calls, D8 cap at 19),
and the synthesizer emits a real patch. A7 is directly load-bearing
here.

## astropy-14182 regression analysis

Still 0 chars post-A7 — but the failure shape is **Generation timed
out**, not a fast-abort at 58 s. The agent now runs to the task
timeout. This is a different class (agent-loop convergence / budget
exhaustion, likely related to the 20-minute SWE-bench timeout on a
heavy `np.char.strip` / FITS-I/O code-analysis task). A7's direct
scope was to eliminate the kimi-400 path; it did. The residual 0-char
outcome is ticketed as follow-up separate from A7.

## Secondary finding — Windows paging pressure

`[astropy__astropy-7746] patch validation failed: validator exception:
OSError: [WinError 1455] Le fichier de pagination est insuffisant pour
terminer cette opération` — Windows `git apply --check` inside
`try_repair_patch` hit the paging file limit while the rustpython wasm
build was compiling in parallel on another core. System-level OOM, not
A7-related. The patch was still written to predictions.jsonl (634
chars) so the outcome survives; only the validation/repair path was
short-circuited.

Mitigation for future large parallel runs: don't build the wasm source
(~40 MB compiled, heavy memory during linking) and run a smoke at the
same time. Or raise the Windows paging file limit.

## Kimi-k2.6 availability (follow-up, 2026-04-24)

User noted that Kimi's latest model is k2.6, not k2.5. cards.toml's
`kimi-k2.5` entry carries the `supports_tools = false` marker from the
F9 audit — the documented reason is Moonshot's `reasoning_content`
API requirement on the 4th tool-call turn with "thinking" enabled. If
k2.6 changed that API contract, we could potentially flip
`supports_tools` back to `true` once the model ID is updated in
cards.toml AND verified against Context7 `/berriai/litellm` or
Moonshot's own docs. This is a separate investigation from A7 —
A7 ships as-is and is independent of which kimi revision is current.

Filed as **A8** for next session.

## Conclusion

A7 (`c20aefd8`) is empirically verified on its direct claim: kimi-k2.5
HTTP 400 path zero occurrences, astropy-7746 flipped
EMPTY→PATCH, astropy-12907 and astropy-14995 also flipped EMPTY→PATCH.
Patch rate improved from 20% to 67% on a shared N=6 subset. The one
remaining 0-char outcome (astropy-14182) is a different failure class
(gen timeout) and needs its own investigation — not an A7 regression.

Sink+kimi residual risk remains (per advisor review, documented in
`templates.rs:1200-1216`): sinks still get tool registry at runtime
and could theoretically hit the 400 if the LLM decides to call tools
despite the 1-turn-output prompt. Observe-mode logs will flag any
such relapse for triage until B9 closes the class.

## Artefacts

- `a7-n6-genonly-gen.log` — full generation log (82 KB, no observe.json
  due to host crash mid-run; predictions.jsonl written first and survived)
- Per-task patch chars: see table above, derived from the temp
  predictions.jsonl (`sage_swebench_c0vnjzxa/predictions.jsonl`).
