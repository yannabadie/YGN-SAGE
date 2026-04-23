# Track 3 — close-out note

**Date:** 2026-04-23
**Context:** Follow-up to the 2026-04-23 emission-format smoke (Track 2).
**Scope:** Track 3 investigates the semantic-miss bucket on SWE-bench Lite
— instances where a patch applies cleanly but fails the FAIL_TO_PASS
tests, or emits structurally-valid-but-semantically-wrong content.
**Outcome:** no N=50 smoke this cycle. Three prompt-hygiene commits shipped
(F3 JSONL field + prompt reframe + SR-missing sidecar + default gen-log),
two breadcrumbs recorded for the next Track 3 cycle.

## What was investigated

| Sub-task | Answer | Source |
|---|---|---|
| 3.1 | Agent **does** read test files on the semantic-miss tracers | `_executed_commands` on astropy-14182 Arm B (37 tool calls incl. `[coder] read_file: astropy/io/ascii/tests/test_rst.py`) and django-10924 Arm A (41 calls incl. `read_file: tests/forms_tests/field_tests/test_filepathfield.py` under `stage_0` and `stage_2`). Hypothesis INVALIDATED. |
| 3.1b | Semantic-miss failure modes are **3 distinct categories**, not a single mode | Gold-vs-emitted diff (hand, ~30 min): (a) context hallucination — emitted diff context `class RST(Table):` where base_commit has `class RST(FixedWidth):`; (b) over-aggressive scope — deleted a helper method mid-rewrite; (c) right intent wrong layer — coerced callable `path` in `__init__` instead of `formfield`, losing Django's migration-autodetector reference. |
| 3.2 | Dead. Adding "Step 0: read test files" addresses a problem that doesn't exist (3.1 outcome), and would not fix any of the 3.1b failure modes either. | — |
| 3.3 | `search_exocortex` is registered, visible to coder/planner/actor nodes, named in both templates, and called **0 times** across 616 tool-call entries in 4 smoke logs. Verdict: prompt-gap. Current framing ("Almost never the right tool for day-to-day bug fixes") is self-defeating. | Read-only subagent audit; earlier 2026-04-21 audit flagged the same gap and applied a partial softening that added the bullet but kept the anti-affordance. |
| 3.4 | `max_steps` is not the lever. 4/6 Arm B Empties have tool_call_count=0 (early abort, not budget exhaustion). No instance hit the 300s timeout ceiling on Arm B. | `_tool_call_count` and `_latency_ms` fields on meta.json. |
| 3.5 | N=50 paired smoke **deferred**. With SE ~2 pp at N=50, needing ≥4 pp lift to beat the noise floor, and no prompt-level change whose lift hypothesis justifies that threshold after 3.1b, the smoke would be measurement theater. | Advisor guidance 2026-04-23. |

## Orthogonal finding from F2 (not Track 3 proper but relevant)

The 2/4 `search-replace-missing` cases from the Track 2 smoke share an
identical log signature: both planner AND coder stalled → sentinel
dropped by `TopologyRunner`; synthesizer ran with 0 tool calls and
emitted 2000+ chars. The two SR-exact cases had tiny synthesizer output
(301 / 635 chars) from the same node pattern. Correlation is solid at
N=4; two candidate mechanisms (verbose-fabrication vs working-memory
starvation) remain indistinguishable until raw responses are persisted.
See `2026-04-23-f2-sr-missing-diagnosis.md` and the SR-missing sidecar
(shipped this cycle) for future diagnosis. Possibly-important:
the synthesizer node also fabricated context in astropy-14182 Arm B
(patch applied with the wrong base class in the context header) — same
shape as F2 but on the unified path, so the fault is not emission-
format-specific.

## What shipped

- **F3** — `_extraction_method` field persisted in `predictions.jsonl`
  (commit `cb03773` on main, post-rebase). Unblocks per-bucket analysis
  from jsonl directly.
- **Prompt hygiene** — `search_exocortex` bullet reframed in both SWE-
  bench templates; SR template opening line + step 7 de-contradicted
  (commit `<TBD>`). Hygiene only, no expected lift.
- **F2 instrumentation** — `SAGE_PERSIST_SR_MISSING=1` env flag writes
  raw response + parsed blocks + per-block match attempts to
  `<out_dir>/sr_missing/<instance_id>.json` when extraction is
  `search-replace-missing`. Default OFF.
- **Gen-log-by-default for `--type swebench`** — the parity-smoke
  gen-phase log was not captured to disk (only the Docker-eval stdout).
  Fixed so the 14182-class post-hoc investigations are log-derivable
  (commit `<TBD>`).

## Breadcrumbs for the next Track 3 cycle

1. **astropy-14182 Arm B context hallucination.** The emitted diff
   context asserts `class RST(Table):`; base_commit has
   `class RST(FixedWidth):`. Model ran 37 tool calls including
   `read_file` on the target file twice. The hallucination survived
   tool use. Next-investigation candidate: pre-emission diff-context
   verifier (read the file around the hunk's line range immediately
   before emission, reject if context mismatches). Cost estimate: a
   new spec + LLM call per emission ≈ 1-2 s latency + $0.01-0.05 per
   task. Unlocks a lift claim worth N=50 measuring.

2. **django-10924 Arm A wrong-layer coercion.** Agent understood the
   semantic problem but placed the fix in `__init__` instead of
   `formfield`, losing Django's migration autodetector. That's domain-
   specific knowledge (Django ORM serialization), not library-API
   surface that `lookup_library_docs` covers. Open question: is this
   addressable at all at the prompt level, or does it require either
   (a) a Django-specific knowledge injection at run time (scope creep),
   or (b) accepting it as an irreducible failure mode? No immediate
   action.

## What this cycle explicitly does NOT do

- No N=50 smoke.
- No noise-floor calibration run (can be computed post-hoc from any
  future N=50+ smoke via resampling — no dedicated spend warranted).
- No verifier-pass implementation. Breadcrumb #1 is the hook for a
  future spec.
- No domain-knowledge injection. Breadcrumb #2 is a question, not a
  plan.
- No Track 2 regression. Unified emission remains the default; SR stays
  operator-opt-in. Per-bucket JSONL + sidecar give us the data for any
  future SR threshold tuning, but no tuning happens in this cycle.
