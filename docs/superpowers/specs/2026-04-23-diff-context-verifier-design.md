# Design — pre-emission diff-context verifier for SWE-bench

**Status:** draft — design only, no implementation in this commit.
**Date:** 2026-04-23
**Motivation:** breadcrumb #1 from Track 3 close-out
(`docs/benchmarks/2026-04-23-track3-closeout.md`).
**Precedent:** partial precedent in the SR-emission path where
`_scan_repo_for_unique_match` already does content-matching (Track 2,
commit `19e6e90`). This spec extends the same discipline to unified
diffs.
**Out of scope:** LLM-based semantic verification (option C from
triage); full trace-time runtime assurance; any change to the SR
extractor.

## Problem

SWE-bench semantic-miss tracer — `astropy-14182 Arm B`:

| Step | Observation | Source |
|---|---|---|
| Agent ran 37 tool calls incl. `read_file: astropy/io/ascii/rst.py` (twice) | Tools clearly available + used | `_executed_commands` in `docs/benchmarks/2026-04-22-swebench-parity-smoke/2026-04-22-parity-typed-meta.json` |
| Base commit `astropy/io/ascii/rst.py` has `class RST(FixedWidth):` | File is authoritative | HF cache `princeton-nlp/SWE-bench_Lite` + `git show` at `base_commit` |
| Emitted diff's hunk header reads `@@ -35,6 +35,7 @@ class RST(Table):` | Hunk **context header asserts the wrong base class** | `model_patch` field in meta.json |
| `git apply` accepted the patch | Context line `class RST(` prefix-matched; single-line addition didn't collide | Arm B outcome: PATCH (applied) |
| FAIL_TO_PASS tests still failed | No actual semantic change — added a marker attribute that doesn't exist in the real base class | 2026-04-22 parity summary, `_repair_stage: failed` |

The hallucination survived 2 reads of the target file. Neither the
existing `git apply --check` (which accepts prefix-matching on a
single-line addition) nor the `try_repair_patch` pipeline (which
repairs malformed hunks, not wrong content) catches this class.

The SR emission path would have caught it: `_scan_repo_for_unique_match`
only emits a diff when the SEARCH text matches file bytes exactly or at
fuzzy≥0.95. On astropy-14182 the SR fallback wasn't triggered because
unified emission came back non-empty. This is precisely the gap — the
unified path has no content-matching guard.

## Proposed verifier

### Scope

**Unified diffs only.** SR emissions already validate content via
`_blocks_to_unified_diff` → `_scan_repo_for_unique_match`. Adding a
second check there would be duplicative.

**What gets verified (A + file-exists):**

1. Each hunk's target file **exists** in the repo clone.
2. Each hunk's context (` `) and removed (`-`) lines match the file
   content at the claimed hunk position (old-side start line + count).
3. Match policy:
   * **Exact** line-by-line equality → pass.
   * **Whitespace-stripped equality** (each line `.strip()`-equal,
     pairwise): pass; record the raw `SequenceMatcher.ratio()` for
     telemetry and emit `kind="fuzzy_below_threshold"` when the raw
     ratio falls below 0.95 so post-hoc analysis can see how much
     whitespace drift the repo tolerates.
   * **Otherwise → `kind="content_mismatch"`**, regardless of ratio.

   **Implementation note (spec correction, 2026-04-23 c05eee0).** The
   earlier draft said "exact or `SequenceMatcher.ratio() ≥ 0.95`",
   deliberately parroting the SR extractor's number. Test 2 caught
   that this literal policy fails the motivating case: the verbatim
   `astropy-14182 Arm B` emitted patch has 1 hallucinated line in a
   6-line body; the raw `SequenceMatcher.ratio()` is 0.956, above the
   0.95 cut-off. A generic "close-enough" gate lets semantic
   hallucinations through. The shipped policy narrows the fuzzy branch
   to **whitespace divergence only**: stripped-line equality is the
   gate, not raw similarity. 0.95 still appears on the `match_ratio`
   field for observability but no longer as an accept/reject threshold
   for anything other than *whitespace-drift reporting* (the
   `fuzzy_below_threshold` kind). Open question #1 in this spec is now
   resolved in favour of "not parallel to SR's number at all"; the
   two verifiers check different things and the symmetry was cosmetic.

**What is NOT verified:**

* Semantic correctness of the added lines (would need a second LLM call
  — option C, out of scope per Directive #2 no-scope-creep).
* Whether the edit actually fixes the FAIL_TO_PASS test (this is what
  the Docker harness measures; the verifier is upstream).
* Path style (a/b prefixes, absolute vs relative) — the existing
  `swebench_ca_patch` already normalises these.

### Hook location

**Pre-`try_repair_patch`, inside `generate_patches`.** The call site is
already in `sage-python/src/sage/bench/swebench_bench.py:1018-1030`
(the `if patch and repo_dir:` block). Current order:

```
emission → try_repair_patch → git apply → predictions.jsonl
```

New order:

```
emission → verify_diff_context → (mismatch? → repair path) → try_repair_patch → git apply
```

The verifier runs BEFORE `try_repair_patch` because the repair
pipeline's LLM one-shot uses `git apply --check` stderr as feedback —
and a hallucinated-context-line hunk applies cleanly at the
prefix-matching level (see the 14182 root cause). Feeding the repair
LLM "apply succeeded" when the hunk is semantically wrong defeats the
repair stage. The context verifier surfaces the mismatch with concrete
line-number + expected-vs-actual diagnostic that the repair LLM can
then use as ground truth.

### Mismatch handling

Three candidate reactions:

* **(X)** Drop the patch → emission becomes EMPTY, bench records
  `_extraction_method = "context_mismatch"`. Harshest; predictable.
* **(Y)** Route to the repair LLM with the mismatch diagnostic as the
  stderr input. Re-emit, re-verify. Max 1 repair attempt to bound cost.
* **(Z)** Log + let it through. Observability only, no behaviour change.

**Choice: (Y) with (Z) as first-deploy default.**

Ship the verifier in Z mode (observability only) for one smoke. Confirm
the verifier's mismatch bucket lines up with expected semantic-miss
cases (astropy-14182 should flag mismatch; cleanly-resolved tasks
should not). Only then flip to (Y) once we've seen the bucket composes
sanely. Flip gated on `SAGE_DIFF_VERIFIER_MODE={off,observe,repair}`.
Default `off` until the observability pass lands data.

### Implementation sketch

```python
# sage-python/src/sage/bench/swebench_diff_verifier.py  (NEW)

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class HunkMismatch:
    file: str
    hunk_index: int
    old_start: int
    old_count: int
    expected: list[str]    # what the agent said was in the file
    actual: list[str]      # what's really there
    kind: Literal["file_missing", "content_mismatch", "fuzzy_below_threshold"]
    match_ratio: float     # SequenceMatcher ratio when kind is fuzzy_*

def verify_diff_context(
    diff: str,
    repo_dir: Path,
    fuzzy_threshold: float = 0.95,
) -> list[HunkMismatch]:
    """Return one HunkMismatch per problematic hunk; empty list = all ok.
    Unified-diff input only. SR blocks should not be passed here
    (they're validated separately in _scan_repo_for_unique_match).
    """
    ...

# sage-python/src/sage/bench/swebench_bench.py  (MODIFY)

if patch and repo_dir:
    mode = os.environ.get("SAGE_DIFF_VERIFIER_MODE", "off")
    mismatches = []
    if mode in {"observe", "repair"}:
        mismatches = verify_diff_context(patch, repo_dir)
        if mismatches:
            log.info(
                "[%s] diff verifier: %d hunk mismatches",
                instance_id, len(mismatches),
            )
    if mode == "repair" and mismatches:
        # Attempt one repair pass with the mismatch diagnostic as
        # context. Reuse try_repair_patch's LLM handle path, but
        # feed our own stderr message instead of git-apply's.
        patch, repair_stage = await _repair_with_verifier_feedback(
            patch, mismatches, repo_dir, llm_handle, instance_id,
        )

    # Existing try_repair_patch path runs regardless (mode=off
    # preserves today's behaviour exactly).
    patch, repair_stage = await try_repair_patch(...)
```

### Instrumentation

New field on the prediction dict: `_diff_verifier_mismatches` —
serialised `HunkMismatch` list (empty on clean, absent on `mode=off`).
Persist through `write_predictions` the same way `_extraction_method`
was piped in F3 (commit `cb03773`). Enables per-bucket analysis on
future smokes without log grep.

### Tests (TDD spec)

In `sage-python/tests/test_swebench_diff_verifier.py` (new):

| # | Case | Expectation |
|---|---|---|
| 1 | Clean diff against a real fixture file | `[]` |
| 2 | `astropy-14182 Arm B` patch against a fixture simulating the base-commit `rst.py` | exactly one `HunkMismatch` with `kind="content_mismatch"`, expected/actual arrays contain `Table` vs `FixedWidth` |
| 3 | Hunk targets a nonexistent file | one `HunkMismatch` with `kind="file_missing"` |
| 4 | Whitespace-only drift (indent) | `kind="fuzzy_below_threshold"` OR clean, depending on ratio; test both sides of 0.95 |
| 5 | Multi-hunk diff, one bad one good | exactly one `HunkMismatch` |
| 6 | Unified diff with no file headers (malformed) | empty list — we defer to `try_repair_patch` for malformed diffs |
| 7 | Line count mismatch in hunk header (`@@ -35,6 +35,7 @@` but only 5 context lines) | empty list for this verifier's scope; the counts-repair path handles it |

In `sage-python/tests/test_swebench_emission_wiring.py`:
* Wire test: `SAGE_DIFF_VERIFIER_MODE=observe` populates
  `_diff_verifier_mismatches` in predictions.jsonl.
* Wire test: `SAGE_DIFF_VERIFIER_MODE=off` does NOT touch the
  prediction dict (byte-identical output to today's).

## Validation plan

### Observability smoke (first ship)

N=10 Lite slice, same as parity smoke. `SAGE_DIFF_VERIFIER_MODE=observe`.
Expected:
* `astropy-14182 Arm B` predictions record non-empty
  `_diff_verifier_mismatches`.
* Cleanly-resolved tasks record empty mismatches.
* Intersection of mismatch-reporting tasks and tests-failed tasks is
  qualitatively coherent.

Cost: ~30 min gen + Docker-eval. ~$5-10. No smoke number is
interpreted as a lift claim — this is observability only.

### Repair-mode validation (second ship, conditional)

Only if observability smoke shows the mismatch bucket aligns with
the semantic-miss bucket. N=50, paired:
* Arm A: `SAGE_DIFF_VERIFIER_MODE=off` (today's behaviour).
* Arm B: `SAGE_DIFF_VERIFIER_MODE=repair`.

Decision gate:
* Arm B resolved-rate ≥ Arm A (non-regression).
* Arm B patch-produced rate not catastrophically lower than Arm A
  (verifier shouldn't drop everything — `(Y)` with 1-attempt repair
  should keep most patches).
* Minimum detectable lift at N=50 is ~4 pp (combined SE ~2 pp); don't
  over-interpret anything smaller.

### Kill criteria

Drop the feature if:
* The observability smoke shows `_diff_verifier_mismatches` non-empty
  on tasks that *resolved* at higher rate than the mismatch-free set.
  That would mean the verifier is triggering on benign prefix-matches
  and the repair mode would hurt.
* The fuzzy threshold is driving the signal (observability data
  dominated by `fuzzy_below_threshold` rather than `content_mismatch`).
  That means we're re-creating the F2 `search-replace-missing`
  threshold-tuning trap in a different coat.
* Per-task wallclock increases by >30 % in repair mode. Cost vs
  hypothetical lift doesn't pencil.

## Alternatives considered

### (B) Path-hallucination verifier (subset of A+file-exists)

Already covered inside the proposed design (file-exists is part of A).

### (C) Semantic verifier via second LLM call

"Does this diff actually fix the problem?" — requires an LLM pass
per emission. Estimated cost: +$0.02-0.10/task, +1-2 s latency. Rejected
for this iteration per Directive #2 (minimal heuristics, no scope
creep) and because the single case we have evidence for
(astropy-14182) is addressable with a mechanical context check. Keep
as a distinct follow-up spec if (A)+(Y) doesn't move the resolved rate.

### (D) Add a new `verify_diff_context` tool and let the agent call it

Relies on agent cooperation. The 2026-04-21 exocortex audit showed
agents reliably ignore unused tools when framing is passive. Tool-level
verifier would need prompt work + is gated by the same kind of
framing challenge we just addressed for `search_exocortex`. Rejected;
the bench-side verifier is unconditional and doesn't require prompt
engineering.

### (E) Move the check into the existing `try_repair_patch`

`try_repair_patch` today repairs malformed hunk counts + asks the LLM
for a one-shot retry on `git apply` failure. Adding context-verification
inside that helper bundles two distinct concerns (structural repair vs
semantic-context sanity). Rejected for separation of concerns; the
verifier runs BEFORE the existing repair so the existing path is a
no-op on clean diffs and a safety net on context-mismatched ones.

## Implementation footprint

| File | Action | LOC est. |
|---|---|---|
| `sage-python/src/sage/bench/swebench_diff_verifier.py` | NEW — core verifier | ~180 |
| `sage-python/src/sage/bench/swebench_bench.py` | MODIFY — `generate_patches` verifier hook; `write_predictions` passthrough | ~30 |
| `sage-python/tests/test_swebench_diff_verifier.py` | NEW — unit tests 1-7 | ~250 |
| `sage-python/tests/test_swebench_emission_wiring.py` | EXTEND — wire tests for env-gate | ~40 |

Total: ~500 LOC. One feature commit + tests. No dependency changes.

## Env-var surface summary

| Var | Values | Effect |
|---|---|---|
| `SAGE_DIFF_VERIFIER_MODE` | `off` (default), `observe`, `repair` | Three-state gate. `off` is byte-identical to today. `observe` logs + annotates predictions but doesn't change emission. `repair` re-prompts the LLM with mismatch diagnostics on first failure (max 1 attempt). |

No other env vars — the fuzzy threshold is a module-level constant
(not a magic-number env knob) per Directive #2.

## Open questions (answer during implementation)

1. **Fuzzy threshold parity.** The SR extractor uses 0.95. Keep the
   verifier at 0.95 for symmetry, or tune separately based on Test 4
   data? Default: keep at 0.95 unless Test 4 reveals a data-driven
   reason to differ. If we differ, both values belong in a single
   `swebench_emission_constants.py` module so the next audit spots
   the divergence.
2. **Repair-mode prompt shape.** The repair LLM in `try_repair_patch`
   today gets `git apply --check` stderr. For the verifier-triggered
   repair pass, build a parallel "diagnostic bundle" message:
   listing the hunk, the expected-vs-actual lines, and a one-sentence
   instruction. Draft during implementation, review before ship.
3. **Interaction with SR emission.** Under `SAGE_EMISSION_FORMAT=search-replace`
   the SR extractor produces a unified diff via `_blocks_to_unified_diff`.
   Should the verifier run on that converted diff? Probably YES — the
   SR extractor already validated the SEARCH text, but the synthesised
   unified diff could still have a bad hunk header. Confirm during
   implementation with a dedicated test (Test 8).

## Decision gate for implementing (not this commit — spec only)

Implement when:
* The observability smoke slot opens (prefer the same day we next run
  a SWE-bench smoke for an unrelated reason, to amortise Docker setup
  cost).
* Or: another semantic-miss tracer lands with the same shape (context
  hallucination despite file-reads). Two tracers vs one changes the
  expected-lift math meaningfully.

Until either condition holds, this spec stays on disk as the record
of reasoning. Shipping it on the strength of one tracer and a single
advisor consult would be speculative.
