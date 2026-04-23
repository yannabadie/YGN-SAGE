# Diff-context verifier — observability smoke, N=10

**Date:** 2026-04-23
**Mode:** `SAGE_DIFF_VERIFIER_MODE=observe`
**Slice:** SWE-bench Lite `--limit 10 --offset 0` (gen-only, Docker not
run — we're validating the verifier's observability signal, not
resolved-rate).
**Wallclock:** ~25 min. Cost: ~$4 Gemini + OpenRouter.

## Headline

| Metric | Count |
|---|---|
| Tasks | 10 |
| PATCH (non-empty emission) | 2 |
| EMPTY | 8 |
| Verifier ran (PATCH + observe mode) | 2 |
| Verifier flagged `content_mismatch` (RAW smoke) | 1 |
| Verifier flagged `content_mismatch` (AFTER fix 711008a) | **2 (both)** |
| False positives | 0 |
| False negatives (fixed in 711008a) | 1 |

## Per-task breakdown

| instance | outcome | verifier ran | raw mismatches | post-fix mismatches |
|---|---|---|---|---|
| astropy__astropy-12907 | EMPTY | No | — | — |
| astropy__astropy-14182 | EMPTY (fast-abort 46s) | No | — | — |
| astropy__astropy-14365 | EMPTY | No | — | — |
| **astropy__astropy-14995** | **PATCH (545)** | **Yes** | **1 content_mismatch @0.0** | 1 content_mismatch @0.0 |
| astropy__astropy-6938 | EMPTY | No | — | — |
| astropy__astropy-7746 | EMPTY (fast-abort 43s) | No | — | — |
| **django__django-10914** | **PATCH (415)** | **Yes** | **0 (FALSE NEGATIVE)** | **1 content_mismatch @0.009** |
| django__django-10924 | EMPTY | No | — | — |
| django__django-11001 | EMPTY | No | — | — |
| django__django-11019 | EMPTY | No | No | — |

## Finding 1 — verifier correctly flags one wrong-file hallucination

**astropy-14995** — model emitted a unified diff for
`astropy/nddata/nddata.py` at line 495. The SWE-bench Lite gold patch
for this instance targets `astropy/nddata/mixins/ndarithmetic.py` at
line 520 (a different file in the same package). The emitted hunk body
at line 495 of `nddata.py` has **zero overlap** with file bytes
(`match_ratio=0.0`) — the verifier correctly flagged
`content_mismatch`. This is a **true positive**.

The failure mode is consistent with the Track 3.1b "wrong file"
hallucination class: the model understood the pattern it needed to
change (the `elif operand is None:` → `elif operand_mask is None:`
rename around `_arithmetic_mask`) but picked the wrong file inside the
package. The emitted diff would apply cleanly against a file that
happens to have a similar line pattern elsewhere, but the semantics
would be wrong.

## Finding 2 — false negative exposed a parser bug (fixed in 711008a)

**django-10914** — model emitted a headerless unified diff (no
`diff --git a/... b/...` prefix, just `--- a/`, `+++ b/`, `@@`). The
patch claimed to edit `django/conf/global_settings.py` at line 379,
but the real file at line 379 has `DATETIME_INPUT_FORMATS = [` (the
`FILE_UPLOAD_PERMISSIONS` declaration lives at line 307). The hunk
body and the file bytes have **virtually no overlap** (ratio 0.009
after the fix). Verifier should have flagged `content_mismatch` but
returned `[]` — the parser's
`if "diff --git " not in diff: return []` gate dropped the whole patch
before any hunk was extracted.

**Fix (commit 711008a):** removed the `diff --git` gate; parser now
keys off `--- `/`+++ `/`@@` triples directly. Re-verified manually
against the cached smoke artifact: post-fix, django-10914 flags
`content_mismatch @0.009` as expected. Added regression test
`test_headerless_unified_diff_is_parsed`. All 29 diff-verifier +
wiring tests green.

The false negative would have been silent in production — the
verifier would have let bad patches through without a peep. The
observability smoke is exactly the gate that caught it. This is
precisely the reason the spec shipped observe-mode BEFORE flipping
to repair.

## Post-fix bucket composition

Both emitted patches (astropy-14995 and django-10914) flag as
`content_mismatch`, meaning **100% of emitted patches in this N=10
slice were semantically wrong at the hunk-content level**. That's
consistent with:
* Neither was resolved by the Docker harness in the parity smoke
  (both had `_repair_stage: failed` when run post-fix gen).
* Both targeted files with naming/line-number patterns the model
  could plausibly confuse with the real answer (`nddata.py` vs
  `mixins/ndarithmetic.py`; headerless patch at a wrong offset).

## Decision-gate implications

For the repair-mode flip (spec §"Validation plan"):

1. **Signal quality is high.** Post-fix, 2/2 patches flag correctly
   with `content_mismatch` at very low ratios (0.0 and 0.009 —
   decisively not "close enough"). Zero false positives. Zero false
   negatives on clean emissions (there were none in this slice to
   verify, admittedly).
2. **Sample is tiny.** N=2 patches is below the minimum detectable
   lift for any repair-mode experiment. Before flipping to repair,
   want ≥10 patches under observe mode across a few smokes to
   confirm the false-positive rate stays low on clean emissions.
3. **8/10 EMPTY rate is a separate concern.** Two fast-aborts
   (14182, 7746 under 60s) suggest provider circuit / classification
   issues orthogonal to the verifier. Not Track 3's scope; flagged
   for the generic "why is EMPTY rate so high" investigation.

**Provisional recommendation:** keep observe mode as default OFF
(spec unchanged). Run observe mode on the next 2-3 SWE-bench smokes
regardless of their purpose — it's zero-cost telemetry. Once we've
accumulated ~10 flagged + ~10 clean emissions with zero false
positives confirmed, revisit the repair-mode flip.

## Artefacts

* `observe.json` — SAGE bench report (latency, cost, routing)
* `observe.jsonl` — SAGE manifest (TaskTrace records)
* `observe-summary.json` — 3-line summary
* `observe-gen.log` — full gen-phase log (~1329 lines) — includes the
  `[astropy__astropy-14995] diff verifier: 1 hunk mismatch(es)` line
  and (notably) the ABSENCE of a similar line for django-10914, which
  was the false-negative smoking gun
* `predictions.jsonl` is at `C:\Users\YANN~1.ABA\AppData\Local\Temp\sage_swebench_7fnqb1ob\predictions.jsonl`
  — contains the 10 records including `_diff_verifier_mismatches`
  for the 2 PATCH entries
