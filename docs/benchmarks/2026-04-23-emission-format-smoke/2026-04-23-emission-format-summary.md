# Emission-format paired smoke — SAGE_EMISSION_FORMAT=unified vs search-replace

**Date:** 2026-04-23
**Slice:** SWE-bench Lite, `--limit 10 --offset 0` (same 10 tasks in both arms)
**Mode:** generate-only (no Docker eval — patch-produce rate only)
**Env common to both arms:** `SAGE_DANGEROUS_TOOLS=0` (post ADR-013 §5 flip default)
**Branch:** feat/search-replace-emission (T2.1-T2.4 implementation)

## Headline

| Metric | Arm A (unified) | Arm B (search-replace) |
|---|---|---|
| Non-empty patches | **4** | **3** |
| SEARCH/REPLACE blocks parsed | — (path inactive) | **4 tasks had 1 block each** |
| SR blocks producing valid diff | — | **2** (match_kind=exact/fuzzy) |
| SR blocks dropped as missing | — | **2** (search_text did not match file) |
| SR path not attempted (model emitted diff fence) | — | ? (unified extractor had priority, no log fires then) |
| Timeout errors | 1 (django-11019) | 1 (astropy-14182) |

## Per-task breakdown

| instance_id | Arm A (unified) | Arm B (search-replace) | SR extractor notes |
|---|---|---|---|
| astropy__astropy-12907 | EMPTY | EMPTY | SR fallback NOT logged |
| astropy__astropy-14182 | EMPTY | ERR timeout | SR fallback NOT logged |
| astropy__astropy-14365 | PATCH (583) | **PATCH (363)** | SR: 1 raw block, 1 normalised, match ok → SR-extractor supplied the diff |
| astropy__astropy-14995 | EMPTY | EMPTY | SR: 1 raw block parsed, but search_text missed file → empty diff |
| astropy__astropy-6938 | PATCH (658) | **PATCH (727)** | SR: 1 raw block, matched → SR-extractor supplied the diff |
| astropy__astropy-7746 | PATCH (674) | EMPTY | SR: 1 raw block parsed, but search_text missed file → empty diff |
| django__django-10914 | EMPTY | **PATCH (28, JUNK)** | SR fallback NOT logged — model emitted `---------------------------` (27 dashes); unified extractor took it |
| django__django-10924 | EMPTY | EMPTY | SR fallback NOT logged |
| django__django-11001 | PATCH (642) | EMPTY | SR fallback NOT logged — model didn't emit anything |
| django__django-11019 | EMPTY | EMPTY | SR: 0 raw blocks — model emitted prose |

## Key findings

1. **The SEARCH/REPLACE directive DOES get partial model compliance.** 4 out of 10 tasks in Arm B had the model emit a SEARCH/REPLACE block (astropy-14365, astropy-14995, astropy-6938, astropy-7746). Two of those produced valid unified diffs via `_blocks_to_unified_diff`; two had a `search_text` that didn't match the actual file (recorded as `missing` match_kind, empty diff).

2. **Unified-format compatibility works as designed.** django-10914 (Arm B, junk patch) and the 2 valid Arm B patches that came from the SR path prove the graceful degradation contract: when the unified extractor finds ANY diff-shaped content, it uses that regardless of the mode. The SR fallback only fires when unified came back empty. T2.4's "unified-first, SR-second" priority is working.

3. **Net patch-rate: unified +1 over SR (4/10 vs 3/10).** The 1-patch delta comes from:
   - astropy-7746: unified got 3x diff fences (674 chars); SR mode got 1 block that didn't match file bytes.
   - django-11001: unified got 642-char diff; SR mode got pure prose.
   - django-10914: unified got nothing; SR mode picked up a junk `---` (not counted as a real win).

4. **JSONL output strips `_extraction_method` field.** T2.4 added the field to the in-memory prediction dict, but the JSONL writer filters to `{instance_id, model_name_or_path, model_patch}`. The method data is still available via the log's `SR fallback:` lines. Follow-up fix for T2.5 / Track 4: unfilter the JSONL writer so post-hoc analysis has the method column natively.

5. **The T2.3 step-7 prompt contradiction (advisor's flag) is real.** The opening line `"producing a minimal unified diff patch"` and Mandatory-Workflow step 7 (`"emit it directly in a ```diff fence"`) that were preserved for byte-identity are probably part of why the model still emits diff fences under the SR variant. The 6/10 tasks where SR fallback did NOT fire in Arm B are likely cases where the model emitted a diff fence despite the SR format directive.

## Decision

**Do NOT flip the `SAGE_EMISSION_FORMAT` default to `search-replace`.**

Net patch-rate is slightly worse (3 vs 4, plus the 28-char junk). Match failure rate among SR-emitted blocks (50%: 2/4 missed file match) is too high to recommend as default. The gate stays present as operator-opt-in infrastructure.

**But the gate works.** 2 patches came through the SR path cleanly — a direct measurement that `_extract_search_replace_blocks` + `_blocks_to_unified_diff` handle real Gemini output correctly.

## Next levers (not blocking T2.5)

1. **Soften the step-7 contradiction** in `SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE`. Remove `"producing a minimal unified diff patch"` from the opening (or replace with `"producing a minimal patch"`), update step 7 to "emit your patch in the format specified above" instead of `"emit it directly in a ```diff fence"`. This breaks the byte-identity with the unified template but should materially improve SR compliance. Re-smoke to measure.

2. **Investigate SR search_text matching failures.** 2/4 blocks missed the file. Inspect whether it's whitespace drift, wrong file, or the model hallucinating code. If whitespace drift, the fuzzy threshold at 0.95 may be too strict for Gemini output.

3. **Restore `_extraction_method` to the JSONL output.** Cheap fix; unlocks per-bucket analysis on future smokes without grep on logs.

## Artefacts

- `2026-04-23-arm-a-unified.log` — Arm A full agent log
- `2026-04-23-arm-a-unified-predictions.jsonl` — 10 records
- `2026-04-23-arm-a-unified-meta.json` — bench metadata
- `2026-04-23-arm-b-search-replace.log` — Arm B full agent log
- `2026-04-23-arm-b-search-replace-predictions.jsonl` — 10 records
- `2026-04-23-arm-b-search-replace-meta.json` — bench metadata

## Wallclock + cost

- Arm A: ~28 min wallclock. 4 Gemini-3.1 non-empty responses + 6 empties + 1 timeout.
- Arm B: ~26 min wallclock. 3 Gemini-3.1 non-empty responses + 6 empties + 1 timeout.
- API cost: approx $5-10 total (Gemini flash-lite + pro for S1/S2 mix).
