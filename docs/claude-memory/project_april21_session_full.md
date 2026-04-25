---
name: April 21 2026 — SWE-bench Lite v13 → v17 full session arc
description: Full session arc 2026-04-21 — v13 0/10 env-broken → v15 1/10 Windows-infra fixed → v16 repair pipeline (no lift) → v17 full smoke (1/10 different composition, fallback fix validated in isolation). Lesson: N=10 too small for total-rate signal; per-bucket attribution works.
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
## TL;DR

**Start**: v13 smoke = 0/10 resolved, user said "intolérable".
**End**: v15 + v17 both = 1/10 resolved, but composition proves fix mechanisms work per-bucket. Honest conclusion: N=10 is noise-dominated; need N=50 for headline signal.

## What shipped (5 commits, all on `main`)

1. `482ea28` — Directive #3 gating of SSL bypasses in `swebench_ca_patch`. Default secure; opt-in via `SAGE_SWEBENCH_ALLOW_INSECURE=1`.
2. `efb8afd` — CRLF LF-safe wrapper for `.sh`/`.bash` in `pathlib.Path.write_text`. Fixes the 0/10 → 1/10 step (Windows eval.sh → Linux container).
3. `bcade10` — UTF-8 default for `swebench.harness.run_evaluation.open`. Fixes pytest Unicode output → Windows cp1252 → v14 regression.
4. `24e51af` — Patch validator + two-stage repair pipeline (programmatic counts-fix + LLM one-shot). 0 lift v16 because validator was too strict; code retained for future relaxation.
5. `4f90a98` — Pipeline Stage 4 fallback: routes to healthy provider via `provider_pool._dead_at`, raises on empty content instead of silently emitting empty patches. **Validated**: 3/5 v13 EMPTYs produced real patches in v17.

Total: 13 unit tests added (swebench_ca_patch + swebench_patch_repair + pipeline_fallback_provider), all pass.

## Per-bucket attribution (v13 → v17)

| Bucket | v13 | v17 | Fix attribution |
|---|---|---|---|
| EMPTY (no patch) | 5 | 4 (−1 direct, +1 via timeout reclass) | `4f90a98` fallback: 3 EMPTYs recovered to PATCH, 1 of those → RESOLVED |
| Timeout/error (no patch) | 0 | 1 | — |
| Apply-error (malformed) | 2 | 1 | Variance across runs; `24e51af` didn't lift |
| Unresolved (applied, tests failed) | 3 | 3 | Semantic-miss bucket, requires deeper work |
| **Resolved** | **0** | **1** | — |

**The fixes work; the N moves aren't statistically meaningful at 10.**

## Key learning: validator / container mismatch

`git apply --check` (what my v16 validator used) is STRICTER than what swebench's container does:
```
swebench container actual chain:
  1. git apply --verbose            (strict)
  2. git apply --verbose --reject   (still strict)
  3. patch -p1 --fuzz=5 -i patch.diff   (lenient fuzz tolerance)
```
So patches that would have been fine at step 3 got flagged as "broken" by my validator and sent to repair. The repair pipeline either produced a cosmetically-different-but-semantically-equivalent patch (no gain) or failed (also no gain). Lesson for future: mirror the container's apply chain when validating.

## Key learning: provider variance across runs

v13 morning: minimax storming 529, openai+gemini healthy.
v17 afternoon: minimax 200 OK, openai+xai SSL-cert-failing.

Same codebase, same tasks — different working-provider set → different routing → different models assigned per node → different agent traces → different patches. This is why N=10 fluctuates 10pp on task flips. The fix targets a bucket; the bucket has variance.

## Artifacts

- v13: `docs/benchmarks/2026-04-21-swebench-smoke-v13-post-phase1-stab.md`
- v15: `docs/benchmarks/2026-04-21-swebench-v15-eval-results.md`
- v16: `docs/benchmarks/2026-04-21-swebench-v16-repair-postmortem.md`
- v17: `docs/benchmarks/2026-04-21-swebench-v17-full-results.md`
- Session commits: `482ea28` → `efb8afd` → `bcade10` → `842b98c` → `172e8dc` → `24e51af` → `4f90a98` → `8bb923b`

## Follow-ups (not this session)

- **N=50 smoke** — only way to distinguish fix lift from N=10 noise.
- **Relax patch-repair validator** to include `patch --fuzz=5` (v16 failure mode).
- **Enrich LLM repair prompt** with ±15 lines around each failing hunk (Aider pattern).
- **Semantic-miss bucket** (astropy-14182, django-11001): deeper planner budget + ExoCortex integration.
- **generation_timeout_300s** → investigate django-11001's agent loop (got stuck; different bug).
