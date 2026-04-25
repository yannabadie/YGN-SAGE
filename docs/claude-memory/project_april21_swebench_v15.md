---
name: April 21 — SWE-bench v15 first Docker-graded pass-rate
description: 2026-04-21 session unlocked the real SWE-bench Lite pass-rate (1/10 = 10%) after v13's 0/10 turned out to be Windows env breakage, not SAGE quality. Chain of 3 infra fixes in sage.bench.swebench_ca_patch + a reusable eval-only re-run script.
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
**Context**: 2026-04-21 session, post-Phase-1 stab (commit `ef8241c` on main). Goal was the first real SWE-bench Lite Docker-graded number. v13 smoke returned 0/10 resolved → diagnosed as Windows env, not SAGE quality.

**Fix chain** (all in `sage-python/src/sage/bench/swebench_ca_patch.py`):

1. **`482ea28`** — gate the SSL bypasses (`wget --no-check-certificate`, `conda ssl_verify=false`, `pip trusted-host`) behind `SAGE_SWEBENCH_ALLOW_INSECURE=1`. The earlier `eb17940` baked them unconditionally into the Dockerfile template, violating CLAUDE.md Directive #3. Default is secure: CA COPY + append-to-ca-certificates only.

2. **`efb8afd`** — CRLF fix. `swebench.harness.run_evaluation.py:199` writes `/eval.sh` with `Path.write_text` (no `newline=`). On Windows that translates `\n` → `\r\n`. bash inside the Linux container fails on `set -uxo pipefail\r` → conda never activates → pytest not in PATH → every FAIL_TO_PASS test reports "command not found". Fix: monkey-patch `pathlib.Path.write_text` to route `.sh`/`.bash` through `write_bytes(data.encode(...))`. No-op on Linux.

3. **`bcade10`** — UTF-8 open fix. Once `/eval.sh` runs (post-CRLF), pytest emits Unicode box chars (`│` U+2502, `├`, `─`). `run_evaluation.py:211` does `open(path, "w")` with no encoding → Windows cp1252 → UnicodeEncodeError → the instance is reported as ERROR even though patch applied and tests ran. Fix: monkey-patch `swebench.harness.run_evaluation.open` to default text-write to `encoding='utf-8'`. No-op on Linux.

**Result**: v15 = **1/10 resolved** (astropy-12907 — 468-char patch `cright[...] = right` vs `= 1` in `separable.py`; SAGE produced the identical fix the maintainer did). Per-task breakdown in `docs/benchmarks/2026-04-21-swebench-v15-eval-results.md`.

**Reusable tooling**:
- `sage-python/scripts/swebench_eval_only_v14.py` — pipes v13 predictions (kept in `C:/Users/yann.abadie/AppData/Local/Temp/sage_swebench_kfhxbz7i/`) straight into `SWEBenchBench.evaluate_with_harness()` without regenerating. Saved ~1h per iteration and dodged minimax 529 variance. Any future Windows infra work on the eval harness can reuse this.
- All three fixes live in one module (`swebench_ca_patch.py`) with 13/13 passing unit tests.

**What's still ugly** (out of scope for this session, real SAGE-quality work):
- 5 EMPTY patches from v13 = minimax 529 storm during generation. Provider-health / fallback-policy work.
- 2 ERROR cases (`astropy-7746`, `django-10914`) — malformed unified-diff hunk headers. LLM hallucinates line counts on large files. Fix: switch SAGE's patch emission to search-and-replace blocks, or add `patch --dry-run` validator before writing predictions.
- 2 UNRESOLVED-applied (`astropy-14182`, `django-11001`) — real quality gaps.

**Disambiguation**: the "70% SWE-bench Lite smoke" in older CLAUDE.md / memory entries is a **patch-generation** rate (v5d/v5e, April 18), NOT a resolution rate. v15 is the first Docker-graded resolution number in this repo. CLAUDE.md §Current State updated in `172e8dc` to call out both numbers separately.

**Where to look**:
- Commits: `482ea28`, `efb8afd`, `bcade10`, `842b98c` (v15 docs), `172e8dc` (CLAUDE.md)
- v15 results: `docs/benchmarks/2026-04-21-swebench-v15-eval-results.md`
- v15 JSON: `docs/benchmarks/2026-04-21-swebench-v15-eval-report.json`
- v13 origin: `docs/benchmarks/2026-04-21-swebench-smoke-v13-post-phase1-stab.md`
