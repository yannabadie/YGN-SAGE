---
title: Tonight's BCB-Hard N=50 evidence run + official Docker re-grade — runbook
date: 2026-04-29
type: runbook
status: pending_execution
related: 2026-04-29-a14-reset cgpro_r6_1a_design path-e-step3
---

# Cycle-7 default-on evidence run — tonight off-corp

cgpro 2026-04-29 Path E postmortem locked the cycle-7 evidence sequence. Run tonight on home network (no corporate proxy ⇒ Docker official harness usable, gRPC SSL clean, no fall-back path needed for the canonical evaluator).

## Pre-flight checks

```bash
# 1. Verify A14 reset is intact
ls ~/.sage/  # should NOT show bandit_state.db / archive_state.db / engine_extras.json
cat ~/.sage/posterior_epoch.json  # epoch should be 1

# 2. Verify codex commits landed (T1 + T6 minimum)
cd /c/Code/YGN-SAGE
git log --oneline -10 | head -10
# Expect commits: ... feat(pipeline): T1 ... feat(bench): SAGE_BENCH_DISABLE_REPAIR ...

# 3. Verify all dependent flags are wired in code
grep -c "SAGE_TOPOLOGY_FORCE_ENGINE\|SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE\|SAGE_TOPOLOGY_LOG_ALL_CANDIDATES" sage-python/src/sage/pipeline.py
grep -c "SAGE_BENCH_DISABLE_REPAIR" sage-python/src/sage/bench/bigcodebench_bench.py

# 4. Verify .env has all 7 API keys
grep -cE "^(DEEPSEEK|GOOGLE|OPENAI|GROK|KIMI|MINIMAX|OPEN_ROUTER)_API_KEY=." .env
# Expect: 7

# 5. Verify Docker is running
docker version
docker images | grep bigcodebench
```

## Phase 1 — Official Docker re-grade of Path E step 3 N=10 predictions

**Purpose**: cross-check the seam evaluator vs the official BCB harness on the N=10 sample we already have.

```bash
cd /c/Code/YGN-SAGE
mkdir -p .tmp/path_e_phase1_$(date +%Y%m%d)
cp docs/benchmarks/2026-04-29-path-e-step3-bcb-canonical-predictions.jsonl \
   .tmp/path_e_phase1_$(date +%Y%m%d)/sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated.jsonl

docker run --rm \
  -v "$(pwd)/.tmp/path_e_phase1_$(date +%Y%m%d):/app" \
  bigcodebench/bigcodebench-evaluate:latest \
  --execution local --split instruct --subset hard \
  --samples sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated.jsonl \
  --no-gt 2>&1 | tee .tmp/path_e_phase1_$(date +%Y%m%d)/docker_official_grade.log
```

**Expected output**: `.tmp/path_e_phase1_<date>/sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated_eval_results.json` with per-task pass/fail.

**Validation**:
```bash
python <<EOF
import json
from pathlib import Path
ph1 = Path('.tmp/path_e_phase1_<date>')
official = json.load((ph1 / 'sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated_eval_results.json').open())
seam = json.load(Path('docs/benchmarks/2026-04-29-path-e-step3-bcb-N10.json').open())
# Compare per-task
print('Official:', official.get('eval', {}))
print('Seam (final, with repair):', {r['task_id']: r['passed'] for r in seam['results']})
EOF
```

**Pass criteria**: official harness per-task agrees with internal evaluator on at least the 5 first-attempt-pass tasks. Repair-rescued tasks may diverge (escalation context).

Commit Phase 1 artifacts:
```bash
cp .tmp/path_e_phase1_*/sage--*_eval_results.json \
   docs/benchmarks/2026-04-29-path-e-step3-bcb-official-grade.json
git add docs/benchmarks/2026-04-29-path-e-step3-bcb-official-grade.json
git commit -m "bench(path-e-step3): Phase 1 official Docker re-grade of N=10 predictions"
git push
```

## Phase 2 — BCB-Hard N=50 cycle-7 evidence run (CLEAN STATE)

**Purpose**: produce the default-on flip evidence on a fresh A14 epoch. NO repair (first-attempt only). NO archive warm-start (epoch=1 starts empty).

```bash
cd /c/Code/YGN-SAGE/sage-python
mkdir -p /c/Code/YGN-SAGE/.tmp/cycle7_evidence_$(date +%Y%m%d)/jsonl_n50
set -a && source ../.env && set +a
export SAGE_ORACLE=1 SAGE_RUN_FRAME=1 SAGE_BENCH_ORACLE_SEAM=1 \
       SAGE_BENCH_DISABLE_REPAIR=1 \
       SAGE_TRACE_JSONL_DIR=/c/Code/YGN-SAGE/.tmp/cycle7_evidence_$(date +%Y%m%d)/jsonl_n50 \
       SAGE_DIFF_VERIFIER_MODE=observe \
       PYTHONUNBUFFERED=1
# OFF-CORP: no SSL bundle env vars needed (default certifi works on home network)

python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50 \
       --output /c/Code/YGN-SAGE/docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50.json \
       2>&1 | tee /c/Code/YGN-SAGE/.tmp/cycle7_evidence_$(date +%Y%m%d)/bench.log
```

**Expected wall time**: 50 tasks × ~50s/task = ~40 min (single-threaded, no repair).

**Mid-run monitoring**:
```bash
# Tasks completed
grep -cE "TopologyRunner.*completed via agent_loop" .tmp/cycle7_evidence_*/bench.log
# Errors
grep -cE "ERROR|Traceback" .tmp/cycle7_evidence_*/bench.log
# Provider distribution
grep -oE "(deepseek|google|openai|kimi|xai|minimax|openrouter)" .tmp/cycle7_evidence_*/bench.log | sort | uniq -c
```

**Post-run validation**:
```bash
python -m sage.bench.path_e_validate \
  --jsonl-dir /c/Code/YGN-SAGE/.tmp/cycle7_evidence_*/jsonl_n50 \
  --bench-report /c/Code/YGN-SAGE/docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50.json \
  --predictions sage-python/docs/benchmarks/2026-04-29-predictions-hard-instruct.jsonl \
  --out-canonical-predictions /c/Code/YGN-SAGE/docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-canonical-predictions.jsonl \
  --out-manifest /c/Code/YGN-SAGE/docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-manifest.json \
  --out-report /c/Code/YGN-SAGE/docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-validation.md
```

**Pass criteria** (cycle-7 default-on flip gate):
1. ≥1 ToolOracle non-abstain or ≥1 Exact non-abstain
2. ≥1 Exact pass trainable (`verdict_source="exact", quality_label="pass", trainable=True`)
3. ≥1 Exact fail trainable
4. No FormalOracle trainable verdict without complete obligation evidence
5. SpecOracle never trains from text substrings
6. Generic/incidental tool fatal never trains fail
7. Event order `final_result < oracle_verdict < run_frame_summary` 50/50
8. No raw stdout/stderr/raw_output/raw_patch in payloads
9. OFF mode produces `runtime_deltas == ()` (separate test)

**Pillar activation observation** (no pass/fail criteria, just data for cycle-7+ tickets):
- topology diversity: count of distinct templates in the 50 runs
- tool calls: total tool_calls across 150 nodes
- memory writes: write_gate accepted vs skipped + reason breakdown (post-T2 if landed)
- adaptation actions: distribution of action codes (post-T4 if landed)
- provider distribution: % calls per provider

## Phase 2bis — Official Docker re-grade of N=50 predictions

```bash
cd /c/Code/YGN-SAGE
cp docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-canonical-predictions.jsonl \
   .tmp/cycle7_evidence_$(date +%Y%m%d)/sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated.jsonl

docker run --rm \
  -v "$(pwd)/.tmp/cycle7_evidence_$(date +%Y%m%d):/app" \
  bigcodebench/bigcodebench-evaluate:latest \
  --execution local --split instruct --subset hard \
  --samples sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated.jsonl \
  --no-gt 2>&1 | tee .tmp/cycle7_evidence_*/docker_official_grade.log

cp .tmp/cycle7_evidence_*/sage--*_eval_results.json \
   docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50-official-grade.json
```

**Calibration outcome**: per-task agreement % between internal seam evaluator and official harness. Target: ≥90% agreement (5% tolerance for matplotlib/timeout edge cases).

## Phase 3 — Conditional: SAGE_TOPOLOGY_FORCE_ENGINE diagnostic (only if Phase 2 still 100% sequential)

If after T1 codex implementation + Phase 2 we still see `topology_id` distribution = {one template only}, run the forced-engine diagnostic:

```bash
cd /c/Code/YGN-SAGE/sage-python
mkdir -p /c/Code/YGN-SAGE/.tmp/phase3_force_engine_$(date +%Y%m%d)/jsonl
set -a && source ../.env && set +a
export SAGE_ORACLE=1 SAGE_RUN_FRAME=1 SAGE_BENCH_ORACLE_SEAM=1 \
       SAGE_BENCH_DISABLE_REPAIR=1 \
       SAGE_TOPOLOGY_FORCE_ENGINE=1 \
       SAGE_TOPOLOGY_LOG_ALL_CANDIDATES=1 \
       SAGE_TRACE_JSONL_DIR=/c/Code/YGN-SAGE/.tmp/phase3_force_engine_$(date +%Y%m%d)/jsonl \
       PYTHONUNBUFFERED=1

python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 5 \
       --output /c/Code/YGN-SAGE/docs/benchmarks/2026-04-29-phase3-force-engine-bcb-N5.json \
       2>&1 | tee /c/Code/YGN-SAGE/.tmp/phase3_force_engine_*/bench.log
```

**Expected**: topology diversity emerges (>=2 distinct templates) OR the cold-start engine itself collapses (would mean a deeper Rust gen issue to triage).

## Final commit + push

```bash
cd /c/Code/YGN-SAGE
git add docs/benchmarks/2026-04-29-cycle7-evidence-bcb-N50*.* \
        docs/benchmarks/2026-04-29-phase3-force-engine-bcb-N5.json 2>/dev/null
git commit -m "bench(cycle7): off-corp BCB-Hard N=50 evidence run + Phase 1+2bis Docker re-grade"
git push
```

## Decision tree post-tonight

| Evidence | Decision |
|---|---|
| All 9 cycle-7 pass criteria green + ≥90% Docker agreement | **FLIP** SAGE_ORACLE default-on. Open ticket cycle 7 default-on smoke. |
| Pass criteria green but Docker disagreement >10% | Investigate evaluator divergence first; defer flip. |
| Pass criteria fail (e.g. FormalOracle trainable without complete evidence) | Push back to cycle 6 R6.1a — bug fix needed. |
| Phase 2 Phase 3 still 100% sequential after force-engine | Open priority ticket for Rust engine cold-start triage; flip can still proceed since safety invariants hold. |

## Rollback if anything goes wrong

```bash
# Restore A14 pre-reset state (loses any clean-epoch updates accumulated)
cp ~/.sage/contaminated_pre_a14_20260429/* ~/.sage/
rm ~/.sage/posterior_epoch.json
```

Document why rollback was needed in `docs/operations/2026-04-29-a14-rollback-decision.md` for audit.
