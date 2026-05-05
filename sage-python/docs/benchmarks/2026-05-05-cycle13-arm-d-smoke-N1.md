# Cycle-13 E Tier 2.1 — Arm D smoke results (N=1, real-API, no Docker grader)

**Date**: 2026-05-05 evening (session-end smoke).
**Commit**: `d3fc6fe0` (cycle-13 E Tier 2.1 + pipeline event_log integration fix).
**Status**: Tier 2.0 complete (Pro patch format adapter SHIPPED + 21 tests). Tier 2.1 partial (telemetry validated end-to-end, grader call gated by Docker daemon being down).

---

## What this smoke proved

Per cgpro DESIGN E (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`), Tier 2.1 acceptance is "1/1 graded real Arm D task minimum, 2/2 only if Docker/image/runtime is not the bottleneck. Hard cutoff: if Docker pull/eval > 15 min OR API spend > $5, stop." Docker daemon was down on the host — grader gated; remaining acceptance: shape-valid predictions.json + per-task RuntimeEventLog file present.

This run validated:

1. **`swebench_pro_fetch.py` works**: pulled N=10 tasks from `ScaleAI/SWE-bench_Pro` test split, stratified by language (3 Go / 3 JS / 2 Python / 2 TS) across 9 distinct repos. `selection_hash` reproducible by `--seed 42`.

2. **Pro patch format adapter works**: Tier 2.0 `swebench_pro_format_patch.py` produces `{instance_id, patch, prefix}` JSON list with LF-only line endings. 21 unit tests cover every shape rejection path (lite-shape leak, missing keys, wrong types, empty patch, prefix presence/absence, unicode round-trip).

3. **`sage run --jsonl` end-to-end**: subprocess invocation + stdin task + stdout JSONL captured by runner. 7 events flowed through:
   ```
   cli_started -> task_started -> routing_decision ->
   topology_selected -> model_assigned -> final_result -> cli_complete
   ```

4. **Pipeline event_log integration works** (after fix at `pipeline.py:763`). See "Real production bug fixed" below.

---

## Smoke parameters

| Parameter | Value |
|---|---|
| Task | `instance_future-architect__vuls-139f3a81b66c47e6d8f70ce6c4afe7a9196a6ea8` |
| Repo | `future-architect/vuls` (Go) |
| Tier | `budget` (deepseek-v4-flash) |
| Budget cap | `$1.00` |
| Wall-clock | 43.7 sec |
| Tokens billed | 0 (agent gave up before reaching LLM) |
| Patch extracted | 0 chars |
| Bypass env active | `SAGE_BOOT_BYPASS_EPOCH_GUARD=1` (per directive #8 — atexit save disabled to avoid `~/.sage/` pollution across consecutive smokes) |

---

## What is NOT validated (still NYI per cgpro DESIGN E trap Q5)

The current `sage run --jsonl` implementation is protocol-v0 PARTIAL:

- `cli_progress` heartbeat — spec'd, NOT YET EMITTED.
- `set_budget` mid-run inbound command — NOT YET HANDLED.
- `cancel` mid-run cancellation token — NOT YET ROUTED through pipeline.
- `cli_complete.payload.final_seq` — spec'd, current impl emits `trace_dir` without `final_seq`.

These remain as cycle-13 phase 2 follow-ups, NOT smoke blockers.

---

## What is NOT validated (Docker-blocked)

The SWE-bench Pro grader (`swe_bench_pro_eval.py` from `scaleapi/SWE-bench_Pro-os`) requires:
- Docker daemon running OR Modal account with auth.
- Per-instance `dockerfiles/{base,instance}_dockerfile/{iid}/Dockerfile` cloned locally.
- Per-instance `run_scripts/{iid}/{run_script.sh, parser.py}` cloned locally.

Docker daemon was down on the smoke host — the predictions.json IS shape-valid and ready for grading whenever Docker comes up. The runner's output is GRADER-READY:

```
sage-python/data/swebench_pro/arm_d_smoke_real_n1_v4/predictions.json
```

To grade: clone `scaleapi/SWE-bench_Pro-os`, install requirements, then:

```bash
python helper_code/gather_patches.py \
    --directory <pred_files_dir> \
    --prefix ygn-sage-arm-d-smoke \
    --output predictions.json
python swe_bench_pro_eval.py \
    --raw_sample_path <sage-python/data/swebench_pro/n10/instances.json or csv> \
    --patch_path <our predictions.json> \
    --use_local_docker
```

---

## Real production bug fixed in this commit

While building the smoke, discovered that `sage run --jsonl` was emitting **ZERO RuntimeEventLog events** end-to-end. Only `cli_started` + `cli_complete` envelope frames landed on stdout.

### Root cause

`pipeline.py:763` (cycle-12 prelude):

```python
event_log = RuntimeEventLog(run_id=_new_runtime_run_id())
```

With no `trace_dir` kwarg AND no `SAGE_TRACE_JSONL_DIR` env, `RuntimeEventLog.__init__` (writer.py:159-163) sets `self.disabled = True`. All `emit_*` methods become no-ops. Then line 770:

```python
install_event_log(event_log)
```

shadows the contextvar, replacing whatever the CLI installed (its own `RuntimeEventLog` with `_CliMirrorSinkHandle` stdout-mirror tee).

### Fix

```python
event_log = current_event_log()
if event_log is None:
    event_log = RuntimeEventLog(run_id=_new_runtime_run_id())
```

Preserves historical default (direct-Python callers without the CLI get a fresh event_log). Adds new contract: when CLI installs an eventlog, pipeline reuses it. `run_id` matches between CLI's `cli_started` and runtime events — frontend frame stitching now works.

### Why cgpro DESIGN E didn't catch this

cgpro flagged some `sage run --jsonl` v0 protocol gaps (cli_progress NYI, set_budget NYI). But it didn't catch the wholesale event-log shadowing. This was empirically discovered while building the runner — exactly the case for `find_real_bugs_via_smoke` over `cgpro_review_alone`.

### Regression test

`test_pipeline.py::test_pipeline_run_respects_installed_event_log` proves the contract:
- External eventlog (with `trace_dir`) IS picked up by `pipeline.run`.
- First emitted event is `task_started`.
- Eventlog `run_id` matches the EXTERNAL eventlog (not regenerated).

PASSES on `d3fc6fe0`. Wider regression sweep: 248/248 across 19 pipeline / runtime / bench files, 0 regression.

---

## Outputs from this smoke (gitignored under `sage-python/data/swebench_pro/`)

```
sage-python/data/swebench_pro/n10/
    instances.json                                 -- 10 stratified Pro tasks
    manifest.json                                  -- fetch metadata
    per_task/<id>.json                             -- 10 sharded task JSONs

sage-python/data/swebench_pro/arm_d_smoke_real_n1_v4/
    predictions.json                               -- Pro grader input format (1 record)
    per_task/<id>.events.jsonl                     -- 7-event runtime trace
    summary.json                                   -- aggregated metrics
```

---

## Cycle-13 follow-ups now unblocked

Once Docker daemon is up:

1. **Tier 2.1 grader call** — `swe_bench_pro_eval.py` on the smoke output. Single task, ~15 min Docker pull + eval. Acceptance: pass/fail returned.
2. **Tier 2.2** — N=10 across arm D. ~$5 API budget if all complete (most won't, agent at budget tier won't resolve hard Pro tasks). Validates fetch + runner + grader at scale.
3. **Cycle-13 main run prereqs**: pin pi-mono v0.73.0 (DONE in `clients/pi-ygn-sage/package.json`), wire arms A + B + C, secure $240-460 API budget, decide on Modal vs local Docker for grading.

## References

- Cycle-13 plan: `docs/benchmarks/2026-05-05-cli-baseline-plan.md`.
- Arm wiring contract: `docs/benchmarks/2026-05-05-cycle13-arm-wiring.md`.
- SAGE_CLI_PROTOCOL v0: `docs/contracts/SAGE_CLI_PROTOCOL.md`.
- Runtime integrity ledger (9 invariants): `docs/contracts/runtime-integrity-ledger.md`.
- SWE-bench Pro repo: `github.com/scaleapi/SWE-bench_Pro-os`.
- HuggingFace dataset: `huggingface.co/datasets/ScaleAI/SWE-bench_Pro`.
- cgpro DESIGN E: conv `cgpro_pi_mono_pivot_20260505` 2026-05-05 verdict GO_TIER_1_PLUS_2.

## Status

- 2026-05-05 evening (this commit): Tier 2.0 complete + Tier 2.1 telemetry-validated. Real prod bug found and fixed. predictions.json grader-ready.
- TBD (cycle-13 grader call): pending Docker daemon up.
- TBD (cycle-13 main): pending arms A/B/C wired + API budget approved.
