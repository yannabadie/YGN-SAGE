# SWE-bench Lite N=50 smoke — stopped at 41/50 (2026-04-21)

**Status:** interrupted at task 41/50 on user request ("on a déjà suffisamment attendu"). The 4 session commits under test were already well-characterized by the partial run; completing the last 9 tasks would add ≈2 pp of statistical power on generation rate, not change attribution.

**Log:** [`2026-04-21-swebench-n50-full.log`](2026-04-21-swebench-n50-full.log) (208 KB, 41 complete task outcomes).

**No JSON / predictions.jsonl:** `sage.bench.swebench_bench` writes predictions at the end of the run (`SWEBenchBench.run()` line 848), so a mid-run kill loses them. Docker pass-rate not measurable from this run.

## Generation outcome (41 tasks)

| Outcome | Count | Share | Meaning |
|---------|-------|-------|---------|
| PATCH | 26 | 63.4 % | Valid-looking unified diff emitted (not yet Docker-graded) |
| EMPTY | 8 | 19.5 % | Final patch was 0 bytes — fallback didn't rescue |
| ERR | 7 | 17.1 % | Stage 0/1 bail or 300 s generation timeout |

## Comparison with prior N=10 smokes on the same slice

| Smoke | Gen rate | EMPTY rate | Stage 4 fallback wired? | Tool-affordance injected? |
|-------|---------:|-----------:|:-----------------------:|:-------------------------:|
| v13 (post Directive #3 + CRLF + UTF-8) | 50 % | 40 % | ✗ | ✗ |
| v17 (+ fallback to healthy provider, commit `4f90a98`) | 60 % | 20 % | ✓ | ✗ |
| N=50 (this run, + tool-affordance `3622ac5`, + MutationStats `2b0c211`) | **63 %** | **19.5 %** | ✓ | ✓ |

**Takeaway:** the EMPTY-bucket shrinkage from v13 (40 %) → v17 (20 %) → N=50 (19.5 %) validates the Stage 4 fallback fix. Tool-affordance and MutationStats are orthogonal changes that don't regress generation rate. The absolute **pass rate** (Docker-graded) stays unknown because the N=50 predictions were never persisted to disk — infrastructure lesson: `SWEBenchBench.run()` needs an incremental `write_predictions()` call per task so that an interrupted smoke yields a partial but gradable predictions file. Logged as follow-up.

## Why stopping here is sound

* The four commits under test (`9eb05b0` AVR enrichment, `3622ac5` native tool-affordance, `0bcb92b` in-run topology/source/memory logging, `2b0c211` MutationStats PyO3) are **orthogonal** — not gated on each other. The 41-task slice already exercises each one.
* The generation-rate delta between N=10 v17 (60 %) and N=41 here (63 %) is within one-task noise (±2.5 pp). Finishing the last 9 tasks was very unlikely to cross a 5 pp detection band.
* The **unit of signal I actually need** for the C2a/C2b refactor is the per-stage log (which I have in full) plus whether new log categories flow (MutationStats events, topology source attribution), not a polished absolute pass rate.

## Follow-up TODO (separate commit, not blocking C2a)

* Teach `SWEBenchBench.run()` to `append_predictions(pred)` per task so an interrupted smoke produces a valid partial jsonl.
* Re-run full N=50 with Docker grading once C2b ships (that's the change where pass-rate attribution is expected to move, not before).
