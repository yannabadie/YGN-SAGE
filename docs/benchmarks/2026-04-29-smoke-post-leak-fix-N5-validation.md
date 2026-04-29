# Path E step 3 — BigCodeBench Hard Instruct N=10 seam validation

**Date**: 2026-04-29
**Cycle**: 6 R6.1a verify Path E (post Gate D)
**Purpose**: prove the live ``verdict_source="exact", trainable=True`` contract on a synchronous-eval bench via the bench-result feedback seam.

## Honest framing locks

- This run is **NOT** a BigCodeBench leaderboard submission. The leaderboard reports **calibrated Pass@1** with greedy decoding through the official ``bigcodebench.evaluate`` harness (or its e2b/gradio backends). On Windows the official ``untrusted_check`` path fails on ``os.killpg`` and coerces every task to ``timeout``; the seam evaluator documents a fall-back to ``BigCodeBenchBench._evaluate_solution_with_stderr`` (matplotlib-headless subprocess, deterministic per (solution, test_code)) and tags ``bench_result.verifier_id`` accordingly.
- This is a **seam validation smoke**, not a value/regression benchmark.
- Per AUDIT2 2026-04-24 framing rule: no "above SOTA" or leaderboard-style claims attached to this number.

## Setup

- ``SAGE_ORACLE=1``, ``SAGE_RUN_FRAME=1``, ``SAGE_BENCH_ORACLE_SEAM=1``, ``SAGE_DIFF_VERIFIER_MODE=observe``
- ``StateCore`` OFF (``SAGE_STATECORE`` unset).
- Throwaway bandit DB: production state moved to ``.tmp/path_e_backup/`` pre-bench, restored post-bench. Production posteriors not polluted.
- SSL: ``SSL_CERT_FILE`` + ``REQUESTS_CA_BUNDLE`` + ``CURL_CA_BUNDLE`` + ``GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`` set to ``C:/Code/certs/windows-full-bundle.pem``.
- Greedy decoding: SAGE pipeline default temperature settings; not the BCB CLI ``--temp 0`` enforcement (separate from the seam contract).
- Single entry point: ``python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 10`` — no parallel scripts.

## cgpro Path E B' minimum pass criteria

| # | Criterion | Result |
|---|---|---|
| 1 | ≥1 ``verdict_source='exact', quality_label='pass', trainable=True`` | PASS |
| 2 | ≥1 ``verdict_source='exact', quality_label='fail', trainable=True`` | PASS |
| 3 | Event order ``final_result < oracle_verdict < run_frame_summary`` on every run | PASS (4/5) |
| 4 | No raw stdout/stderr/raw_output/raw_patch leaks in any payload | PASS (0 leaks) |

## Seam-vs-bench cross-check (per task)

Escalation/repair may turn a first-attempt seam fail into a final bench pass; the seam captures the **first-attempt** verdict, the bench report captures the **final** outcome.

| task_id | seam source | seam label | seam score | seam trainable | bench passed (final) | runtime_deltas | cross-check |
|---|---|---|---|---|---|---|---|
| BigCodeBench/13 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/15 | None | None | None | None | False | None | seam_abstain |
| BigCodeBench/17 | exact | fail | 0.0 | True | False | 0 | agree |
| BigCodeBench/19 | exact | pass | 1.0 | True | True | 3 | agree |
| BigCodeBench/34 | exact | pass | 1.0 | True | True | 4 | agree |

**Cross-check totals**: agree=4, diverged_likely_escalation=0, unknown/abstain=1.

## Verdict-source distribution

- ``exact/fail``: 2
- ``exact/pass``: 2
- ``None/None``: 1

## Reproducibility

- Repo: https://github.com/yannabadie/YGN-SAGE
- All artifacts SHA-256-hashed in the manifest below.
- Bench command (canonical, single entry point):

```bash
SAGE_ORACLE=1 SAGE_RUN_FRAME=1 SAGE_BENCH_ORACLE_SEAM=1 \
  SAGE_TRACE_JSONL_DIR=.tmp/path_e_artifacts/jsonl_n10 \
  python -m sage.bench --type bigcodebench --subset hard --split instruct \
                       --limit 10 --output report.json
```

## Manifest (SHA-256)

```json
{
  "bench_report_sha256": "c952a8a4588b7f8b568d732371d2e644c2ecad6772d133b8c38f7a03d2ef250b",
  "canonical_predictions_jsonl_sha256": "b6768c671ae45954f759e0fc7ef28dfba0ad10979130962dc6d87f8aa694f3bb",
  "jsonl_traces": {
    "01KQD97DQEPH6D7V54MB81XAVM.jsonl": "9e066625d6b7e95202c856eb0bca30fbf68fe0d3f665e23a21cfe215a597f236",
    "01KQD9801561FYTBXCBMVRK72S.jsonl": "072dd1f2b947e6f9eea9c19e9963100a5ad962366c15e6eb873131a488f05e42",
    "01KQD9BN764N1AE78764FC4MR3.jsonl": "0558c18deaf04aa16924736cc85b7d4d37e505c4cf106095527f9a969256193f",
    "01KQD9FBWGVN6XE23J8CW843ZM.jsonl": "32bbe052e73de4b0dbdcb884d08a4598ac520fdcaad3c0f32c6bf5baa731de27",
    "01KQD9FS3Q7VHZY5MZ4Q7FBWBK.jsonl": "b75c57d6e2037ef86df90d68d9a61acf6cb2c4258ac34bb3798f18d59b58f990"
  },
  "predictions_jsonl_sha256": "f374a7bdbd3035f0ad45a16f1876767ec17da0b5483b513ace5ad56424f16c11",
  "validator_version": "path_e_step3_v1"
}
```