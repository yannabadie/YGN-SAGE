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
| 3 | Event order ``final_result < oracle_verdict < run_frame_summary`` on every run | PASS (10/10) |
| 4 | No raw stdout/stderr/raw_output/raw_patch leaks in any payload | PASS (0 leaks) |

## Seam-vs-bench cross-check (per task)

Escalation/repair may turn a first-attempt seam fail into a final bench pass; the seam captures the **first-attempt** verdict, the bench report captures the **final** outcome.

| task_id | seam source | seam label | seam score | seam trainable | bench passed (final) | runtime_deltas | cross-check |
|---|---|---|---|---|---|---|---|
| BigCodeBench/13 | exact | pass | 1.0 | True | True | 0 | agree |
| BigCodeBench/15 | exact | fail | 0.0 | True | True | 0 | diverged_likely_escalation |
| BigCodeBench/17 | exact | fail | 0.0 | True | True | 0 | diverged_likely_escalation |
| BigCodeBench/19 | exact | pass | 1.0 | True | True | 0 | agree |
| BigCodeBench/34 | exact | pass | 1.0 | True | True | 0 | agree |
| BigCodeBench/37 | exact | pass | 1.0 | True | True | 0 | agree |
| BigCodeBench/82 | exact | fail | 0.0 | True | False | 0 | agree |
| BigCodeBench/89 | exact | fail | 0.0 | True | True | 0 | diverged_likely_escalation |
| BigCodeBench/92 | exact | pass | 1.0 | True | True | 0 | agree |
| BigCodeBench/93 | exact | fail | 0.0 | True | False | 0 | agree |

**Cross-check totals**: agree=7, diverged_likely_escalation=3, unknown/abstain=0.

## Verdict-source distribution

- ``exact/pass``: 5
- ``exact/fail``: 5

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
  "bench_report_sha256": "85270c3f00007000f4dc2a574d87ddd6bdccff60605113877186b8e01b475a97",
  "canonical_predictions_jsonl_sha256": "4723d37fe64157ec8bb321afc0bedf644e0f8a37c0a699a814880a0ccba5e20e",
  "jsonl_traces": {
    "01KQCFTX6380AT1018J5E4965M.jsonl": "8122d483ac2767715c233083062c1258d4ba299801e8345921c222a979f63962",
    "01KQCFW5KRJHJ34JJRW7F8NKJ1.jsonl": "0458ed021b9f36b1d5cf039046324a9cc9503b83fdc8e56ebbcf43e2b237b692",
    "01KQCFYPRPWR0A3S9NJG1HVRG2.jsonl": "50663660ac89614b67306adcaf8487f1f293448b9b7baeb7ba2f296abfa20bb1",
    "01KQCG0JJBKVM1CC7M448GTW1C.jsonl": "27c773c9cbde914288e48a6084a1b72a04e0abb5c05890e42227d3d3ba86cb91",
    "01KQCG2B98AV6KY820E6K7V2TD.jsonl": "fa11b7e8db9c2686cee1e04bc47bf53102022033defe13676d4043bde8a16a57",
    "01KQCG3WHY85Z46BZ9B3JN2YBP.jsonl": "d87adfffbff0c9a3adb73b1c0fd903fa772a61b7976d2f1d7e2b5ccaffef7094",
    "01KQCG5B38910ZHNPDKXGMPHR9.jsonl": "59c722065bfe8714aaed7ad94db327853ef7eba04bb98449d8cb7ef466ca999f",
    "01KQCG7EADBZXD8CFV0S77EZ3Z.jsonl": "af73406c5df12799d368ff2cbb4dc540db46ae4cb53403281e751fd7243fe437",
    "01KQCGBFXFTVJ9FZ1TRMMA0BFQ.jsonl": "9972226f187b9f63716b583743f4dd8b8f315d0a1fb8f470d2caf44743d2b0dc",
    "01KQCGCYHKK3E2NJ9REHV2PP2G.jsonl": "78d51a2e7ad2d8fa3fbefa8381501050ab76a55dd2f916aa79cbf2867c91ac76"
  },
  "predictions_jsonl_sha256": "ef66bfe282fa46154dd8da98da7654576c64e61ea7f75da21d4d426ec4967e5c",
  "validator_version": "path_e_step3_v1"
}
```
## Official BCB harness re-grade — attempted, blocked by corporate proxy

Per BCB protocol the canonical re-grade path uses `bigcodebench/bigcodebench-evaluate:latest` via Docker:

```bash
docker run --rm \
  -v "C:/Code/YGN-SAGE/.tmp/path_e_artifacts/bcb_official:/app" \
  -v "C:/Code/certs:/certs:ro" \
  -e SSL_CERT_FILE=/certs/windows-full-bundle.pem \
  -e REQUESTS_CA_BUNDLE=/certs/windows-full-bundle.pem \
  bigcodebench/bigcodebench-evaluate:latest \
  --execution local --split instruct --subset hard \
  --samples sage--bigcodebench-instruct--multi-0-1-sanitized_calibrated.jsonl \
  --no-gt
```

The 9.27 GB official image was pulled successfully. However the run fails inside the container with `SSLCertVerificationError(self-signed certificate in certificate chain)` when downloading `bigcode/bigcodebench-hard` from HuggingFace Hub — even with the Windows trusted store bundle mounted as `windows-full-bundle.pem`. Root cause: Docker Desktop on Windows routes container egress through its own NAT, which presents a different cert chain than the host (the corporate proxy intercepts host traffic but not container traffic; the bundle that works on the host doesn't match the cert chain HF.co presents to the container).

**Reproducibility path** for an independent third party:

- On a Linux host without the corporate proxy: clone the repo, check out `path_e_step3` (this commit), copy `docs/benchmarks/2026-04-29-path-e-step3-bcb-canonical-predictions.jsonl`, run the official Docker command above (or directly via `python -m bigcodebench.evaluate`), compare per-task pass/fail with the cross-check table.
- Or upload `docs/benchmarks/2026-04-29-path-e-step3-bcb-canonical-predictions.jsonl` to https://bigcode-bigcodebench-evaluator.hf.space (the BCB gradio backend) for remote calibrated grading.

The on-host SAGE seam evaluator (subprocess fallback path with verifier_id `bcb_internal_subprocess_fallback`) is documented as a Windows-compatibility substitute for `bigcodebench.eval.untrusted_check`, deterministic per (solution, test_code). Cross-check rows tagged `agree` and `diverged_likely_escalation` provide the per-task self-consistency evidence; calibrated Pass@1 is intentionally out of scope for this seam validation (cgpro 2026-04-29 lock + AUDIT2 2026-04-24 framing rule).

