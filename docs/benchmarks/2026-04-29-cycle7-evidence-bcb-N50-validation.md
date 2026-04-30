# Cycle 7 default-on flip — BigCodeBench Hard Instruct N=50 validation

**Date**: 2026-04-29 (evening, post commit `128e1b89` default-on flip + commit `f6711385` raw-reason leak fix)
**Cycle**: 7 default-on flip evidence (`SAGE_ORACLE` unset = ON)
**Purpose**: prove the live ``verdict_source="exact", trainable=True`` contract under cycle-7 default-on on a synchronous-eval bench via the bench-result feedback seam.

## Headline

- **Internal pass@1 = 30%** (15 exact/pass, 34 exact/fail, 1 abstain on `BigCodeBench/227`)
- **Official Docker pass@1 = 32%** (16/50, see `2026-04-29-cycle7-evidence-bcb-N50-official-pass-at-k.json`)
- **Per-task agreement = 49/50 = 98%** (the single divergence is the abstain task; bench fail = direction agreement)
- **0 raw stdout/stderr/raw_output/raw_patch leaks** across all events
- **Event order `final_result < oracle_verdict < run_frame_summary`** holds on every run that emitted an `oracle_verdict` (49/50); `BigCodeBench/227` abstained and therefore has no verdict event, so the ordering is vacuously satisfied for that run.

## Honest framing locks

- This run is **NOT** a BigCodeBench leaderboard submission. The leaderboard reports **calibrated Pass@1** with greedy decoding through the official ``bigcodebench.evaluate`` harness (or its e2b/gradio backends). On Windows the official ``untrusted_check`` path fails on ``os.killpg`` and coerces every task to ``timeout``; the seam evaluator documents a fall-back to ``BigCodeBenchBench._evaluate_solution_with_stderr`` (matplotlib-headless subprocess, deterministic per (solution, test_code)) and tags ``bench_result.verifier_id`` accordingly.
- This is a **seam validation smoke**, not a value/regression benchmark.
- Per AUDIT2 2026-04-24 framing rule: no "above SOTA" or leaderboard-style claims attached to this number.

## Setup

- ``SAGE_ORACLE`` **unset** (cycle-7 default-on contract — `oracle_enabled()` returns True). Kill-switch via ``SAGE_ORACLE=0|false|off|no|disable|disabled``.
- ``SAGE_RUN_FRAME=1``, ``SAGE_BENCH_ORACLE_SEAM=1``, ``SAGE_DIFF_VERIFIER_MODE=observe``.
- ``SAGE_BENCH_DISABLE_REPAIR=1`` (T6) for clean first-attempt measurement.
- ``StateCore`` OFF (``SAGE_STATECORE`` unset).
- Throwaway bandit DB: production state moved to ``.tmp/path_e_backup/`` pre-bench, restored post-bench. Production posteriors not polluted. **A14 reset paired with the flip** — pre-existing off-policy posteriors discarded; production starts fresh at Posterior epoch=1 (commit `128e1b89` and ops runbook `docs/operations/2026-04-29-a14-reset.md`).
- SSL: ``SSL_CERT_FILE`` + ``REQUESTS_CA_BUNDLE`` + ``CURL_CA_BUNDLE`` + ``GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`` set to ``C:/Code/certs/windows-full-bundle.pem``.
- Greedy decoding: SAGE pipeline default temperature settings; not the BCB CLI ``--temp 0`` enforcement (separate from the seam contract).
- Single entry point: ``python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50`` — no parallel scripts.

## controller_decision payload note (cgpro 2026-04-30 cycle-7 VERIFY round-1)

This N=50 evidence was produced with the **pre-allowlist** controller_decision payload writer (commit history `b6820f2b` and earlier). 157/157 events therefore carry a top-level ``payload.reason`` field; in 137/157 events it is empty (`""`), in 20/157 events it carries short non-PII ratio strings (e.g. ``"quality=0.25 < 0.3"``, ``"importance=0.09 < 0.2"``, ``"error-like output"``). cgpro 2026-04-30 cycle-7 VERIFY round-1 PUSH BACK closed this class by tightening the writer to **allowlist-only** forced payloads (no free-form ``reason``). Future regenerated N=X evidence (post-allowlist, e.g. round-2 verify) will not contain the ``reason`` key.

## cgpro Path E B' minimum pass criteria

| # | Criterion | Result |
|---|---|---|
| 1 | ≥1 ``verdict_source='exact', quality_label='pass', trainable=True`` | PASS |
| 2 | ≥1 ``verdict_source='exact', quality_label='fail', trainable=True`` | PASS |
| 3 | Event order ``final_result < oracle_verdict < run_frame_summary`` on every run | PASS (49/50) |
| 4 | No raw stdout/stderr/raw_output/raw_patch leaks in any payload | PASS (0 leaks) |

## Seam-vs-bench cross-check (per task)

Escalation/repair may turn a first-attempt seam fail into a final bench pass; the seam captures the **first-attempt** verdict, the bench report captures the **final** outcome.

| task_id | seam source | seam label | seam score | seam trainable | bench passed (final) | runtime_deltas | cross-check |
|---|---|---|---|---|---|---|---|
| BigCodeBench/13 | exact | fail | 0.0 | True | False | 4 | agree |
| BigCodeBench/15 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/17 | exact | fail | 0.0 | True | False | 1 | agree |
| BigCodeBench/19 | exact | pass | 1.0 | True | True | 6 | agree |
| BigCodeBench/34 | exact | pass | 1.0 | True | True | 7 | agree |
| BigCodeBench/37 | exact | pass | 1.0 | True | True | 5 | agree |
| BigCodeBench/82 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/89 | exact | fail | 0.0 | True | False | 8 | agree |
| BigCodeBench/92 | exact | pass | 1.0 | True | True | 8 | agree |
| BigCodeBench/93 | exact | fail | 0.0 | True | False | 8 | agree |
| BigCodeBench/99 | exact | pass | 1.0 | True | True | 3 | agree |
| BigCodeBench/100 | exact | fail | 0.0 | True | False | 2 | agree |
| BigCodeBench/101 | exact | fail | 0.0 | True | False | 10 | agree |
| BigCodeBench/108 | exact | fail | 0.0 | True | False | 4 | agree |
| BigCodeBench/120 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/123 | exact | fail | 0.0 | True | False | 0 | agree |
| BigCodeBench/124 | exact | fail | 0.0 | True | False | 9 | agree |
| BigCodeBench/129 | exact | pass | 1.0 | True | True | 5 | agree |
| BigCodeBench/139 | exact | pass | 1.0 | True | True | 7 | agree |
| BigCodeBench/147 | exact | fail | 0.0 | True | False | 7 | agree |
| BigCodeBench/161 | exact | fail | 0.0 | True | False | 6 | agree |
| BigCodeBench/162 | exact | pass | 1.0 | True | True | 6 | agree |
| BigCodeBench/177 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/184 | exact | pass | 1.0 | True | True | 3 | agree |
| BigCodeBench/187 | exact | fail | 0.0 | True | False | 9 | agree |
| BigCodeBench/199 | exact | pass | 1.0 | True | True | 5 | agree |
| BigCodeBench/208 | exact | pass | 1.0 | True | True | 0 | agree |
| BigCodeBench/211 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/214 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/227 | None | None | None | None | False | None | seam_abstain |
| BigCodeBench/239 | exact | fail | 0.0 | True | False | 1 | agree |
| BigCodeBench/241 | exact | fail | 0.0 | True | False | 6 | agree |
| BigCodeBench/267 | exact | fail | 0.0 | True | False | 6 | agree |
| BigCodeBench/273 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/274 | exact | fail | 0.0 | True | False | 4 | agree |
| BigCodeBench/287 | exact | fail | 0.0 | True | False | 8 | agree |
| BigCodeBench/302 | exact | pass | 1.0 | True | True | 7 | agree |
| BigCodeBench/308 | exact | pass | 1.0 | True | True | 11 | agree |
| BigCodeBench/310 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/313 | exact | fail | 0.0 | True | False | 2 | agree |
| BigCodeBench/324 | exact | fail | 0.0 | True | False | 2 | agree |
| BigCodeBench/326 | exact | fail | 0.0 | True | False | 2 | agree |
| BigCodeBench/341 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/346 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/360 | exact | fail | 0.0 | True | False | 6 | agree |
| BigCodeBench/367 | exact | pass | 1.0 | True | True | 7 | agree |
| BigCodeBench/368 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/374 | exact | fail | 0.0 | True | False | 5 | agree |
| BigCodeBench/399 | exact | fail | 0.0 | True | False | 3 | agree |
| BigCodeBench/401 | exact | pass | 1.0 | True | True | 6 | agree |

**Cross-check totals**: agree=49, diverged_likely_escalation=0, unknown/abstain=1.

## Verdict-source distribution

- ``exact/fail``: 34
- ``exact/pass``: 15
- ``None/None``: 1

## Reproducibility

- Repo: https://github.com/yannabadie/YGN-SAGE
- All artifacts SHA-256-hashed in the manifest below.
- Bench command (canonical, single entry point):

```bash
# cycle-7 default-on (SAGE_ORACLE unset)
SAGE_RUN_FRAME=1 SAGE_BENCH_ORACLE_SEAM=1 SAGE_BENCH_DISABLE_REPAIR=1 \
  SAGE_TRACE_JSONL_DIR=.tmp/path_e_artifacts/jsonl_n50 \
  python -m sage.bench --type bigcodebench --subset hard --split instruct \
                       --limit 50 --output report.json
```

## Manifest (SHA-256)

```json
{
  "bench_report_sha256": "75d300ec61d3a9f9eacc8f04eb2554913784ef45e154fb3e82012650a4eb3242",
  "canonical_predictions_jsonl_sha256": "60656e0f46aec10c3fc59de00200bed9b20b7d9eb926a9fa765959457b86b8e3",
  "jsonl_traces": {
    "01KQD2YG2N7WVTZ70GETD6CYA8.jsonl": "6b5e5dbba6719ad8f1e2a6bc220e5a9219a31125c7321086433dd66e445d6630",
    "01KQD2Z61VKFW6D6J4D360TWP2.jsonl": "b848e133d2bad7c857f0875b4c414ea2e74c038a65bee6ae70a2d01017a0d6d9",
    "01KQD30005W9FHDHSVPJWBDCHG.jsonl": "139d42e2e69d9b595ea6137e5aeceed71d2a41c82d033acc2b957ce16f47eb3a",
    "01KQD32DM22RG16XV27X6ZQC6V.jsonl": "87aeb60161c36299cdc0cba531e7d598e0a7258a11be18606045a6b0e6731c12",
    "01KQD330SSQFQSAPN8RWFWXR2V.jsonl": "846eb89df95a7442afd30f655d76ce9a8a1badc8e5410ed064fefd8bbe279dbc",
    "01KQD33MD89BNPKWPHRHAEQH04.jsonl": "035ff61026158e818bbb6e46b94ae1cfcada26f48caee3c8c82dfd317e691267",
    "01KQD34JBHWC9A95F5M1EKBPA6.jsonl": "8af914cc48c9fb49df4ac4d6be9489fdc4c606d003300000203f8d8756709216",
    "01KQD359Q8REY9S7E30VGJ40M0.jsonl": "4e1d3013b4229337148300a6ef66873b4ccff22ecdbad055ec437dc26ae5a900",
    "01KQD37QNT9W1180WWHZTFXYTZ.jsonl": "531c59510a2d8751bbd1e0ed469bf99dc7b5c17765da63847b2b080ec607d938",
    "01KQD3A8PDMK1KQY6A3RH40488.jsonl": "ec632e1a267c1db068e848621926d00895927713e3e6daed0960aa4ec67e9f05",
    "01KQD3B321M140EPS0SADMH3YH.jsonl": "9037ea1f88686e33d15a2df4f625dddce608d23c9e044515d9799ee9f670cbbb",
    "01KQD3BSNYXSADC065K3J0FAM9.jsonl": "5a2759cf7e109107eeaa14dad4b127409a7f4c6b8d713a9a126e093649c8ed4e",
    "01KQD3CBBJJ6CGJ6BWTERXAKDK.jsonl": "8930edee0d0c859e1e05c604f62e23652f4d916ad0c7eadb748eeabc0199f160",
    "01KQD3DP3TWS9KTG483A6V5D8R.jsonl": "b63859a0d737db8a03237ca928adaca33a158cd54ee47ee3625aa09228064666",
    "01KQD3EM7ZM7Q5B2X6JME34AD0.jsonl": "3bc50ebc8da69318fee63062ac18642aa55bd265129943155b36d13d83740fe7",
    "01KQD3FC3C9E8X6BCE03Q6BPCX.jsonl": "c6404493ebe109608ff2b89bd48bcc43a8536ac9f852f8a1c36428969a559db3",
    "01KQD3JRWJWGZA87TRZGF41AW7.jsonl": "2dca1a6a04421478a4186542c2f158fc93e9946b763424b702bd7c423377c7f7",
    "01KQD3KRH70VB9597911SKVHJQ.jsonl": "c29b5743b5002c0b4820231613378d2e9e459caec689c6902ec6b820c26227cd",
    "01KQD3MGV0CQ02XVWHY3Q6WA40.jsonl": "f4fe29f60a0779a04eac02e80009614929958aeac4c1a34765e72869aa8580fb",
    "01KQD3N4NHKGFJA6FHB4B9F7NR.jsonl": "de7299e149a8c4739149ff6c3cccd17f4944a9f9846556b17b34693b34963cf9",
    "01KQD3NVQ7QMZY7S6T9C5J5788.jsonl": "42867f8d7571ca73db3f2600241692dc5232f9be82a9539e2fd61942743f88b7",
    "01KQD3PH5E8T8QK0K9TR2CJVTT.jsonl": "e2ecc3153b0bb0a35503e9a82920cc9518819a464c7b5c1253fac37de8db7265",
    "01KQD3Q5RTGVRM3002N6MPN54A.jsonl": "c7e1f11cf1b5c2855205f066529c0fa1816dfbadfa0eaa6a92c131f1497809fa",
    "01KQD3QW93SCT4557G3SWZF1ZE.jsonl": "09605d3f4107b5f30426a08831f2c1203c40480e4ba63069e75edd1a32a11aef",
    "01KQD3RMYQN6SXNWE0HNNV6FD3.jsonl": "af307bb1fc21b9eb7e49e194ae394a0dffee765f2cdd4c25f4c291bbfbb526f9",
    "01KQD3SSZEEG7QXNZV8KXKF89A.jsonl": "cdfc13f26330ace490f7372e49d7b93f9925ff8f337031c255a2a00001c76b50",
    "01KQD3TG9ETTG5XG8HKAVEG0ZG.jsonl": "09e9c4404020969282059e6a85c196004593ad9d1b06e89a9418c10762d95929",
    "01KQD3XGXEGE1N4YZQ4CCBD1YS.jsonl": "876316a804f86c1af320bdd2774ebe8450e6d26136ceb5a1686286f6cf4993d1",
    "01KQD3YZWN72WSEET3KK7A7DQV.jsonl": "54f058700f882642da3b4d15c7e44a1a064fc47d9d693e961e596e6b4b826fd1",
    "01KQD3ZRRMVPG548N2GNT8FVN5.jsonl": "e6de604029af8954c7a3cf624a8a4dad19ddf88ff04f62d62e9d894af3c75b6e",
    "01KQD43DYXRZYW9J1110RKKJP4.jsonl": "d6f2e04c000b6d2bf31dde98577d64addd4bc2b5c8fef03c946a3acd1e3565ed",
    "01KQD467XBQ2ZXNQ3XK766MRD4.jsonl": "e1486783d3fb70921f6f3f9c979858840658abcbb533b8fc32dca7499f722be4",
    "01KQD46YTHKV61S07QGM5GR87E.jsonl": "49b6088151bc310e955b32321236679106e9924214257175b9549914433f01ff",
    "01KQD47H6QQRSR9GNEVTX3GHXR.jsonl": "640e3194ed92ee95cae2d1336d197df5b1405aa0f86ceed76c0722b78390a56e",
    "01KQD48YF4AQQGEAN3QHDZDVG5.jsonl": "2a0434ba2db26a03504f66fd768154e3580eba717d3a9cecd7ff5e3c28785e5d",
    "01KQD49S7PZM66GB365168R3PC.jsonl": "8338d04bd7cfbfa27da1e811880f069ff451881626180ea3166f73f93893e06a",
    "01KQD4AMZZKYAPBTBRBHZ4KM1P.jsonl": "af7c323f0b2e5de97d52b48073da4be753c6379af1e8a5ec8237b02f7b19478b",
    "01KQD4BHA9R5YGAYY6W3A702NE.jsonl": "197e67c56d18e6114650c6affb473f4e5706b9feb74502446dff06bda4a4883e",
    "01KQD4CJK4QJ173C92MX4Q31MA.jsonl": "f3f455408f474416d08fd41fa2b92d571c1155055a4f7d04f84eeaa52652c901",
    "01KQD4D6V6TPTXRWJ8KA8210JD.jsonl": "ca5f3b210f4f8e2ec8cb12cdf5f69d60567cceca3acd1d3ac820a6fcf80dffac",
    "01KQD4E5D0XEPF4EDH3X6TMHQQ.jsonl": "389fdfddad6d3ee4cd8c7f67eac62520ef72ab2da27af0422e2ea818e3d5e17f",
    "01KQD4EKRQZ2X467K70D4SP9HS.jsonl": "e4814abb3a0d0a65fab79c0f8eb6fca50d8fb3bfe0ceb0e181deab39edf09f1e",
    "01KQD4F1K0WC29KBZQB0X9BDMF.jsonl": "8b18c04a458226ea9ece5cb164841d17cbb6f88186f98d0fc0ed7c8c01aac2a4",
    "01KQD4FKBZNCTKQ9AR2Q3XHHY1.jsonl": "242766f8aaffb356af2bc8ff740bb68ef10413513cd128522d96e35a930deb1e",
    "01KQD4G02FKR8TV2BNQXX6BM47.jsonl": "26a733f01234e53a5e9b2411e3920427b0f40ce744c858bd4fa9eb2e20c56350",
    "01KQD4J5P901Z174C4EBF0QBCK.jsonl": "d4091ae5c644a5ac77d968259d69f5ca8dd27b6afd9ae7556937fa8c128d4fde",
    "01KQD4JX7JECHY2P94AFT7Z76C.jsonl": "db1819fec99c2228de0235a1187af861029fb8560619d54850c5b89ef91ba491",
    "01KQD4KBFQK3EPZ34DAD4D4VC0.jsonl": "77e2e92cc7390f490a9023557fe99beaac111e80e6c1a4f8f400d18364b20f2a",
    "01KQD4M6DZWNBB7M1FH9AWBKC6.jsonl": "f0b60047e2ec1876bb379bde95368d74a1e9513025bc4a8f31565108164f0d8d",
    "01KQD4MM8HNSHJVTG9W6NA7VNF.jsonl": "59b19719dc30244a0701a5e38f24b77005bb892a97c73f8f5f9617ca2b94b588"
  },
  "predictions_jsonl_sha256": "90ebf836184a805a91ef92df400c4dea40b156ea4a0a1ebe5336afcc80f8b25e",
  "validator_version": "path_e_step3_v1"
}
```