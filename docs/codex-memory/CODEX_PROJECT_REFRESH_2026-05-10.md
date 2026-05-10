# YGN-SAGE Codex Project Refresh - 2026-05-10

This is the Codex-facing continuation note for the last 48 hours of work.
Treat it as directional memory; current source, tests, claims, and artifacts
remain authoritative.

## Current Base

- Repo: `C:\Code\YGN-SAGE`
- Branch: `main`
- Recent pushed base before the provider/model-catalog ticket:
  `76f5a54be4293d8845b08ed0d95b72e0d43fcb27`
- Current dated status note:
  `docs/status/2026-05-10-current-state.md`

## What Changed Since The 2026-05-07 Memory Refresh

- RC1 work established the Evidence-Pareto framing: optimization matters, but
  runtime integrity and proof boundaries still gate learning and claims.
- `clients/pi-ygn-sage/` and the CLI v0 backend have real subprocess/cancel
  evidence from 2026-05-09. That is transport/adapter evidence, not benchmark
  evidence.
- SWE-bench Pro official local grading remains blocked. The latest preflight
  is `docs/benchmarks/2026-05-10-grader-preflight-76f5a54b.json` with
  decision `NO_GO_GRADER_REPO_DIRTY` and blockers `host_disk_below_swebench_minimum`
  plus dirty `external/SWE-bench_Pro-os`.
- A real N=1 canary preflight was launched at
  `docs/benchmarks/2026-05-10-canary-n1-preflight-76f5a54b/`; it timed out at
  120s, produced 0 patch, and ended `BLOCKED`. Do not cite it as performance.
- Provider/model-catalog work corrected current model names against live
  discovery and provider docs:
  `deepseek-v4-flash`, `deepseek-v4-pro`, `MiniMax-M2.7`,
  `kimi-k2.6`, `qwen/qwen3.5-plus-02-15`.
- The final catalog keeps lowercase `minimax-m2.7` as a non-selectable
  compatibility alias, so the post-review total is 24 model cards.
- `cards.toml` is now the explicit source of truth for runtime model IDs,
  context windows, runtime selectability, replacements, and model runtime
  settings; `connector.py` is connection settings only.
- Live provider preflight after the catalog refresh is 10/10 OK for small
  responses, including OpenAI `gpt-5.4` and `gpt-5.5-pro`.

## Evidence To Reuse

- `docs/benchmarks/2026-05-10-live-model-discovery.json`
- `docs/benchmarks/2026-05-10-provider-preflight-post-model-catalog.json`
- `docs/benchmarks/2026-05-10-grader-preflight-76f5a54b.json`
- `docs/benchmarks/2026-05-10-canary-n1-preflight-76f5a54b/summary.json`
- `docs/status/current.json`
- `docs/status/2026-05-10-current-state.md`

## Open Next Steps

1. Wait for cgpro VERIFY on
   `.tmp/cgpro_model_catalog_provider_settings_verify_20260510.md`.
2. Apply any required corrections, then commit the provider/model-catalog and
   documentation sync atomically.
3. Resolve SWE-bench Pro grading blockers on a clean Linux Docker host or via
   authenticated Modal before claiming official benchmark results.
4. Re-run the canary only after the grader gate and CI gate have clean evidence.
