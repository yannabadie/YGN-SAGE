# Audit remediation status (2026-04-22)

Status of the 24-item action plan from `2026-04-22-audit-verification-master.md` after this session's execution. Session commit range: `91d6eed..HEAD` (starting from the consolidated verification doc).

## Summary

| Tier | Count | ✅ Shipped | 📝 Spec only | ❌ Deferred |
|------|------:|-----------:|-------------:|------------:|
| **P0 critical** | 4 | 2 | 2 | 0 |
| **P1 high** | 7 | 7 | 0 | 0 |
| **P2 medium** | 9 | 7 | 0 | 1 (external) + 1 (no-op) |
| **P3 low** | 4 | 4 | 0 | 0 |
| **Total** | **24** | **20** | **2** | **2** |

20 action items shipped in 7 commits. 2 items landed as specs (need follow-up sprints). 1 deferred (external process, no code). 1 no-op (audit conflated two citations; our code was already correct).

## Per-item status

### P0 — critical security

| # | Action | Commit | Status | Notes |
|---|---|---|---|---|
| P0.1 | Replace `execute_bash` with typed tools (read_file / search_repo / run_tests / apply_patch / git_diff) | — | 📝 spec only | Full design in [`docs/superpowers/specs/2026-04-22-safe-sandbox-redesign-spec.md`](../superpowers/specs/2026-04-22-safe-sandbox-redesign-spec.md). Implementation is a 1-2-week sprint. Mitigated in the interim by (a) P0.2 env allowlist, (b) chat mode excluding bash by default (commit `d4f337a`), (c) deprecation NOTICE added to the `execute_bash` tool description so LLMs see it |
| P0.2 | Subprocess env allowlist | `0c7969a` | ✅ shipped | `_BASH_ENV_ALLOWLIST` + `_safe_subprocess_env()` in boot.py. API keys cannot leak via bash commands |
| P0.3 | Gate `execute_raw()` opt-in | `0c7969a` | ✅ shipped | Now denies by default; requires `SAGE_UNSAFE_RAW_EXEC=1`. Rust test `test_execute_raw_gated_by_env_var` covers both paths |
| P0.4 | Make Wasm sandbox mandatory + fail-closed | — | 📝 spec only | Same spec doc as P0.1 (they share the threat model + fix). ~2 weeks of engineering |

### P1 — high (marketing calibration)

| # | Action | Commit | Status |
|---|---|---|---|
| P1.1 | "LTL Model Checking" → "graph-property checks" | `2edccaf` | ✅ |
| P1.2 | "CEGAR synthesis" → "candidate enumeration with syntactic weakening" | `2edccaf` | ✅ |
| P1.3 | "S-MMU (Selective Memory Management Unit)" → "Structured Memory Multi-view graph" | `2edccaf` | ✅ |
| P1.4 | "Zero heuristics" QualityLabeler → "Hybrid quality scoring" with honest 80/20 split documented | `2edccaf` | ✅ |
| P1.5 | Fake 0.5 reward → explicit abstention when quality unknown | `0c7969a` | ✅ |
| P1.6 | Real per-token cost tracking | `44d6d9a` | ✅ | Verified already wired in `think._extract_step_cost`; added clarifying docstring on `_estimate_topology_cost` to distinguish predictive budget-gate from actual cost tracker |
| P1.7 | Remove "Unique among agent frameworks" overclaim | `2edccaf` | ✅ |

### P2 — medium (research integrity)

| # | Action | Commit | Status |
|---|---|---|---|
| P2.1 | Remove PILOT ghost paper (arxiv 2508.21141) citations from active code | `91d6eed` | ✅ |
| P2.2 | Fix ShinkaEvolve arxiv ID | — | ❌ no-op | Audit was wrong here — our code already cites ShinkaEvolve correctly at 2509.19349; 2601.04170 is cited separately for Agent Drift (which IS that paper's subject). README had the misassignment and was fixed as a bonus in commit `2edccaf` |
| P2.3 | Fix ETH-SRI Cascade venue (ICML 2025 → ICLR 2025) | `91d6eed` | ✅ |
| P2.4 | Qualify OpenSAGE ICML 2026 as pre-notification | `91d6eed` | ✅ |
| P2.5 | Honest kNN 92% attribution split (internal 50-task GT vs paper) | `2edccaf` | ✅ |
| P2.6 | Rename internal `MASBENCH` → `sage-mas-bench`, `TopologyBench` → `sage-topo-bench` in docs | `c63f7fb` | ✅ | Docs only; Python class names retained for API compat, breaking-rename is a separate follow-up |
| P2.7 | Correct HumanEval+ 89.6% → 84.1% (aspirational figure was never measured) | `c63f7fb` | ✅ |
| P2.8 | Add commit hash + feature flags + provider metadata to every benchmark report | `44d6d9a` | ✅ | Auto-populated at `BenchReport.from_results()` via new `_discover_git_sha()` + `_discover_feature_flags()` helpers |
| P2.9 | Submit 37.8% BCB Hard Instruct budget result to official leaderboard | — | ❌ deferred | External process (submission portal + protocol compliance), not a code task. Out of scope for this audit batch |

### P3 — low (polish)

| # | Action | Commit | Status |
|---|---|---|---|
| P3.1 | Python version consistency | `91d6eed` | ✅ | pyproject.toml → `>=3.12` (matches README) |
| P3.2 | Security workflow branch fix | `91d6eed` | ✅ | `security.yml` now triggers on `main` push + PR + weekly cron |
| P3.3 | MCP SQL gateway relabel ("mathematically proven" → "heuristic structural guard") | `0c7969a` | ✅ |
| P3.4 | Persistent bandit default-on | `44d6d9a` | ✅ | `cognitive` feature moved into sage-core default features; SQLite persistence works out of the box |

## Commits this session

1. **`8642f30`** — consolidated audit verification report (Phase 2 output; 40 assertions verified against HEAD)
2. **`91d6eed`** — P3.1 + P3.2 + P2.1 + P2.3 + P2.4 (config + citation fixes)
3. **`2edccaf`** — P1.1 + P1.2 + P1.3 + P1.4 + P1.7 + P2.5 + P2.7 (public-doc marketing calibration)
4. **`c63f7fb`** — P2.6 + P2.7 completion (benchmark rename + HumanEval+ figure)
5. **`0c7969a`** — P0.2 + P0.3 + P1.5 + P3.3 (code changes: env allowlist, execute_raw gate, reward abstention, SQL guard relabel)
6. **`44d6d9a`** — P1.6 + P2.8 + P3.4 (cost-tracker docstring, bench metadata, persistent-bandit default)
7. **this commit** — safe-sandbox redesign spec (P0.1 + P0.4) + this status doc + execute_bash deprecation notice

## Tests

After each batch the affected test suites were run:
* Rust `cargo test --features smt,tool-executor --lib`: **485/485** pass (includes new `test_execute_raw_gated_by_env_var`)
* Python `pytest tests/test_input_* tests/test_chat_* tests/test_context7_* tests/test_agent_system_* tests/test_perceive_* tests/test_swebench_* tests/test_bench_*`: **145/145** pass
* Ruff lint on every file I touched: clean

## What's still on the table

**Follow-up sprint (1-2 weeks) to close P0.1 + P0.4:**

1. Implement 6 typed repo tools (read_file, search_repo, list_files, run_tests, apply_patch, git_diff) in `sage-python/src/sage/tools/typed_repo.py`.
2. Compile them to a Wasm component (`componentize-py` or Rust rewrite).
3. Refactor `sage-core/src/sandbox/tool_executor.rs` to take `Arc<Component>` (not `Option`) and remove the subprocess fallback from `validate_and_execute`.
4. Paired smoke: N=50 SWE-bench Lite with typed tools vs `execute_bash` baseline. Ship if pass-rate parity within ±2 pp.
5. Flip `dangerous_tools` default: `False` for all configs except benches that explicitly opt in.

**External work (not code):**

1. Prepare BCB Hard Instruct submission package: predictions.jsonl + run log + commit hash. Submit to bigcode-bench.github.io official leaderboard per its protocol.

**The 2026-04-22 audit has been fully acknowledged, triaged, and either shipped-fixed or spec'd-for-follow-up.** Every claim traces to a commit or a dated design doc; there is no remaining item that silently slipped through the audit's filter.
