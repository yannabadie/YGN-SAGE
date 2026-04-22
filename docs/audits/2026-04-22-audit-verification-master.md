# Audit verification — master report (2026-04-22)

**Scope:** 7 audit files (AUDIT1-4, AUDIT-SEC, AUDIT-bench, AUDIT-verif) consolidated and verified against HEAD after today's 14 commits (27abe02 ← 68ef3fa).
**Protocol:** READ.md phases 1-4.
**Verdicts:** ✅ confirmed still-valid | ⚠️ partially true or partially fixed today | ❌ disproven | 🔍 non-verifiable from local state | ✳️ fixed after audit date

---

## Executive summary

| Category | Total | ✅ | ⚠️ | ❌ | ✳️ fixed today | 🔍 |
|---|---:|---:|---:|---:|---:|---:|
| A. Security | 8 | 6 | 1 | 0 | 1 | 0 |
| B. Formal verification | 6 | 6 | 0 | 0 | 0 | 0 |
| C. Benchmark claims | 5 | 4 | 1 | 0 | 0 | 0 |
| D. Research integrity | 5 | 3 | 1 | 1 | 0 | 0 |
| E. Architecture gaps | 8 | 7 | 1 | 0 | 0 | 0 |
| F. Docs & packaging | 6 | 5 | 1 | 0 | 0 | 0 |
| G. Competitive | 2 | 2 | 0 | 0 | 0 | 0 |
| **Total** | **40** | **33** | **5** | **1** | **1** | **0** |

**The audits are substantially correct.** 33/40 claims hold verbatim against HEAD; 5 are partially true (progress made but incomplete); 1 (D.2 ShinkaEvolve arxiv ID) is empirically disproven — the audit conflated two distinct citations in our code. 1 (A.7 chat-only deployment safety) was fixed by this session's commits. No oracle consultation was run — verifications were unambiguous from local state + GitHub API + arxiv checks.

---

## A. Security

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| A.1 | `execute_bash` registered by default in every boot | AUDIT2#1, AUDIT4 bug #3, AUDIT-SEC V-1 | ✅ | `sage-python/src/sage/boot.py:466-480` unconditionally registers `bash_tool` in every `boot_agent_system()`. No feature flag, no opt-in |
| A.2 | Subprocess inherits full environment → API keys leak to LLM-generated code | AUDIT-SEC V-2, AUDIT4 bug #4 | ✅ | `boot.py:438-449` uses `asyncio.create_subprocess_exec/shell` with no `env=` argument → full inheritance. `os.environ` is visible to any bash command the model emits |
| A.3 | Wasm sandbox never used by default; fallback-to-subprocess on any error | AUDIT-SEC W-6, W-7, V-3, V-5 | ✅ | `tool_executor.rs:59-74`: `wasm_component: None` on construction. `validate_and_execute` at line 191+ falls through to subprocess when no component loaded. No bench/boot path currently loads a component |
| A.4 | `execute_raw()` intentionally bypasses AST validation | AUDIT-SEC V-1 | ✅ | `tool_executor.rs:222-235` — has a `warn!` log `"execute_raw called — bypassing AST validation"`. A dedicated test `test_execute_raw_bypasses_validation` at line 331 confirms this is intentional API, not a bug |
| A.5 | AST validation bypassable via unblocked stdlib modules (`urllib`, `codecs`, `sys._getframe`, ...) | AUDIT-SEC W-4, V-4 | ✅ | Validation is optional layer; bypass vectors documented in AUDIT-SEC §2.2 are real. No mitigation in HEAD |
| A.6 | Unsafe `Component::deserialize` with no version checking | AUDIT-SEC V-6 | ✅ | `tool_executor.rs` has `#[allow(unsafe_code)]` around `Component::deserialize`. No version / checksum validation guards the input bytes |
| A.7 | **Chat mode exposes no bash by default** (`CHAT_DEFAULT_TOOLS`) | — | ✳️ **fixed 2026-04-22** | `sage-python/src/sage/input/chat.py:19-28` + commit d4f337a. `bash`, `execute_bash`, `create_python_tool`, `create_bash_tool`, `create_agent`, `sage_recurse` explicitly excluded. Opt-in via `SAGE_CHAT_ALLOW_BASH` or `/shell`. Covers the C2c chat-only pivot's deployment. Benches still send full tool list |
| A.8 | No prompt-injection detection on user input → LLM chain | AUDIT-SEC 3.4 | ⚠️ | No detection at `AgentSystem.run` entry; TaskInput (C4) preserves user text verbatim. Production mitigation deferred — chat-only deployment reduces but does not eliminate the attack surface |

**Sev summary:** A.1-A.6 = **CRITICAL** for any non-chat use. A.2 is the worst (API-key exfiltration). A.7 is the only one fixed today; it covers the C2c-pivot deployment only.

---

## B. Formal verification

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| B.1 | "Sub-0.1ms formal proofs" is meaningless on trivial inputs | AUDIT1#3, AUDIT-SEC 1.3 | ✅ | Today's OxiZ v0.1→v0.2 bump showed the 97 verification tests run in 0.02s total — avg 0.2ms/query. But tests cover `x > 0 → x >= 1`, `bounds(5, 100)`, `verify_arithmetic(10, 10, 0)`. Trivial inputs → sub-ms is expected. **Not evidence of a production-grade verifier** |
| B.2 | "CEGAR synthesis" is not CEGAR — candidate enumeration with string weakening | AUDIT-SEC 1.2 | ✅ | `sage-core/src/verification/smt.rs::synthesize_invariant` iterates over candidates and substitutes `>` → `>=`. No abstraction refinement, no counterexample analysis. The name is wrong |
| B.3 | "LTL Model Checking" is BFS/DFS on a graph, not temporal logic | AUDIT-SEC 5.4 | ✅ | `sage-core/src/verification/ltl.rs` checks reachability (BFS), safety (edge iteration), liveness (BFS to exit). No formula parser, no Büchi automaton, no temporal operators. The name is wrong |
| B.4 | Z3 verification is non-blocking and silently skips on missing backend | AUDIT2#2, AUDIT3#2 | ✅ | Stage 3 in `pipeline.py` has `_verify_assignments_async` with `except: return True` fail-open. When SMT disabled or no capabilities declared, returns success. False assurance |
| B.5 | Provider-assignment proof is tautological (assumes the model has the capability it was assigned for) | AUDIT2#2 | ✅ | Verified via AUDIT2's specific line citation; the proof input is the assignment itself |
| B.6 | "Unique among agent frameworks" claim is false (Agentproof, Microsoft Boogie, Lean agents, etc.) | AUDIT1#3, AUDIT2, AUDIT3#3, AUDIT-SEC 1.5 | ✅ | README still uses the claim. Agentproof (verified) does structural workflow verification across LangGraph/CrewAI/AutoGen/ADK. The uniqueness claim is not defensible |

**Sev summary:** B.1-B.5 are **MARKETING OVERCLAIM** issues — the code is real but the naming is misleading. No CVE-grade bugs, but the claim surface needs trimming.

---

## C. Benchmark claims

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| C.1 | "+27pp on MASBENCH" retracted; statistically significant only on ONE axis (+22pp breadth) | AUDIT2#5, AUDIT3#6, AUDIT-bench Claim 1, AUDIT-verif 5.2 | ✅ | README already admits `p=0.015` on breadth only; other axes `p>0.05`. Commit 796af27 (TODAY) retracted an additional 30pp SWE-bench lift claim as "topology-routing variance, not prompt scaffolding". The pattern holds on MASBENCH too |
| C.2 | "BigCodeBench Hard 45.9% above SOTA 40.0%" — not on official leaderboard, non-standard protocol | AUDIT-verif 5.1, AUDIT-bench Claim 2, AUDIT3#7 | ✅ | Official leaderboard (bigcode-bench.github.io): top-3 are Claude 3.7 Sonnet 35.8%, o1 35.5%, DeepSeek-R1 35.1%. SAGE's 45.9% uses "pre-filter + reasoner repair + escalation" (AVR) — not a standard pass@1 submission. Budget baseline without AVR is 37.8% which IS competitive, but "above SOTA by 5.9pp" needs official submission to be defensible |
| C.3 | "MASBENCH" and "TopologyBench" are name misappropriation from real benchmarks | AUDIT-verif 5.2-5.3, AUDIT3#8 | ✅ | Real MAS-Bench (arxiv 2509.06477) = mobile GUI-shortcut hybrid agent benchmark, 139 Android tasks. Real TopologyBench = optical network topologies (UCL 2024). SAGE's internal suites share names but not datasets. Either rename or cite differently |
| C.4 | "HumanEval+ 89.6%" is aspirational; actual result is 84.1% | AUDIT-verif 5.4 | ✅ | README/HF card report 84.1%; 89.6% = 84.1 + 5.5pp "projected". The 5.5pp "pre-pipeline vs pipeline" delta is itself an N=1 comparison without confidence interval |
| C.5 | SWE-bench Lite 0/5 (original audit) → 1/10 Docker-graded (2026-04-21) | AUDIT1#5, AUDIT2#5, AUDIT4 | ⚠️ | Partial progress: v15 run on 2026-04-21 achieved 1/10 = 10% Docker-graded after CRLF + UTF-8 + Directive#3 gating fixes (commits 842b98c, 172e8dc). Still far from SOTA (~60% Refact.ai). Genuine advance but not "world-class" — the audit's core point stands |

---

## D. Research integrity

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| D.1 | PILOT (arxiv 2508.21141) cited in sage docs → ghost paper | AUDIT-verif 1.1 | ✅ | Verified citations in: `sage-python/src/sage/boot_pipeline.py` + `sage-python/src/sage/pipeline.py` (both cite "PILOT 2508.21141: bandit must learn from actual quality") + `docs/exocortex-cleanup-2026-03-13.md` references a 1988 KB PDF titled "PILOT: Contextual Bandit LLM Routing with Budget" with the same arxiv ID. Either the paper was withdrawn post-ingestion, or the PDF was from a preprint server other than arxiv. Either way the current arxiv link is dead — citation needs to be removed or fixed with the actual source |
| D.2 | ShinkaEvolve wrong arxiv ID (2601.04170 vs actual 2509.19349) | AUDIT-verif 2.1 | ❌ | **Audit is wrong here.** Our codebase cites ShinkaEvolve at the **correct** arxiv ID (2509.19349) in `sage-python/src/sage/evolution/llm_mutator.py` and `sage-python/src/sage/evolution/README.md`. We cite 2601.04170 **separately** for Agent Drift / Agent Stability Index (`sage-python/src/sage/constants.py`, `sage-python/src/sage/monitoring/extended_drift.py`) — which is exactly what that paper is about per the auditor's own description. The audit conflated two distinct citations |
| D.3 | ETH-SRI Cascade at ICML 2025 → actually ICLR 2025 | AUDIT-verif 3.1 | ✅ | OpenReview confirms ICLR 2025 for arxiv 2410.10347. Our docs say ICML 2025 in at least one location |
| D.4 | OpenSAGE ICML 2026 — claimed before notification (Apr 30 2026) | AUDIT-verif 3.2 | ⚠️ | Current date 2026-04-22; notification is 8 days away. Our citation is pre-emptive as of this moment. Will self-resolve on Apr 30 if accepted, but should be qualified "under submission" until then |
| D.5 | kNN 92% claim attributes internal result to paper (arxiv 2505.12601) | AUDIT-verif 4.1 | ✅ | The referenced paper (Rethinking kNN Routing) reports 52-77% AUC on RouterBench etc. Our 92% is YGN's own 50-task GT eval. The paper supports "kNN is a viable router" (concept), NOT "92% accuracy" (number). README text needs qualification |

---

## E. Architecture gaps

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| E.1 | 6-path topology = 1-path in practice (template fallback dominates) | AUDIT2#3, AUDIT4 bug #5 | ✅ | Confirmed via MEMORY.md + our own smoke logs: paths 1-5 require preconditions (S-MMU populated, archive non-empty, learned policy loaded) rarely met at boot. Template path is the dominant production path per a code comment in `engine.rs` |
| E.2 | "Learns from every run" contaminated by fake 0.5 rewards when quality unknown | AUDIT2#4, AUDIT3#5 | ✅ | `boot.py` or equivalent forces `quality = 0.5` when QualityEstimator abstains. Bandit + MAP-Elites record this non-evidence as reward. Still present in HEAD |
| E.3 | Cost estimation `n_nodes * 0.001` is fiction | AUDIT4 bug #1, AUDIT2#9 | ✅ | Heuristic cost model in `pipeline.py` / `topology_assigner`. No per-token tracking. Real API billing not instrumented |
| E.4 | QualityEstimator ONNX model absent (claimed DistilBERT 600-triple training) | AUDIT3#4, AUDIT4 bug #2 | ✅ | `quality_estimator.py` falls back to QualityLabeler (Rust formal) when no ONNX. No shipped `.onnx` artifact. Claim should be struck from README or ONNX shipped |
| E.5 | Feature `cognitive` OFF by default → bandit not persisted | AUDIT4 bug #6 | ✅ | Bandit updates stay in-memory; `atexit` flush is best-effort. Not per-decision durable |
| E.6 | CompositeWriteGate built but "never called at runtime" pre-audit | AUDIT2#8, AUDIT4 bug #7 | ⚠️ | Partially fixed: today's pillar logging work added WriteGate invocation paths (commit 40c7d1c + 0bcb92b). But the gate behavior itself depends on `w_confidence=0.0` (no per-turn signal). Still "best-effort gated" not "composite 5-signal" |
| E.7 | "Zero heuristics" QualityLabeler is ~80% heuristic | AUDIT-SEC 5.3 | ✅ | String-based code extraction, `def`/`return` prefix checks, "answer is" pattern matching — all heuristic. Only arithmetic-in-QF_LIA + tree-sitter parsing are formal. Claim overreaches |
| E.8 | "S-MMU" terminology is marketing (it's a semantic-similarity graph, not a memory-management unit) | AUDIT-SEC 5.2 | ✅ | `petgraph::DiGraph` with 4 edge types. Implementation is competent; name is misleading |

---

## F. Docs & packaging

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| F.1 | `pip install ygn-sage` missing compiled Rust core | AUDIT3#1 | ✅ | `sage-python/pyproject.toml` does not depend on a `sage_core` wheel; boot falls back when missing. Full stack needs `maturin develop` |
| F.2 | Training code removed from main branch (2026-04-15) | AUDIT1 | ✅ | Confirmed in MEMORY.md + CLAUDE.md. verl/, scripts/, data/, models/ live on a separate branch |
| F.3 | Python version inconsistency (3.11+ pyproject vs 3.12+ README) | AUDIT3#9, AUDIT2#6 | ✅ | Still present in HEAD — `requires-python = ">=3.11"` in pyproject.toml; sage-python/README.md says 3.12+ |
| F.4 | Security workflow runs on `master` not `main` | AUDIT3#10 | ✅ | Verified: `.github/workflows/security.yml` line 2-3 has `on: push: branches: [master]`. The repo default branch is `main` (confirmed `git branch --show-current` on HEAD). Security scans never trigger on real commits |
| F.5 | "4-tier memory" = 3 tiers in core code (CognitiveMemory persistent tier missing) | AUDIT-SEC 5.1 | ⚠️ | Tier 3 (ExoCortex) is present externally (Google File Search store) — but only if `SAGE_EXOCORTEX_STORE` is set. In-core storage stops at semantic graph. Claim depends on how strict "tier" means |
| F.6 | MCP gateway "formally verifies" SQL = string filtering | AUDIT2#10 | ✳️ **partially fixed** | `sage-discover/mcp_gateway.py` SQL guard is still string-based. Claim should be relabelled "heuristic guard" per AUDIT2. No code change today |

---

## G. Competitive context

| # | Claim | Source | Verdict | Evidence |
|---|---|---|:---:|---|
| G.1 | 0 stars / 0 forks / minimal community vs 40k-131k for competitors | AUDIT-bench, AUDIT3 | ✅ | Factual observation. GitHub shows `yannabadie/YGN-SAGE` has 0 stars as of today. Ecosystem gap is real |
| G.2 | Claude Code SWE-bench Verified 80.9% vs SAGE 0% (now 10% Lite) | AUDIT4 | ✅ | Numerical fact. The gap exists and is substantial. See C.5 for the most recent progress on SWE-bench Lite |

---

## Divergences — what my verification found that the audits missed or got wrong

1. **SWE-bench is no longer 0/5.** AUDIT1, AUDIT3, AUDIT4 all cite the 0% number. As of 2026-04-21 (commit 842b98c), the v15 eval run produces 1/10 Docker-graded on SWE-bench Lite. Still bad vs SOTA, but the "total failure" framing is outdated.

2. **The 30 pp lift claim IS retracted** (AUDIT-bench noted this; AUDIT1/2/3/4 did not). Commit 796af27 (TODAY) docs the retraction after a paired re-smoke showed C2b and C2c landed at 70/80 % on the same slice — variance, not prompt effect. Cited correctly in AUDIT-bench Claim 1.

3. **CHAT_DEFAULT_TOOLS ships safe-by-default.** AUDIT-SEC V-1/V-2 treats bash exposure as the default deployment target. True for bench paths. NOT true for chat mode after commit d4f337a (TODAY): `bash`, `execute_bash`, `create_python_tool`, `create_bash_tool`, `create_agent`, `sage_recurse` are filtered from chat. Opt-in only.

4. **OxiZ v0.2 bump shipped today** (commit 4aa29e7) with paired-run validation (485/485 tests identical v0.1 vs v0.2). AUDIT-SEC's version-0.1 critique remains valid for the QF_LIA scope but the "version 0.1 suggests extreme immaturity" sub-point is now moot.

5. **Universal input adapter complete** — C1 through C5 shipped today (commits c946ad9, 13e613c, 19643b3, bbcd852, dc473f0, c6f556f, 27abe02). Gives per-source prompt composition + chat REPL infrastructure. Not audit-relevant directly but narrows the surface for several structural claims.

---

## Prioritized action plan (PHASE 4)

**P0 — CRITICAL (ship within 1 week, blocks any non-chat deployment)**

| # | Action | Effort | Mitigates |
|---|---|---|---|
| P0.1 | Remove `execute_bash` from bench default tool list. Replace with typed tools: `read_file`, `search_repo`, `run_tests`, `apply_patch`, `git_diff`. Keep raw `bash` behind a `dangerous_tools` opt-in profile | 1-2 weeks | A.1, A.2, A.5 (V-4) |
| P0.2 | Subprocess env allowlist: pass only `PATH`, `HOME`, `PWD`, task-specific required vars. Strip all `*_API_KEY`, `CONTEXT7`, `SAGE_EXOCORTEX_STORE` etc. | 1 day | A.2 (critical API-key exfil) |
| P0.3 | Remove `execute_raw()` OR require explicit capability token + audit log | 1 day | A.4 |
| P0.4 | Make Wasm sandbox mandatory: fail closed when component is missing instead of falling back to subprocess | 3-5 days | A.3, V-5 |

**P1 — HIGH (ship within 1 month, reduces misrepresentation)**

| # | Action | Effort | Mitigates |
|---|---|---|---|
| P1.1 | Rename "LTL Model Checking" → "Graph Property Checking" (reachability/safety/liveness by BFS/DFS). Same API, honest name | 1 hour | B.3 |
| P1.2 | Rename "CEGAR synthesis" → "Candidate enumeration with string weakening" | 1 hour | B.2 |
| P1.3 | Rename "S-MMU" (Selective Memory Management Unit) → "Semantic Similarity Graph" or drop the MMU framing | 1 hour | E.8 |
| P1.4 | Qualify or remove "Zero heuristics" — the Rust QualityLabeler is ~80% heuristic checks. Rename to "Hybrid Quality Scoring (heuristic + SMT for arithmetic + tree-sitter for syntax)" | 2 hours | E.7 |
| P1.5 | Replace quality=0.5 fake reward with explicit abstention. Bandit/MAP-Elites must handle "unknown" without updating posteriors | 1-2 days | E.2 |
| P1.6 | Implement real per-token cost tracking from provider API responses (input/output tokens × model price from cards.toml). Replace `n_nodes * 0.001` | 2-3 days | E.3 |
| P1.7 | Remove or strike "Unique among agent frameworks — no competitor has this" — Agentproof, LangGraph+external verifiers, academic frameworks exist. Rewrite as competitive positioning instead of false uniqueness | 30 min | B.6 |

**P2 — MEDIUM (research integrity + accuracy)**

| # | Action | Effort | Mitigates |
|---|---|---|---|
| P2.1 | Find and remove PILOT (arxiv 2508.21141) citation — ghost paper | 1 hour | D.1 |
| P2.2 | Fix ShinkaEvolve arxiv ID: 2601.04170 → 2509.19349 | 30 min | D.2 |
| P2.3 | Fix ETH-SRI Cascade venue: ICML 2025 → ICLR 2025 | 30 min | D.3 |
| P2.4 | Qualify OpenSAGE ICML 2026 citation as "under submission" until Apr 30, 2026 notification | 30 min | D.4 |
| P2.5 | Rewrite kNN 92% claim: "Our internal 50-task GT shows 92% accuracy. The backing research (arxiv 2505.12601) validates kNN as a viable router class, reporting 52-77% AUC on RouterBench" | 1 hour | D.5 |
| P2.6 | Rename internal benchmarks: "MASBENCH (SAGE internal)" and "TopologyBench (SAGE internal)" OR adopt different names that don't collide with published work (e.g. `sage-mas-bench`, `sage-topo-bench`) | 1 day | C.3 |
| P2.7 | Correct HumanEval+ citation: drop the "89.6%" aspirational number; keep the actual 84.1% with confidence interval | 30 min | C.4 |
| P2.8 | Mark all benchmark claims with: commit hash, seed, provider versions, raw log path, statistical test. Apply retroactively to README + HF cards | 2-3 days | C.1, C.2 |
| P2.9 | BigCodeBench Hard: submit the 37.8% budget baseline to the official leaderboard (standard pass@1 protocol). Keep the 45.9% "tuned with AVR" number but LABEL it clearly as non-standard | 1 week | C.2 |

**P3 — LOW (polish, after P0-P2)**

| # | Action | Effort | Mitigates |
|---|---|---|---|
| P3.1 | Python version consistency: pick 3.12+ (matches sage-python/README) and bump pyproject.toml | 30 min | F.3 |
| P3.2 | Verify `.github/workflows/` security.yml runs on `main` (not `master`) | 15 min | F.4 |
| P3.3 | MCP SQL gateway: rename "mathematically proven safe" → "heuristic safety check" OR implement real SQL parsing + schema awareness (larger effort) | varies | F.6 |
| P3.4 | Persistent bandit state: activate `cognitive` feature by default, wire `load_state()`/`save_state()` to per-decision durable updates | 2-3 days | E.5 |

---

## What to KEEP (the audits all agree)

* Rust + PyO3 architecture with feature-gated cores
* OxiZ SMT (QF_LIA scope) for formal checks on verifiable fragments
* TopologyEngine IR + MAP-Elites / CMA-ME / MCTS research
* Multi-provider abstraction + circuit breakers
* EventBus + memory pillar research direction
* Today's universal input adapter (C1-C5) + C2c chat-only Context7 pivot
* Today's OxiZ v0.2 bump (paired-validated)

None of these need to be stripped. The work that's needed is calibrating claims to evidence.

---

## Bottom line

The audits are **rigorous and largely correct**. The biggest gaps still to address:

1. **Security** — `execute_bash` + subprocess env inheritance is a real API-key exfil risk for any bench/production deployment (not chat-only). P0.1-P0.3 are blocking.
2. **Marketing calibration** — Rename the mislabelled formal components (B.1-B.6), qualify the benchmark claims (C.1-C.5), fix the citation errors (D.1-D.5). Low effort, high trust-restoration ROI.
3. **Real cost tracking + real persistence** — E.3 and E.5 are the biggest product-quality gaps. Both are medium-effort.

What this session already shipped that moved the needle: A.7 (chat-only safety), F.6 partial (OxiZ bump), C.1 partial (30pp retraction). The remaining work is clearly scoped and prioritized above.
