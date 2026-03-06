# YGN-SAGE Technical Audit V3 — Independent Assessment

**Auditor:** Claude Opus 4.6 (acting as Principal Engineer / Research Scientist)  
**Date:** 2026-03-06  
**Methodology:** Full repository clone + source-level inspection of every key component  
**Commit range:** 9680aeb → f5ccd9c (54 commits, March 2–6, 2026)

---

## EXECUTIVE SUMMARY

**YGN-SAGE is a 4-day, AI-generated prototype that presents itself as a production-grade SOTA agent framework.** The gap between claim and reality is severe across nearly every dimension. The documentation-to-code ratio (16,593 lines of plans/docs vs 8,736 lines of Python source) is the defining signature: this project was built by iterative LLM prompting, and the plans are the real artifact — the code is a sketch that followed.

That said, the *ideas* are often sound, the component architecture is well-structured, and several pieces (eBPF sandbox, Arrow memory tier, multi-provider registry) represent genuine engineering work. The path forward is not to discard but to confront what's real and what isn't.

**Verdict:** Honest PoC with interesting ideas. Not remotely SOTA. Not a competitor to AdaptOrch, OpenSage, or AgentConductor. Could become one with 3–6 months of focused, honest engineering.

---

## 1. AUDIT — Component-by-Component

### 1.1 CognitiveOrchestrator (orchestrator.py, 265 lines)

**Architecture:** Sound decomposition pattern — assess complexity, route to S1/S2/S3, select model, build topology. The implementation is clean and readable.

**Failure modes:**
- Task decomposition uses a *cheap model* to parse a *cheap prompt* that outputs `[CODE]`/`[REASON]`/`[GENERAL]` tags — then regex-parses the result. This is fragile. A single model response that doesn't follow the format silently falls back to single-task mode.
- Dependency handling is binary (has_deps → sequential, else → parallel). No hierarchical or hybrid topologies. The claim "Hierarchical" exists in TopologySpec but is never routed to.
- The `synthesize` aggregator for parallel results is just string concatenation with markdown headers. No consistency scoring, no conflict resolution, no quality gating.

**Test coverage:** 22/47 test files use mocks/patches. The orchestrator tests mock the LLM layer entirely, which means the decomposition→routing→selection pipeline has never been tested end-to-end against real models in the test suite.

**Rating:** 4/10 — Reasonable skeleton, not production-viable.

---

### 1.2 Z3 Topology Verification (z3_topology.py, 140 lines)

**This is the single most misleading claim in the project.**

The "Z3-based topology verification" is:
1. Kahn's algorithm (textbook cycle detection) — ~25 lines
2. A longest-path DFS — ~15 lines
3. A cosmetic `_z3_verify()` that calls `sage_core.Z3Validator().validate_mutation(constraints)` with a single bounds check

The proof string is a formatted f-string containing the word "sat" — not an actual SAT solver output. The `try/except ImportError: pass` means Z3 is effectively never used. The Rust Z3 validator (`sage-core/src/sandbox/z3_validator.rs`, 132 lines) takes string constraints and returns a boolean — there is no SMT encoding of DAG properties (reachability, termination, deadlock freedom).

**Claim:** "Proves DAG properties (termination, no cycles, no deadlock) BEFORE executing multi-agent topologies. Uses Kahn's algorithm + optional Z3 SMT solver."  
**Reality:** Kahn's algorithm detects cycles. That's it. Z3 is decorative.  
**What AdaptOrch does:** Provable termination guarantees via their Adaptive Synthesis Protocol with formal convergence bounds.

**Rating:** 2/10 — Graph cycle detection marketed as formal verification.

---

### 1.3 Evolution Engine (engine.py, 223 lines) + SAMPO Solver (solvers.py, 279 lines)

**Architecture:** MAP-Elites QD search with an LLM mutator and game-theoretic action selection. The idea is strong — use SAMPO-style policy optimization to choose between mutation strategies (code, prompt, hyperparameter, etc.) and let the engine self-modify.

**Failure modes:**
- `SAMPOSolver.update()` references `List[Dict[str, Any]]` but `Dict` is never imported from `typing`. This is a **runtime NameError** on Python <3.10 and a latent type-checking failure on 3.10+ (works only because annotations aren't eagerly evaluated).
- The SAMPO "sequence-level importance sampling" is actually just a per-action additive shift with clipping. There's no actual importance ratio computation, no KL divergence constraint, no trust region. The PPO-style claim is marketing.
- The DGM "self-modification" modifies `mutations_per_generation` and `clip_epsilon` by fixed deltas. This is hyperparameter scheduling, not self-modifying code in the Gödel Machine sense.
- `VolatilityAdaptiveSolver` (VAD-CFR) has inline comments in French ("Taux de décroissance de 0.1") and "SOTA Mandate" annotations that read like Gemini CLI prompt artifacts. The magic numbers (1.1 boost, -20.0 cap, 500 warm-start) appear to be LLM-confabulated rather than derived from theory or ablation.

**Test coverage:** `test_evolution.py` and `test_dgm_sampo.py` exist but test in isolation with mock evaluators. The full evolution loop (mutate → Z3 gate → eBPF evaluate → SAMPO update) has never been run end-to-end.

**Rating:** 5/10 — Interesting idea space, but the math doesn't survive scrutiny and the solvers have real bugs.

---

### 1.4 Memory System (4 tiers, ~560 lines Python + ~550 lines Rust)

**Architecture:** Tier 0 (Arrow buffer in Rust) → Tier 1 (SQLite episodic) → Tier 2 (Semantic, entity-relation graph) → Tier 3 (ExoCortex, Google GenAI File Search).

**The Rust Arrow tier is real engineering.** `compact_buffer_to_arrow()` builds proper RecordBatches with typed columns (Utf8, Timestamp, Boolean), registers chunks in the S-MMU with keywords and embeddings, and manages temporal ranges. The S-MMU (`smmu.rs`, 295 lines) implements multi-view memory management with chunk metadata indexing. This is the most solid component in the entire project.

**Failure modes:**
- The Python-side `SemanticMemory` (112 lines) stores entity-relation triples in a `dict`. No actual graph database, no embedding search, no vector similarity.
- The `MemoryAgent` (memory_agent.py) uses an LLM to extract entities from text — but there's no grounding, no deduplication, and the entity types are whatever the LLM decides to output.
- Tier 3 (ExoCortex) depends on Google GenAI File Search API availability and auto-configuration. Tight vendor coupling with no fallback.
- No memory eviction policy, no garbage collection, no memory pressure handling.

**Rating:** 6/10 — Arrow/SMMU tier is genuine; upper tiers are scaffolding.

---

### 1.5 Rust Core (sage-core, 1,727 lines)

**eBPF Sandbox (ebpf.rs, 161 lines):** Uses `solana_rbpf 0.8.5` for real ELF/raw instruction loading and VM execution. Properly handles `MemoryMapping`, `TestContextObject`, instruction counting. The `SnapBPF` CoW snapshot system is clean and well-tested (4 Rust tests with edge cases). **This is real.**

**Wasm Sandbox (wasm.rs, 90 lines):** Uses `wasmtime 29.0` with `Store`, `Engine`, `Module`, `Instance`, `Func` — a real Wasm execution path. Thin but functional.

**Arrow Tier (arrow_tier.rs, 110 lines):** Real Apache Arrow RecordBatch construction with proper schema, typed builders, S-MMU registration. Solid.

**Z3 Validator (z3_validator.rs, 132 lines):** As noted, this is the weakest Rust component. Parses constraint strings (e.g., `"assert bounds(depth, 30)"`) with regex and returns a boolean. No actual Z3 bindings — the `Cargo.toml` comment says "using Python z3-solver for formal verification (avoids LLVM/CMake build deps on Windows)."

**Rating:** 7/10 — The eBPF/Wasm/Arrow work is genuine and well-integrated. Z3 is a stub.

---

### 1.6 Benchmarks

**HumanEval:** 85% pass@1 on 20 problems using Gemini Flash Lite + S2 routing. Average latency: 42,409ms per problem. All 20 problems were routed to S2 (the routing "intelligence" never kicked in for this benchmark). No S1 or S3 routing occurred.

**Issues:**
- 20/164 problems is 12% of HumanEval. The claimed "85% pass@1" needs the asterisk "on a 20-problem subset." Industry-standard is full 164.
- 42-second average latency per problem makes this practically useless for real workloads.
- No comparison to baseline (direct Gemini Flash Lite call without the framework). The framework overhead is unknown.
- No error analysis of the 3 failures. No retry/reflection mechanism.
- No SWE-bench, GPQA, or GAIA results — the benchmarks that AdaptOrch and AgentConductor actually report on.

**Routing Accuracy:** 30/30 (100%) "with heuristic assessment." This means the heuristic (regex + keyword matching in `metacognition.py`) classified 30 hand-picked tasks correctly. This is not a meaningful benchmark — it tests that the heuristic agrees with whatever classification the developer expected.

**Rating:** 3/10 — Partial, non-comparable, no baselines.

---

### 1.7 Dashboard (ui/, 2 files)

Single-file HTML with Tailwind + Chart.js, 9 REST/WS endpoints via `app.py`. This is standard monitoring UI. Well-built for what it is.

**Rating:** 7/10 — Functional, appropriate scope.

---

### 1.8 Knowledge Pipeline (sage-discover, 3,647 lines)

arXiv/Semantic Scholar/HuggingFace ingestion, curation, and ExoCortex upload. `ModelWatcher` detects new unprofiled models across providers. This is a useful component for keeping the system's model knowledge current.

**Failure modes:**
- Discovery pipeline depends on Google GenAI File Search for ExoCortex storage — single vendor dependency.
- No rate limiting, no retry logic, no deduplication of ingested papers.
- `curator.py` uses an LLM to assess paper relevance — no ground truth, no evaluation of curation quality.

**Rating:** 5/10 — Useful scaffolding, not robust.

---

## 2. COMPARISON — vs. Competitors

### 2.1 vs. AdaptOrch (arXiv 2602.16873)

| Dimension | AdaptOrch | YGN-SAGE | Gap |
|-----------|-----------|----------|-----|
| **Topology routing** | O(|V|+|E|) formal algorithm with 4 canonical topologies, provable termination | Keyword heuristic → S1/S2/S3, 2 topologies (sequential/parallel) | **Massive** |
| **Benchmarks** | SWE-bench (+22.9%), GPQA, RAG tasks, full results | 20/164 HumanEval partial run | **Incomparable** |
| **Formal guarantees** | Convergence Scaling Law, termination proofs | Kahn's algorithm labeled "Z3" | **Massive** |
| **Synthesis** | Adaptive Synthesis Protocol with consistency scoring | String concatenation | **Massive** |
| **Paper** | 21 pages, 10 figures, 6 tables, peer-reviewable | No paper | N/A |

**YGN-SAGE does NOT surpass AdaptOrch on any dimension.**

### 2.2 vs. OpenSage (arXiv 2602.16891)

| Dimension | OpenSage | YGN-SAGE | Gap |
|-----------|----------|----------|-----|
| **Self-programming** | S-DTS stochastic tree search | DynamicAgentFactory (LLM generates agent blueprints) | **Large** — S-DTS has formal properties, YGN's factory is prompt-based |
| **Tool isolation** | Container-based | eBPF + Wasm sandbox | **YGN-SAGE may have an edge here** — hardware-level isolation is stronger than containers |
| **Memory** | Hierarchical memory agent | 4-tier memory with Arrow/S-MMU | **Close** — both are hierarchical, YGN has better low-level implementation |

**Potential genuine advantage:** The eBPF/Wasm sandbox approach is architecturally stronger than container-based isolation for code execution. This is a real differentiator if hardened.

### 2.3 vs. AgentConductor (arXiv 2602.17100)

| Dimension | AgentConductor | YGN-SAGE | Gap |
|-----------|----------------|----------|-----|
| **Optimization** | RL-optimized layered DAG topology | Game-theoretic (SAMPO) action selection | **Different approaches** — both interesting, neither proven in YGN |
| **Results** | +14.6% pass@1, -68% token cost | No comparable benchmark | **Cannot compare** |
| **Cost efficiency** | Explicitly optimized | cost_sensitivity parameter but no measured cost reduction | **Large** |

---

## 3. ROADMAP

### Phase 1: Brutal Honesty (1 week)

**P1.1: Strip all false claims (Day 1)**
- Remove "Z3 formal verification" from all docs and README. Replace with "graph-based topology validation."
- Remove "surpasses AdaptOrch/OpenSage/AgentConductor" claims. Replace with "inspired by."
- Remove "SAMPO game-theoretic solver" label. Call it "adaptive policy gradient action selector."
- Fix the `Dict` import bug in `solvers.py`.
- **Why:** Credibility is the #1 asset a new framework needs. False claims destroy it permanently.
- **Complexity:** 1 day.

**P1.2: Run full HumanEval 164 and publish raw results (Days 2–3)**
- Run complete HumanEval with direct model baseline (Gemini Flash Lite alone, no framework) and YGN-SAGE.
- Measure: framework overhead in latency and cost.
- Publish raw JSON with every problem result.
- **Why:** This is the minimum credible benchmark. Anything less is vaporware.
- **Novelty:** Honest frameworks are rare. Transparency itself is a differentiator.
- **Complexity:** 2 days.

**P1.3: Implement AdaptOrch's topology routing algorithm (Days 4–5)**
- The paper provides Algorithm 1 (topology routing) in full detail. Implement it: compute parallelism width (ω), critical path depth (δ), coupling ratio (γ), then route to parallel/sequential/hierarchical/hybrid.
- Wire it into CognitiveOrchestrator as a replacement for the keyword heuristic.
- **Why:** This is the single highest-impact upgrade. Topology routing is the thesis of the field right now.
- **Novelty:** Not novel — you're catching up. But it makes everything else meaningful.
- **Complexity:** 2 days.

**P1.4: Add Adaptive Synthesis Protocol (Days 6–7)**
- Replace string concatenation in ParallelAgent with consistency scoring across parallel outputs.
- Implement: pairwise similarity → consistency score → threshold → accept or re-route.
- **Why:** Without synthesis quality gating, parallel execution is meaningless.
- **Complexity:** 2 days.

---

### Phase 2: Differentiation (1 month)

**P2.1: eBPF-based code evaluation pipeline (Week 1–2)**
- This is YGN-SAGE's *only genuine technical moat*. No competitor uses eBPF for agent code evaluation.
- Build a proper pipeline: Python → compile to eBPF-compatible bytecode → load into EbpfSandbox → execute with memory isolation → extract result.
- Measure: sub-millisecond evaluation vs. subprocess-based execution (HumanEval's `subprocess.run`).
- **Why:** If you can evaluate candidate solutions 100–1000× faster than subprocess, the evolution engine becomes practical.
- **Novelty:** First agent framework with hardware-level sandbox for evolved code evaluation. Publishable.
- **Complexity:** 10 days.

**P2.2: Real Z3 topology verification (Week 2–3)**
- Replace the stub with actual SMT encoding: encode agent transitions as integer variables, edges as constraints, prove termination (bounded execution steps), deadlock freedom (no unreachable agents), and liveness (all outputs eventually produced).
- Use the Python `z3-solver` package directly since the Rust bindings aren't viable on Windows.
- **Why:** "Verified topologies" becomes a real claim instead of a false one. No competitor does this.
- **Novelty:** Formally verified agent topologies with SMT proofs. Publishable.
- **Complexity:** 5 days.

**P2.3: Convergence-aware model selection (Week 3)**
- Implement AdaptOrch's Convergence Scaling Law: measure pairwise model agreement, compute ε-convergence, and switch optimization target from model selection to topology selection when models converge.
- This gives the orchestrator a principled decision framework instead of hardcoded cost_sensitivity values.
- **Why:** Makes the S1/S2/S3 routing data-driven instead of heuristic.
- **Novelty:** Integrating convergence scaling into a live orchestrator (AdaptOrch theorized it, you'd operationalize it).
- **Complexity:** 5 days.

**P2.4: SWE-bench evaluation (Week 4)**
- Run SWE-bench Verified with the improved orchestrator.
- Compare against AdaptOrch's reported results with identical model backends.
- **Why:** SWE-bench is the standard for agent coding benchmarks. Without it, you're invisible.
- **Complexity:** 5 days.

---

### Phase 3: SOTA Push (3 months)

**P3.1: Evolutionary topology search with formal guarantees (Month 1)**
- Combine the evolution engine with the Z3 verifier: evolve topologies (agent structure, connections, routing rules) using MAP-Elites, but gate every candidate through Z3 verification before evaluation.
- This closes the loop: evolution proposes, formal methods verify, eBPF evaluates.
- **Why:** No framework does formally-verified evolutionary topology search. This is the paper.
- **Novelty:** Provably-safe topology evolution. Novel contribution to the field.
- **Complexity:** 20 days.

**P3.2: Multi-provider convergence detection + auto-arbitrage (Month 2)**
- Use DriftMonitor + ModelWatcher + convergence scaling to detect when providers converge in capability, then automatically shift traffic to the cheapest provider.
- Publish cost savings vs. static model selection.
- **Why:** Operationalizes the convergence scaling law for real cost reduction.
- **Novelty:** Live convergence-aware cost arbitrage across 7+ providers. Practical SOTA.
- **Complexity:** 15 days.

**P3.3: SnapBPF-accelerated evolution with CoW rollback (Month 2–3)**
- Use SnapBPF's CoW snapshots to enable microsecond-level state rollback during evolution. Snapshot before mutation → evaluate → rollback if rejected.
- Benchmark: mutations/second with vs. without SnapBPF.
- **Why:** Makes the evolution loop fast enough for real-time topology adaptation.
- **Complexity:** 10 days.

**P3.4: Write and submit the paper (Month 3)**
- Title suggestion: "Formally Verified Evolutionary Agent Topologies with Hardware-Isolated Evaluation"
- Core contributions: (1) Z3-verified topology evolution, (2) eBPF sandbox for sub-ms code evaluation, (3) Convergence-aware multi-provider orchestration.
- Target: ICML, NeurIPS, or AAAI 2027 workshop track.
- **Complexity:** 15 days.

---

## 4. BLIND SPOTS

### 4.1 The AI-generation problem is existential

This codebase has the fingerprints of Gemini CLI generation everywhere: French-language inline comments, "SOTA Mandate" annotations, magic numbers without derivation, documentation that describes capabilities the code doesn't have. The `docs/plans/` directory contains 16,593 lines of architectural plans written *before* implementation — this is LLM-first development where the plan *is* the deliverable and the code is a byproduct.

**Risk:** Every future contributor (human or AI) will inherit confabulated architecture descriptions and assume they're true. The plans describe a system that doesn't exist. This will compound with every iteration.

**Mitigation:** Delete or archive all plan documents. Replace with a single honest `ARCHITECTURE.md` that describes only what's implemented and tested.

### 4.2 The testing problem is structural

413 Python test functions across 47 files sounds impressive. But 22/47 files use mocks, and the end-to-end integration tests (`test_integration.py`, `test_integration_v2.py`) mock the LLM layer. This means the system has never been tested as a system. Individual components pass their unit tests, but the composed behavior (orchestrate → route → select → execute → synthesize) is untested.

**Risk:** The first real user will discover failures that no test catches.

**Mitigation:** Write 5–10 integration tests that call real LLM APIs (with cost budgets) and verify end-to-end correctness.

### 4.3 Research you haven't integrated that would change the architecture

- **AgentTrek (Feb 2026):** Automated agent trajectory generation for GUI agents. If you're serious about tool use, this changes how you generate training data for the evolution engine.
- **Agent-as-a-Judge (2025–2026):** Using LLM agents to evaluate other agents' outputs. Relevant to your synthesis/aggregation problem — replace string concatenation with agent-based quality assessment.
- **ADAS (Automated Design of Agentic Systems, 2025):** Meta-learning for agent architecture search. Directly relevant to your MAP-Elites topology evolution — ADAS provides a grounded framework for what you're attempting ad-hoc.
- **Trace (2025):** A programming framework for optimizing computational graphs of LLM calls. Could replace your entire orchestrator with a learned, differentiable orchestration graph.

### 4.4 The biggest risk: competing with Google ADK

The README claims YGN-SAGE "surpasses Google ADK." Google ADK has thousands of engineers, unlimited compute for benchmarking, and integration with Google's model stack. Competing head-to-head is suicide.

**Alternative framing:** YGN-SAGE should position as "the formally verified agent framework" — the one framework where you can prove your agent topology terminates before you run it. This is a niche Google won't fill because they don't need to. It's also the one genuine technical moat this project could build.

### 4.5 Single-developer bus factor

54 commits, 1 contributor, 4 days. This is a solo project built at sprint speed. The code quality is consistent (suggesting a single AI tool was used throughout), but there's zero community, zero external validation, zero code review. The `memory-bank/` and `research_journal/` directories suggest this was built using an AI coding assistant with persistent context.

**Risk:** Burnout, context loss, or a single architectural decision could kill the project.

**Mitigation:** Open-source with proper contribution guidelines, find 1–2 collaborators who can validate the core claims independently.

---

## FINAL ASSESSMENT

| Dimension | Score | Notes |
|-----------|-------|-------|
| Architecture Design | 7/10 | The 5-pillar model is well-conceived |
| Implementation Quality | 4/10 | Thin, buggy, heavily mocked |
| Benchmark Credibility | 2/10 | 20/164 HumanEval, no baselines, no comparisons |
| Claims vs. Reality | 2/10 | Pervasive overclaiming, false marketing |
| Rust Core | 7/10 | eBPF/Wasm/Arrow is real engineering |
| Novelty Potential | 6/10 | eBPF sandbox + Z3 verification (if implemented) is genuinely novel |
| Competitive Position | 2/10 | Behind all named competitors on every measured dimension |
| Path to SOTA | 5/10 | Possible in 3–6 months with honest, focused execution |

**The project has one genuine technical moat: hardware-level sandbox (eBPF/Wasm) for agent code evaluation. Everything else is either a stub, a false claim, or a reimplementation of known techniques. The path to SOTA runs through doubling down on the sandbox + formal verification story and being brutally honest about everything else.**
