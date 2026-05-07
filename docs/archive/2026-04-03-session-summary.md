# Session Summary — April 1-3, 2026

## What was done (16 commits, main branch)

### Commits (chronological)

| Commit | Description |
|--------|-------------|
| `b1b5957` | 3 new Rust topology templates (robust, horizon_pipeline, parallel_fanout) + DAG-driven selection via omega/delta/gamma |
| `c1e7848` | Adaptive context budget (removes 1000-char truncation) + cosine similarity gate + bandit→assigner feedback + multi-turn debate (reset_node + open_gate) |
| `00a1f3c` | Per-node streaming (run_stream) + HITL approval callback + documentation update (7 files) |
| `5ebcccd` | A2A v1.0 streaming + task lifecycle + cancel support |
| `4cebc51` | 10 audit fixes: bandit constraint bypass (CRITICAL), cosine dedup, security_label, write-behind SQLite queue |
| `731fb2c` | Audit6 cleanup: 0 except Exception in god files, 162MB→8MB data (moved to HuggingFace), GitHub metadata |
| `841a9da` | Split god files: boot.py 1243→647 lines (5 sub-modules), agent_loop.py 1190→576 (3 sub-modules) + remove 27 mypy overrides |
| `c12c6f6` | Clean repo root: removed 40+ personal/stale files |
| `7d1b05f` | Extend QualityLabeler to 3 verification canaux (code + math text + numeric answer via OxiZ) |
| `7714308` | Replace regex with OxiZ parser probe in QualityLabeler (island grammar pattern) |
| `d3c4d5c` | Fix pipeline budget DEFAULT_BUDGET_USD/2 → full 10.0 |
| `67e0c82` | Add SINK_NODE_PROMPT to all template exit nodes |
| `d3643d1` | 5 critical bugs from triple audit: context_window overwrite, LLMConfig context_window field, bare max_tokens 256→2048, open_gate emission, bandit self-degrading loop removed |
| `3cd6080` | Benchmark methodology: fresh boot per axis, incremental save, resumable |
| `ac737b8` | 5 evolution fixes: role tiers, quality floor 0.3 on MAP-Elites, MCTS invalid→0.0 |
| `ba31ff2` | Thompson sampling per mutation operator |
| `4c4379f` | 4 integration bugs: boot provider detection, dashboard concurrence lock, CLI serve.py import, A2A pipeline wiring |
| `2605401` | 10 iGSM exemplars in kNN → math routes to S1 |
| `6ac1eaf` | ONNX embedder singleton (1 session shared, 58000x speedup on subsequent calls) |
| `07b1957` | 4 findings fixes: sandbox bypass, gateway crash, debug exposure, MCP chain |
| `ccfaac9` | iGSM deterministic solver (later partially reverted — overfit) |
| `ea5aa7f` | Generic formal_solver topology: LLM formalizes → Rust solves |
| `fbd246d` | S1 math tasks use formal_solver instead of single-agent bypass |

---

## Current state of MASBENCH benchmarks

### Last benchmark results (20 tasks breadth, April 3)

```
Bare:  13/20 = 65%
SAGE:   3/20 = 15%
Delta: -50pp (SAGE is WORSE than bare)
```

### Previous results for reference

| Date | Bare breadth | SAGE breadth | Delta | Notes |
|------|-------------|--------------|-------|-------|
| March 31 | 40% | 72% | +32pp | Budget=5 → topology degraded to 1 node (= same as bare via pipeline) |
| April 2 (run 1) | 30% | 36% | +6pp | Budget=10, multi-node active, but verbose outputs |
| April 2 (run 2) | 12% | 10% | -2pp | Depth axis, same problems |
| April 3 (formal_solver) | 65% | 15% | -50pp | Formalizer works but solver fails on complex equations |

---

## Open problems (priority order)

### P0 — solve_equation_system() cannot handle equality chains

**File**: `sage-core/src/verification/smt.rs`, function `solve_equation_system()`

**Problem**: The topological evaluator cannot resolve variables defined by equality to other unsolved variables. Example:
```
bread_rye = price_chopper_product    ← neither is a constant
price_chopper_product = ???           ← never defined independently
```
The iterative solver resolves 10/14 variables but the remaining 4 form a dependency cycle that requires algebraic substitution or constraint propagation, not just bottom-up evaluation.

**Impact**: The formal_solver topology produces wrong answers on ~70% of MASBENCH tasks because the formalizer correctly extracts equations but the Rust solver can't resolve them.

**Solution needed**: Either extend `solve_equation_system()` to handle equality propagation (when `a = b`, unify the two variables), or use OxiZ SAT checking with binary search for unknown values.

### P0 — _execute_solver_node() answer extraction is fragile

**File**: `sage-python/src/sage/topology/runner.py`, method `_execute_solver_node()`

**Problem**: The solver node parses `ANSWER = variable_name` from the formalizer's output, but the formalizer doesn't always produce this line. When missing, the solver returns the last resolved value (often wrong).

**Impact**: Even when all equations are correctly solved, the wrong variable is returned as the answer.

**Solution needed**: Better answer extraction — parse the original question ("How many X does Y have?") to identify the target variable, or require the formalizer prompt to always end with ANSWER.

### P1 — Some tasks get domain=general instead of domain=math

**File**: `sage-python/src/sage/pipeline_stages.py`, function `_infer_domain()`

**Problem**: `_infer_domain()` uses keyword matching. Tasks with entity names like "Cerberus's Ganglia" or "Price Chopper's Bread" don't match math keywords → domain="general" → no formal_solver.

**Impact**: 5/20 tasks (T1, T3, T7, T15, T19) bypassed formal_solver and used single_agent path.

**Evidence**: These 5 tasks show `topo=None(0n)` in the traces.

### P1 — Provider timeout/fallback adds ~60s latency

**File**: `sage-python/src/sage/topology/runner.py`, `_execute_node()` fallback logic

**Problem**: OpenRouter fails on ~50% of calls (circuit breaker opens), triggering a 60s timeout + fallback to DeepSeek. The formal_solver should use DeepSeek directly for the formalizer node.

**Impact**: Tasks that should take 10s take 130s due to provider failures.

### P2 — SINK_NODE_PROMPT on solver node is wrong

**File**: `sage-core/src/topology/templates.rs`, `formal_solver()` function

**Problem**: The solver node has SINK_NODE_PROMPT ("You are the final synthesizer...") which is an LLM prompt, but the solver is a Rust deterministic node — it doesn't call an LLM. The prompt is ignored but it's conceptually wrong.

**Solution**: Set `solver.prompt = ""` (empty) since the solver node dispatches to `_execute_solver_node()` which doesn't use the prompt.

### P2 — Bare model max_tokens=2048 still truncates some answers

**File**: `sage-python/src/sage/bench/masbench.py`, line 238

**Problem**: Some complex chain-of-thought responses need >2048 tokens. The bare baseline is still slightly truncated.

---

## Architecture state

### Templates: 12
sequential, parallel, AVR, selfmoa, hierarchical, hub, debate, brainstorming, robust, horizon_pipeline, parallel_fanout, **formal_solver**

### Routing
- kNN with 60 exemplars (20 S1, 20 S2, 20 S3)
- 10 iGSM exemplars route math → S1
- S1 + math domain → formal_solver topology
- S1 + non-math → single-agent (no topology)
- S2/S3 → TopologyEngine 6-path

### Quality verification
- QualityLabeler: 3 canaux (code blocks, text arithmetic via OxiZ parser probe, numeric answer)
- Quality floor 0.3 on MAP-Elites archive
- FrugalGPT cascade on quality < 0.3

### Evolution
- Role tier system (4 tiers: input < processing < evaluation < synthesis)
- Thompson sampling per mutation operator
- HybridVerifier rejects role ordering violations
- MCTS scores invalid mutations at 0.0

### Memory
- ONNX embedder singleton (1 session, 58000x speedup)
- Write-behind asyncio.Queue for EpisodicMemory
- Bandit + MAP-Elites state persisted to ~/.sage/

---

## Log files

| File | Content |
|------|---------|
| `docs/benchmarks/2026-04-03-masbench/masbench_traces.jsonl` | 20 detailed task traces (topology, nodes, edges, outputs, scores) — **THE KEY FILE** |
| `masbench_traced.log` | Local root DEBUG log removed during 2026-05-07 root cleanup |
| `masbench_traced_console.log` | Local root console log removed during 2026-05-07 root cleanup |
| `masbench_official.log` | Stale local root official ablation log removed during 2026-05-07 root cleanup |
| `~/.sage/archive_state.db` | PURGED (was corrupted by random mutations) |
| `~/.sage/bandit_state.db` | PURGED |
| `~/.sage/episodic.db` | Episodic memory (active) |
| `~/.sage/semantic.db` | Semantic memory (active) |
| `~/.sage/causal.db` | Causal memory (active) |

---

## Test counts

- **Rust**: 436 tests, 0 failures
- **Python**: ~2000+ tests (not re-run since god file split)
- **New this session**: QualityLabeler 3-canal tests, iGSM parser tests, mutation stats tests, role tier tests

---

## Key research references used

| Paper | arXiv | How used |
|-------|-------|----------|
| AdaptOrch | 2602.16873 | Var_tau/Var_M formula: topology adds zero value when omega=1 |
| SatLM | 2305.09656 | Formalization/solving separation: +23% on hard math |
| MALT | 2412.01928 | Generation→Verification→Refinement loop (multi-turn) |
| From Spark to Fire | 2603.04474 | Error cascade mechanics in multi-agent |
| S2-MAD | 2502.04790 | Similarity gate: -94% tokens |
| AgentConductor | 2602.17100 | Role ordering, graded penalties, topology evolution |
| Graph-GRPO | 2603.02701 | Edge-level credit assignment |
| AgentDropout | 2503.18891 | Policy gradient on adjacency matrices |
| MonoScale | 2601.23219 | Trust-region Thompson sampling |
| TalkHier | 2502.11098 | Structured communication carrying full outputs |
| iGSM (Meta) | 2407.20311 | Single GPT-2 achieves 99% on dependency-graph math |
| MAS-Orchestra | 2601.14652 | MASBENCH benchmark definition |

---

## Next steps (recommended)

1. **Fix `solve_equation_system()` equality propagation** — when encountering `a = b`, unify variables before evaluation. This is the #1 blocker for formal_solver accuracy.

2. **Fix answer extraction** in `_execute_solver_node()` — parse the question text to find the target variable instead of relying on ANSWER= line.

3. **Fix domain inference** — add math keywords that cover iGSM vocabulary ("number of each", "equals", "times as much").

4. **Pin DeepSeek as default provider for formalizer** — avoid OpenRouter timeouts.

5. **Re-run benchmark** after fixes 1-4 to measure if formal_solver beats bare.

6. **If formal_solver still doesn't beat bare**: the honest conclusion is that for iGSM-style tasks, a single strong model with chain-of-thought is optimal. SAGE's value is on composite tasks (code + test + review) where roles are genuinely complementary.
