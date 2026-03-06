# YGN-SAGE V2 "Evidence-First Rebuild" — Final Design

> Validated by 3 independent reviews: Opus 4.6 (V3 self-audit), GPT-5.4 Pro (chat audit), GPT-5.4 Codex (design review with xhigh reasoning, 317K tokens).

**Date**: 2026-03-06
**Author**: Yann Abadie + Claude Opus 4.6
**Status**: Approved for implementation

---

## Core Principle

**"Evidence before feature. Contracts before topology. Baselines before claims."**

Every component emits a structured EvidenceRecord. The README describes only what EXISTS with reproducible proof. No "surpasses X" claims until an external benchmark proves it.

---

## Architecture — 4 Layers

```
Layer 3: Evidence & Policy   — EvidenceRecord, PolicyVerifier (Z3 SMT), PCAS-inspired info-flow
Layer 2: Orchestration       — TaskNode IR (VeriMAP-inspired), DyTopo routing INSIDE verified envelope
Layer 1: Runtime Substrate   — Wasm (fuel+limits), eBPF (experimental), auth dashboard
Layer 0: Data Plane          — Rust Arrow/S-MMU (kept, already solid)
```

---

## Phase Order (corrected per GPT-5.4)

### Phase 0: Kill Dangerous Defaults (2 days)
1. Remove ALL silent degradation — hard-fail when capabilities missing
2. Remove ALL false claims from README, create honest ARCHITECTURE.md
3. Archive obsolete docs/plans/ (they describe systems that don't exist)
4. Relabel routing benchmark as "heuristic self-consistency test"
5. Fix provider semantic loss: stop rewriting tool-role messages silently

### Phase 1: Benchmark & Baseline Truth (5 days)
1. Full HumanEval 164 WITH bare-model baseline (no framework) — measure overhead
2. Benchmark truth pack: per-task JSONL with git SHA, model version, seed, cost, latency
3. Provider adapter conformance tests: verify each provider actually supports claimed features
4. Semantic CapabilityMatrix (not boolean): structured_output, tool_role, file_search, grounding — per provider, verified at boot
5. Dashboard hardening: HTTPBearer auth, CORS, multi-run queue, EventBus.clear() API

### Phase 2: Contract IR + Policy Verification (20 days)
1. **TaskNode IR** (VeriMAP-inspired):
   - Typed named I/O schemas (JSON)
   - Verification functions (Python) per subtask
   - Capability requirements per node
   - Side-effect permissions + read/write sets
   - Security labels (info-flow)
   - Failure policy: retry count, replan trigger, compensation
   - Budget constraints (tokens, cost, wall time)
   - Idempotence annotation
   - Provenance outputs
   - Reducer semantics for fan-in

2. **Contract & Policy Verification** (NOT "topology proofs"):
   - Z3 SMT/Optimize: capability coverage, type compatibility, budget feasibility, shared-resource serialization, quorum legality, info-flow constraints
   - Z3 Datalog/fixedpoint: reachability, provenance tracking
   - Do NOT claim proof of semantic correctness for Python VFs or LLM outputs
   - Inspired by PCAS (arxiv 2602.16708): reference monitor, explicit policy layer

3. **EvidenceRecord** (NOT simple enum):
   ```python
   @dataclass
   class EvidenceRecord:
       level: str          # heuristic|checked|model_judged|solver_proved|empirically_validated
       proof_strength: float  # 0.0-1.0
       external_validity: bool
       freshness: datetime
       coverage: float     # what fraction of cases does this evidence cover
       assumptions: list[str]
       artifacts: list[str]  # paths to reproducible evidence
   ```
   README projection uses the simple level string. Runtime uses full record.

4. **Wasm Sandbox Hardening** (declarative interface):
   - Input: fuel_limit, wall_timeout_ms, memory_limit_bytes, host_call_budget, io_bytes_budget, module_hash, entrypoint, env_allowlist
   - Output: status, fuel_used, peak_memory, trap_kind, artifacts
   - wasmtime: consume_fuel + StoreLimits + async timeouts for host calls
   - eBPF stays experimental until memory-region model is complete

### Phase 3: Dynamic Routing + Memory (30 days)
1. **DyTopo-inspired round-level routing INSIDE verified envelope**:
   - Static Task-DAG decides stages and legal dataflow
   - DyTopo rewires communication ONLY within authorized stages
   - Every proposed edge passes legality checks: info-flow labels, budget, fan-in/out limits, channel policy
   - Failed VFs trigger retry/replan, NOT arbitrary topology mutation

2. **Causal Memory** (AMA-Bench + OpenSage insights):
   - Causality Graph: preconditions -> actions -> outcomes -> counterfactuals
   - Write gating: memory decides when NOT to write (abstention)
   - Domain-specific schemas (code-memory types differ from reasoning-memory types)
   - Keyword + graph-node hybrid retrieval
   - MANDATORY benchmark vs long-context baseline — disable memory modes that lose

3. **Planner/Executor separation** (Plan-and-Act, arxiv 2503.09572):
   - Planner generates TaskNode DAG with contracts
   - Executor runs within contracts
   - Clear separation prevents planner-executor confusion

### Phase 4: Repair + Validation (30+ days)
1. **Counterexample-guided repair** with hard fences:
   - Require idempotent or compensable effect nodes
   - Repair budget (max retries, max cost)
   - Taboo memory for failed repair attempts
   - Delta regression checks before accepting "minimal" fixes
   - Guard against: verifier gaming, wrong fault localization, oscillation, duplicated side effects, provenance leaks

2. **Synthetic failure lab** targeting MAST taxonomy (arxiv 2503.13657):
   - Known multi-agent failure categories, not self-invented demos
   - Adversarial tool environments, memory contamination, degraded providers

3. **Paper** (narrow thesis):
   - Title: "Contract-Verified Agent Orchestration with Deterministic Tool Sandboxes"
   - Claim: contract-verified orchestration with deterministic sandboxes reduces policy violations while preserving task success
   - Requires: full baselines, artifact release, at least one external benchmark

---

## Research References

| Paper | ID | Contribution to Design |
|-------|-----|----------------------|
| AdaptOrch | 2602.16873 | Topology routing formalism, convergence scaling |
| DyTopo | 2602.06039 | Round-level semantic routing (within envelope) |
| VeriMAP | 2510.17109 | TaskNode IR, verification functions, structured I/O |
| AMA-Bench | 2602.22769 | Causal memory, long-context baseline requirement |
| ADAS | 2408.08435 | Formal search space for agent design |
| AgentConductor | 2602.17100 | Layered DAG cost optimization |
| OpenSage | 2602.16891 | Domain-specific memory schemas |
| PCAS | 2602.16708 | Policy compiler, reference monitor, info-flow |
| MAST | 2503.13657 | Multi-agent failure taxonomy |
| Plan-and-Act | 2503.09572 | Planner/executor separation |

---

## Explicit YAGNI

- No RL/policy network (insufficient data, would be confabulation)
- No SWE-bench until TaskNode IR + sandbox are proven (complexity is not a moat)
- No Neo4j/vector DB (unjustified until causal memory beats long-context)
- No "surpasses X" in ANY document until reproducible external benchmark proves it
- eBPF stays experimental until memory-region model is complete

---

## Honest Timeline

GPT-5.4 correctly identified that 75 days for 6 coupled research threads is fiction. Honest estimate:

- Phase 0: 2 days (cleanup, honesty)
- Phase 1: 5-7 days (benchmarks, hardening, conformance)
- Phase 2: 20-30 days (IR + verification + evidence + Wasm — these are coupled)
- Phase 3: 30-45 days (routing + memory + planner separation — need Phase 2 stable)
- Phase 4: 30+ days (repair + failure lab + paper — research-grade work)

Total: **90-120 days** for a credible, publishable result. Not 75.

---

## Success Criteria

1. Full HumanEval 164 with published per-task traces AND bare-model baseline
2. Every component emits EvidenceRecord — dashboard shows evidence levels
3. TaskNode IR with VFs, tested against real LLM APIs (not mocked)
4. Z3 verifies contracts/policies (not topology shape) — real SAT/UNSAT, not f-strings
5. Wasm sandbox with measurable fuel/memory limits
6. Causal memory beats long-context baseline on at least one task class OR is disabled
7. Paper-ready artifact package: code, benchmarks, traces, negative results
