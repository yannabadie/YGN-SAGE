# YGN-SAGE v2 "Convergence" — Design Document

**Date**: 2026-03-06
**Author**: Yann Abadie + Claude Opus 4.6
**Status**: Approved
**Objective**: Transform YGN-SAGE from research prototype to functional ADK that surpasses Google ADK, OpenAI Agents SDK, and LangGraph on every dimension.

---

## Vision

A functional, benchmarkable, observable Agent Development Kit built on 3 structural advantages no competitor has:

1. **Cognitive Routing S1/S2/S3** — LLM-assessed task complexity drives agent topology selection
2. **Z3 Formal Guardrails** — Mathematical proofs, not heuristic checks
3. **Integrated Real-Time Dashboard** — Built-in observability (no external service)

Primary demonstrator: **Code Agent** (HumanEval + SWE-bench Lite benchmarks).

---

## Architecture

```
                    +----------------------------------------------+
                    |     Dashboard (:8000) -- Single HTML          |
                    |  WebSocket <-- EventBus (real-time push)      |
                    +---------------------+------------------------+
                                          | WS bidirectional
                    +---------------------v------------------------+
                    |            EventBus (in-proc)                |
                    |  All components emit here. Subscribers:       |
                    |  - WebSocket (dashboard)                     |
                    |  - JSONL logger (optional)                   |
                    |  - BenchmarkRunner (metrics capture)         |
                    +---------------------+------------------------+
                                          |
              +---------------------------v----------------------------+
              |              AgentSystem.run(task)                     |
              |                                                       |
              |  1. MetacognitiveController.assess(task)               |
              |     +-- LLM assessment (Gemini Flash Lite)            |
              |     +-- Heuristic fallback                            |
              |                                                       |
              |  2. route() -> S1 / S2 / S3                           |
              |     +-- S1: fast model, no validation                 |
              |     +-- S2: sandbox AVR loop, empirical validation    |
              |     +-- S3: Z3 formal verification                   |
              |                                                       |
              |  3. AgentLoop.run(task, routing_decision)              |
              |     perceive -> think -> act -> learn                  |
              |     +-- Tools (registry + dynamic creation)           |
              |     +-- Sub-agents (handoff / parallel / sequential)  |
              |     +-- Memory (STM + Episodic + Semantic + ExoCortex)|
              |     +-- Guardrails (input + output + Z3 runtime)     |
              |     +-- Evolution (DGM + SAMPO + SnapBPF)            |
              +-------------------------------------------------------+
```

---

## Chantier 0: Nettoyage

Remove all artifacts that undermine credibility.

### Deletions
- `docs/plans/official_benchmark_proof.json`
- `docs/plans/benchmark_results.json`
- `docs/plans/benchmark_dashboard.html`
- `docs/plans/cybergym_benchmark_proof.json`
- `docs/plans/real_benchmark_proof.json`
- `sage-discover/official_benchmark_suite.py`
- `sage-discover/cybergym_benchmark_suite.py`
- `research_journal/` (86 files of Gemini logs)
- `debug/` stale files (keep useful ones after triage)

### Fixes
- Align license: set `Cargo.toml` workspace to `license = "Proprietary"`
- Add `LICENSE` file at repo root
- Clean GitHub "About" section

---

## Chantier 1: EventBus -- Central Nervous System

Replace JSONL file polling with in-process event bus.

### Interface

```python
# sage-python/src/sage/events/bus.py
class EventBus:
    def emit(self, event: AgentEvent) -> None
    def subscribe(self, callback: Callable[[AgentEvent], None]) -> str
    def unsubscribe(self, sub_id: str) -> None
    def stream(self) -> AsyncIterator[AgentEvent]  # For WebSocket
    def query(self, phase: str = None, last_n: int = 50) -> list[AgentEvent]
```

### Event Types

| Source | Event Types |
|--------|------------|
| MetacognitiveController | `ROUTING` |
| AgentLoop | `PERCEIVE`, `THINK`, `ACT`, `LEARN` |
| SandboxManager | `SANDBOX_EXEC` |
| MemoryCompressor | `MEMORY_PRESSURE`, `MEMORY_COMPRESS` |
| EpisodicMemory | `MEMORY_STORE`, `MEMORY_SEARCH` |
| ExoCortex | `EXOCORTEX_QUERY`, `EXOCORTEX_GROUND` |
| EvolutionEngine | `EVOLVE_STEP`, `EVOLVE_ACCEPT` |
| AgentPool | `AGENT_SPAWN`, `AGENT_HANDOFF`, `AGENT_COMPLETE` |
| Guardrails | `GUARDRAIL_CHECK`, `GUARDRAIL_BLOCK` |
| BenchmarkRunner | `BENCH_START`, `BENCH_RESULT`, `BENCH_SUMMARY` |

### Integration
- `boot.py` creates EventBus and injects into all components
- `AgentLoop._emit()` sends to EventBus instead of writing JSONL
- JSONL logger becomes an optional EventBus subscriber

---

## Chantier 2: Multi-Agent Composable

### Composition Patterns

```python
# sage-python/src/sage/agents/sequential.py
class SequentialAgent:
    """Execute N sub-agents in series. Output of each feeds the next."""
    def __init__(self, name: str, agents: list[Agent], shared_state: dict = None)

# sage-python/src/sage/agents/parallel.py
class ParallelAgent:
    """Execute N sub-agents in parallel. Aggregate results."""
    def __init__(self, name: str, agents: list[Agent], aggregator: Callable = None)

# sage-python/src/sage/agents/loop.py
class LoopAgent:
    """Execute an agent in loop until exit condition."""
    def __init__(self, name: str, agent: Agent, max_iterations: int, exit_condition: Callable)
```

### Handoffs

```python
# sage-python/src/sage/agents/handoff.py
@dataclass
class Handoff:
    target: Agent
    description: str
    input_filter: Callable = None
    on_handoff: Callable = None
```

### Cognitive Handoffs (unique to YGN-SAGE)

The MetacognitiveController routes BEFORE the LLM decides. S1/S2/S3 determines which agent topology to deploy:
- S1: single fast agent, no sub-agents
- S2: code agent with sandbox AVR, may spawn sub-agents
- S3: parallel agents (coder + verifier), Z3 merge

### agents-as-tools

Any agent can be exposed as a tool for another agent via `agent.as_tool(name, description)`.

---

## Chantier 3: Guardrails -- Z3 Formal Advantage

### 3-Layer Architecture

```
Input -> [InputGuardrail] -> Agent -> [RuntimeGuardrail] -> Output -> [OutputGuardrail]
              |                           |                              |
         LLM check                  Z3 during sandbox              Z3 + LLM check
```

### Built-in Guardrails

| Guardrail | Type | Method | Competitor comparison |
|-----------|------|--------|---------------------|
| `SafeCodeGuardrail` | Runtime | Z3 bounds/loop/arithmetic | Nobody has this |
| `InjectionGuardrail` | Input | LLM prompt injection detection | OpenAI does this |
| `HallucinationGuardrail` | Output | Cross-check ExoCortex sources | LangGraph via RAG, not formal |
| `ResourceGuardrail` | Runtime | eBPF memory/cpu limits | Nobody has this |
| `CostGuardrail` | Runtime | Budget tracking per S1/S2/S3 tier | LangSmith tracks, doesn't block |
| `SchemaGuardrail` | Output | Pydantic validation | Parity with OpenAI/ADK |

### Composable Z3 Constraints

```python
agent = Agent(
    name="sql_generator",
    guardrails=[
        SafeCodeGuardrail(constraints=[
            "assert bounds(row_count, 10000)",
            "assert invariant('no_delete', 'no_drop')",
        ]),
        CostGuardrail(max_usd=0.50),
    ],
)
```

---

## Chantier 4: Memory v2 -- 4 Tiers

### Architecture

```
Tier 0: Working Memory (STM)     <- Rust Arrow buffer, per session
        MEM1 internal state every step
        Pressure-triggered compression

Tier 1: Episodic Memory          <- SQLite, cross-session
        CRUD + keyword search + temporal queries
        Auto-store significant outputs (LEARN phase)

Tier 2: Semantic Memory (NEW)    <- Entity graph, in-memory
        Entity extraction via MemoryAgent -> relations -> graph
        Graph traversal for context enrichment

Tier 3: ExoCortex (Persistent RAG) <- Google GenAI File Search
        500+ sources, passive grounding in _think()
        Active grounding via search_exocortex tool
```

### Key Changes
- `episodic.py`: add `aiosqlite` backend with `db_path` parameter (default `~/.sage/episodic.db`)
- `semantic.py`: new module, in-memory entity graph built by MemoryAgent
- `agent_loop.py` LEARN phase: wire MemoryAgent extraction -> SemanticMemory
- All memory events emitted on EventBus

### New Dependency
- `aiosqlite` (async SQLite, lightweight)

---

## Chantier 5: Dashboard -- Everything Wired

Single-file HTML (`ui/static/index.html`) + FastAPI backend (`ui/app.py`).

### Layout Sections

| Section | Data Source | Render |
|---------|-----------|--------|
| Routing indicator | `ROUTING` events | 3 bars S1/S2/S3, animated active |
| Response pane | `THINK` events | Markdown rendered, color by system |
| Memory tiers | `MEMORY_*` events | 4 progress bars + sparkline pressure |
| Agent topology | `AGENT_*` events | Mini SVG graph (nodes + edges) |
| Evolution grid | `EVOLVE_*` events | Canvas heatmap MAP-Elites |
| Guardrails | `GUARDRAIL_*` events | Green/red checklist, real-time |
| Event stream | All events | Scrollable table, filterable |
| Benchmarks | `BENCH_*` events | Progress bar + tabulated results |

### Backend Changes
- `WS /ws`: push from EventBus (replaces JSONL file reading)
- `POST /api/benchmark`: launch benchmark `{"type": "humaneval"|"swebench"|"routing"}`
- `GET /api/memory/stats`: 4-tier memory stats
- `GET /api/topology`: active agent graph (JSON nodes+edges)
- `GET /api/evolution`: MAP-Elites grid + DGM stats

### Frontend Libs (CDN, zero build)
- Tailwind CSS (styling)
- Chart.js (sparklines, bars)
- Canvas API (MAP-Elites heatmap)
- SVG inline (agent topology)

---

## Chantier 6: Benchmark Pipeline

### Benchmarks

| Benchmark | Problems | What it proves |
|-----------|----------|---------------|
| HumanEval | 164 | Code generation, S1/S2 routing value |
| SWE-bench Lite | 300 | Full cycle S1->S2->S3, sandbox, memory |
| Routing Accuracy | 100 labeled | MetacognitiveController precision |

### CLI

```bash
python -m sage.bench --type humaneval --limit 20       # Smoke test
python -m sage.bench --type humaneval                   # Full (164)
python -m sage.bench --type swebench --limit 10         # SWE subset
python -m sage.bench --type routing                     # Routing accuracy
python -m sage.bench --type all                         # Everything
```

### Metrics Captured Per Task

| Metric | HumanEval | SWE-bench | Routing |
|--------|-----------|-----------|---------|
| Pass rate | Y | Y | Y |
| Avg latency | Y | Y | Y |
| Avg cost/task | Y | Y | - |
| Routing breakdown S1/S2/S3 | Y | Y | Y |
| Escalation rate S2->S3 | Y | Y | - |
| Sandbox executions/task | Y | Y | - |
| Memory utilization | Y | Y | - |
| Z3 verifications | - | Y | - |

Results saved to `docs/benchmarks/YYYY-MM-DD-<type>.json`.

---

## Chantier 7: Testing Strategy

### Test Pyramid

| Level | Count | Requires API key | In CI |
|-------|-------|-----------------|-------|
| Unit (existing) | 200+ | No | Yes |
| Integration v2 | ~15 | No | Yes |
| E2E real | 3 | Yes | No (opt-in) |
| Benchmark smoke | ~5 | No | Yes |
| **Total** | **~223+** | - | **~220 in CI** |

### E2E Tests (3)
1. `test_s1_simple_question` -- S1 routing, fast model, <3s
2. `test_s2_code_with_sandbox` -- S2 routing, AVR sandbox, <15s
3. `test_s3_formal_verification` -- S3 routing, Z3, <30s

### Integration Tests (15)
- EventBus receives all phases
- Memory events emitted across 4 tiers
- Sequential/Parallel/Loop agents compose correctly
- Handoffs transfer context
- Guardrails block correctly
- Episodic memory persists cross-session
- Benchmark runner pipeline works end-to-end

---

## Dependency Graph

```
0 (Cleanup) --> base propre
      |
      v
1 (EventBus) --> required by everything
      |
  +---+---+--------+
  v   v   v        v
  2   3   4     (parallel)
  |   |   |
  +---+---+
      |
      v
5 (Dashboard) --> consumes events from 1-4
      |
      v
6 (Benchmarks) --> uses complete system (1-5)
      |
      v
7 (Tests) --> validates everything (0-6)
```

---

## Competitive Matrix

| Dimension | Google ADK | OpenAI Agents | LangGraph | **YGN-SAGE v2** |
|-----------|-----------|---------------|-----------|-----------------|
| Routing | Static (agent type) | LLM picks handoff | Graph edges | **S1/S2/S3 cognitive + LLM assessment** |
| Guardrails | LLM instructions | Input/Output heuristic | Human-in-loop | **Z3 formal proofs, composable** |
| Memory | Session in-memory | SQLiteSession (1 tier) | Checkpointer (1 tier) | **4 tiers (STM/Episodic/Semantic/ExoCortex)** |
| Sandbox | None | None | None | **Wasm + eBPF + Docker (3 backends)** |
| Evolution | None | None | None | **DGM + SAMPO + MAP-Elites** |
| Dashboard | GCP Console | Traces viewer (cloud) | LangSmith (cloud, paid) | **Integrated, real-time, free, local** |
| Benchmarks | Eval framework | Not integrated | Via LangSmith | **HumanEval + SWE-bench + Routing built-in** |
| Composition | Seq/Par/Loop | Handoffs + as_tool | StateGraph | **Seq/Par/Loop + Handoffs + Cognitive routing** |
| Cost tracking | None | None | Via LangSmith | **Per-tier tracking + budget guardrail** |

---

## Target Metrics

| Metric | Target |
|--------|--------|
| HumanEval pass@1 | >40% |
| SWE-bench Lite | >15% |
| Routing accuracy | >80% |
| Total tests | 223+ (220 in CI) |
| Dashboard startup | `python ui/app.py` -> all visible <2s |
| Avg latency S1 | <1s |
| Avg latency S2 | <5s |
| Avg latency S3 | <30s |
| Avg cost/task S1 | <$0.001 |
| Avg cost/task S2 | <$0.005 |
| Avg cost/task S3 | <$0.03 |
