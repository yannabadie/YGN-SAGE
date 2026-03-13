# API Reference

This page documents the key classes and interfaces in the YGN-SAGE framework. For complete API details, refer to the source code in `sage-python/src/sage/` and `sage-core/src/`.

---

## AgentSystem

The top-level container returned by `boot()`. Wires all five pillars together.

```python
from sage.boot import boot

system = await boot()
```

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `agent` | `AgentLoop` | The main agent loop |
| `event_bus` | `EventBus` | Central event system |
| `guardrail_pipeline` | `GuardrailPipeline` | 3-layer guardrail chain |
| `topology_engine` | `PyTopologyEngine` | Rust topology generation |
| `bandit` | `ContextualBandit` | Model selection bandit |

**Boot sequence:**

1. Auto-discovers available LLM providers via `registry.refresh()`
2. Loads model configuration from TOML (`config/models.toml`)
3. Populates the CapabilityMatrix
4. Wires EventBus, GuardrailPipeline, TopologyEngine, ContextualBandit
5. Generates topology on every task (Phase 6)
6. Records outcomes to S-MMU + MAP-Elites + bandit learning loop

---

## AgentLoop

The runtime that implements the PERCEIVE -> THINK -> ACT -> LEARN pipeline.

```python
result = await system.agent.run("Your task here")
```

**Pipeline phases:**

| Phase | What happens |
|-------|-------------|
| PERCEIVE | Input guardrails check the task |
| THINK | Routing (S1/S2/S3) + S-MMU context injection |
| ACT | Topology execution + AVR self-refinement (S2 code tasks) |
| LEARN | Output guardrails + entity extraction + memory write |

**Key features:**

- SLF-based S2 AVR (Act-Verify-Refine) with syntax-first validation and stagnation detection
- Z3 S3 prompts for formal verification tasks
- CRAG-gated memory injection (threshold=0.3)
- Code-task detection for specialized handling
- DriftMonitor wired into `_emit()` with sliding-window analysis
- 6 circuit breakers for subsystem failure isolation

---

## ComplexityRouter

Heuristic S1/S2/S3 routing via word-boundary regex matching.

```python
from sage.strategy.metacognition import ComplexityRouter

router = ComplexityRouter()
system = router.classify("Write a sorting algorithm")  # Returns "S2"
```

**Features:**

- Word-boundary regex (`\b`) keyword matching
- CGRS self-braking
- Speculative zone detection (0.35--0.55)
- Provider injection for LLM assessment (no vendor lock-in)

---

## AdaptiveRouter

5-stage learned routing with duck-type compatibility with ComplexityRouter.

```python
from sage.strategy.adaptive_router import AdaptiveRouter

router = AdaptiveRouter()
system = router.classify("Prove that x > 0 implies x >= 1")  # Returns "S3"
```

**Stages:**

1. Structural features (Rust, zero-cost)
2. kNN embeddings (arctic-embed-m, 92% accuracy)
3. BERT ONNX classifier
4. Entropy probe
5. Cascade fallback to heuristic

Falls back to `ComplexityRouter` if `sage_core[onnx]` is unavailable.

---

## KnnRouter

kNN-based routing using pre-computed exemplar embeddings.

```python
from sage.strategy.knn_router import KnnRouter

router = KnnRouter()  # Loads from config/routing_exemplars.npz
system = router.classify("Implement binary search")  # Returns "S2"
```

**Properties:**

- Uses snowflake-arctic-embed-m embeddings (768-dim)
- Distance-weighted majority vote
- 92% accuracy on 50 GT tasks versus 52% for keyword heuristics
- Auto-builds from ground truth at boot if `.npz` is missing
- Refuses hash embeddings (must be real embeddings)

---

## DynamicTopologyEngine (Rust)

Generates and evolves multi-agent topologies via PyO3 bindings.

```python
from sage_core import PyTopologyEngine

engine = PyTopologyEngine()
result = engine.generate(task_description="Sort a list of numbers")
topology_id = result.topology_id  # Opaque ID for lazy-load
```

**6-path generation strategy:**

1. S-MMU recall (similar prior topology)
2. Archive lookup (MAP-Elites quality-diversity)
3. LLM synthesis (3-stage pipeline)
4. Mutation (7 operators)
5. MCTS (UCB1, 50 sims / 100ms)
6. Template fallback (8 built-in)

**Evolution:**

```python
engine.evolve(topology_id, fitness_score=0.95)
```

Uses MAP-Elites + CMA-ME refinement (50% random / 50% CMA-sampled mutations).

---

## TopologyExecutor (Rust)

Executes generated topologies with dual-mode scheduling.

```python
from sage_core import PyTopologyExecutor

executor = PyTopologyExecutor()
result = await executor.execute(topology)
```

**Scheduling modes:**

- **Static**: Kahn's topological sort for acyclic DAGs
- **Dynamic**: Gate-based readiness scheduling for cyclic topologies

---

## MemoryAgent

Autonomous entity extraction wired to the LEARN phase.

```python
from sage.memory.memory_agent import MemoryAgent

agent = MemoryAgent()
entities = await agent.extract_entities(text="Paris is the capital of France")
```

**Features:**

- Heuristic or LLM-based entity extraction
- Provider injection (no vendor lock-in)
- Falls back to GoogleProvider if none injected
- Writes to Tier 2 semantic memory

---

## ExoCortex

Persistent RAG via Google GenAI File Search API.

```python
from sage.memory.remote_rag import ExoCortex

exo = ExoCortex()  # Auto-configured with DEFAULT_STORE
results = await exo.search("multi-agent topology optimization")
```

**Configuration resolution:**

1. Explicit parameter
2. Environment variable `SAGE_EXOCORTEX_STORE`
3. `DEFAULT_STORE` (hardcoded)

500+ research sources indexed. Available as the `search_exocortex` agent tool.

---

## SmtVerifier (Rust)

Pure-Rust SMT solver via OxiZ with QF_LIA integer solving.

```python
from sage_core import SmtVerifier

verifier = SmtVerifier()

# Memory safety proof
result = verifier.prove_memory_safety("x >= 0", "x < buffer_size")

# Loop bound verification
result = verifier.check_loop_bound("i < n", max_iterations=1000)

# Invariant synthesis (CEGAR)
result = verifier.synthesize_invariant(
    pre="x > 0",
    post_candidates=["x >= 1", "x > -1"],
    max_rounds=5
)
```

**10 PyO3 methods:**

| Method | Purpose |
|--------|---------|
| `prove_memory_safety` | Verify memory access bounds |
| `check_loop_bound` | Verify loop termination |
| `verify_arithmetic` | Check arithmetic constraints |
| `verify_arithmetic_expr` | Check arithmetic expressions |
| `verify_invariant` | Verify loop invariants |
| `verify_invariant_with_feedback` | Invariant check with clause-level diagnostics |
| `synthesize_invariant` | CEGAR invariant synthesis (max 5 rounds) |
| `verify_array_bounds` | Check array access bounds |
| `validate_mutation` | Verify topology mutation correctness |
| `verify_provider_assignment` | SAT check for provider assignment |

Sub-0.1ms latency (0.024ms PRM, 0.060ms mutation validation).

---

## BenchmarkRunner

Entry point for running benchmarks.

```bash
# CLI interface
python -m sage.bench --type evalplus --dataset humaneval
python -m sage.bench --type ablation --limit 20
python -m sage.bench --type routing_gt
```

**Benchmark types:**

| Type | Description |
|------|-------------|
| `evalplus` | EvalPlus HumanEval+ / MBPP+ (official) |
| `ablation` | 6-config pillar contribution measurement |
| `routing_gt` | 50-task non-circular ground truth |
| `memory_ablation` | 4-config memory tier measurement |
| `evolution_ablation` | 3-config evolution search measurement |
| `humaneval` | Legacy HumanEval (164 problems) |
| `routing` | Legacy routing accuracy (30 tasks) |

---

## EventBus

Central in-process event system for observability.

```python
from sage.events.bus import EventBus

bus = EventBus()

# Subscribe to events
bus.subscribe("ROUTING", callback)

# Emit an event
bus.emit("ROUTING", {"system": "S2", "model": "gemini-2.5-flash"})

# Query recent events
events = bus.query(phase="ROUTING", last_n=10)

# Async streaming (for WebSocket)
async for event in bus.stream():
    send_to_client(event)
```

**Event types:** PERCEIVE, THINK, ACT, LEARN, ROUTING, GUARDRAIL_CHECK, GUARDRAIL_BLOCK, BENCH_RESULT, DRIFT, and more.

---

## GuardrailPipeline

3-layer guardrail chain wired at boot.

```python
from sage.guardrails.base import GuardrailPipeline
from sage.guardrails.builtin import OutputGuardrail, CostGuardrail

pipeline = GuardrailPipeline([
    CostGuardrail(max_cost=1.0),
    OutputGuardrail(),
])

result = await pipeline.check(output, phase="output")
```

**Layers:**

| Layer | Phase | Purpose |
|-------|-------|---------|
| Input | PERCEIVE | Block dangerous or off-topic tasks |
| Runtime | ACT | Validate code before sandbox execution |
| Output | LEARN | Check for empty, too-long, or refusal outputs |

**Built-in guardrails:**

- `CostGuardrail` -- cumulative cost cap
- `OutputGuardrail` -- text output validation (default)
- `SchemaGuardrail` -- JSON mode output validation

---

## Agent Composition

### SequentialAgent

```python
from sage.agents.sequential import SequentialAgent

pipeline = SequentialAgent(agents=[researcher, coder, reviewer])
result = await pipeline.run("Build a REST API")
# researcher output -> coder input -> reviewer input
```

### ParallelAgent

```python
from sage.agents.parallel import ParallelAgent

ensemble = ParallelAgent(agents=[agent_a, agent_b], aggregator=best_of)
result = await ensemble.run("Solve this problem")
```

### LoopAgent

```python
from sage.agents.loop_agent import LoopAgent

refiner = LoopAgent(agent=coder, exit_condition=tests_pass, max_iterations=5)
result = await refiner.run("Fix the failing tests")
```

### Handoff

```python
from sage.agents.handoff import Handoff

handoff = Handoff(target=math_expert, input_filter=extract_math, on_handoff=log_handoff)
result = await handoff.run("Calculate the integral of x^2")
```
