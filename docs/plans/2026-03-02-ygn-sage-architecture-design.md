# YGN-SAGE Architecture Design

**Date**: 2026-03-02
**Status**: Approved
**Author**: Yann Abadie + Claude Opus 4.6

## Overview

**YGN-SAGE** = Yann's Generative Neural Self-Adaptive Generation Engine

A next-generation Agent Development Kit (ADK) + flagship Research & Discovery agent, built on 5 unified cognitive pillars. Designed to surpass OpenSage by integrating evolutionary self-improvement (AlphaEvolve) and game-theoretic multi-agent strategy (PSRO/DCH).

## Research Foundation

| Paper | Key Insight | Integration in YGN-SAGE |
|-------|------------|------------------------|
| OpenSage (2602.16891) | Self-programming ADK: auto-topology, dynamic tools, graph memory | Pillars 1-3 foundation |
| AlphaEvolve (2506.13131) | LLM + evolution + automated evaluation for discovery | Pillar 4: Evolution Engine |
| PSRO/DCH (1711.00832) | Game-theoretic multi-agent framework | Pillar 5: Strategy Engine |
| LLM-evolved MARL (2602.16928) | LLMs discover better algorithms (VAD-CFR, SHOR-PSRO) | Evolution + Strategy synergy |
| Gerbicz (2505.16105) | Structural insight > parametric search | Hybrid human-AI approach |

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YGN-SAGE System                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              sage-core (Rust)                        │   │
│  │  Central orchestrator: scheduling, IPC, lifecycle    │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────┼──────────────────────────────────────┐   │
│  │  Topology │ Tools │ Memory │ Evolution │ Strategy   │   │
│  │  Engine   │Engine │ Engine │  Engine   │  Engine    │   │
│  └──────────────┼──────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            sage-python (Python SDK)                  │   │
│  │  Agent API, LLM providers, tool builders, eval       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         sage-discover (Flagship Agent)               │   │
│  │  Research & Discovery agent built on the ADK         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Packages

| Package | Language | Role |
|---------|----------|------|
| `sage-core` | Rust | High-performance orchestrator: scheduling, agent lifecycle, IPC (gRPC), sandboxing |
| `sage-python` | Python | SDK for building agents: LLM abstraction, tool API, memory API, evolution API |
| `sage-discover` | Python | Flagship Research & Discovery agent built on the SDK |

## Pillar 1: Topology Engine

### Agent Model

```python
@dataclass
class AgentConfig:
    name: str
    model: str                    # "claude-opus-4-6", "gpt-5", "gemini-3-pro", etc.
    system_prompt: str
    tools: list[str]              # tool names or paths
    memory_scope: MemoryScope     # isolated, shared, inherited
    max_steps: int
    parent_id: str | None
```

### Supported Topologies

1. **Vertical**: Parent -> sequential sub-agents (task decomposition)
2. **Horizontal**: Multiple parallel agents on same task -> ensemble
3. **Mesh**: Interconnected agents with message board (new vs OpenSage)
4. **Hierarchical**: Deep Cognitive Hierarchies - recursive levels (PSRO-inspired)

### Inter-Agent Communication

- **Message Board**: Shared file with locks (horizontal topology)
- **gRPC streams**: Real-time communication (mesh topology)
- **Event bus**: Publish events for consumption by other agents

### Agent Pool

Unified pool with operations:
- `create_agent(config)` -> Agent ID
- `search_agents(query)` -> Matching agents
- `run_agent(id, task)` -> Result
- `ensemble(ids, task, merge_strategy)` -> Merged result
- `terminate(id)` / `resume(id, checkpoint)`

## Pillar 2: Tools Engine

### Tool Levels

1. **Meta-tools** (built-in): `create_tool`, `search_tool`, `execute_tool`, `inspect_tool`
2. **System tools**: bash, file I/O, git, HTTP, etc.
3. **Domain tools**: static analysis, fuzzing, debugging, coverage
4. **AI-generated tools**: Created dynamically by agents during execution

### Hierarchical Structure

```
tools/
├── meta/              # Tool creation & management
├── system/            # OS-level tools
├── analysis/          # Code analysis (static + dynamic)
├── research/          # Research-specific tools
└── generated/         # AI-created tools (runtime)
```

### Sandboxing (via sage-core)

- Each tool set gets its own Docker container with isolated requirements
- **Snapshots**: Docker states saved for fast reuse
- **Async execution**: Long-running tools with handles (poll/retrieve/kill)
- **Shared workspace**: Docker volume mounted for data sharing

## Pillar 3: Memory Engine

### 3-Level Architecture

| Level | Type | Backend | Lifetime | Content |
|-------|------|---------|----------|---------|
| **Working** | Graph | In-memory (Rust) | Session | Execution events, tool calls, responses |
| **Episodic** | Graph + Vector | Neo4j + Qdrant | Cross-session | Past experiences, learned patterns |
| **Semantic** | Graph + Vector | Neo4j + Qdrant | Permanent | Structured knowledge, facts, relations |

### Working Memory (short-term)

In-memory graph with `AgentRun` -> `Event` -> `ToolCall`/`Response` nodes, stored in performant Rust graph (not Neo4j, to avoid latency).

### Episodic Memory (medium-term)

Compressed and indexed past executions:
- Embedding of summaries via text-embedding-3-small or local embedding
- Retrieval: hybrid (graph traversal + similarity search)
- Auto-compression when context grows

### Semantic Memory (long-term)

Persistent knowledge graph:
- Nodes: concepts, classes, functions, facts, Q&A
- Edges: DEFINED_IN, CALLS, DEPENDS_ON, ANSWERS, ABOUT
- Extensible types via Memory Agent
- Retrieval: embedding-based (top-N) + pattern-based (grep on labels)

### Memory Agent

Dedicated agent that:
- Receives natural language instructions
- Decides which memory level to query
- Manages store/update/delete with deduplication
- Working memory: read-only (auto-updated)
- Episodic + Semantic: read/write

## Pillar 4: Evolution Engine

### Evolutionary Pipeline

```
Population DB -> LLM Mutation -> Evaluation Cascade -> Selection & Archive -> loop
```

### Evolvable Targets

1. **Tools**: Evolve better tools for tasks
2. **Prompts**: Co-evolve agent system prompts (meta-prompt evolution)
3. **Strategies**: Evolve meta-strategies (like VAD-CFR discovered by AlphaEvolve)
4. **Evaluation functions**: Improve evaluators themselves
5. **Algorithms**: Discover new algorithms (flagship use case)

### Evaluation Cascade (AlphaEvolve-inspired)

- Easy tests first -> rapid elimination of bad candidates
- Progressively harder/costlier tests
- Massive parallelization of evaluations
- Multi-objective support (diversity + performance)

### Population Management (MAP-Elites inspired)

- Niche grid based on behavioral features
- Diversity maintenance: each niche keeps its best program
- Island model: isolated populations with periodic migration

## Pillar 5: Strategy Engine

### Role

When multiple agents/strategies are available for a task, the Strategy Engine decides *how to combine them* optimally.

### Meta-Strategy Solvers

- **Regret Matching**: For convergence toward correlated equilibria
- **Projected Replicator Dynamics (PRD)**: For minimizing exploitability
- **Softmax/Boltzmann**: For aggressive exploitation of best strategies
- **SHOR-PSRO** (hybrid discovered by AlphaEvolve): Adaptive blending

### Practical Application

When the flagship agent tackles a research problem:
1. Launch N agents with different approaches (horizontal topology)
2. Strategy Engine evaluates intermediate results
3. Dynamically adjust resources (more compute to promising agents)
4. Orchestrate final ensemble with optimal meta-strategy

## Reinforcement Loops

- Evolution improves meta-strategies (like VAD-CFR)
- Memory feeds Evolution (past experiences accelerate search)
- Strategy orchestrates Topology (game-theoretic solver decides topology)
- Tools are co-evolved with the agents that use them

## LLM Abstraction Layer

Multi-provider support with unified interface:

```python
class LLMProvider(Protocol):
    async def generate(self, messages, tools, config) -> Response: ...
    async def embed(self, text) -> list[float]: ...

# Supported providers:
# - Anthropic (Claude Opus, Sonnet, Haiku)
# - OpenAI (GPT-5, GPT-5 Mini)
# - Google (Gemini 3 Pro, Flash)
# - Local (Ollama, vLLM for open-source models)
```

### Heterogeneous Model Collaboration

Following OpenSage's pattern:
- Strong model (Pro/Opus) for planning and review
- Fast model (Flash/Haiku) for execution and exploration
- Automatic model routing based on task complexity

## Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Orchestrator | Rust | Performance, memory safety, concurrency |
| Agent SDK | Python 3.12+ | ML/LLM ecosystem, rapid prototyping |
| IPC | gRPC + tokio | Cross-language, async, streaming |
| Graph Memory | Neo4j | Mature graph DB, Cypher queries |
| Vector Memory | Qdrant | Fast similarity search, Rust-native |
| Metadata | SQLite | Lightweight, embedded, zero-config |
| Sandboxing | Docker | Container isolation, snapshots |
| LLM APIs | httpx/aiohttp | Async HTTP for all providers |
| Serialization | Protocol Buffers | Efficient cross-language serialization |

## Implementation Phases

| Phase | Deliverable | Priority |
|-------|-----------|----------|
| Phase 1 | `sage-core` Rust: orchestrator, agent lifecycle, basic IPC | Foundation |
| Phase 2 | `sage-python`: LLM abstraction multi-provider, tool API, basic agent | SDK |
| Phase 3 | Memory Engine: working + episodic + semantic with Neo4j/Qdrant | Memory |
| Phase 4 | Tools Engine: meta-tools, Docker sandboxing, tool synthesis | Tools |
| Phase 5 | Topology Engine: vertical/horizontal/mesh + agent pool | Multi-agent |
| Phase 6 | Evolution Engine: pipeline, population DB, evaluation cascade | Evolution |
| Phase 7 | Strategy Engine: PSRO solvers, meta-strategy selection | Strategy |
| Phase 8 | `sage-discover`: flagship Research & Discovery agent | Flagship |
