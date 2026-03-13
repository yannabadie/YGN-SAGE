# Getting Started

This guide walks you through installing YGN-SAGE and running your first agent.

## Prerequisites

- **Python 3.12+**
- **Rust 1.90+** (for building sage-core)
- **maturin** (`pip install maturin`)
- **A Google API key** (for Gemini models) or other LLM provider credentials

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE
```

### Step 2: Install the Python SDK

```bash
cd sage-python
pip install -e ".[all,dev]"
```

The `[all]` extra installs all LLM provider dependencies (Google, OpenAI, etc.). The `[dev]` extra adds test and lint tools.

### Step 3: Build the Rust Core

The Rust core provides the high-performance memory system, routing engine, topology engine, and SMT verifier. It is optional but strongly recommended.

```bash
cd ../sage-core
maturin develop
```

For additional features:

=== "ONNX Embeddings"

    ```bash
    maturin develop --features onnx
    ```

    Enables native ONNX Runtime embeddings (snowflake-arctic-embed-m, 768-dim). Auto-discovers `onnxruntime.dll` from pip package.

=== "Tool Executor"

    ```bash
    maturin develop --features tool-executor
    ```

    Enables tree-sitter AST validation + subprocess execution with timeout and kill-on-drop.

=== "Full Sandbox"

    ```bash
    maturin develop --features sandbox,tool-executor
    ```

    Adds Wasm WASI sandbox (wasmtime v36 LTS, deny-by-default) on top of the tool executor.

!!! note "Without Rust core"
    SAGE falls back to Python mock implementations with a warning. The agent loop, routing, and memory all work, but without native performance (Arrow SIMD, ONNX embeddings, SMT verification).

### Step 4: Configure Environment

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Optional overrides:

```bash
export SAGE_MODEL_FAST="gemini-2.5-flash"       # Override any tier model ID
export SAGE_DASHBOARD_TOKEN="your-token"         # Dashboard auth (no token = open dev mode)
```

Model IDs are resolved in order: environment variable `SAGE_MODEL_<TIER>` > `config/models.toml` > hardcoded defaults. TOML is searched in `cwd/config/`, `sage-python/config/`, and `~/.sage/`.

## Your First Agent

```python
import asyncio
from sage.boot import boot

async def main():
    # Boot wires all 5 pillars together
    system = await boot()

    # Run a task through the full cognitive pipeline
    result = await system.agent.run("Write a Python function to compute fibonacci numbers")
    print(result)

asyncio.run(main())
```

The `boot()` function:

1. Auto-discovers available LLM providers
2. Loads model configuration from TOML
3. Wires EventBus, GuardrailPipeline, TopologyEngine, ContextualBandit
4. Populates the CapabilityMatrix
5. Returns an `AgentSystem` ready to process tasks

Each task goes through the full pipeline: **PERCEIVE** (input guardrails) -> **THINK** (routing + memory injection) -> **ACT** (execution + sandbox) -> **LEARN** (output guardrails + memory write).

## Agent Composition

SAGE provides four composition primitives for building multi-agent systems:

```python
from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.agents.loop_agent import LoopAgent
from sage.agents.handoff import Handoff

# Chain agents in series (output feeds next input)
pipeline = SequentialAgent(agents=[researcher, coder, reviewer])

# Run agents concurrently with pluggable aggregator
ensemble = ParallelAgent(agents=[agent_a, agent_b], aggregator=my_aggregator)

# Iterate until exit condition or max_iterations
refiner = LoopAgent(agent=coder, exit_condition=tests_pass, max_iterations=5)

# Transfer control to specialist with input filtering
handoff = Handoff(target=math_expert, input_filter=extract_math_parts)
```

## Agents as Tools

Any agent can be wrapped as a tool and given to other agents:

```python
from sage.tools.agent_tool import AgentTool

# Wrap a specialist agent as a callable tool
research_tool = AgentTool.from_agent(
    agent=research_agent,
    name="deep_research",
    description="Performs in-depth research on a topic"
)

# Now other agents can use it as a tool
main_agent.tools.append(research_tool)
```

## Running Benchmarks

```bash
cd sage-python

# Official benchmarks (EvalPlus -- 80x more tests than HumanEval)
python -m sage.bench --type evalplus --dataset humaneval          # HumanEval+ (164 problems)
python -m sage.bench --type evalplus --dataset humaneval --limit 20  # Quick smoke test
python -m sage.bench --type evalplus --dataset mbpp               # MBPP+ (378 problems)

# Ablation study (proves each pillar's value vs bare LLM)
python -m sage.bench --type ablation --limit 20                   # 6 configs x 20 tasks

# Routing ground truth (non-circular, human-labeled)
python -m sage.bench --type routing_gt                            # 50 tasks
```

See [Benchmark Results](benchmarks/results.md) for detailed results and [Methodology](benchmarks/methodology.md) for evaluation protocols.

## Starting the Dashboard

```bash
python ui/app.py
```

Opens at [http://localhost:8000](http://localhost:8000). The dashboard shows real-time routing decisions (S1/S2/S3), memory tier activity, guardrail events, and benchmark results via WebSocket push.

Set `SAGE_DASHBOARD_TOKEN` to require authentication. Without it, the dashboard runs in open dev mode.

## Running Protocol Servers

SAGE supports both MCP (Model Context Protocol) and A2A (Agent-to-Agent) protocols:

```bash
pip install ygn-sage[protocols]

# MCP server
python -m sage.protocols.serve --mcp --mcp-port 8001

# A2A agent
python -m sage.protocols.serve --a2a --a2a-port 8002

# Both simultaneously
python -m sage.protocols.serve --mcp --a2a
```

See [Protocols](protocols.md) for integration details.

## Running Tests

=== "Python SDK"

    ```bash
    cd sage-python
    python -m pytest tests/ -v     # ~1300 tests
    ruff check src/                # Lint
    mypy src/                      # Type check
    ```

=== "Rust Core"

    ```bash
    cd sage-core
    cargo test --no-default-features --lib            # ~211 baseline tests
    cargo test --no-default-features --features smt --lib  # +25 SMT tests
    cargo clippy --no-default-features                # Lint
    ```

=== "Discovery Pipeline"

    ```bash
    cd sage-discover
    pip install -e .
    python -m pytest tests/ -v     # 52 tests
    ```

=== "End-to-End"

    ```bash
    python tests/e2e_proof.py      # Full-stack: 25/25 tests, ~35s, real LLM
    ```

## Next Steps

- [Architecture Overview](architecture/overview.md) -- understand the system design
- [Five Pillars](architecture/pillars.md) -- deep dive on each cognitive pillar
- [Benchmark Results](benchmarks/results.md) -- see what SAGE achieves
- [API Reference](api/core.md) -- explore key classes and interfaces
