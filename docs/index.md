# YGN-SAGE

**Yann's Generative Neural Self-Adaptive Generation Engine** -- an Agent Development Kit built on five cognitive pillars.

---

## What is YGN-SAGE?

YGN-SAGE is a research-grade Agent Development Kit (ADK) that treats agent orchestration as a systems engineering problem rather than a prompt engineering one. It combines a high-performance Rust core with a flexible Python SDK to build agents that route tasks through cognitive systems, evolve their own topology, verify correctness with SMT solvers, and learn from every interaction.

The system is organized around **five cognitive pillars**:

| Pillar | What it does |
|--------|-------------|
| **Topology** | Evolves multi-agent graph structures using MAP-Elites, CMA-ME, MCTS, and LLM synthesis |
| **Strategy** | Routes tasks through S1/S2/S3 cognitive systems using kNN embeddings, contextual bandits, and BERT classifiers |
| **Memory** | 4-tier memory hierarchy from Rust Arrow working memory to persistent ExoCortex RAG |
| **Tools** | Wasm WASI sandboxed execution with tree-sitter AST validation and subprocess fallback |
| **Evolution** | Self-modifying agent topologies via SAMPO-directed mutations with formal verification |

## Key Differentiators

**Evolutionary topology search.** Unlike frameworks that use fixed agent pipelines, SAGE evolves multi-agent topologies using MAP-Elites quality-diversity search, CMA-ME directional optimization, and Monte Carlo Tree Search. The `DynamicTopologyEngine` tries six paths before falling back to templates.

**Cognitive routing, not keyword matching.** Tasks are classified into S1 (fast/intuitive), S2 (analytical), or S3 (formal verification) cognitive systems using a 4-stage learned pipeline: structural features, kNN embeddings, BERT ONNX classifier, and entropy probing (with cascade fallback to heuristic). The kNN router achieves 92% accuracy on human-labeled ground truth versus 52% for keyword heuristics.

**SMT formal verification.** S3 tasks get genuine satisfiability checking via OxiZ (pure-Rust SMT solver) with Z3 fallback. This includes memory safety proofs, loop bound verification, CEGAR invariant synthesis, and LTL temporal model checking on topology graphs.

**Rust-native performance core.** The orchestrator, memory system (Arrow/SIMD), embedder (ONNX Runtime), router, topology engine, and SMT verifier all run in Rust via PyO3 bindings. Python is the SDK layer; the hot path is native.

**7 LLM providers, zero vendor lock-in.** Auto-discovers Google, OpenAI, xAI (Grok), DeepSeek, MiniMax, Kimi, and Codex CLI at boot. FrugalGPT cascade fallback across providers. Any provider can be injected into any component.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yannabadie/YGN-SAGE.git
cd YGN-SAGE

# Install Python SDK with all providers
cd sage-python
pip install -e ".[all]"

# Build Rust core with Python bindings
cd ../sage-core
maturin develop

# Set your API key
export GOOGLE_API_KEY="your-key-here"

# Run your first agent
cd ../sage-python
python -c "
import asyncio
from sage.boot import boot

async def main():
    system = await boot()
    result = await system.agent.run('What is the capital of France?')
    print(result)

asyncio.run(main())
"
```

See the [Getting Started](quickstart.md) guide for detailed installation instructions.

## Project Structure

```
YGN-SAGE/
  sage-core/       # Rust orchestrator (PyO3 bindings)
  sage-python/     # Python SDK for building agents
  sage-discover/   # Knowledge Pipeline (arXiv -> ExoCortex)
  ui/              # Control Dashboard (FastAPI + WebSocket)
  docs/            # Documentation
```

## Benchmark Highlights

| Benchmark | Score | Context |
|-----------|-------|---------|
| EvalPlus HumanEval+ | **84.1%** pass@1 | 164 problems, 80x harder tests |
| EvalPlus MBPP+ | **75.1%** pass@1 | 378 problems, 35x harder tests |
| Routing accuracy (kNN) | **92%** | 50 human-labeled tasks |
| TopologyBench best | **96.3%** | Evolved topology on HumanEval+ |
| Ablation: full vs baseline | **+15pp** | Same model, framework value proof |

SOTA context for HumanEval+ pass@1: O1 ~89%, GPT-4o ~87%, Qwen2.5-Coder-32B ~87%, **YGN-SAGE 84.1%** (using budget Gemini 2.5 Flash), Claude Sonnet 3.5 ~82%.

See [Benchmark Results](benchmarks/results.md) for the full breakdown.

## Tech Stack

- **Rust 1.90+** -- core orchestrator via PyO3 (Arrow memory, SIMD, SMT, topology engine)
- **Python 3.12+** -- SDK, agents, benchmarks
- **7 LLM providers** -- Google Gemini, OpenAI Codex, xAI Grok, DeepSeek, MiniMax, Kimi
- **OxiZ** -- pure-Rust SMT solver for formal verification
- **Apache Arrow / PyArrow** -- zero-copy memory compaction
- **wasmtime v36 LTS** -- Wasm WASI sandbox (Component Model, deny-by-default)
- **ONNX Runtime** -- native embeddings (snowflake-arctic-embed-m, 768-dim)
- **FastAPI + WebSocket** -- real-time dashboard

## License

MIT License. See [GitHub repository](https://github.com/yannabadie/YGN-SAGE) for details.
