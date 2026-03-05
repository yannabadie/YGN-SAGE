# YGN-SAGE

**YGN-SAGE** is an Enterprise Agent Development Kit (ADK) that combines a high-performance Rust execution core (`sage-core`) with a Python orchestration layer (`sage-python`).

It is designed to solve a critical problem in the AI agent ecosystem: **Secure, fast, and auditable execution of LLM-generated logic.**

## Core Architecture

Instead of relying on slow Docker containers or unsafe `exec()` calls, YGN-SAGE provides:
1. **Rust/PyO3 Core**: A compiled backend handling zero-copy memory (Apache Arrow).
2. **eBPF & Wasm Sandboxing**: Sub-millisecond isolation for executing dynamically generated code safely.
3. **Formal Verification (Z3)**: Mathematical proofing of agent logic before execution to prevent critical failures.
4. **MCP Gateway**: Exposes the execution engine and RAG capabilities via the Model Context Protocol (MCP) for seamless integration with modern enterprise workflows (ERP, MES, etc.).

## Quickstart

### Prerequisites
- Rust (`rustup default stable`)
- Python 3.12+

### Installation
```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -e sage-python[all]

# 3. Build the Rust core
pip install maturin
cd sage-core
maturin develop
```

### Running the MCP Gateway
The MCP server exposes our secure execution sandboxes to external agents (like Claude or Cursor).
```bash
export GOOGLE_API_KEY="your_api_key"
python sage-discover/mcp_gateway.py
```

## Structure
- `sage-core/`: Rust library exposing PyO3 bindings for memory, Wasm, eBPF, and Z3.
- `sage-python/`: Python SDK for agent orchestration, topology planning, and vector memory.
- `sage-discover/`: Reference applications, benchmarks, and the MCP server.

## Status & Philosophy
This repository was initially an experimental ground for autonomous discovery loops. We have pivoted to focus entirely on **pragmatic B2B integrations**: providing a secure execution runtime and grounded RAG for enterprise solutions. 
