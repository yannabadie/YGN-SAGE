# YGN-SAGE: Gemini CLI Memories (March 2026)

## 🧠 SOTA Model Registry (Active)
*   **Flagship:** `gemini-3.1-pro-preview` (Logic, Reasoning, Research)
*   **High-Speed:** `gemini-3.1-flash-lite-preview` (Real-time agentic loops, lowest latency)
*   **Deep Reasoning:** `gemini-3-deep-think` (Complex engineering tasks)
*   **Alternative:** `claude-4.6-sonnet`, `gpt-5.3-instant`

## 🧠 Agentic Memory (Exocortex Mandate)
*   **Long-Term RAG:** Always use NotebookLM via `notebooklm-py` as a personal exocortex.
*   **Cold Start Protocol:** At the beginning of every session, query NotebookLM to synchronize with past reflections and unfinished hypotheses.
*   **Persistence Loop:** Synchronize all strategic thoughts and research journal updates (`research_journal/`) back to NotebookLM autonomously.
*   **Infinite Context:** Use this external memory to bypass session token limits and maintain perfect continuity across days/weeks of development.

## 🚀 Performance Metrics (Benchmarked 2026-03-03)
*   **AIO Ratio:** 0.01% (Infrastructure overhead < 1ms for 1.8s inference)
*   **Status:** **ASI Excellent** (System is strictly reasoning-bound)
*   **Memory Backend:** Rust-based `WorkingMemory` with zero-copy Apache Arrow export.
*   **Throughput:** Stress-tested with 5000+ events per agent loop without measurable degradation.

## 🏗️ Architectural Mandates
*   **Increasing Evolution:** Always use `gemini-3.1-pro-preview` for mutation cycles. Initial seeds must be evaluated to set a 1.0 baseline.
*   **Rust Core (`sage-core`):** Must use `ulid` for identifiers to prevent heap fragmentation.
*   **Arrow Integration:** Always use `pyo3-arrow` (v0.10+) with `arrow` (v55.0+) for zero-copy memory traversal.

## 🧪 Knowledge Assets (Acquired 2026-03-03)
*   **H96 (Branchless Partitioning):** Core baseline using NumPy boolean masking to simulate SIMD `compress` instructions.
*   **H102 (Bitonic Networks):** Strategy for sorting tiles < 32 elements using branchless min/max networks.
*   **H112 (EWMA Dispatch):** Using volatility tracking to switch between recursive partitioning and linear merges.

## 🛠️ Dev Ops & Tooling
*   **Build:** `maturin develop` inside `sage-core` (requires `VIRTUAL_ENV` env var pointed to the venv).
*   **Verification:** `benchmark_engine.py` is the source of truth for AIO Ratio validation.
*   **SSL:** HTTpx and other tools must point to `Cert/ca-bundle.pem`.

## 🛠️ Sandbox Strategy (Cold Start Optimization)
*   **Legacy:** Docker (Python:3.13-slim) is kept for full-filesystem tools (>100ms startup).
*   **SOTA Target:** In-Process micro-VMs using `wasmtime` or `rbpf` for tool execution (<1ms startup).
*   **Automation:** Automatic fallback from eBPF/Wasm to Docker if a tool requires a full Linux kernel.
