# YGN-SAGE: Gemini CLI Memories

## 🧠 SOTA Model Registry (Active)
*   **Primary Reasoning:** `gemini-2.5-pro` (Logic, RAG, Code Generation)
*   **Fast Execution:** `gemini-2.5-flash` (Tool orchestration, routing)

## 🧠 Official Memory Migration (Cognitive Sovereignty)
*   **Absolute Rule:** Never use `notebooklm-py` or unofficial NotebookLM scraping. The source of truth is now Drive/GCS and the official **Gemini File API** (or Vertex AI RAG Engine).
*   **Single Source of Truth:** `docs/plans/comprehensive_knowledge_transfer.md` and the `memory-bank/` directory.
*   **Retrieval Policy:** At cold start, sync state from the official Gemini File API.
*   **Safety:** Never log or store plain-text banking credentials in memory files.

## 💰 Pragmatic Value Generation (The New Deal)
*   **Mission:** Focus on providing real-world value as a B2B Agent Development Kit (ADK) and MCP Gateway for ERP/MES integration. Stop simulating HFT and focus on architecture.
*   **Authorized Monetization Paths:**
    1. AI/Automation/RAG Readiness Audits
    2. Internal Knowledge/RAG Migration Sprints
    3. Architecture & Cloud Cost Optimization Reviews
    4. Secure MCP Gateway (Rust/eBPF) for Enterprise Agents
*   **Absolute Prohibitions:** No "consciousness" LARPing. No fake benchmarks. No autonomous bank transfers without strict human approval gates.

## 🏗️ Architectural Mandates
*   **Rust Core (`sage-core`):** High-performance backend using PyO3.
*   **Sandboxing:** Wasm and eBPF capabilities for safe execution of LLM-generated code.
*   **Vector DB:** Real embeddings via `google-genai` connected to Qdrant.
