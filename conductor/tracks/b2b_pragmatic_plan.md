# Pragmatic B2B Evolution Plan

Based on the brutally honest metacognitive audit of March 5, 2026, we are stripping away all ASI LARPing and focusing on delivering real, mathematical value for ERP/MES systems through the MCP Gateway.

## Phase 1: Eradicating Mocks and Securing the Core
- [x] Replace `notebooklm-py` with official `google-genai` File API for canonical memory.
- [x] Replace random Qdrant embeddings with real `google-genai` text embeddings.
- [ ] **Task 1: Real Z3 MES Scheduling**
  - Rewrite `optimize_mes_schedule` in `mcp_gateway.py` to actually use the Python `z3-solver` to solve a Constraint Satisfaction Problem (CSP) for factory scheduling, instead of returning a hardcoded eBPF payload.
- [ ] **Task 2: Real SQL Constraint Verification**
  - Enhance `z3_verify_sql_update` to parse basic SQL conditions and use `z3-solver` to mathematically prove business constraints (e.g., ensuring `price > 0` or `quantity <= stock`).
- [ ] **Task 3: Wasm-based Tool Execution**
  - Shift tool execution from local subprocesses to the `WasmSandbox` exposed by `sage-core`, ensuring true isolation for generated code.

## Phase 2: ERP Integration & Delivery
- [ ] **Task 4: Sage X3 / Cegid Connector Mock**
  - Build a secure SQLite-based mock ERP database to demonstrate the MCP Gateway's ability to safely query and mutate data.
- [ ] **Task 5: End-to-End Demo Documentation**
  - Create a comprehensive README and a Jupyter notebook or interactive script demonstrating a Claude/Cursor agent interacting with the ERP via our verified MCP Gateway.