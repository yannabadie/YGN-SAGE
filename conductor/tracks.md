# Tracks Registry

This registry tracks the major development efforts (tracks) of the YGN-SAGE project.

| Track ID | Title | Status | Goal |
| :--- | :--- | :--- | :--- |
| `scaffold-initial` | Project Scaffold & SOTA Solvers | [DONE] | Establish the project structure and core SOTA MARL algorithms. |
| `memory-graphrag` | Memory Pillar: GraphRAG & Compressor | [DONE] | Implement the Neo4j/Qdrant synchronization and MemoryCompressor agent. |
| `tools-sandboxing` | Tools Pillar: Docker Checkpointing | [DONE] | Implement fast Docker snapshots for agent sandboxing. |
| `evolution-engine` | Evolution Pillar: MAP-Elites Mutator | [DONE] | Develop the code mutation and evaluation engine. |
| `topology-dynamic` | Topology Pillar: Parent-Child Multi-Agents | [DONE] | Implement dynamic multi-agent delegation patterns. |
| `flagship-discover` | Sage-Discover: Integrated Workflow | [DONE] | Wire up the flagship research agent to use all SOTA pillars. |
| `hardware-awareness` | ASI Pillar: Hardware Auto-Discovery | [DONE] | Detect host CPU features (SIMD, AVX-512, RAM) and GPUs dynamically. |
| `memory-arrow` | ASI Pillar: Zero-Copy Arrow Memory | [DONE] | Replace UUIDs with ULIDs and migrate `WorkingMemory` to Apache Arrow contiguous buffers. |
| `tools-ebpf` | ASI Pillar: eBPF & Wasm Sandboxing | [DONE] | Replace Docker with sub-millisecond eBPF and Wasm execution environments. |
| `knowledge-grounded-asi` | Cognitive Pillar: NotebookLM Grounding | [ACTIVE] | Anchor the `ResearchAgent` to NotebookLM via `knowledge-bridge` (SDK 2026). |
| `topology-liquid` | ASI Pillar: Liquid Neural Routing | [PLANNED] | Implement a dynamic LNN-based routing layer to replace static topologies. |
