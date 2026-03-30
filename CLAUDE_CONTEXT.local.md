# feat/runtime-pipeline — 5-Stage Pipeline + Agent Runtime

## Scope
CognitiveOrchestrationPipeline, TopologyRunner, TopologyController, boot.py, agent_loop, phases.

## Key Files
- `sage-python/src/sage/pipeline.py` — 5-stage pipeline (primary execution path)
- `sage-python/src/sage/boot.py` — system bootstrap, 3 paths (pipeline/legacy/mock), capability logging
- `sage-python/src/sage/topology/runner.py` — node execution, code node dispatch (HyEvo), 60s timeout, DeepSeek fallback
- `sage-python/src/sage/topology_controller.py` — 5 adaptation actions (continue/upgrade/prune/reroute/spawn)
- `sage-python/src/sage/topology/llm_caller.py` — Path 6 policy loader (V1 Phi-4-mini, V2 Nemotron-8B)
- `sage-python/src/sage/agent_loop.py` — PERCEIVE/THINK/ACT/LEARN, causal memory wiring, consolidation trigger
- `sage-python/src/sage/phases/` — perceive.py, think.py, act.py, learn.py
- `sage-python/src/sage/quality_estimator.py` — Z3/ONNX quality estimation

## Recent Changes (March 2026)
- Per-node timeout 60s (was no timeout)
- DeepSeek fallback on provider failure
- kNN Rust router (92%) now active in Stage 0
- Predecessor context truncated to 1000 chars per node
- _last_execution_path tracking (pipeline/legacy/mock)
- MASBENCH: SAGE 67% vs bare 40% (+27pp)

## Commands
```bash
cd sage-python && python -m pytest tests/test_boot_topology.py tests/test_execution_path.py tests/test_e2e_integration.py -v
```

## Out of Scope
Rust internals, training, UI.
