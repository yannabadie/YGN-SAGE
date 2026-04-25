---
name: April 7-8 Sessions — Phase A→B + CORAL + SWE-bench prep
description: 35 commits, BigCodeBench 37.2%→45.9%, MASBENCH stats, EvolutionMemory, provider health check, truststore SSL, SWE-bench diagnostic
type: project
---

## Key Results

### BigCodeBench Hard (4 iterations)
- v1: 37.2% (55/148) — baseline
- v3b: 35.8% (53/148) — bypass too aggressive  
- v4: **45.9% (68/148)** — bypass + repair + escalation
- 3 provider bugs found & fixed: json_schema, FrugalGPT model_id, MiniMax SSL

### MASBENCH Statistical Analysis (N=50, computed on existing data)
- **breadth: +22pp, p=0.015** — ONLY statistically significant axis
- depth/horizon/parallel/robustness: all p>0.05
- ADR-006: BigCodeBench omega=1.3, topology NOT the lever there

### SWE-bench Lite Diagnostic (5 tasks)
- 5/5 patches generated, 0 tools used, 100% S2 routing, 100% bypassed
- **3 critical gaps**: routing S2→S3, no tool use, one-shot (no iteration)
- Agent has execute_bash + ToolForge but prompt doesn't tell it

## Architecture Changes (35 commits)

### Provider Robustness
- **truststore** at boot: OS cert store → MiniMax SSL fixed (proxy *.adgroupe.com)  
- **Health check** at boot: probes all providers, circuit breaker for dead ones
- **Rust ModelAssigner.exclude_providers()**: dead providers excluded at source
- **ProviderPool.infer_provider()**: centralized model→provider mapping (no duplicated string matching)
- **json_schema**: only sent to OpenAI (DeepSeek rejects it)
- **FrugalGPT cascade**: validates provider before model upgrade

### Pipeline Intelligence
- **Adaptive bypass**: S2+sequential → single-agent (AdaptOrch-backed)
- **Topology escalation**: bypass → repair → topology fallback (Conductor-inspired)
- **Pipeline tracing**: last_context exposed, per-task trace in JSONL

### Evolution (CORAL Phase 1)
- **EvolutionMemory**: SQLite WAL, mutations + skills, lazy init
- **Engine wired**: records mutations after population.add()
- **LLMMutator wired**: injects skills into mutation prompt

### Agent Capabilities
- **ToolForge wired** into agent loop (was commented out)
- **execute_bash** tool registered at boot
- **13 tools** total: bash, 2 meta-tools, 8 memory, 2 knowledge

### Documentation
- **Obsidian vault** tracked (48 files, 20+ corrections)
- **CORAL paper fiche** added
- **ADR-006** (BigCodeBench limits)
- **Monitoring protocol** in CLAUDE.md rules

## Next Session: SWE-bench

3 problems to solve:
1. **Routing**: SWE-bench tasks must route to S3 (complex), not S2
2. **Tool use**: Agent must know about execute_bash and iterate
3. **Agent loop**: Need multi-turn tool-call loop, not one-shot generation

The agent has the tools (execute_bash, create_python_tool, ToolForge).
The pipeline has the capabilities (S3, topology, memory).
What's missing: the PROMPT and the AGENT LOOP that connects them.
