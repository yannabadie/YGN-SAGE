# sage-python

Python SDK for YGN-SAGE (Self-Adaptive Generation Engine) — an Agent Development Kit built on five cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Installation

```bash
cd sage-python
pip install -e ".[all,dev]"    # All providers + dev tools
pip install -e ".[google]"     # Google Gemini only
pip install -e ".[z3]"         # Z3 formal verification
```

Requires Python 3.12+. **Requires `sage_core` (Rust extension)** — see root README for build instructions.

## Quick Start

```python
from sage.boot import boot_agent_system

system = boot_agent_system()           # Auto-detects providers from .env
result = await system.run("Solve X")   # S1/S2/S3 routing + full agent loop
```

Or via the CLI:

```bash
sage serve          # Start A2A + MCP server
sage bench          # Run benchmarks
sage chat           # Interactive chat (bash off by default; SAGE_CHAT_ALLOW_BASH=1 to enable)
```

## Testing

```bash
python -m pytest tests/ -v             # 2940 collected (source of truth: docs/status/current.json)
ruff check src/                        # Lint (clean)
mypy src/sage/                         # Type check (0 errors)
```

## Benchmarks

```bash
# Primary benchmarks (use these to prove framework value)
python -m sage.bench --type routing_gt                            # kNN routing accuracy (60-task GT set, instant)
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 20  # BCB Hard
python -m sage.bench --type ablation --limit 10 --tier budget    # Ablation study (6 configs)

# SWE-bench (all pillars, observe mode is default for smoke runs)
SAGE_DIFF_VERIFIER_MODE=observe \
  python -m sage.bench --type swebench --dataset lite --limit 10 \
    --output docs/benchmarks/$(date +%F)-observe.json
```

## Package Structure

| Subpackage | Description |
|------------|-------------|
| `sage/` | Core runtime: boot sequence, agent loop, resilience |
| `sage/agents/` | Composition patterns: sequential, parallel, loop, handoff |
| `sage/bench/` | BigCodeBench, SWE-bench, routing_gt, ablation, EvalPlus, GAIA, APPS |
| `sage/contracts/` | Contract IR, DAG verification, Z3 SMT, CEGAR repair |
| `sage/cli.py` | Root CLI dispatcher (`sage serve` / `sage bench` / `sage chat`) |
| `sage/events/` | EventBus: in-proc event system for observability |
| `sage/guardrails/` | 3-layer guardrails: input, runtime, output |
| `sage/input/` | Normalizers: BCB, SWE-bench, chat → TaskInput |
| `sage/llm/` | LLM providers: 7 API providers + Codex CLI |
| `sage/memory/` | 4-tier memory: working (Arrow), episodic (SQLite), semantic (graph), ExoCortex (RAG) |
| `sage/phases/` | PERCEIVE → THINK → ACT → LEARN (OODA loop) |
| `sage/providers/` | connector.py (single source of truth), OpenAI-compat adapter |
| `sage/runtime/` | Typed runtime spine: event_log, state, run_frame, oracle (default-on since cycle-7) |
| `sage/routing/` | ShadowRouter (dual Rust/Python traces) |
| `sage/sandbox/` | Sandbox manager (subprocess isolation) |
| `sage/strategy/` | AdaptiveRouter: kNN (92% GT), ComplexityRouter (heuristic fallback) |
| `sage/tools/` | ToolForge, AgentTool, typed repo tools, memory tools, ExoCortex tools |
| `sage/topology/` | TopologyRunner (code node dispatch), LLM caller (Path 6 V1/V2), controller |
| `sage/verl/` | Training: topology_env, reward, manifest, cascaded_eval (training branch only) |

## Environment Variables

```bash
export DEEPSEEK_API_KEY="..."           # Primary budget provider
export GOOGLE_API_KEY="..."             # Reliable fallback
export OPENAI_API_KEY="..."             # Best quality
export GROK_API_KEY="..."               # xAI Grok
export KIMI_API_KEY="..."               # Moonshot Kimi
export MINIMAX_API_KEY="..."            # MiniMax
export OPEN_ROUTER_API_KEY="..."        # OpenRouter (200+ models)

# Optional feature flags
export SAGE_ENABLE_PATH6=1              # Enable learned topology policy
export SAGE_OTEL_EXPORTER=console      # OTel spans to stdout
export SAGE_DIFF_VERIFIER_MODE=observe # SWE-bench diff verifier (observe mode)
export SAGE_EXOCORTEX_STORE=fileSearchStores/ygnsageresearch-wii7kwkqozrd  # ExoCortex RAG
```

## Key Runtime Flags (bench + sandbox)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAGE_DANGEROUS_TOOLS` | `0` | Register `execute_bash` at boot (off by default since 2026-04-23) |
| `SAGE_UNSAFE_RAW_EXEC` | unset | Allow `ToolExecutor.execute_raw` (bypasses both AST + Wasm sandbox) |
| `SAGE_ORACLE` | on | OracleStack training gate. Kill-switch: `0\|false\|off\|no\|disable\|disabled` |
| `SAGE_BOOT_BYPASS_EPOCH_GUARD` | unset | Forensic inspection bypass for A14 topology state guard |

## A14 Epoch Guard

State files under `~/.sage/` (bandit_state.db, archive_state.db, engine_extras.json, topology_state_manifest.json) are protected by a fail-closed epoch guard. Normal boot requires `posterior_epoch.json` (epoch=1) + `topology_state_manifest.json`. Reset: `python -m sage.ops.a14_reset --reason "..."`.

## Dependencies

Core: `httpx`, `pydantic`, `rich`, `anyio`, `aiosqlite`, `numpy`.

Optional: `google-genai` (Gemini), `openai` (DeepSeek/OpenAI/xAI via compat), `pydantic-ai` (typed LLM client), `pyarrow` (Arrow working memory), `z3-solver` (formal verification), `fastapi`/`uvicorn` (dashboard + A2A server), `onnxruntime` (Rust ONNX embedder DLL), `sage_core` (required at runtime — Rust extension: TopologyEngine, SystemRouter, QualityLabeler, ContextualBandit, S-MMU, sandbox).
