# Multi-Provider Dynamic Routing & Agentic Topology — Design Document

> **Approved**: 2026-03-08 by Yann Abadie
> **Option chosen**: B — Score-based routing with `ModelRegistry.select()`

## Goal

Wire all 6+ LLM providers (Google, OpenAI, xAI, DeepSeek, MiniMax, Kimi, Codex CLI) into the main control plane with auto-discovery at boot, cascade fallback, provider quirk handling, and dynamic topology selection per task.

## Problem Statement

YGN-SAGE has `.env` keys for 6 providers but only uses 2 (Codex CLI + Google Gemini). The infrastructure exists (`ModelRegistry`, `CognitiveOrchestrator`, `ProviderConnector`, `OpenAICompatProvider`) but is **not wired** into the main control path (`AgentSystem.run()`). Three modules have hardcoded `GoogleProvider` usage.

## Architecture

### Two Coexisting Systems (Current)

| System | Status | Role |
|--------|--------|------|
| **Legacy `ModelRouter`** | ACTIVE | Tier→model static mapping (Codex + Google only) |
| **Modern `ModelRegistry` + `CognitiveOrchestrator`** | BUILT, UNUSED | Score-based selection across all providers |

### Target Architecture

The `CognitiveOrchestrator` becomes the **primary routing engine**. Legacy `ModelRouter` is kept as fallback only. `ProviderConnector.discover_all()` runs at boot to populate `ModelRegistry` with live models.

```
AgentSystem.run(task)
  ├─ ComplexityRouter → S1/S2/S3
  ├─ ModelRegistry.select(needs) → best available model
  │   ├─ Live discovery (ProviderConnector)
  │   ├─ TOML knowledge base (model_profiles.toml)
  │   └─ Feedback loop (DynamicRouter history)
  ├─ Cascade fallback: try → escalate on failure
  └─ DyTopo: topology selection per task structure
```

## 4 Axes of Implementation

### Axe 1 — Wire Multi-Provider as Primary Path

**Files**: `boot.py`, `orchestrator.py`, `registry.py`, `connector.py`

1. **Boot refresh**: `registry.refresh()` at startup → discover all available models
2. **Wire orchestrator**: `AgentSystem.run()` uses `CognitiveOrchestrator` as primary, `ModelRouter` as fallback
3. **Fix vendor lock-ins** (3 hardcoded GoogleProvider usages):
   - `memory/memory_agent.py` — entity extraction LLM
   - `memory/remote_rag.py` — ExoCortex query model
   - `strategy/metacognition.py` — complexity assessment LLM
4. **Cascade fallback** (FrugalGPT pattern): Try cheapest viable model first, escalate on failure up to 3 attempts

### Axe 2 — Provider Quirks in OpenAICompatProvider

**Files**: `providers/openai_compat.py`, `config/model_profiles.toml`

Provider-specific behaviors to handle:

| Provider | Quirk |
|----------|-------|
| DeepSeek | `temperature` ignored on `deepseek-reasoner`; reasoning in `reasoning_content` field |
| Grok | Reasoning tokens in `reasoning_content` (separate from `content`) |
| Kimi | `temperature` must be in [0, 1] range (not 0-2) |
| MiniMax | `<think>` tags in `content` body, not `reasoning_content` |
| OpenAI | `reasoning_effort` param for GPT-5.4 class models |

Implementation: quirk dispatcher keyed on `(provider, model_family)`.

### Axe 3 — Dynamic Topology (DyTopo + AdaptOrch)

**Files**: `routing/dynamic.py`, `topology/evo_topology.py`, new `topology/dytopo.py`

1. **DyTopo semantic matching**: Per-round topology rewiring via query/key descriptors + cosine similarity (all-MiniLM-L6-v2 — same embedder already used by S-MMU)
2. **AdaptOrch topology routing**: O(|V|+|E|) algorithm analyzing TaskDAG structure → selects parallel/sequential/hierarchical/hybrid topology
3. **Self-MoA finding**: Single strong model with multiple samples (+6.6%) beats multi-model ensembles → prefer best model with retries over mixing providers

### Axe 4 — Validation & Benchmarks

**Files**: `bench/`, tests, CI

1. Per-provider smoke test (can generate? can tool-call? latency baseline?)
2. Routing benchmark expansion: 30→50 tasks covering all providers
3. HumanEval per-provider comparison
4. Cost tracking: actual spend vs predicted

## Auto-Refresh at Boot

**User requirement**: "l'idéal serait que a chaque debut de session les modeles soient mis a jour/évalués"

`ModelRegistry.refresh()` at boot:
1. Call `ProviderConnector.discover_all()` → list all live models
2. Merge with TOML knowledge base (scores, pricing)
3. Mark availability (API key present + model responds to ping)
4. Cache results for session duration
5. Log discovered models + availability summary

## Selection Algorithm (Option B)

```python
score = quality^(2 - cost_sensitivity) / cost^cost_sensitivity
```

Where:
- `quality` = weighted blend of `code_score`, `reasoning_score`, `tool_use_score` per task needs
- `cost` = `cost_input` (normalized)
- `cost_sensitivity` ∈ [0, 1]: 0 = pure quality, 1 = cheapest viable

## Key Constraints

1. **No eval() anywhere** — all scoring is formula-based
2. **Best-effort provider discovery** — missing API key = skip (no crash)
3. **Backward compatible** — `use_mock_llm=True` path unchanged
4. **Stateless selection** — each `select()` call is independent (feedback stored externally)
5. **Self-MoA**: Prefer retries on best model over multi-model ensemble

## Research References

| Paper | Use |
|-------|-----|
| FrugalGPT | Cascade fallback: cheap→expensive |
| Self-MoA | Single model + retries > multi-model mix |
| DyTopo (2602.06039) | Per-round semantic matching |
| AdaptOrch (2602.16873) | DAG-based topology selection |
| AMA-Bench (2602.22769) | Multi-agent baseline comparisons |

## Success Criteria

1. All 6 providers discoverable at boot (when API keys present)
2. `ModelRegistry.select()` returns appropriate model for S1/S2/S3 needs
3. Cascade fallback survives provider outage (3 retries, different models)
4. Provider quirks handled transparently (no semantic loss)
5. HumanEval ≥ 75% pass@1 maintained across routing changes
6. Boot time < 5s with all providers (parallel discovery)
