# Cognitive Orchestrator — Design Document

**Date**: 2026-03-06
**Author**: Yann Abadie + Claude Opus 4.6
**Status**: Approved
**Prerequisites**: v2 Convergence + Phase 2 Wiring completed

---

## Goal

Build a Cognitive Orchestrator that dynamically selects the optimal LLM model
for each subtask based on auto-discovered model profiles. The system must:

1. Auto-discover available models from all configured providers at every boot
2. Maintain rich model profiles (capabilities, cost, latency, compatibility)
3. Decompose complex tasks into subtasks with different cognitive requirements
4. Match each subtask to the optimal model based on its profile
5. Build and execute multi-agent topologies with per-agent LLM selection

## Validated Providers (Live-Tested 2026-03-06)

| Provider | API Key Env | Base URL | SDK | Discovery | Status |
|----------|------------|----------|-----|-----------|--------|
| Google | GOOGLE_API_KEY | Gemini API | google-genai | client.models.list() | OK (23 models) |
| OpenAI | OPENAI_API_KEY | api.openai.com/v1 | openai | client.models.list() | OK (42 models) |
| xAI | GROK_API_KEY | api.x.ai/v1 | openai-compat | client.models.list() | OK (11 models) |
| DeepSeek | DEEP_SEEK_API_KEY | api.deepseek.com | openai-compat | client.models.list() | OK (2 models) |
| MiniMax | MINIMAX_API_KEY | api.minimaxi.chat/v1 | openai-compat | Hardcoded list | OK (chat works) |
| Kimi | KIMI_API_KEY | api.moonshot.cn/v1 | openai-compat | client.models.list() | FAIL 401 (key expired) |
| Codex CLI | PATH | Local subprocess | subprocess | shutil.which() | OK |

## Discovered Models (Live 2026-03-06)

### Google Gemini (23 models, key ones)
- gemini-3.1-pro-preview (1M ctx, thinking)
- gemini-3.1-flash-lite-preview (1M ctx, thinking)
- gemini-3-flash-preview (1M ctx, thinking)
- gemini-2.5-flash (1M ctx, thinking)
- gemini-2.5-pro (1M ctx, thinking)

### OpenAI (42 GPT-5+ models, key ones)
- gpt-5.4 (2026-03-05, LATEST)
- gpt-5.4-pro (2026-03-05)
- gpt-5.3-codex
- gpt-5.2, gpt-5.2-pro, gpt-5.2-codex
- gpt-5-mini, gpt-5-nano, gpt-5-pro

### xAI Grok (11 models, key ones)
- grok-4-1-fast-reasoning (2M ctx, $0.20/$0.50)
- grok-4-fast-reasoning
- grok-3, grok-3-mini
- grok-code-fast-1

### DeepSeek (2 models)
- deepseek-chat (V3.2, 128K ctx)
- deepseek-reasoner (V3.2 thinking, 128K ctx)

### MiniMax (hardcoded, chat-verified)
- MiniMax-M2.5 (1M ctx, SWE-bench 80.2%)
- MiniMax-M2.5-highspeed (200K ctx)
- MiniMax-M2.1

### Codex CLI
- gpt-5.3-codex (via subprocess)

## Known API Constraints (verified)

1. Google Gemini: structured_output + tools are MUTUALLY EXCLUSIVE
2. MiniMax: no models.list() endpoint; use hardcoded model list
3. Kimi: API key expired, provider disabled until renewed
4. Codex CLI: 120s timeout, subprocess overhead, not suitable for batch

## Model Profile Schema

```python
@dataclass
class ModelProfile:
    id: str                         # "gemini-3.1-pro-preview"
    provider: str                   # "google"
    family: str                     # "gemini-3.1"
    available: bool                 # Discovered at boot

    # Capability scores (0.0-1.0, from TOML knowledge base)
    code_score: float = 0.5
    reasoning_score: float = 0.5
    tool_use_score: float = 0.5

    # Economics (from TOML, $/1M tokens)
    cost_input: float = 1.0
    cost_output: float = 5.0
    latency_ttft_ms: float = 2000
    tokens_per_second: float = 100

    # Technical (from API discovery)
    context_window: int = 128000
    max_output_tokens: int = 8192
    supports_thinking: bool = False

    # Compatibility (from TOML)
    supports_structured_output: bool = True
    supports_tools: bool = True
    structured_output_tools_compatible: bool = False
    supports_file_search: bool = False

    # Access
    base_url: str | None = None
    api_key_env: str = ""
    sdk: str = "openai"            # "google-genai" | "openai" | "subprocess"
```

## Architecture

### 1. ProviderConnector (per-provider discovery)

```python
class ProviderConnector:
    provider: str
    api_key_env: str
    base_url: str | None
    sdk: str

    async def discover(self) -> list[DiscoveredModel]
    # Google: google-genai client.models.list()
    # OpenAI/xAI/DeepSeek/Kimi: openai client.models.list()
    # MiniMax: hardcoded list (no discovery endpoint)
    # Codex: shutil.which("codex") check
```

### 2. ModelRegistry (auto-refresh at boot)

```python
class ModelRegistry:
    async def refresh(self):
        # 1. Load TOML knowledge base
        # 2. For each provider with valid API key:
        #    a. Call connector.discover()
        #    b. Merge API metadata + TOML scores
        # 3. Unknown models get conservative defaults (0.5)
        # 4. Log warnings for new/removed models

    def select(self, needs: TaskNeeds) -> ModelProfile:
        # Score each available model against task needs
        # Optimize: quality * need_weight - cost * cost_sensitivity

    def list_available(self) -> list[ModelProfile]
```

### 3. CognitiveOrchestrator

```python
class CognitiveOrchestrator:
    async def run(self, task: str) -> str:
        # 1. Decompose task into subtasks (cheap LLM call)
        # 2. For each subtask: assess needs + select model
        # 3. Build topology (Seq/Par based on dependencies)
        # 4. Execute with EventBus observability
```

## Optimal Routing (based on verified data)

| Role | Best Model | Why |
|------|-----------|-----|
| S1 fast/cheap | grok-4-1-fast ($0.20/$0.50) | Frontier quality, cheapest, 2M context |
| S2 code | MiniMax-M2.5 ($0.30/$1.20) | SWE-bench 80.2% at 50x less than Claude |
| S2 code alt | deepseek-chat ($0.28/$0.42) | Even cheaper, good quality |
| S3 reasoning | deepseek-reasoner ($0.28/$0.42) | Best reasoning per dollar |
| S3 max | gpt-5.4-pro | Latest OpenAI, maximum capability |
| Routing assessment | gemini-3.1-flash-lite ($0.025/$1.50) | Cheapest with structured output |
| Large context | grok-4-1-fast (2M) or MiniMax-M2.5 (1M) | Biggest windows |

## Implementation Tasks

| # | Task | Files | Complexity |
|---|------|-------|-----------|
| 12 | ProviderConnector + discovery | sage/providers/ | Medium |
| 13 | ModelRegistry + TOML profiles | sage/providers/registry.py, config/model_profiles.toml | Medium |
| 14 | CognitiveOrchestrator | sage/orchestrator.py | Medium |
| 15 | Wire into boot.py + agent_loop | boot.py | Low |
| 16 | Tests | tests/ | Medium |
