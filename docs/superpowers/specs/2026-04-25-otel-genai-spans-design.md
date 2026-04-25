# roadmap-B1 — OpenTelemetry GenAI spans (design spec)

**Status:** brainstormed 2026-04-25, awaiting user review.
**Author:** Claude (multi-source consultation: Codex gpt-5.5 xhigh + Context7 OTel + advisor).
**Scope:** sage-python orchestration path. sage-core bridge, sage-discover, ui/FastAPI deferred to B1.b/c/d.

---

## 1. Goal

Adopt OpenTelemetry GenAI semantic conventions on the sage-python orchestration path so multi-agent cascades become diagnosable in one trace view instead of a 2-hour log-mining session.

**Direct motivation:** the 2026-04-24 roadmap-A2 investigation cost ~2h to diagnose a Stage 4 multi-agent → Kimi HTTP 400 → fallback empty cascade. With proper OTel spans, the same diagnosis would take ~5 minutes from a Jaeger / logfire view.

**Spec source of truth:** OpenTelemetry GenAI Semantic Conventions, https://opentelemetry.io/docs/specs/semconv/gen-ai/ (stability: "Development", attribute set frozen enough to ship against).

---

## 2. Constraints

- **Default off** — no observability overhead for users who don't opt in. Env-gated activation via `SAGE_OTEL_EXPORTER`.
- **Redaction by default** — sensitive payloads pass through the existing A16 `RedactionFilter` before emission.
- **No breaking change to AgentEvent / EventBus** — dashboard (ui/app.py), bench analytics (sage.bench), tests stay byte-identical.
- **Python orchestration only** for B1; Rust core (sage-core) gets a separate ticket B1.b.
- **No new top-level deps** — OTel SDK + HTTP exporter are already on disk via the logfire 4.32.1 transitive dependency.

---

## 3. Architecture

### 3.1 Span hierarchy

```
sage.pipeline.run                       [span name; gen_ai.operation.name="invoke_agent"]
├─ sage.classify                        [gen_ai.operation.name="sage.classify"]
├─ sage.decompose                       [gen_ai.operation.name="sage.decompose"]
├─ sage.topology_select                 [gen_ai.operation.name="sage.topology_select"]
├─ sage.assign_models                   [gen_ai.operation.name="sage.assign_models"]
├─ sage.execute                         [gen_ai.operation.name="sage.execute"]
│  ├─ sage.node.<node_name>             [gen_ai.operation.name="invoke_agent"; per TopologyNode; <node_name> falls back to f"node_{idx}" if the node has no name]
│  │  ├─ sage.chat                      [gen_ai.operation.name="chat"; per LLM call]
│  │  └─ sage.tool                      [gen_ai.operation.name="execute_tool"; per tool call]
│  ├─ sage.node.<another>               [invoke_agent]
│  │  └─ sage.chat (kimi-k2.5)          [error.type set when HTTP 400]
│  └─ sage.node.fallback                [invoke_agent — Stage 4 single-agent fallback]
│     └─ sage.chat (gemini-flash-lite)
└─ sage.learn                           [gen_ai.operation.name="sage.learn"]
```

**Custom op names** for the 6 SAGE pipeline stages use the `sage.*` namespace (allowed by spec for non-well-known operations).

### 3.2 AgentEvent ↔ OTel coexistence (phased)

`AgentEvent` and `EventBus` remain unchanged in this spec. Existing consumers (dashboard WebSocket, bench analytics, tests) are untouched.

OTel spans are a **complementary independent layer** emitted at the same call sites via a shared context manager:

```python
# sage/observability/spans.py
from contextlib import contextmanager
from typing import Any
from sage.observability import _init_tracer, _get_tracer

def _otel_enabled() -> bool:
    """True iff a TracerProvider is configured (any non-`none` exporter)."""
    _init_tracer()  # idempotent
    return _get_tracer() is not None


@contextmanager
def sage_span(name: str, op: str, **attrs: Any):
    """Emit an OTel span if a tracer is configured; no-op otherwise.

    Independent of AgentEvent emission — both can fire at the same
    call site without coupling.
    """
    if not _otel_enabled():
        yield None
        return
    tracer = _get_tracer()
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("gen_ai.operation.name", op)
        for k, v in attrs.items():
            span.set_attribute(k, v)
        yield span
```

A future session may consolidate AgentEvent → OTel span events; that migration is **out of scope** for B1 and not blocked by it.

### 3.3 Periphery (incremental rollout)

| Sub-item | Scope | Cost | Prerequisite |
|---|---|---|---|
| **B1** (this spec) | sage-python orchestration | shipped via writing-plans | — |
| **B1.b** | sage-core via `tracing-opentelemetry` bridge. Rust spans carry counts/metrics only — no payload, no A16-in-Rust needed. | 1–2 days | B1 |
| **B1.c** | sage-discover MCP server + Qdrant retrieval. `gen_ai.operation.name=retrieval` + MCP attrs. | 0.5–1 day | B1 |
| **B1.d** | ui/app.py FastAPI auto-instrumentation via `opentelemetry-instrumentation-fastapi`. HTTP `/api/task` linked to pipeline.run via context propagation. | 0.5 day | B1 |

---

## 4. Sensitive payload handling

### 4.1 Attribute classification

| Class | Examples | Default policy |
|---|---|---|
| **Safe** (low cardinality, no payload) | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `error.type`, response status_code | Emitted verbatim. Always. |
| **Sensitive payload** | `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.input.messages`, `gen_ai.output.messages` | Pass through `RedactionFilter` (A16). Truncate to 4 KiB UTF-8. |
| **Internal high-cardinality** | full prompts, full memory snapshots | **Not on spans at all.** Reference via AgentEvent if needed (B2 future). |

### 4.2 Redaction integration

```python
# sage/observability/spans.py
import json, os
from typing import Any
from sage.security.redaction import RedactionFilter

_REDACTOR = RedactionFilter()  # honors SAGE_REDACT_SECRETS env

def _safe_str(value: Any, max_bytes: int = 4096) -> str:
    raw_payloads = os.environ.get("SAGE_OTEL_RAW_PAYLOADS", "0").strip().lower() in {
        "1", "true", "yes",
    }
    if raw_payloads:
        s = value if isinstance(value, str) else str(value)
    else:
        if isinstance(value, dict):
            redacted = _REDACTOR.redact_dict(value) if _REDACTOR.enabled else value
            s = json.dumps(redacted, ensure_ascii=False)
        elif isinstance(value, str):
            s = _REDACTOR.redact_text(value) if _REDACTOR.enabled else value
        else:
            s = str(value)
    encoded = s.encode("utf-8")
    if len(encoded) > max_bytes:
        truncated = encoded[: max_bytes - 16].decode("utf-8", errors="ignore")
        return truncated.rsplit(" ", 1)[0] + "…[truncated]"
    return s
```

### 4.3 Edge cases

| Scenario | Behavior |
|---|---|
| `SAGE_REDACT_SECRETS=0` AND `SAGE_OTEL_RAW_PAYLOADS=0` (default) | Truncated but **not redacted**. Log a **once-per-process** WARN at first `sage_span` call (pattern: module-level `_WARNED_SECRETS_DISABLED = False`, set on first hit; reset by tests via `_reset_warn_flag_for_tests()`): "OTel spans active but secret redaction disabled — payloads on spans may contain secrets." |
| `SAGE_OTEL_RAW_PAYLOADS=1` | Skip both redaction and truncation. Documented as dev-only. |
| Multi-turn input messages list | Recursive redaction via `redact_dict` (A16 already traverses dicts). |
| Tool result of 50 KB read_file | Span attribute capped at 4 KiB; full content stays in `ToolResult.output` for in-process consumers. |
| `gen_ai.response.finish_reasons=["content_filter"]` | Set `error.type="content_filter"` on the span. No payload needed for this signal. |

---

## 5. Provider naming

| SAGE provider id | `gen_ai.provider.name` | Rationale |
|---|---|---|
| `google` | `gcp.gemini` | Well-known. |
| `openai` | `openai` | Well-known. |
| `deepseek` | `deepseek` | Well-known. |
| `xai` | `x_ai` | Well-known. **Underscore, not `xai`.** |
| `kimi` | `moonshot.ai` | Custom. Dotted-namespace pattern (matches `gcp.gemini`, `aws.bedrock`). Forward-compatible if OTel adds it to the enum. |
| `minimax` | `minimax.ai` | Custom. Same pattern. |
| `openrouter` | `openrouter.ai` | Custom. OpenRouter is an aggregator; `gen_ai.request.model` carries the underlying routed model id. |

Centralized in `_OTEL_PROVIDER_NAME_MAP` in `sage/observability/spans.py`. Adding a provider = one line.

---

## 6. Exporter strategy

Env var: **`SAGE_OTEL_EXPORTER`** (default `none`).

| Value | Exporter | Use case |
|---|---|---|
| `none` (default) | NoOp. `_otel_enabled()` returns False. Zero runtime overhead. | Default for PyPI users not opting in. |
| `console` | `ConsoleSpanExporter` → stdout, human-readable | Dev local, bench debug, CI logs. |
| `otlp_http` | `OTLPSpanExporter` (HTTP) reading `OTEL_EXPORTER_OTLP_ENDPOINT` | Prod — user wires a collector (Jaeger, Tempo, Honeycomb). |
| `logfire` | `logfire.configure(service_name="ygn-sage")`. logfire bridges OTel automatically. | User wants a managed dashboard without infra setup. |

No gRPC exporter (HTTP-only). The `opentelemetry-exporter-otlp-proto-http` package is already in deps via logfire.

### 6.1 Lazy boot

```python
# sage/observability/__init__.py
import importlib.metadata, logging, os
log = logging.getLogger(__name__)

_INITIALIZED = False
_TRACER = None

def _init_tracer() -> None:
    global _INITIALIZED, _TRACER
    if _INITIALIZED:
        return
    _INITIALIZED = True

    exporter_kind = os.environ.get("SAGE_OTEL_EXPORTER", "none").strip().lower()
    if exporter_kind == "none":
        return  # _TRACER stays None; sage_span yields None

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource

    try:
        version = importlib.metadata.version("ygn-sage")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0+dev"

    resource = Resource.create({"service.name": "ygn-sage", "service.version": version})
    provider = TracerProvider(resource=resource)

    if exporter_kind == "console":
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter, SimpleSpanProcessor,
        )
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter_kind == "otlp_http":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif exporter_kind == "logfire":
        import logfire
        logfire.configure(service_name="ygn-sage")
        # logfire installs its own TracerProvider and bridges to OTel.
        # We must still resolve a tracer, but we do NOT call
        # trace.set_tracer_provider — that would clobber logfire's setup.
        _TRACER = trace.get_tracer("sage", version)
        return
    else:
        log.warning("Unknown SAGE_OTEL_EXPORTER=%r; no exporter active", exporter_kind)
        return

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("sage", version)


def _get_tracer():
    """Return the configured tracer, or None if no exporter is active."""
    return _TRACER
```

### 6.2 Sampling

Default: `ParentBasedAlwaysOn` (OTel SDK default). No custom sampler in B1.

Justification: bench volumes (N=10/N=50) are trivial. In production, the user's collector / SaaS controls sampling, not us. A user who needs a sampler can override `TRACER_PROVIDER` after our init.

### 6.3 Async context propagation

OTel SDK propagates context via `contextvars` automatically across `await`, `asyncio.create_task()`, `asyncio.gather()`. No special handling needed.

Edge case verified: A15 single-flight consolidation (background task spawned via `asyncio.create_task`) — context is captured at task creation, so the consolidation span attaches to its trigger's trace.

---

## 7. Testing strategy

### 7.1 Unit tests — `tests/observability/test_sage_span.py`

- `sage_span()` is a no-op when `SAGE_OTEL_EXPORTER=none` (yields None, no provider initialized).
- `sage_span()` emits a span when `SAGE_OTEL_EXPORTER=console` (capture via `InMemorySpanExporter`).
- `_safe_str()` redacts OpenAI/AWS/GCP/Bearer/JWT secrets (one assertion per A16 class).
- `_safe_str()` truncates payloads >4 KiB and ends with `…[truncated]`.
- `SAGE_OTEL_RAW_PAYLOADS=1` skips both redaction and truncation.
- Provider name mapping returns the documented value for each of the 7 SAGE providers.

### 7.2 Integration tests — `tests/observability/test_pipeline_spans.py`

`InMemorySpanExporter` captures spans during a mocked `pipeline.run("hello")`:

- 6 stage spans emitted in order (`sage.classify` → `sage.learn`).
- `sage.execute` contains nested `invoke_agent` spans, one per TopologyNode.
- `chat` spans carry `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`.
- `execute_tool` spans carry redacted `gen_ai.tool.call.arguments` + `gen_ai.tool.call.result`.

### 7.3 Golden-trace replay — `tests/observability/test_a2_cascade_replay.py`

Replay the 2026-04-23 cascade A2 log against a mock pipeline:

- Top-level `sage.pipeline.run` duration > 40s.
- `sage.execute` contains an `invoke_agent` Stage 4 multi-agent span with `error.type` set.
- The Kimi `chat` span carries `error.type` and `gen_ai.response.finish_reasons` reflecting the HTTP 400.
- The fallback `chat` span (gemini-flash-lite) shows empty content.

### 7.4 AgentEvent coexistence regression

- `test_otel_does_not_replace_agent_event`: with `SAGE_OTEL_EXPORTER=console`, all `AgentEvent` types (PERCEIVE/THINK/ACT/LEARN/EXECUTE_BUDGET_EXCEEDED/PROMPT_INJECTION_DETECTED) are still emitted on `EventBus`.
- `test_disabling_otel_keeps_agent_event`: with `SAGE_OTEL_EXPORTER=none`, all `AgentEvent` types are still emitted.

### 7.5 Test infrastructure

All tests use `InMemorySpanExporter` from `opentelemetry.sdk.trace.export.in_memory_span_exporter`. **Zero network calls, zero collector dependency.**

---

## 8. Documentation updates

| File | Update |
|---|---|
| `CLAUDE.md` | Quick Commands: example with `SAGE_OTEL_EXPORTER=console python -m sage.bench --type swebench --dataset lite --limit 10`. |
| `.claude/rules/development.md` | Env table: add `SAGE_OTEL_EXPORTER` (none/console/otlp_http/logfire) + `SAGE_OTEL_RAW_PAYLOADS` (default 0; debug only). |
| `docs/observability/otel-genai-spans.md` (new) | Authoritative usage doc: provider mapping table, span hierarchy diagram, redaction policy, integration examples (Jaeger, logfire, Honeycomb). |
| `README.md` | Brief "Observability" mention: "OTel GenAI conventions supported, set `SAGE_OTEL_EXPORTER=console` for local debug." |
| `roadmap.md` | B1 → Closed; add B1.b/c/d/e as open sub-items with cost estimates. |

---

## 9. Cost / non-regression analysis

| Metric | Pre-B1 | Post-B1 (`none`) | Post-B1 (`console`) |
|---|---|---|---|
| Latency `pipeline.run` | baseline | **0 overhead** (sage_span yields None) | +1–3 ms per span (formatting + stdout write) |
| Memory | baseline | 0 | ~50 spans × ~1 KB each = ~50 KB transient per run |
| Wheel size | unchanged | unchanged (deps already pulled by logfire/pydantic_ai) | unchanged |
| Existing tests | baseline | must remain green | must remain green |

---

## 10. Out of scope

| Topic | Reason |
|---|---|
| OTel metrics (`opentelemetry-metrics`) | Separate concern. Ticketed B1.f if needed. |
| OTel logs (`opentelemetry-logs`) | sage uses stdlib `logging`. Bridge can be added later if a user needs it. |
| Distributed tracing across processes (A2A serve, MCP gateway) | HTTP `traceparent` propagation. Ticketed B1.g. |
| Span persistence / replay | That's roadmap-B2 (durable trace+replay). B1 produces the schema; B2 persists it. |
| AgentEvent → OTel migration | Phased Q5=D explicitly defers this. |

---

## 11. Acceptance criteria

The implementation plan (next step: `superpowers:writing-plans`) must produce code that satisfies:

1. `pytest tests/observability/ -v` — all tests pass.
2. `SAGE_OTEL_EXPORTER=console python -m sage.bench --type swebench --dataset lite --limit 1` — outputs a hierarchical trace to stdout matching §3.1's structure.
3. `pytest sage-python/tests/ -x` — no regression on the existing ~2428 tests.
4. `grep -r "import opentelemetry" sage-python/src/sage/` only finds matches in `sage/observability/`. (Instrumentation isolated.)
5. Replay of the 2026-04-23 A2 cascade logs through the OTel pipeline produces a trace where the cascade is visible in one view.
