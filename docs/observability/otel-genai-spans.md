# OpenTelemetry GenAI spans

YGN-SAGE emits OpenTelemetry GenAI semantic-convention spans on the
sage-python orchestration path. Default off — opt in via env.

## Quickstart

```bash
# Console exporter (dev)
SAGE_OTEL_EXPORTER=console python -m sage.bench --type swebench --dataset lite --limit 1

# OTLP HTTP collector (prod)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
SAGE_OTEL_EXPORTER=otlp_http python -m sage.bench --type bigcodebench --limit 5

# Pydantic logfire (managed dashboard)
export LOGFIRE_TOKEN=<your token>
SAGE_OTEL_EXPORTER=logfire python -m sage.bench --type swebench --limit 1
```

## Span hierarchy

```
sage.pipeline.run                       [invoke_agent]
├─ sage.classify                        [sage.classify]
├─ sage.decompose                       [sage.decompose]
├─ sage.topology_select                 [sage.topology_select]
├─ sage.assign_models                   [sage.assign_models]
├─ sage.execute                         [sage.execute]
│  └─ sage.node.<name>                  [invoke_agent — per TopologyNode]
│     ├─ sage.chat                      [chat — per LLM call]
│     └─ sage.tool                      [execute_tool — per tool call]
└─ sage.learn                           [sage.learn]
```

## Provider names (`gen_ai.provider.name`)

| SAGE provider | OTel value |
|---|---|
| google | `gcp.gemini` |
| openai | `openai` |
| deepseek | `deepseek` |
| xai | `x_ai` |
| kimi | `moonshot.ai` (custom) |
| minimax | `minimax.ai` (custom) |
| openrouter | `openrouter.ai` (custom) |

## Sensitive payloads

`gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`,
`gen_ai.input.messages`, `gen_ai.output.messages` are passed through
the A16 `RedactionFilter` and truncated to 4 KiB UTF-8.

To disable redaction (dev only): `SAGE_OTEL_RAW_PAYLOADS=1`. Logged
warning at startup; never set this in shared environments.

Exception stack traces on `sage.chat` and `sage.tool` spans go through
A16 redaction explicitly via `record_exception=False` + manual event
emission (closes the auto-record info-leak that affected raw OTel
auto-record).

## Rust spans (B1.b — landed 2026-04-25)

`sage-core` Rust hot paths (engine, system_router, knn, model_assigner,
write_gate, mutations, wasm_python, tool_executor) emit `tracing` spans
that nest under their Python parent in the OTel trace tree.

### Build recipe

By default, the Rust OTel bridge is **opt-in**. Build with:

```bash
cd sage-core && maturin develop --features otel,smt,onnx
```

Without `--features otel`, `sage_core.init_otel` returns False and Rust
spans are not exported. Python spans continue to flow.

### Exporter compatibility

| `SAGE_OTEL_EXPORTER` | Python | Rust (with `--features otel`) |
|---|---|---|
| `none` | no spans | no spans |
| `console` | spans to stdout | spans to stderr (opentelemetry-stdout SimpleSpanProcessor) |
| `otlp_http` | OTLP HTTP → collector | OTLP HTTP → same collector (correlated by trace_id) |
| `logfire` | spans to logfire | **not exported** (B1.b.7 — see roadmap) |

### How parent linkage works

At each `sage_span` enter on the Python side, the current span's
`SpanContext` is serialized as a W3C traceparent string and passed to
Rust via `sage_core.bridge_python_span(traceparent, name)`. Rust
attaches the traceparent as the active OTel context, then creates a
`tracing` span whose existing `info_span!` / `#[instrument]` children
inherit the parent automatically.

### Span audit

All 27 existing Rust span attributes are counts/IDs/numeric values —
zero raw user content. See spec §4.1 for the full table:
`docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md`.

## Sub-items (future)

- `B1.c` — sage-discover MCP server retrieval spans
- `B1.d` — ui/FastAPI auto-instrumentation
- `B1.e` — sampler tuning (gated on production volume data)
- `B1.b.1` — rename Rust span names to `sage.<crate>.<op>` form (cosmetic)
- `B1.b.7` — logfire-mode Rust export (auth header contract)
- `B1.b.9` — OTLP batch exporter with explicit tokio runtime ownership
