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

## Sub-items (future)

- `B1.b` — sage-core Rust spans via `tracing-opentelemetry` bridge
- `B1.c` — sage-discover MCP server retrieval spans
- `B1.d` — ui/FastAPI auto-instrumentation
