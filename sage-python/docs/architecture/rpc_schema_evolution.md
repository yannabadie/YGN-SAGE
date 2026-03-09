# RPC Schema Evolution Strategy (Backward Compatible)

## Scope
This strategy applies to line-delimited JSON RPC/event payloads (for example Codex CLI `--json` output consumed by `sage.llm.codex`).

## Envelope Contract
Every payload should carry a stable envelope:

- `type`: event/message discriminator (required)
- `schema_version`: integer schema version (required for producers, optional for consumers)
- `item` or payload body: typed content

Consumers must treat missing `schema_version` as `1`.

## Compatibility Rules
- Minor additive changes are allowed within the same major version:
  - new optional fields
  - new object members
  - new enum values (if unknown-safe)
- Breaking structural changes require a new `schema_version`.
- Producers should emit the latest version.
- Consumers must support at least the previous major version for one deprecation window.

## Migration Pattern
1. **Parse envelope first**: read `type` and `schema_version`.
2. **Route to decoder**: dispatch to `decode_v1`, `decode_v2`, etc.
3. **Normalize to internal model**: convert all versions to one internal representation.
4. **Log unknown versions**: fail soft where possible; hard-fail only when critical fields are absent.

## Current Decoder Policy
In `src/sage/llm/codex.py`:

- v1 supported: `item.completed` with `item.text`.
- v2 supported: `item.completed` with `item.content[]` text parts.
- Unknown/invalid lines are skipped to preserve stream robustness.

## Schema Authoring Policy (Structured Output)
When passing JSON Schema to providers:

- Set `additionalProperties: false` on all object nodes recursively.
- Include nested objects in `properties`, `items`, `$defs`/`definitions`, and composition keywords (`allOf`/`anyOf`/`oneOf`/`not`).
- This ensures strict and stable decoding across model/provider upgrades.

## Deprecation Lifecycle
1. Introduce new schema version with dual-read support.
2. Emit telemetry for legacy reads.
3. Announce deprecation date.
4. Remove old decoder only after deprecation window and no observed legacy traffic.
