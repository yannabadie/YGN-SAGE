# A2A Protocol Research — Integration Guide for YGN-SAGE

**Date:** 2026-03-10
**Status:** Research complete, integration planned

## Protocol Overview
- **Name:** Agent2Agent (A2A) Protocol
- **Version:** v0.3.0 (Jul 30, 2025), v1.0 draft in progress
- **Governance:** Linux Foundation (donated by Google, Feb 2026)
- **Stars:** 22.4k GitHub, 143 contributors
- **License:** Apache 2.0
- **Repo:** https://github.com/a2aproject/A2A

## Agent Card Schema (key fields)
```json
{
  "id": "string",
  "name": "string (REQUIRED)",
  "description": "string (REQUIRED)",
  "version": "string (REQUIRED)",
  "url": "string (REQUIRED)",
  "provider": { "name": "string (REQUIRED)", "url": "string", "support_contact": "string" },
  "capabilities": {
    "a2aVersion": "0.3",
    "streaming": true,
    "pushNotifications": true,
    "mcpVersion": "string (optional)",
    "teeDetails": { "type": "Intel SGX", "attestationEndpoint": "...", "publicKey": "..." }
  },
  "skills": [{
    "id": "string",
    "name": "string",
    "description": "string",
    "tags": ["routing", "coding"],
    "examples": ["Write a sort function"],
    "inputModes": ["text/plain"],
    "outputModes": ["text/plain", "application/json"],
    "input_schema": {},
    "output_schema": {}
  }],
  "securitySchemes": ["APIKey", "HTTPBearer", "OAuth2", "OpenIdConnect", "MutualTLS"],
  "interfaces": ["JSON-RPC", "gRPC", "HTTP+JSON/REST"]
}
```

## A2A vs MCP (coexist, not compete)
- **MCP** (Anthropic): Agent-to-tool connection (client-server, stateless)
- **A2A** (Google): Agent-to-agent coordination (peer-to-peer, stateful tasks)
- **Three-Layer Stack:** WebMCP → MCP (tools) → A2A (agents)
- Most production systems use BOTH

## Task Lifecycle
```
SUBMITTED → WORKING → COMPLETED | FAILED | CANCELED
                    → INPUT_REQUIRED → WORKING (resume)
                    → AUTH_REQUIRED → WORKING (resume)
         → REJECTED
```

## Rust Crates Available
| Crate | Version | Description | Relevance |
|-------|---------|-------------|-----------|
| `a2a-types` | 0.1.3 | Serde types only, MIT, 76.7% docs | Best for sage-core integration |
| `a2a-rs` | 0.1.0 | Hexagonal arch + a2a-mcp bridge | Reference implementation |
| `acai` | 0.2.0 | Server framework (Hyper+Tokio) | Production server |
| `a2a` | 0.1.0 | Minimal types + InMemoryTaskStore | Early stage |

## Python SDK
- `pip install a2a-sdk[all]` (v0.3.25, March 2026)
- Extras: http-server, grpc, telemetry, encryption, signing, sqlite

## Mapping to YGN-SAGE

| A2A Concept | YGN-SAGE Equivalent | Integration |
|-------------|---------------------|-------------|
| Agent Card | ModelCard (Rust) | Extend ModelCard to publish as `/.well-known/agent-card.json` |
| Skills + tags | S1/S2/S3 systems | Map cognitive systems to A2A skill tags |
| Capability discovery | `registry.refresh()` | Add A2A card fetching to boot discovery |
| Task lifecycle | Agent loop phases | Map TaskState to perceive/think/act/learn |
| Push notifications | EventBus | Bridge A2A push → EventBus subscribers |
| SSE streaming | WebSocket dashboard | Bridge A2A SSE → EventBus `stream()` |
| Security schemes | SAGE_DASHBOARD_TOKEN | Extend to OAuth2/mTLS for production |

## Adoption (March 2026)
Google, Microsoft, AWS, Salesforce, SAP, IBM, Cisco, Adobe, LangChain, MongoDB, PayPal, Confluent (Kafka), Huawei (A2A-T telecom)

## Key Insight
ModelCard in sage-core SHOULD be A2A Agent Card compatible. This means:
1. Export ModelCards as A2A Agent Cards for external discovery
2. Import remote A2A Agent Cards into ModelRegistry
3. Use A2A skill tags as routing input in SystemRouter
