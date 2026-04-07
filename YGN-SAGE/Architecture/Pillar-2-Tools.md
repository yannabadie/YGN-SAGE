---
title: "Pilier 2 — Tools"
type: architecture
pillar: 2
tags:
  - architecture
  - tools
  - sandbox
updated: 2026-04-07
---

# Pilier 2 — Tools

## Sandbox 3 couches (defense-in-depth)

### Couche 1 — tree-sitter (Rust)
- Validation AST statique
- 23 modules bloques, 21 appels bloques, 14 dunders bloques
- Rapide, pas d'execution

### Couche 2 — Wasm WASI (Rust, wasmtime v43)
- Isolation component model deny-by-default
- Cranelift JIT compilation
- Feature gate : `--features sandbox,cranelift`

### Couche 3 — subprocess (Rust/Python)
- Kill-on-drop avec timeout
- bwrap (bubblewrap) sur Linux
- **Sur Windows** : subprocess simple avec blocklist regex

> [!warning] Securite sandbox
> La couche subprocess utilise une blocklist regex (mkfs, rm-rf, dd, etc.).
> C'est du hardening, pas de la securite formelle.
> `sage-python/src/sage/tools/meta.py:129` execute `/bin/bash -c` avec blocklist.
> Les tests valident la blocklist mais pas l'exhaustivite.

## AgentTool

`AgentTool.from_agent()` : wrappe n'importe quel agent comme un outil appelable.
Permet la composition recursive d'agents.

## ToolForge

Generation dynamique d'outils au runtime :
- **GapDetector** : identifie les capacites manquantes (queue bornee, TTL, deduplication)
- **BuildLoop** : synthetise code + tests via LLM, dual-gate (AST + sandbox), max 3 rounds
- **Tool.run()** : methode d'execution ajoutee (avril 7, bug "Tool not callable" corrige)
- **E2E valide** : gap detection → synthese → registration → utilisation dans un pipeline run
- **Limites** : MAX_CREATIONS_PER_RUN=2, MAX_BUILD_ROUNDS=3

## Dynamic sub-agent creation

`agent_mgmt.py` : creation de sous-agents a la OpenSAGE (self-programming).
