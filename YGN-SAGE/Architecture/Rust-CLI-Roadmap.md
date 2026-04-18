---
title: Rust CLI / TUI Roadmap
type: architecture
tags:
  - roadmap
  - cli
  - tui
  - rust
  - product
updated: 2026-04-18
---

# Rust CLI / TUI — Roadmap

> Vision produit : **Claude Code × infiniment plus puissant et efficace**.
> Binaire Rust autonome, portable, sans overhead Python runtime pour l'utilisateur
> final. Conserve le moteur YGN-SAGE (self-adaptive topology, 5 piliers)
> mais expose une UX end-user proche de Claude Code / Codex CLI / Cursor CLI.

## UI Mockup (Apr 18)

```
┌──────────────────────────────────────────────────────┬──────────────────┐
│                                                      │ ┌──────────────┐ │
│                                                      │ │   Topology   │ │
│                                                      │ │   / tasks    │ │
│                                                      │ └──────────────┘ │
│          MAIN CHAT / TRANSCRIPT                      │                  │
│                                                      │ ┌──────────────┐ │
│                                                      │ │ Mémoire et   │ │
│          (user ↔ sage, streaming output,             │ │ pensées de   │ │
│           tool-call previews, diffs, etc.)           │ │ l'agent      │ │
│                                                      │ │ YGN-SAGE     │ │
│                                                      │ └──────────────┘ │
│                                                      │                  │
│                                                      │ ┌──────────────┐ │
│                                                      │ │   (cost /    │ │
│                                                      │ │   status /   │ │
│                                                      │ │   provider)  │ │
│                                                      │ └──────────────┘ │
├──────────────────────────────────────────────────────┴──────────────────┤
│  > prompt input ____________________________________________________    │
└──────────────────────────────────────────────────────────────────────────┘
```

Trois panneaux droite + main + input bas = classique TUI (Claude Code,
OpenCode, Cursor). Ratatui est le framework Rust de référence.

## Références étudiées

| Projet | URL | Apports pour YGN-SAGE |
|--------|-----|----------------------|
| **pi-mono** (TypeScript) + **pi_agent_rust** | github.com/badlogic/pi-mono | Architecture modulaire : pi-ai (unified LLM API), pi-agent-core (tool exec + event streaming), pi-tui (TUI différentielle flicker-free). Philosophie : minimaliste, résiste au bloat. Claude Code refactor-proof. |
| **openclaude** | github.com/Gitlawb/openclaude | Multi-provider (OpenAI/Gemini/DeepSeek/Ollama/Codex/200+ via OpenAI-compatible). Headless gRPC mode permet d'intégrer YGN-SAGE comme backend pour d'autres UIs. |
| **claude-code-rust** | github.com/srothgan/claude-code-rust | Ratatui natif remplaçant le binaire Claude Code. |
| **claurst** | github.com/Kuberwastaken/claurst | Clean-room reimplementation de Claude Code en Rust : TUI pair programmer, plugin system, multi-provider. |
| **OpenAI Codex CLI** (Rust) | (leur implem) | Référence industrielle : agent CLI Rust en production. |
| **mini-swe-agent** | github.com/SWE-agent/mini-swe-agent | 100 lignes, bash-only, 74% SWE-bench. Validation du pattern « simple loop + 1 tool » pour cas tâche unique. |

## Architecture cible

### Couches (bottom-up)

1. **sage-core** (Rust, existant) — TopologyEngine, SystemRouter,
   ModelAssigner, QualityLabeler. 441 tests.
2. **sage-runtime** (Rust, **nouveau**) — porter progressivement les
   composants Python critiques :
   - AgentLoop (run/stream)
   - phases (think/act/learn/perceive)
   - LLM providers (via reqwest + SSE streaming)
   - TopologyRunner
3. **sage-cli** (Rust, **nouveau**) — binaire utilisateur :
   - CLI: `sage`, `sage code`, `sage chat`, `sage bench`, `sage eval`
   - Ratatui TUI par défaut si stdin est tty, headless sinon
   - Gestion config/profils (`~/.sage/config.toml`, `~/.sage/profiles/`)
   - Streaming événements moteur → UI (topology updates, thoughts, cost)
4. **sage-ipc** (Rust, **nouveau**) — gRPC bidi stream pour headless
   intégration (CI/CD, MCP, hooks). Inspiration : openclaude.

### Couche de compatibilité

Pendant la transition (phase actuelle), **sage-python reste le moteur** ;
**sage-cli l'invoque via PyO3** comme hôte Python embarqué. Quand sage-runtime
rattrape feature-parity, sage-cli bascule vers Rust pur → on retire
la dépendance Python pour le binaire end-user.

## Ordre de migration (progressif, pas big-bang)

| Phase | Livrable | Durée estimée | Bloqueur |
|-------|----------|---------------|----------|
| **0** (actuelle) | Moteur Python stable, telemetry réelle, per-model routing, TTL exclusion | ✅ | — |
| **1** | `sage-cli` minimal (Rust) : `sage chat` + `sage code` qui invoque le moteur Python via PyO3. TUI Ratatui 3-panneaux. | 2-3 semaines | PyO3 bindings complets sur `AgentSystem.run()` (existent partiellement via sage-core) |
| **2** | Provider abstraction en Rust (`sage-llm` crate) : reqwest + SSE, traits pour Gemini/OpenAI/DeepSeek/MiniMax/OpenRouter/xAI/Kimi. Retirer LiteLLM Python pour les providers core. | 3-4 semaines | litellm remains for exotic providers only |
| **3** | AgentLoop + phases portés en Rust (`sage-agent` crate). Interop : Rust appelle tools Python via PyO3 *dans l'autre sens* (ToolForge reste Python au début). | 4-6 semaines | Memory tiers (Arrow/SQLite) sont déjà 50% Rust |
| **4** | TopologyRunner Rust. Meta-Harness reste Python (search-time only, pas runtime). | 3 semaines | Rust bindings pour ToolRegistry |
| **5** | Binaire single-file. `cargo install ygn-sage-cli` ou `.exe`/`.macho`/`.elf` téléchargeable. | 1-2 semaines | Packaging cross-platform |
| **Total** | **~3-5 mois** | | |

## Commandes CLI cibles

```bash
# Install
cargo install ygn-sage-cli
# ou : curl -LsSf https://ygn-sage.dev/install.sh | sh

# Chat interactif (TUI)
sage chat
sage chat --model gemini-3.1-flash-lite-preview

# Tâche coding one-shot (style Claude Code)
sage code "Fix the bug in src/auth.rs where TOTP validation rejects valid codes"

# Headless (pour CI/CD, hooks, IDE integration)
sage code --headless "..." --output json
sage serve --grpc --port 50051   # sage-ipc

# Bench / eval (dev only)
sage bench swebench-lite --limit 20 --offset 0
sage eval predictions.jsonl  # Docker eval local

# Meta-Harness search
sage tune start --benchmark masbench
sage tune status
sage tune apply <candidate_id>

# Provider management
sage provider list              # live health + TTL
sage provider reprobe <name>    # force re-probe
sage provider set-priority ...  # weight override

# Memory / ExoCortex
sage memory stats
sage memory recall "quel est le state des tests"
sage discover --domain llm-routing --papers 10
```

## Composants du TUI (ratatui)

### Main pane (gauche)
- **Chat transcript** : messages user/assistant alternés, rendu markdown, diff syntax highlighted
- **Streaming** : tool calls affichés en temps réel avec preview tronqué, réponse progressive
- **Approval prompts** : HITL inline (y/n/edit/abort) — matche le hook `hitl_callback` existant

### Side pane 1 : Topology / tasks
- **Graphe topo en cours** : rendu ASCII du DAG (nœuds + flèches), état par nœud (pending/running/done/failed)
- **Tool calls live** : liste des `execute_bash` en cours et terminés (scrollable, truncated 120 chars)
- **Provider par nœud** : `[planner: deepseek-reasoner] [coder: minimax] [synth: gemini]` — voir les décisions de ModelAssigner en action

### Side pane 2 : Mémoire et pensées
- **Working memory** (STM) : 10 derniers événements USER/ASSISTANT/TOOL
- **Internal state** (MEM1 rolling summary) : ~200 chars auto-résumé par l'agent
- **Memory recalls** : search hits ExoCortex + semantic memory en contexte
- **CGRS / self-brake** : indicateur visuel quand l'agent s'auto-frêne

### Side pane 3 : Status / cost
- **Total cost $** du run courant + cumul session
- **Tokens input/output** par provider
- **Step count** / timeout progress bar
- **Provider health** : live dead-list + TTL counters

### Bottom : input
- Multi-line prompt (Shift+Enter pour newline)
- Autocomplete commandes slash (`/clear`, `/reset`, `/save`, `/model`, `/provider`)
- File paste protection (> 100 lines → confirm)

## Optimisation coûts (objectif parallèle)

| Levier | Où | Gain estimé |
|--------|----|-----|
| **Diversity routing** (Apr 18 58ec0d8) | ModelAssigner Rust | -40% saturation 1 provider, évite timeouts cascadés |
| **FrugalGPT-on-rate-limit** (Apr 18 58ec0d8) | LiteLLMProvider | Évite l'over-pay quand un provider ralentit → fallback cheap-fast |
| **Budget scaling dynamique par plateau detection** | Agent loop Rust | -30% tokens sur tâches faciles, +capacité sur tâches dures |
| **Context compression sémantique** | Memory consolidation | -50% context tokens sur tâches à historique long |
| **Meta-Harness tuning** | `config/harness.json` sélectif | +10-15pp benchmark à coût constant |
| **Cache de prompts système** (Gemini / Anthropic) | Provider layer | -70% coût prompts répétés |
| **Routing S1/S2/S3 par kNN + bandit** (déjà fait) | SystemRouter | -60% vs full-reasoner-always |
| **Outil execute_bash local** (mini-swe-agent pattern, pas via Docker) | Existant | ~ gratuit |

## État actuel vs cible

| Composant | État Apr 18 | Cible |
|-----------|-----------|-------|
| Rust core (TopologyEngine + SystemRouter + ModelAssigner) | ✅ 441 tests | ✅ (continuer) |
| Python SDK | ✅ 1906 tests | ⚠️ maintenir pendant transition, ne pas le durcir |
| CLI `sage` | ❌ (uniquement `python -m sage.bench`) | ✅ binaire Rust |
| TUI | ❌ (logs stdout uniquement) | ✅ Ratatui 3-panneaux |
| Streaming events | ⚠️ event_bus existe, pas de consumer UI | ✅ SSE/WS → UI |
| Headless gRPC | ❌ | ✅ (inspiration openclaude) |
| Packaging end-user | ❌ `pip install ygn-sage` | ✅ `cargo install` / binaire standalone |
| Approval HITL | ✅ callback existe | ✅ UI inline |
| Meta-Harness | ✅ wired Apr 18 (58ec0d8) | ✅ `sage tune` command |

## Open questions

1. **Memory tier en Rust** : Arrow STM est déjà Rust, mais SQLite episodic
   est Python. Port natif (rusqlite) vs wrapper ?
2. **ExoCortex / knowledge pipeline** : reste Python (Google GenAI SDK,
   pas encore équivalent Rust).
3. **sage-discover** (arXiv ingestion) : Python définitif ou port ?
   Probable : Python, appelé offline, pas runtime-critical.
4. **Bandit (Thompson sampling)** : déjà Rust dans sage-core, OK.
5. **CLI argparse** : `clap` (Rust). **TUI** : `ratatui`. **gRPC** : `tonic`.
   **LLM HTTP** : `reqwest` + `eventsource-stream`.

## Prochaines actions concrètes

1. **Aujourd'hui / cette semaine** : stabiliser moteur Python (en cours) —
   derniers bugs `_cost_usd=0.0`, dynamic step budget, consolidation memoire.
2. **Semaine+1** : scaffold `sage-cli/` crate avec clap + ratatui, hello-world
   qui invoque `AgentSystem.run()` via PyO3. Commit « Phase 1 kickoff ».
3. **Semaine+2** : wire l'event_bus Python → Rust channel → UI panels.
4. **Semaine+3** : chat loop interactif fonctionnel sur une tâche.
5. **Semaine+4** : `sage code <prompt>` one-shot, feature-parity partielle.

## Non-objectifs (pour éviter scope creep)

- Pas de web UI (pour l'instant). Terminal first.
- Pas de cloud sync. Tout local.
- Pas de modèle propriétaire / fine-tuné dans le CLI (reste via providers externes).
- Pas de plugin marketplace (MCP existant suffit).
- Pas de support Windows WSL spécifique (le binaire Rust est cross-platform nativement).
