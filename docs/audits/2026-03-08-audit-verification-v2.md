# YGN-SAGE — Audit Verification Report V2

> **Verifier:** Claude Opus 4.6
> **Date:** 2026-03-08
> **Scope:** 4 independent audits (AuditC, AuditG, AuditQ, AuditCG) — 44 verifiable assertions
> **Method:** Codebase search + code review + Context7 docs + Gemini 3.1 Pro + GPT-5.4 Codex oracles

---

## Synthese

| Metric | Value |
|--------|-------|
| Assertions verifiees | 44 |
| Confirmees (audit correct) | 28 (64%) |
| Partiellement vraies | 8 (18%) |
| Infirmees (audit incorrect) | 8 (18%) |
| Non verifiables | 0 |

---

## Matrice Complete

### Securite

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| S1 | eval() RCE dans kg_rlvr.py | AuditQ | INFIRME | Safe AST evaluator en place (whitelist 12 noeuds, `__builtins__: {}`) |
| S2 | ~15 except silencieux | AuditQ | PARTIELLEMENT | 4 reels (pas 15). Audit surestime x3.75 |
| S3 | WebSocket token query param | AuditQ | CONFIRME | Token dans logs/historique. Solution: First-Message Auth |
| S4 | Sandbox allow_local=False | AuditQ | CONFIRME | Fix A1 applique correctement |
| S5 | wasmtime v29 obsolete | AuditQ | INFIRME | Upgrade v36 LTS complete |
| S6 | solana_rbpf commente | AuditC/Q | CONFIRME | Deferre Phase 2, documente |
| S7 | snap_bpf.c stub 22 lignes | AuditQ | CONFIRME | 21 lignes, marque STUB |

### Routing

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| R1 | Keyword substring matching | AuditQ/C/CG | CONFIRME | `"test" in lower` (pas word-boundary) |
| R2 | Seuils hardcodes S1<=0.35 S3>0.7 | AuditQ | CONFIRME | Parametres constructeur |
| R3 | Speculative zone non implementee | AuditQ | PARTIELLEMENT | Loggee avec _log.info(), pas silencieuse |
| R4 | Routing benchmark circulaire | AuditC/Q/CG | CONFIRME | Docstring dit "self-consistency NOT accuracy" |
| R5 | AgentSystem.run() bypass orchestrateur | AuditC/CG | CONFIRME | By design, documente |
| R6 | Orchestrateur construit jamais appele | AuditC/CG | CONFIRME | Cree a boot, jamais dans execution |
| R7 | "test" declenche tool_required | AuditC | CONFIRME | Substring match, question conceptuelle = faux positif |
| R8 | 3 plans de controle concurrents | AuditCG | PARTIELLEMENT | Consolide (fix B8) mais legacy retenu |

### Benchmark

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| B1 | HumanEval 20/164 seulement | AuditC/Q | CONFIRME | Publie: 10 et 20 problemes |
| B2 | Tous routes vers S2 | AuditQ | CONFIRME | 100% S2 dans traces |
| B3 | Pas de baseline publie | AuditC/Q/CG | CONFIRME | Code existe, jamais execute |
| B4 | Badges test count perimes | AuditQ | INFIRME | Badge = 691 = actuel |
| B5 | Aucun benchmark concurrent | AuditCG | CONFIRME | Aucune comparaison externe |

### Z3/Contrats

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| Z1 | capability_coverage trivial | AuditC/Q | PARTIELLEMENT | Z3 syntaxiquement correct, probleme = set membership |
| Z2 | budget_feasibility trivial | AuditC/Q | PARTIELLEMENT | Inegalite lineaire encodee en Z3 Reals |
| Z3 | type_compatibility trivial | AuditC/Q | PARTIELLEMENT | Check transitif, legitime mais simple |
| Z4 | provider_assignment genuine SAT | AuditC/Q/CG | CONFIRME | CSP multi-variable avec exclusions mutuelles |
| Z5 | z3_topology.py silent catch | AuditQ | CONFIRME | `except Exception: return ""` sans log |

### Memoire

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| M1 | Working memory fallback silencieux | AuditQ/CG | PARTIELLEMENT | Mock existe mais log WARNING bruyant (fix A3) |
| M2 | SemanticMemory pas de persistence | AuditQ | INFIRME | SQLite save/load implemente et wire |
| M3 | CausalMemory pas de persistence | AuditC/Q | CONFIRME | In-memory uniquement |
| M4 | S-MMU retourne IDs pas contenu | AuditCG | CONFIRME | Chunk {id} sans texte |
| M5 | S-MMU O(n2) edges | AuditC | CONFIRME | Itere TOUS les chunks par cosine |
| M6 | ExoCortex hardcode DEFAULT_STORE | AuditC/Q/CG | CONFIRME | Store ID en dur |
| M7 | sentence-transformers pas dans deps | AuditCG | CONFIRME | Pas dans extras, fallback gracieux |
| M8 | WriteGate unbounded | AuditQ | INFIRME | Deja fixe OrderedDict(10000) |

### Guardrails/Architecture

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| G1 | SchemaGuardrail misconfigured | AuditC/Q/CG | CONFIRME | Attend JSON, recoit texte |
| G2 | Cost estimation 4 chars/token | AuditC | CONFIRME | `len(text)//4` hardcode |
| G3 | Cost per 1K hardcode | AuditC | CONFIRME | Dict _COST_PER_1K |
| G4 | Dashboard single-task (409) | AuditQ | CONFIRME | HTTP 409 Conflict |
| G5 | Provider tool->user rewrite | AuditCG | CONFIRME | GoogleProvider reecrit avec warning |

### Evolution

| # | Assertion | Source | Verdict | Detail |
|---|-----------|--------|---------|--------|
| E1 | ~500 LOC non valide | AuditC/Q/CG | PARTIELLEMENT | 1159 LOC, tests unitaires OK, pas de baseline |
| E2 | eBPF evaluator importe code desactive | AuditC | CONFIRME | Import avec fallback gracieux |
| E3 | Seul SAMPO utilise | AuditC | INFIRME | 4 solvers ont des appelants |
| E4 | Magic constants solvers.py | AuditC | CONFIRME | 8+ constantes heuristiques documentees |
| E5 | CI skip cargo test | AuditQ/CG | INFIRME | cargo test bien present |
| E6 | License contradiction | AuditC/Q | INFIRME | MIT partout, deja fixe |

---

## Divergences Majeures

### Audits incorrects

1. **eval() RCE (AuditQ P0-1)** — Le plus gros faux positif. L'audit dit "UNFIXED after 5 audits" alors que le safe AST evaluator est en place depuis le debut (commit T75). Aucune action necessaire.

2. **"~15 silent exceptions" (AuditQ)** — Surestime x3.75. Reel: 4 patterns silencieux (1 continue, 3 returns sans log). Impact reduit de CRITICAL a LOW.

3. **"Z3 = snake oil/decorative" (AuditG/Q)** — Trop severe. Z3 est le gold standard (Context7 confirme). Les 3 checks triviaux sont over-engineered mais pas faux. Provider assignment est genuinely non-trivial.

4. **"SemanticMemory no persistence" (AuditQ)** — Faux. SQLite save/load implemente et wire dans boot.py. Audit perime.

5. **"Only SAMPO used, 3 solvers dead" (AuditC)** — Faux. Les 4 solvers (SAMPO, RegretMatcher, VAD-CFR, SHOR-PSRO) sont tous instancies dans StrategyEngine.

6. **"CI skips cargo test" (AuditQ/CG)** — Faux. ci.yml ligne 27: `cargo test --no-default-features`.

### Audits partiellement corrects mais exageres

1. **AuditG "SMMU = tech-word salad"** — L'architecture multi-vue est solide (confirmee par 3 oracles). Le vrai probleme est O(n2), pas le concept.

2. **AuditG "eBPF = CVE generator"** — solana_rbpf est userspace (pas kernel Ring-0), commente, et wrappe en try/except. Deja neutralise.

3. **AuditG "SIMD = performative theater"** — Pour le FFI tax, oui. Mais l'ANN index (usearch-rs) resout ca proprement.

---

## Solutions SOTA — Consensus 3 Oracles (mars 2026)

| Probleme | Solution | Source |
|----------|----------|--------|
| S-MMU O(n2) | `usearch-rs` HNSW ANN, top-k=16, rerank exact | Gemini + Codex |
| SchemaGuardrail | OutputGuardrail pour texte + structured output google-genai | Gemini |
| Routing keywords | fastText shadow mode (1.6MB, 0.92 acc), pas BERT direct | Codex |
| Cost estimation | `usage_metadata` post-request + `count_tokens()` pre-request. PAS tiktoken | Gemini |
| ExoCortex vendor lock | `KnowledgeStore(Protocol)` + adapters (Google, Qdrant) | Gemini |
| WebSocket auth | First-Message Auth (accepter WS, auth sur 1er message, timeout 3s) | Gemini |
| Z3 trivial checks | Python natif (~2000x speedup). Garder Z3 pour provider_assignment SAT | Codex |
| Z3 SAT bug | `z3.PbEq([(v,1) for v in vars], 1)` — exactly-one, pas at-least-one | Codex |
| Evolution validation | 4 bras (full/random/MAP-sans-SAMPO/seed-only), 20-30 seeds, 3 familles | Codex |
| Dashboard mono-slot | Court terme: asyncio.Queue. Long terme: Postgres + Redis Streams + Temporal | Codex |
| Dependencies | ort load-dynamic OK, wasmtime v36 LTS OK, z3-solver = gold standard | Context7 |

---

## Plan d'action

Voir: `docs/plans/2026-03-08-audit-response-v2.md`

| Sprint | Tasks | Effort | Impact |
|--------|-------|--------|--------|
| Sprint 1 | 8 quick wins | 1 jour | Fix guardrails, Z3, routing, cost, SMMU, WS auth, causal persistence |
| Sprint 2 | 5 structural | 3-5 jours | ANN index, ExoCortex protocol, Z3 bug, dashboard queue, docs |
| Sprint 3 | 5 evidence | 1-2 semaines | HumanEval 164, routing proof, evolution validation, memory ablation |

*Verification conduite le 2026-03-08 contre le dernier commit master.*
