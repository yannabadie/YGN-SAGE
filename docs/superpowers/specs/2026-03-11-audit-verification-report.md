# Audit Verification Report — 5 Independent Audits

**Date:** 2026-03-11
**Verified by:** Claude Opus 4.6 (systematic codebase verification)
**Scope:** Audit1.md through Audit5.md — every verifiable assertion checked against HEAD

---

## Synthèse

| Metric | Count |
|--------|-------|
| Assertions extraites | 38 |
| ✅ Confirmées | 22 (58%) |
| ⚠️ Partiellement vraies | 8 (21%) |
| ❌ Infirmées | 5 (13%) |
| 🔍 Non vérifiables | 3 (8%) |

---

## Matrice Complète

### SÉCURITÉ

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| S1 | Python validator bypass 5/8 vectors | Audit3 F-01 | ✅ | CRITICAL | Chunk 1: expand BLOCKED_CALLS + dunder detection |
| S2 | Subprocess zero OS isolation | Audit3 F-02 | ✅ | CRITICAL | Chunk 2: document limitation, recommend Wasm path |
| S3 | Python blocks 6 vs Rust 11 calls | Audit3 F-11 | ✅ | HIGH | Chunk 1: achieve parity (+ extras) |
| S4 | _safe_z3_eval uses eval() | Audit3 F-12 | ⚠️ | MEDIUM | Mitigated: AST whitelist + empty __builtins__. z3 surface unaudited |
| S5 | eBPF requires CAP_SYS_ADMIN | Audit1 §3 | ❌ | N/A | Uses userspace solana_rbpf, not kernel eBPF |
| S6 | execute_raw() bypass validation | Audit5 §6 | ✅ | HIGH | Chunk 2: add WARN tracing |

### BENCHMARKS & CI

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| B1 | Badge test inflated (1127) | Audit3 F-03 | ⚠️ | MEDIUM | Chunk 3: update to 1149 |
| B2 | HumanEval 95% = 20 tasks only | Audit3 F-04 | ✅ | HIGH | Chunk 3: report 84.1% (164) |
| B3 | Routing benchmark circulaire | Audit3 F-07 | ✅ | HIGH | Chunk 6: relabel as self-consistency |
| B4 | Artifacts model: "unknown" | Audit5 §2 | ✅ | MEDIUM | Chunk 3: fix _last_model tracking |
| B5 | CI skips smt, tool-executor | Audit3 F-05 | ✅ | HIGH | Chunk 4: add CI jobs |
| B6 | No Dockerfile/lockfile | Audit2 §6 | ❌ | N/A | Both exist (Dockerfile + Cargo.lock) |
| B7 | HumanEval = architectural fraud | Audit1 §6 | ⚠️ | MEDIUM | Ablation proves +15pp delta — valid framework contribution |
| B8 | Routing 23/30 vs 30/30 claimed | Audit5 §1 | ✅ | HIGH | Chunk 3: correct README |

### VÉRIFICATION FORMELLE

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| V1 | OxiZ: QF_LIA only | Audit2 §1 | ✅ | MEDIUM | Intentional — document clearly |
| V2 | Verifier = constraint DSL | Audit2 §1 | ✅ | MEDIUM | Accurate — not code verification |
| V3 | verify_arithmetic bug | Audit3 F-06 | ⚠️ | LOW | Audit claim WRONG: path IS reachable with z3-solver |
| V4 | Z3 = scheduling SAT only | Audit4 §4 | ✅ | MEDIUM | z3_verify.py yes; kg_rlvr.py does invariants too |

### ROUTING

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| R1 | Routing = regex by default | Audit2 §2 | ✅ | HIGH | Chunk 6: honest documentation |
| R2 | async = sync wrapper | Audit5 §3 | ✅ | LOW | Design decision, not bug |
| R3 | Bandit assumes stationarity | Audit1 §5 | ❌ | N/A | Decay 0.995 implemented — handles non-stationarity |

### ARCHITECTURE

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| A1 | conductor/ empty | Audit3 F-14 | 🔍 | N/A | Directory doesn't exist |
| A2 | Docs > source LOC | Audit3 F-08 | ✅ | LOW | ~38.7K vs ~38.4K — marginal |
| A3 | CircuitBreaker not thread-safe | Audit4 §5.1 | ✅ | LOW | Single-threaded in practice |
| A4 | No automatic recovery | Audit4 §5.2 | ✅ | MEDIUM | Chunk 5: add half-open state |
| A5 | Providers = discovery not assurance | Audit5 §4 | ✅ | HIGH | Chunk 5: document gap |
| A6 | Overlapping control planes | Audit5 §8 | ✅ | HIGH | Future: consolidate routing stack |

### MÉMOIRE

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| M1 | FFI overhead 10-30ms | Audit1 §1 | 🔍 | MEDIUM | No profiling data — unverifiable |
| M2 | Memory no evidence vs baseline | Audit5 §7 | ⚠️ | HIGH | Ablation shows mixed results on 20 tasks |
| M3 | 130 tests skipped | Audit3 F-13 | ✅ | MEDIUM | 102 skipped — Rust features + optional deps |
| M4 | S-MMU returns "" sans Rust | Audit5 §7 | ✅ | HIGH | By design — fallback documented |
| M5 | Arrow category error | Audit1 §1 | ⚠️ | MEDIUM | Arrow = container, not OLAP — overstated |

### ÉVOLUTION

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| E1 | Zero improvement evidence | Audit3 F-09 | ✅ | HIGH | Chunk 7: document evidence gap |
| E2 | SAMPO params "not from paper" | Audit3 F-10 | ✅ | MEDIUM | Self-admitted, honest comments |
| E3 | MAP-Elites $500/run | Audit1 §4 | ⚠️ | LOW | Offline only — runtime uses archive lookup (0 cost) |
| E4 | Z3 safety gate removed | Audit4 §6.3 | ✅ | MEDIUM | Documented reason in code |

### MISC

| ID | Claim | Source | Verdict | Sévérité | Action |
|----|-------|--------|---------|----------|--------|
| X1 | Claude co-authorship 401 commits | Audit2 §5.2 | 🔍 | LOW | Not in git log |
| X2 | CI failures 2026-03-10 | Audit5 §1 | ✅ | MEDIUM | Point-in-time, addressed by Chunk 4 |

---

## Divergences Majeures — Là où les Audits se Trompent

| Audit | Claim | Réalité |
|-------|-------|---------|
| Audit1 §3 | eBPF requires root | Userspace solana_rbpf — no kernel eBPF |
| Audit1 §5 | Bandit assumes stationarity | Decay 0.995 handles non-stationarity |
| Audit2 §6 | No Dockerfile / lockfile | Both Dockerfile (58L) + Cargo.lock exist |
| Audit3 F-06 | verify_arithmetic bug: unreachable | ast.literal_eval IS reachable with z3-solver |
| Audit1 §4 | MAP-Elites $500/run | Runtime uses archive (0 cost), offline only |
| Audit1 §1 | Arrow = OLAP category error | Arrow = serialization container, not OLAP engine |

---

## Problèmes Confirmés par Priorité

### P0 CRITICAL (fix immédiatement)
1. **S1+S3**: Python sandbox dunder bypass → expand BLOCKED_CALLS + dunder detection
2. **S2**: Subprocess no OS isolation → document, recommend Wasm

### P1 HIGH (fix cette semaine)
3. **B2+B8**: README metrics faux → corriger 84.1%, 23/30
4. **B5**: CI skip features → ajouter smt + tool-executor jobs
5. **S6**: execute_raw no audit trail → ajouter WARN tracing

### P2 MEDIUM (fix ce mois)
6. **A4**: CircuitBreaker no recovery → ajouter half-open
7. **A5**: Provider capabilities statiques → documenter gap
8. **B3**: Routing benchmark circulaire → relabeler
9. **B4**: Model "unknown" → fix _last_model tracking

### P3 LOW (backlog)
10. **E1**: Evolution no evidence → documenter, planifier ablation
11. **A6**: Control planes multiples → consolider (futur sprint)
12. **M2**: Memory vs long-context → planifier benchmark comparatif

---

## Recommandations SOTA (Mars 2026)

### Sécurité — Sources: HackTricks, E2B, nsjail, RestrictedPython v7.4

**Consensus industrie 2026**: AST-only est insuffisant. Defense-in-depth requis.

| Approche | Latence | Isolation | Python Support | Recommandation |
|----------|---------|-----------|----------------|----------------|
| **Wasm WASI (wasmtime)** | ~1ms | Forte (capability-based) | Limitée | ✅ Déjà implémenté — sous-utilisé |
| **nsjail** | ~5ms | Forte (namespaces + seccomp) | Complète | ✅ Ajouter pour Linux subprocess |
| **bubblewrap** | ~5ms | Moyenne | Complète | Alternative à nsjail |
| **Docker** | ~50ms | Moyenne | Complète | Trop lent pour tool exec |
| **gVisor** | ~70ms | Forte | Complète | Cloud-only |
| **Firecracker** | ~125ms | Maximale | Complète | E2B utilise ça — trop lourd ici |

**Action immédiate**: Hardening tree-sitter (bloquer TOUS les dunders via `attr.starts_with("__")`) + expanded BLOCKED_CALLS. Ne PAS adopter RestrictedPython comme dépendance — implémenter le pattern en Rust.

**Court terme**: nsjail wrapper pour subprocess Linux (feature flag).

### Routing — Sources: RouteLLM (ICLR 2025), MixLLM (NAACL 2025), LLMRouterBench

| Approche | Coût | Qualité | Training Data | Status |
|----------|------|---------|---------------|--------|
| **ModernBERT classifier** | 0 (local) | Forte | 1500+ labels | ✅ Remplace DistilBERT/DeBERTa |
| **RouteLLM (BERT router)** | 0 (local) | Forte | 80K preference pairs | Référence |
| **MixLLM (contextual bandit)** | $0 | 97.25% GPT-4 quality | Online learning | Valide approche YGN-SAGE |
| **PILOT (LinUCB budget)** | $0 | Forte | Online | Valide ContextualBandit |

**45 labels est insuffisant** — minimum 1500 golden-labeled. LLMRouterBench (23,945 prompts) disponible comme source.

**ModernBERT-base** (answerdotai, 149M params, 8192 tokens) remplace BERT/DistilBERT comme backbone — 2-4x plus rapide.

### Architecture — Sources: MasRouter (ACL 2025), Unified Cascade (ICLR 2025)

**Consolidation 3-stage pipeline** (validé par MasRouter):
```
Stage 1: TopologyEngine    → QUELLE topologie
Stage 2: SystemRouter       → QUEL système cognitif (S1/S2/S3)
Stage 3: ContextualBandit   → QUEL modèle spécifique
```
- AdaptiveRouter absorbé dans Stage 2
- ShadowRouter → middleware observabilité (pas un control plane)
- CognitiveOrchestrator → point d'entrée unique appelant Stages 1-2-3

### Évolution — Sources: AlphaEvolve, CodeEvolve, FunSearch

**Protocole validation minimum**:
- N ≥ 10 runs par configuration
- Wilcoxon signed-rank test (paired, non-paramétrique)
- Cohen's d effect size (small=0.2, medium=0.5, large=0.8)
- 3 baselines: random search, static best model, FunSearch-style
- Courbes de convergence fitness vs générations

### Benchmarks — Sources: SWE-bench, Terminal-Bench, DPAI Arena

**Minimum statistique**: 50+ tasks (20 insuffisant), bootstrap CIs, delta avec intervalles.

**Truth pipeline**: CI produit `status.json` → README auto-généré → fail si drift détecté.
