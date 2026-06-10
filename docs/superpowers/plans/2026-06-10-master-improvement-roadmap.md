# Master Improvement Roadmap — reprise juin 2026 (couvre Amelioration-fable.md)

> **Nature du document** : roadmap à gates de décision, PAS un plan d'implémentation.
> Chaque phase exécutable reçoit son plan détaillé (style superpowers:writing-plans,
> steps + code exact) **just-in-time**, pour trois raisons : (1) la règle
> no-placeholder — on ne peut pas écrire les steps exacts d'une branche dont
> l'existence dépend d'un résultat empirique futur ; (2) la convention projet —
> tout bloc touchant la sémantique runtime passe par cgpro DESIGN→VERIFY avant
> SHIP ; (3) chaque bloc nécessite sa reconnaissance code propre au moment T.
> Source d'analyse : `Amelioration-fable.md` (audit du 2026-06-10, recherche
> arXiv/HF/Context7 + leaderboards juin 2026).

**But global** : combler l'écart entre la sophistication du runtime (ingénierie
saine : mypy 0, 4 346 tests, 11 invariants, claims auditées) et l'évidence
benchmark encore manquante — sans diluer la niche « verified adaptive
orchestration runtime ».

---

## Phase 0 — Dette sécurité + hygiène (P0.1, P0.2, P0.4-partiel) — ✅ FAIT 2026-06-10

Plan exécuté : `docs/superpowers/plans/2026-06-10-security-deps-hygiene.md`.
Livré (`c1974e0d..4f08b2fd`, 8 commits) : wasmtime-wasi 44.0.2
(RUSTSEC-2026-0149 HIGH), rand patchés + deny.toml nettoyé, constraints
full-regen (10 advisories pip-audit fermées, pydantic-ai-slim 1.106.0),
re-pin cargo-deny-action (débloque Dependabot), artefacts canary 11-12 mai
commités, current.json 3659→3662, wheel local rebuilt (`matches:true`,
smoke L2/L3/L4 verts), 57 Go disque libérés (`target/debug` — était à 100 %,
blocker documenté du grader SWE-bench Pro).

Reste de Phase 0 (suivi dans cette session) : CI verte confirmée sur
`4f08b2fd` (Security + CI + latest-deps dispatché), fermeture issue #14,
merge PR Dependabot #13.

## Phase 1 — `B2_RERUN_UNBLOCKERS` (P0.3) — PRÊT, gratuit, prochain bloc

**Scope verrouillé par cgpro le 2026-05-12** (post-push B2, conv
`cgpro_i11_design_20260511`) — le DESIGN existe déjà, pas besoin d'un
nouveau round avant implémentation ; un VERIFY pré-push reste dû (sémantique
runtime).

Les 3 bugs de prod (tous confirmés encore présents — zéro commit depuis) :

1. **Provider `unknown` sur tier reasoner** — `gemini-3.1-pro-preview` exécute
   avec `provider_final=unknown` alors que `cards.toml` déclare
   `provider="google"`. Fall-through dans la chaîne d'attribution
   runtime/canary. Casse `provider_gate` (NO_GO `execution_outside_allowlist`).
   Contrainte directive #7 : le fix doit sourcer la vérité depuis cards.toml,
   pas hardcoder un mapping.
2. **Perte de coût hors-timeout** — `total_cost_usd=0` sur échecs non-timeout
   ET pertes partielles sur succès (réel $0.79 vs rapporté $0.30, 2.6×).
   Le fix `eeb3a7fb` ne couvrait que le chemin timeout ; généraliser la
   récupération depuis `_observed_event_cost_usd` avec source explicite.
3. **Diff-verifier silencieux** — `_diff_verifier_outcome=None` partout malgré
   `SAGE_DIFF_VERIFIER_MODE=observe` : propagation env vers le subprocess
   canary OU gate interne. Doit produire un outcome non-null avec patch, et
   `skipped_no_patch` explicite sans patch.

**Contrat de sortie (déjà spécifié par cgpro)** : 9 tests requis ;
allowed_files = `run_dryrun_arm_d.py`/tests + `sage/**/{model,provider,routing}*.py`
+ doc postmortem ; forbidden = toucher I-11/ledger, affaiblir provider_gate,
allowlister `unknown`, désactiver les gates diff-verifier, rerun payant,
refonte prompt/topologie ; stop_conditions si le fix exige un changement de
sémantique policy ou sort du périmètre runner+verifier.

**Livrable final de phase** : dry-run gratuit (fixture/mock) prouvant les
3 champs peuplés (`provider_final` correct, coût intégral, outcome verifier).
Plan détaillé : `2026-06-10-b2-rerun-unblockers.md` (écrit au démarrage du bloc).

## Phase 2 — Échelle d'évidence payante (P1) — 🔒 GATE D'AUTORISATION YANN

Prérequis : Phase 1 close + CI verte. **Aucun run payant sans GO explicite.**

| Étape | Coût | Gate de passage |
|---|---|---|
| 2.a Canary N=5 gradé (reasoner, patch_focused, manifest frozen) | ~$1 | `patches_extracted ≥ 3/5` ET provider_gate PASS ET coûts intègres ET verifier non-silencieux |
| 2.b **Arm A (single-agent fort) vs Arm D (SAGE complet), apparié, N=10** | ~$15-20 | — c'est LE test de la proposition de valeur |
| 2.c (optionnel, seulement si 2.b positif) N=50 4-arm | $240-460 | décision séparée, jamais avant 2.b |

**Pourquoi 2.b avant tout score absolu** : les leaderboards juin 2026 sont
dominés par des scaffolds minimaux (Mini-SWE-Agent 59,1 % officiel,
Live-SWE-agent 45,8 %) et les ablations internes (cycle-10 : « v7 gap =
sample variance ») n'ont jamais démontré le delta d'orchestration. 2.b
tranche la question avec le minimum d'argent. **Les deux issues sont
exploitables** — c'est un gate de direction, pas un pari.

## Phase 3 — Branche selon le résultat de 2.b

### 3α — Delta d'orchestration démontré → renforcer l'existant
- **GEPA sur les prompts des 12 templates** (DSPy/GEPA via le meta-harness
  externe déjà cloné ; métrique BCB-hard ; MASS prédit +9pp joint
  prompt+topology). Meilleur ratio gain/effort qualité.
- **TwinRouterBench** (arXiv 2605.18859) : évaluer ModelAssigner + bandit sur
  le track statique (970 décisions) puis dynamique — évidence routing externe
  à coût faible, pile dans la niche « show why it chose ».
- **Z3 inter-nœuds** : QualityLabeler en step-level entre les nœuds (façon
  VPRMs/FoVer) au lieu de post-hoc seulement — gap déjà documenté dans
  research-decisions.md, jamais exécuté.

### 3β — Pas de delta → pivot du pilier Evolution vers les skills
- **Bibliothèque de compétences testables evidence-gated** (pattern
  MUSE-Autoskill/AutoSkill/Atomic Skills) : cycle création → mémoire →
  évaluation (tests unitaires) → promotion gated par `verdict.trainable`.
  SAGE a déjà les 3 briques (ToolForge, oracle Z3, learning-side-effects
  ledger) ; il manque le cycle de vie. S'aligne avec la niche « verified »
  au lieu de la diluer. Nouvel invariant ledger probable (« skill promue ⇒
  évidence de test liée »).
- La niche se recentre sur auditabilité/intégrité — validée par la
  convergence du champ (FormalJudge, VeriGuard, Pro2Guard, Agent libOS).

### 3-commun (indépendant de la branche, opportuniste)
- **ARG-Designer comme refonte apprise du Path 6** (AAAI 2026 oral,
  génération autorégressive conditionnelle du DAG) : à lancer seulement
  quand les données accumulées (archive MAP-Elites + outcomes bandit avec
  provenance) suffisent pour le fine-tuning conditionnel ω/δ/γ + domaine +
  tier. Caveat : ses benchmarks (HumanEval/GSM8K/MMLU) sont saturés —
  valider sur BCB-hard/Pro.
- **Pro2Guard-style** : pré-checks probabilistes (DTMC) dans le
  TopologyController — prédire l'état unsafe avant qu'il survienne.

## Phase 4 — Maintenance continue (P0.4-reste + P2-maintenance, sans dépendance)

- **Compaction CLAUDE.md** : 8 sections « Current State » empilées → garder
  la plus récente + pointeurs d'archives datées sous `docs/status/`.
  (Attention narrative_guard : sections historiques taguées.)
- **Compaction MEMORY.md** : 59 KB / 327 lignes, dépasse la limite de
  chargement — une ligne par mémoire, détails dans les fichiers topic.
- **mypy 2.x migration** : ticket dédié (cap `<2` posé le 2026-05-12 pour le
  narrowing isinstance ; `extended_drift.py:229`).
- **pyo3 0.25 → courant** : vérifier matrice de compat maturin/abi3 d'abord.
- **R2-Router-style** : router aussi le budget de sortie/reasoning-effort
  (extension naturelle de cards.toml — `thinking` y est déjà par modèle).
- Veille hebdo : latest-deps + Security restent les détecteurs ; ne plus
  laisser 4 semaines de rouge s'accumuler.

## Ordre d'exécution effectif

```
Phase 0 ✅ → Phase 1 (B2, gratuit) → [GO Yann] Phase 2.a → gate ≥3/5
  → Phase 2.b (arm A vs D) → branche 3α OU 3β → (2.c N=50 si justifié)
Phase 4 en tâche de fond entre les blocs.
```

Chaque bloc : plan détaillé dédié + cgpro VERIFY pré-push si sémantique
runtime + claims/ledger mis à jour si une capability change d'état.
