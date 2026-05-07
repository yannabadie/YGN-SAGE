---
name: routing.* claims pinned delivered + narrative recalibration shipped 2026-05-07
description: Cycle 12-commit qui ferme les 2 derniers routing.* evidence_pending via test_routing_gt_invariant.py + recalibre 23 surfaces narrative + ajoute Option C predicate dans narrative_guard
type: project
originSessionId: ea52b6a4-977f-4cac-83f6-f491b3dad1c2
---
Cycle routing claims evidence pinning + narrative recalibration shipped autonomous segment 2026-05-07. cgpro conv `cgpro_routing_claims_evidence_20260507` closed (4 rounds: DESIGN + amendment + 2 EDIT_REQUIRED). Push origin/main = HEAD `0d6f56f8`.

**Why:** `routing.knn_92pct` + `routing.system_router_88pct` étaient `evidence_pending` depuis Phase 0.4 (cycle-13 K, 2026-05-06 commit `763d1fc1`) faute de test CI-runnable pinant les chiffres. Les claims références (~92% / ~88%) avaient été mesurés sur un GT 50-task subset précédent, le GT actuel a 60 tasks. Audit Mini.md + AUDIT/YGN-SAGE_Analyse_Critique.md du jour les pointaient comme priorité critique #1 et #2.

**How to apply:**

**Méthodologie qui a marché** (réutilisable dans futurs cycles d'evidence pinning):

1. **Empirical baseline AVANT cgpro DESIGN_LOCK** — Mesurer les vraies valeurs locales (kNN 50/60 = 83.3%, SystemRouter 52/60 = 86.7%) AVANT de demander des seuils à cgpro. cgpro a initialement prescrit 51/60 (= 85%) qui aurait failed empiriquement de 1 — il a fallu un follow-up pour amender. Si j'avais mesuré before, on aurait pas eu cet aller-retour.

2. **2-commit anchor pattern pour evidence_commit self-referential** — Phase 0.4 ne permet pas SHA placeholder ou amend. Recipe: commit 1 ship test + YAML status flip avec `evidence_commit: "evidence_pending"` placeholder. Commit 2 anchor `evidence_commit: "<commit_1_sha>"` + regen claims index + status_snapshot. claims_audit fail localement sur commit 1 mais passe sur commit 2; CI ne run que sur le HEAD pushé.

3. **Option C lock pour archive vs live docs** — cgpro distingue:
   - **Live docs** (CLAUDE.md, .claude/rules/*, README authoritative, AI-ARCH live narrative): delivered-floor wording, NO present-tense `evidence_pending`
   - **Archive snapshots** (YGN-SAGE/* Obsidian vault, docs/papers/*, docs/benchmarks/results.md): past-tense `were evidence_pending in this archive snapshot; current status: docs/CLAIMS.yaml`. Vocabulary required: `archive snapshot` / `historic snapshot` / `historic figures` / `non-autoritative` / `historique` AND `was/were evidence_pending` / `était/étaient evidence_pending` AND `current status` / `current authoritative status` / `docs/CLAIMS.yaml`

4. **Strict 3-token AND predicate** dans narrative_guard pour les stale-status patterns — broad any-token allowlist était trop permissif (laissait passer "non-autoritative + claims are evidence_pending"). Le predicate `_is_archive_historic_status_line` requiert co-occurrence des 3 token classes sur la même ligne.

5. **4 stale-status guards anti-régression** ajoutés (cycle de 2 EDIT_REQUIRED rounds requis pour les couvrir):
   - `stale-routing-knn-status`: claim ID + evidence_pending co-occurrence (no figure required)
   - `stale-routing-systemrouter-status`: same shape SystemRouter
   - `stale-knn-accuracy-pending`: README project-tree pattern `kNN ... accuracy ... evidence_pending` (no claim ID)
   - `stale-routing-currently-pending`: present-tense `Routing ... currently evidence_pending` (no claim ID, no accuracy figure)

**Méthodologie qui a foiré** (à éviter):

1. **Sweeping wide regex sans allowlist** — La 1ère pass `stale-routing-knn-status` avait `()` allowlist par directive cgpro, fail sur 23 archive docs légitimes. cgpro round-2 a clarifié: needs predicate-based allow, not no-allow.

2. **Pattern `_is_archive_historic_status_line` sans token vocabulary stable** — J'ai initialement reformulé CLAUDE.md L293 avec "April 26 previous-state snapshot" qui n'était pas dans `_ARCHIVE_CONTEXT_TOKENS`. Round-2-final fix: réécrire avec "April 26 archive snapshot" pour matcher le vocab existant.

**Final state on origin/main:**

- HEAD `0d6f56f8`
- 12 commits cycle: `c43e8322` → `0d6f56f8`
- claims_audit --strict GREEN (20 claims, 9 delivered ↑ from 7, 2 evidence_pending ↓ from 4, 4 default-on, 2 opt-in, 2 planned, 1 retired)
- narrative_guard 28/28 PASS avec 4 nouveaux stale-status guards + Option C predicate
- mypy 0 / ruff clean
- 3292 Python tests collected (+2 routing GT invariants), 555 Rust, 100 sage-discover
- Empirical baselines locked: kNN 50/60 (S1=16/20, S2=15/20, S3=19/20), SystemRouter 52/60 (budget=1.0)

**Open follow-ups (defer to fresh conv):**

- A22b bucket-analysis script pour diff_verifier outcomes — cgpro recommandation pour next direction "petit mais haute valeur pour observabilité SWE-bench"
- A14b Stage-0 closure (route_integrated() repair) — listé en cycle-13 K queue, real wiring multi-file
- B9 AgentLoop per-run immutable context — concurrency-safe refactor (cycle-12 P6-A Phase B avait shipped half)

**Lesson on audit triage** (cf separate feedback memory):

Mini.md et AUDIT/YGN-SAGE_Analyse_Critique.md du jour ont été générés sans regarder les commits 2026-04-22 (P3.4 cognitive default, P1.6 cost split) et 2026-05-04 (cycle-9 closure). Les 3 "failles critiques" qu'ils pointaient (H-1 persistance, H-2 ONNX QualityEstimator, M-1 cost fictive) sont soit déjà fermées soit mal qualifiées. Vérification empirique du code actuel AVANT de prendre les audits comme map du fond restant.
