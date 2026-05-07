---
name: Verify external audits empirically against current code BEFORE acting on them
description: Standing directive — audits externes peuvent être stale. Yann challenge 2026-05-07 sur "tu n'as pas avancé sur le fond". Vérifier code actuel avant traiter audit comme map du fond restant.
type: feedback
originSessionId: ea52b6a4-977f-4cac-83f6-f491b3dad1c2
---
Yann challenge directive 2026-05-07: "j'ai comme l'impression que tu n'as pas beaucoup avancé sur le fond technique" après ~8 commits de doc/test pinning. Provoqué une enquête empirique des 3 "failles critiques" listées par AUDIT/Mini.md + AUDIT/YGN-SAGE_Analyse_Critique.md du jour. Toutes les 3 étaient stale ou mal qualifiées.

**Why:** External audits (cgpro analysis, Anthropic-style critiques, kimi/grok reviews) sont produits depuis le clone du repo + git log à un instant T. Si un cycle de fixes a shipped entre la date de génération de l'audit et le moment où on agit dessus, les claims de l'audit peuvent être déjà fermées sans qu'il le sache. Un audit qui dit "perdue au restart" pour une feature qui est default-on depuis 15 jours est noise, pas signal.

**How to apply:**

Avant d'engager un cycle dev sur une "faille critique" identifiée par un audit:

1. **Lire les commits récents touchant la zone concernée**. `git log --since="<audit date - 2 weeks>" -- <area>`. Si l'audit dit "X is missing" et qu'un commit récent dit "feat(X): ship X", l'audit est stale.

2. **Run le code actuel pour vérifier le claim empiriquement**. Si "persistence est perdue", load + save + reload + verify in-memory state matches. Si "cost is fictive", read le code de cost tracking et trace le chemin runtime vs pre-exec.

3. **Vérifier les feature flags vs default**. Beaucoup d'audits citent "feature behind flag X" mais oublient que X est dans `default = [...]` du Cargo.toml depuis cycle Y. Le commentaire AUDIT4 bug #6 dans Cargo.toml lines 75-99 est une preuve documentary que la "faille perdue au restart" était fermée 2026-04-22.

4. **Triage des findings de l'audit en 3 buckets**:
   - **Confirmed live gap**: vrai trou, ouvre un ticket
   - **Already-closed gap**: l'audit cite un état antérieur, juste documenter dans le ticket "closed by commit Z"
   - **Misqualified**: l'audit conflate 2 mécanismes (ex M-1 conflate pre-exec predictor vs post-exec real tokens). Documenter la distinction.

5. **Ne pas pivoter sur un audit avant le triage**. Si je commence à coder une "fix" pour une faille déjà fermée, c'est du temps perdu + churn pour le repo.

**Examples:**

Cycle 2026-05-07 routing claims pinning. Yann a challengé après ~8 commits "tu n'as pas avancé sur le fond". J'ai investi 30min sur l'enquête empirique:

- **H-1 "Persistance bandit/MAP-Elites perdue au restart"** (Mini.md critique #1): `cognitive` feature in default depuis 2026-04-22 P3.4 (Cargo.toml lines 75-99 commentaire AUDIT4 bug #6 explicite). `boot_topology.py:164` load au boot, `:191` atexit save, `pipeline_v2/learn.py:319` flush mid-pipeline avec `ensure_clean_epoch_before_save`. Tests existent: `test_engine_persistence.py::TestBanditSqlitePersistence` + `test_bandit.rs::test_bandit_save_load_round_trip` + `wheel_smoke.py::_check_save_state_manifest_contract`. **Already-closed gap**.

- **H-2 "ONNX QualityEstimator non livré"**: TRUE mais déjà gated — `SAGE_QUALITY_ONNX=1` opt-in (cycle-10 P7), capability table `planned, not shipped`. **Already-handled gap**.

- **M-1 "Coût budgétaire fictif $0.001 * n_nodes"**: TRUE for pre-exec only. Runtime cost = `AgentLoop.total_cost_usd` via `response.usage.prompt_tokens + completion_tokens × cards.toml rate` (real tokens). P1.6 audit remediation a séparé pre-exec PREDICTOR de post-exec ACTUAL. **Misqualified gap**.

Les 3 "failles critiques" ne nécessitaient aucun code change. Yann challenge était fair — j'aurais dû faire ce triage AVANT de me lancer sur le routing claims pinning, ou au moins en parallèle. Pas après ~8 commits.

**Methodology cgpro pour eviter ce trou** (suggéré par cgpro round-2 sur conv `cgpro_routing_claims_evidence_20260507`): toujours faire le `git log --since` + grep sur le code actuel AVANT d'envoyer un DESIGN_LOCK basé sur claims d'audit. cgpro lui-même ne pull pas les commits récents quand on lui donne juste le repo URL — il faut explicitement lui donner le `git log --oneline | head -50` dans le prompt si on veut qu'il triage.

**Anti-pattern à éviter**: prendre une liste de "failles critiques" comme map du fond restant sans vérification. Le fond technique est dans `roadmap.md` Horizon A/B + cycle-13 K queue, pas dans les audits externes du jour. Les audits sont commentary, pas roadmap.
