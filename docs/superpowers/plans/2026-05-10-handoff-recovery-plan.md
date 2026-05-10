# YGN-SAGE — Plan de récupération handoff harness/codex (2026-05-10)

**Status**: roadmap living document. Update as blocks ship.
**HEAD au plan**: `dbc76813536ac8919ec59d52bf8ec8f00e8a0bfd` (2026-05-10 16:40, "fix(providers): refresh model catalog evidence")
**Branche**: `main`, aligned with `origin/main`
**Working tree**: dirty (CGPRO.md, Cargo.lock + 14 dossiers `docs/benchmarks/2026-05-{08,09,10}-*` non suivis)

## Origine & contexte

Ce plan synthétise deux consultations cgpro back-to-back sur la même conversation `cgpro_ygn_sage_global_analysis_20260510` (id ChatGPT `6a00a5a1-96e8-8396-ad88-46d0c6b46623`):

- **cgpro #1 — analyse globale + sous-systèmes + interdépendances** — archive `.tmp/cgpro_global_analysis_20260510_response.md` (37KB)
- **cgpro #2 — findings 48h + roadmap structurée** — archive `.tmp/cgpro_roadmap_response_20260510.md` (37KB)

Findings 48h analysés (commits `beb0889b` → `dbc76813`):
- **Phase 1 (Claude harness MiniMax-M2.7 / DeepSeek-v4-pro, ~38 commits)**: REVIEW.md audit P0/P1 (engine bypass `e009cd9e`, ToolCapability.DANGEROUS missing `f01c8e0c`, SSL global bypass éliminé `e4f5b39a`+`b901ce4c`+`8e8b6aaf`, OpenAI gpt-5.5-pro regex `c5605439`, oracle/bandit health counters `19b657b7`, prompt-injection ingress `83938ada`, ModelAssigner top-3 trace via PyO3 `d221c3ad`, claims_audit `--resolve-pytest --resolve-git` `f7cf2988`).
- **Phase 2 (codex GPT-5.5 xhigh, ~42 commits)**: provider policy 3-couches (537 LOC pipeline_v2 + 303 LOC Rust + 153 LOC bootstrap), learning side-effect ledger v0 (`65f7c5c5` 1456 LOC, schema + writer + validate + 6 side_effects + 4 decisions + 24 reason_codes), SMMU persistence cross-restart (`a75af341` 526 LOC), pi-ygn-sage TS bridge strict JSONL (`9db4380f`), centralisation OpenAI Responses routing (`83e4e036`), A14 Windows atomic rename + e2e isolation (`9175eeaf`).

**Protocole cgpro respecté**: 14+ DESIGN docs sous `.tmp/cgpro_*_design_2026-05-09/10*` confirment cgpro DESIGN avant chaque feature majeure.

**Problèmes 48h identifiés**:
- 🔴 CI rouge `rust-core-coverage` : `sage-core/tests/test_model_card.rs:39,75,107` + `_load_real_cards_toml_helper` construisent `ModelCard` LITTÉRALEMENT sans les 6 nouveaux champs `runtime_*` ajoutés à `model_card.rs:163-200` dans `dbc76813`.
- 🔴 N=5 réel canary (`ec0b775e`, 2026-05-10 09:01-09:11): 5/5 timeouts à 120s, 0 patches, `model_id_final=null` ET `provider_final=null` pour TOUTES — providers JAMAIS appelés. canary_decision=NO_GO. provider_audit_failed avec `missing_provider_or_model` pour 5/5.
- 🔴 SWE-bench Pro grader bloqué local: host disk 11.14GB / 120GB requis, `external/SWE-bench_Pro-os` dirty (`M swe_bench_pro_eval.py`), Modal CLI installé mais pas authentifié.
- 🟡 CI band-aid rounds 8 mai (`31e4ba3c → 06c891a3 → d4064a9e` avant `f1c46dff` cgpro systemic CI remediation). Pattern interdit par `feedback_no_sweeping_under_rug.md`.
- 🟡 Bundle commit HEAD `dbc76813`: 46 fichiers / 2434 insertions sous label "refresh model catalog evidence" — difficile à rollback chirurgicalement.
- 🟡 MiniMax: `models.list` retourne `count=0` mais smoke direct OK — discoverability caveat.
- 🟡 Working tree dirty: `M CGPRO.md` + `M Cargo.lock` + 14 répertoires `docs/benchmarks/2026-05-*` non suivis.

## Verdict cgpro #1 — synthèse (full archive en .tmp)

Architecture macro = **verified adaptive orchestration runtime** confirmé. Pipeline cognitif `CLASSIFY → DECOMPOSE → SELECT_TOPOLOGY → ASSIGN_MODELS → EXECUTE → LEARN` correct.

**Verdict piliers** :
- **Strategy** = le plus empirique aujourd'hui (kNN/SystemRouter delivered, bandit attribution delivered)
- **Topology** = partiel (6-path engine evidence_pending, divergence templates 11 vs 12)
- **Tools** = boundary fort, sémantique faible (capability déclarative seule)
- **Memory** = persist S-MMU delivered, qualité long terme non démontrée
- **Evolution** = mécanisme oui, gain non démontré sur canaries actuels

**Maturité 10 invariants ledger** :
- HAUTE : I1 event payload, I2 oracle evidence, I3 posterior epoch, I6 bandit attribution, I9 CLI protocol
- MOYENNE-HAUTE : I5 RunFrame, I10 tool capability
- MOYENNE : I4 contaminated backup, I7 timeout enforcement, I8 control-surface

**6 invariants candidats proposés** (gap declared→verified) :
- I11 Provider declaration & execution grant
- I12 Model catalog freshness/provenance
- I13 Learning side-effect completeness
- I14 Benchmark claim hygiene
- I15 UI observe-only enforcement
- I16 Provider transcript integrity

**Top-15 couplings critiques** (cf archive). Le plus fragile : **provider selection → execution** (3 couches + monde externe instable).

**Non supprimables (niche claim)** : ledger runtime, oracle/evidence, bandit attribution, A14/posterior manifest, provider policy, sandbox/tool grant, CLI protocol.

## Roadmap

22 blocs, format CGPRO.md, ordonnés dépendances-aware. **NEXT_BLOCK_ID=clean-room-worktree-and-artifacts**.

---

## §A BLOCAGES IMMÉDIATS (≤ 24h, AVANT tout autre cycle)

### A1. `clean-room-worktree-and-artifacts` ⭐ NEXT
- **PRIORITY**: 1
- **GOAL**: Assainir le working tree pour que les prochains résultats CI/bench soient attribuables à un commit propre.
- **ALLOWED_FILES**: `CGPRO.md`, `Cargo.lock`, `docs/benchmarks/2026-05-{08,09,10}-*/**`, `.tmp/cgpro_*`, `docs/status/current.json` lecture seule sauf snapshot explicite
- **FORBIDDEN_SCOPE**: Ne pas modifier sage-core/, sage-python/, clients/, ui/, sage-discover/, cards.toml, ni les contrats runtime
- **REQUIRED_TESTS**: `git status --short` ; `git diff -- CGPRO.md Cargo.lock` ; `git ls-files --others --exclude-standard docs/benchmarks` ; `git rev-parse HEAD` ; `git merge-base --is-ancestor origin/main HEAD`
- **STOP_CONDITIONS**: CGPRO.md classé intentionnel ou stashed ; Cargo.lock justifié OU revert ; artefacts benchmark classés keep/quarantine/delete
- **ACCEPTANCE_GATE**: `.tmp/clean_tree_decision_20260510.md` liste chaque fichier avec décision explicite ; `git status --short` ne contient plus que les fichiers acceptés pour la suite
- **EFFORT**: 1h
- **DEPENDENCIES**: []
- **RISK**: low
- **LINKED_INVARIANT**: none
- **RATIONALE**: Pattern declared≠verified étendu — des artefacts NO_GO non classés violent l'esprit du ledger

### A2. `fix-model-card-ci-red`
- **PRIORITY**: 2
- **GOAL**: Réparer CI rouge Rust en mettant à jour les fixtures d'intégration ModelCard
- **ALLOWED_FILES**: `sage-core/tests/test_model_card.rs` ; éventuellement `sage-core/src/routing/model_card.rs` SEULEMENT pour helper factory partagée
- **FORBIDDEN_SCOPE**: Pas toucher cards.toml, provider policy, model routing, coûts, scores, runtime selection semantics, ni les nouveaux champs `runtime_*` eux-mêmes
- **REQUIRED_TESTS**: `cargo test -p sage-core --test test_model_card` (avec et sans features) ; `cargo test --list --features smt,cognitive,sandbox,cranelift,tool-executor | rg "model_card"`
- **STOP_CONDITIONS**: Tests construisent ModelCard avec les 6 champs `runtime_*` OU via builder/helper partagée ; aucune assertion supprimée
- **ACCEPTANCE_GATE**: rust-core-coverage local passe ; diff = fix chirurgical fixture, pas affaiblissement modèle
- **EFFORT**: 1h
- **DEPENDENCIES**: [A1]
- **RISK**: low
- **LINKED_INVARIANT**: none
- **RATIONALE**: 6 champs `runtime_selectable, runtime_settings, runtime_replacement, runtime_replacement_settings, runtime_evidence, runtime_retire_after` ajoutés `model_card.rs:163-200`

### A3. `unblock-swebench-pro-grader`
- **PRIORITY**: 3
- **GOAL**: Rendre possible un vrai grading SWE-bench Pro via 1 des 3 voies : libérer ≥120GB local, OU authentifier Modal, OU runner Linux cloud
- **ALLOWED_FILES**: `sage-python/src/sage/bench/swebench_pro_grader_preflight.py`, `scripts/run_dryrun_arm_d.py`, `docs/benchmarks/*grader-preflight*.json`, `external/SWE-bench_Pro-os/**` uniquement pour revert/clean
- **FORBIDDEN_SCOPE**: Pas modifier le runtime YGN-SAGE pour "passer" le preflight ; pas modifier les résultats du grader ; pas committer secrets Modal ; pas ignorer le repo grader dirty
- **REQUIRED_TESTS**: `python -m sage.bench.swebench_pro_grader_preflight` ; `docker system df` ; `docker run hello-world` ; `modal token info` si voie Modal ; smoke N=1 prediction vide
- **STOP_CONDITIONS**: Preflight ne retourne plus NO_GO_GRADER_REPO_DIRTY / host_disk_below_swebench_minimum / modal_not_authenticated
- **ACCEPTANCE_GATE**: `docs/benchmarks/<date>-grader-preflight-*.json` avec décision GO + preuve grading sur prediction minimale
- **EFFORT**: 3h
- **DEPENDENCIES**: [A1]
- **RISK**: medium
- **LINKED_INVARIANT**: I7

### A4. `provider-gate-no-go-forensic-repro`
- **PRIORITY**: 4
- **GOAL**: Reproduire en minimal le `provider_gate=NO_GO missing_provider_or_model` ; isoler entre assigner / pool / policy / canary runner
- **ALLOWED_FILES**: `sage-python/src/sage/pipeline_v2/provider_policy.py`, `assign_models.py`, `runtime_events.py` ; `sage-python/src/sage/llm/provider_pool.py` ; `sage-core/src/routing/model_assigner.rs` ; `scripts/run_dryrun_arm_d.py` ; tests provider/assigner
- **FORBIDDEN_SCOPE**: Pas bypasser policy ; pas forcer provider en dur dans runner ; pas rendre missing_provider_or_model silencieux
- **REQUIRED_TESTS**: `pytest sage-python/tests -k "provider_policy or assign_models or provider_pool"` ; `cargo test model_assigner` ; dry-run N=1 avec event log complet ; dry-run avec/sans allowlist
- **STOP_CONDITIONS**: DAG non vide → chaque `model_assigned` a model_id non vide et provider_id résolu avant EXECUTE, OU event failure explicite
- **ACCEPTANCE_GATE**: Artefact forensic montre la chaîne `routing_decision → topology_selected → model_assigned → provider_policy_decision → provider_call_attempt` ou point d'arrêt exact
- **EFFORT**: 4h
- **DEPENDENCIES**: [A2]
- **RISK**: high
- **LINKED_INVARIANT**: none (préfigure I11)
- **RATIONALE**: 5/5 canaries N=5 ont fini avec model_id_final=null et provider_final=null

### A5. `canary-stage-timing-triage`
- **PRIORITY**: 5
- **GOAL**: Distinguer "timeout 120s trop court" / "hang/deadlock avant provider" / "scoring/boot impossible"
- **ALLOWED_FILES**: `scripts/run_dryrun_arm_d.py` ; `sage-python/src/sage/bench/{watchdog,event_ledger}.py` ; `sage-python/src/sage/cli/run.py` ; `docs/benchmarks/*canary*.json` ; tests
- **FORBIDDEN_SCOPE**: Pas augmenter timeouts globalement pour masquer un deadlock ; pas compter run sans provider invocation comme tentative SWE-bench réelle
- **REQUIRED_TESTS**: N=1 dry-run avec timestamps par stage ; N=1 avec provider smoke court ; N=1 avec --no-grader ; pytest watchdog/event_ledger/cli_progress
- **STOP_CONDITIONS**: Chaque timeout canary catégorisé avec last_stage, elapsed_ms_by_stage, provider_attempted, model_id_final, provider_final, reason_code
- **ACCEPTANCE_GATE**: Les 5 timeouts historiques reclassés ; prochaine N=5 ne peut plus produire model_id_final=null sans event explicatif
- **EFFORT**: 3h
- **DEPENDENCIES**: [A4]
- **RISK**: medium
- **LINKED_INVARIANT**: I7

---

## §B SHORT-TERM (≤ 7 jours, fix CI propre + premier graded result + headline metric)

### B1. `canonical-ci-green-after-p0`
- **PRIORITY**: 1, **EFFORT**: 6h, **DEPS**: [A2, A1], **RISK**: medium, **INV**: I1
- **GOAL**: CI canonique verte sur commit propre après P0, sans band-aid ni suppression de tests
- **STOP**: Tous tests canoniques passent ; current.json distingue collect/list/pass ; aucun nouveau xfail/ignore ; tree clean
- **GATE**: Commit CI vert + artefact local/CI avec commandes exactes, durée, SHA, features Rust

### B2. `first-graded-swebench-pro-n5` ⭐ HEADLINE
- **PRIORITY**: 2, **EFFORT**: 8h, **DEPS**: [A3, A4, A5, B1], **RISK**: high, **INV**: I8
- **GOAL**: Premier résultat SWE-bench Pro gradé N=5+ du cycle-13, même si taux résolution faible
- **STOP**: Run contient instances_submitted, completed, resolved, unresolved, with_empty_patches, with_errors, resolution_rate
- **GATE**: Artefact gradé N=5+ avec commit SHA propre, provider/model non null pour chaque tentative, grader résultat ≠ NO_GO

### B3. `provider-execution-evidence-v0`
- **PRIORITY**: 3, **EFFORT**: 6h, **DEPS**: [A4, B1], **RISK**: medium, **INV**: I1
- **GOAL**: Preuve runtime minimale liant modèle assigné → provider résolu → provider appelé → réponse
- **STOP**: Run réel produit witness `assigned_model_id, resolved_provider_id, call_provider_id, call_model_id, call_started_ts, call_result_status`
- **GATE**: model_id_final=null/provider_final=null devient impossible sans event failure ou provider_execution_witness_missing

### B4. `swebench-pro-timeout-envelope`
- **PRIORITY**: 4, **EFFORT**: 4h, **DEPS**: [A5], **RISK**: medium, **INV**: I7
- **GOAL**: Séparer timeout "boot/provider smoke 120s" du timeout "long-horizon SWE-bench Pro graded"
- **STOP**: Runner expose 2 profils ; chaque timeout contient phase, elapsed, gate-quality status
- **GATE**: N=5 ne peut plus échouer 5/5 à 120s sans dire s'il a tenté provider

### B5. `claims-after-graded-evidence-only`
- **PRIORITY**: 5, **EFFORT**: 3h, **DEPS**: [B2], **RISK**: low, **INV**: I1
- **GOAL**: Update claims SEULEMENT après artefact gradé + audit strict résolu
- **STOP**: Chaque claim modifié pointe vers artefact réel, test collectable, commit ancestor
- **GATE**: `claims_audit --strict --resolve-pytest --resolve-git` vert + diff docs limité

### B6. `minimax-discoverability-guardrail`
- **PRIORITY**: 6, **EFFORT**: 3h, **DEPS**: [B1, B3], **RISK**: medium
- **GOAL**: Encadrer MiniMax-M2.7 comme runtime-selectable seulement si runtime_evidence explicite que `models.list=0` mais smoke direct fonctionne
- **STOP**: Carte MiniMax encode caveat dans `runtime_evidence` ; preflight distingue discoverable=false de callable=true

---

## §C MID-TERM (≤ 30 jours, niche claim consolidation, 11ème invariant, ablation graded N=10/N=50)

### C1. `invariant-11-provider-execution-witness`
- **PRIORITY**: 1, **EFFORT**: 12h, **DEPS**: [B3, B2], **RISK**: high
- **GOAL**: Formaliser et implémenter I11 "Provider execution witness"
- **STOP**: Tout appel provider possède `provider_decision_id` ; refusé si assigned/resolved/call diverge sans replacement déclaré
- **GATE**: Test adversarial reproduisant model_id=null échoue avec event explicite ; ledger documente I11

### C2. `topology-engine-bypass-regression-lock`
- **PRIORITY**: 2, **EFFORT**: 10h, **DEPS**: [B1], **RISK**: high, **INV**: I8
- **GOAL**: Verrouiller par tests que les 6 chemins TopologyEngine ne sont plus bypassés par select_topology pour hints standards
- **STOP**: Chaque hint attendu appelle `engine.generate()` OU exception testée ; bench TASK_END.control_surface complet pour DAG non vide
- **GATE**: Test reproduit régression P0-2 et échoue si engine.generate() n'est pas appelé

### C3. `graded-ablation-n10-n50` ⭐
- **PRIORITY**: 3, **EFFORT**: 24h, **DEPS**: [B2, C1, C2, B4], **RISK**: high, **INV**: I8
- **GOAL**: Ablation gradée N=10 puis N=50 comparant sequential baseline / topology engine adaptive / provider-policy on/off / learning audit-only
- **STOP**: Chaque instance a commit SHA, model/provider witness, topology witness, timeout, patch, grader, oracle/learn status
- **GATE**: Rapport ablation avec resolution, errors, empty patches, cost, latency, intervals ; aucun claim public ne dépasse l'ablation

### C4. `learning-side-effect-completeness-gate`
- **PRIORITY**: 4, **EFFORT**: 16h, **DEPS**: [C1, B1], **RISK**: high, **INV**: I2
- **GOAL**: Passer du sidecar audit-only à vérification complétude des sinks d'apprentissage
- **STOP**: Tout write vers bandit, MAP-Elites, archive, S-MMU learning, training memory émet side-effect record OU est explicitement non_learning
- **GATE**: Test "write sans sidecar" échoue ; "oracle abstain mais write tenté" échoue

### C5. `posterior-smmu-a14-regression-matrix`
- **PRIORITY**: 5, **EFFORT**: 12h, **DEPS**: [B1], **RISK**: medium, **INV**: I3
- **GOAL**: Matrice régression A14/SMMU/posterior epoch : fresh load / old checkpoint sans S-MMU / tamper SHA / stale binary / backup contaminé / Windows tmp orphan
- **STOP**: Chaque état incompatible cold-start ou fail-close selon contrat ; aucun état chaud ne survit à un load cold
- **GATE**: Matrice tests verte + artefact `smmu_state.json` hashé quand présent / ignoré proprement quand absent

### C6. `rust-first-hotpath-budget`
- **PRIORITY**: 6, **EFFORT**: 20h, **DEPS**: [C3], **RISK**: medium, **INV**: I6
- **GOAL**: Identifier et déplacer en Rust uniquement les hotpaths bloquants pour graded N=50, sans feature narrative
- **STOP**: Hotpath déplacé seulement si profil avant/après montre gain mesurable ; chaque PyO3 wrapper a test boundary
- **GATE**: Réduction mesurée latence/variance N≥10 sans régression I1, I6, I8, I11

---

## §D LONG-TERM (≤ 90 jours, scaling, opt-in features, niche claim production)

### D1. `production-provider-registry-and-catalog`
- **PRIORITY**: 1, **EFFORT**: 32h, **DEPS**: [C1, B6], **RISK**: high
- **GOAL**: Model cards en catalogue runtime vérifiable avec freshness, endpoint smoke, replacement semantics, provider witness complet (préfigure I12)
- **GATE**: Aucun modèle ne peut être choisi pour exécution fraîche si runtime_selectable=false / stale / provider denied / smoke obligatoire manquant

### D2. `scaled-swebench-pro-eval-platform`
- **PRIORITY**: 2, **EFFORT**: 40h, **DEPS**: [C3, A3], **RISK**: high, **INV**: I7
- **GOAL**: Plateforme évaluation répétable N=50/N=100+ avec local Docker ou Modal/cloud, artefacts hashés, coûts suivis
- **GATE**: Tableau headline stable: resolution rate, patches empty, errors, mean cost, p95 latency, distribution provider/topology

### D3. `opt-in-online-evolution-production-gate`
- **PRIORITY**: 3, **EFFORT**: 36h, **DEPS**: [C4, C5, C3], **RISK**: high, **INV**: I2
- **GOAL**: Évolution online opt-in production sans jamais apprendre depuis run non prouvé
- **STOP**: `should_evolve()` dépend de oracle trainable + attribution bandit + provider witness + timeout gate-quality + clean posterior epoch
- **GATE**: Ablation gain ou absence régression ; side-effects entièrement retraçables

### D4. `ui-adapter-observe-only-production-contract`
- **PRIORITY**: 4, **EFFORT**: 20h, **DEPS**: [B3, C1], **RISK**: medium, **INV**: I9 (préfigure I15)
- **GOAL**: ui/ et clients/pi-ygn-sage/ restent observe-only ou veto-only, sans souveraineté sur modèle/topo/learning gate
- **STOP**: Frontend peut cancel/deny/tighten budget ; ne peut pas loosen budget / override model / topology / oracle gate
- **GATE**: Test adversarial frontend essayant de muter modèle/topo/learn refusé

### D5. `strategic-simplification-without-sage-lite`
- **PRIORITY**: 5, **EFFORT**: 24h, **DEPS**: [D2, D3], **RISK**: medium
- **GOAL**: Supprimer/parquer doublons qui n'aident pas "show why", sans pivot Sage-Lite ni extraction sous-produit
- **FORBIDDEN**: Ne pas supprimer ledger, oracle/evidence, bandit attribution, provider policy, sandbox, A14/SMMU, CLI protocol
- **GATE**: Surface réduite sans perte invariants, sans baisse headline metric, sans nouvelle divergence Rust/Python

---

## Exécution

**Discipline par bloc** :
1. Re-vérifier pre-conditions (DEPENDENCIES done)
2. Faire le travail dans ALLOWED_FILES strictement
3. Run REQUIRED_TESTS
4. Vérifier STOP_CONDITIONS et ACCEPTANCE_GATE
5. Commit avec message conventionnel `<type>(<scope>): <subject>` + body référençant BLOCK_ID
6. `python scripts/narrative_guard_phase22.py` + `python scripts/sync_doc_counters.py --check`
7. cgpro VERIFY pre-commit (`--resume cgpro_ygn_sage_global_analysis_20260510`) si bloc impacte runtime contracts
8. Push + cgpro post-push report (même resume)
9. Mark block `[shipped]` ici + advance to next BLOCK_ID

**Pattern interdit** : sweeping under rug. Si un test échoue ou un fichier rouge, fix root cause (cf. `feedback_no_sweeping_under_rug.md`).

**External AI consultations** :
- cgpro (`cgpro_ygn_sage_global_analysis_20260510`) = stratégie / arbitrages risqués / verify post-bloc runtime
- codex (rescue agent) = quand bloc bloqué ou veut second avis implémentation
- advisor = check stratégique en cours

**Source of truth tests** : `docs/status/current.json` régénéré post-CI ou par `python scripts/status_snapshot.py`.

**Niche claim** : *"the coding agent that can show why it chose a topology, why it trusted or rejected a result, and why it did or did not learn from the run"*. Toute proposition diluant ce claim = HARD_NO.

---

**NEXT_BLOCK_ID = `clean-room-worktree-and-artifacts` (A1)**

Archive cgpro #1 + #2 dans `.tmp/cgpro_global_analysis_20260510_response.md` et `.tmp/cgpro_roadmap_response_20260510.md`. Conv ChatGPT id `6a00a5a1-96e8-8396-ad88-46d0c6b46623` (save name `cgpro_ygn_sage_global_analysis_20260510`) — resume pour VERIFY chaque bloc impactant runtime contracts.
