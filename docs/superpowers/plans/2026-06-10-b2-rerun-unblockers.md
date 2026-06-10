# B2_RERUN_UNBLOCKERS Implementation Plan (Phase 1 du master roadmap)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. TDD strict (superpowers:test-driven-development) : chaque fix commence par son test rouge.

**Goal:** Clore les 3 bugs de prod exposés par le canary N=5 du 2026-05-12 (provider `unknown` sur tier reasoner, perte de coût hors-timeout, diff-verifier silencieux) et produire le dry-run gratuit prouvant les 3 champs peuplés — condition d'entrée de la Phase 2 payante.

**Architecture:** Scope verrouillé par cgpro 2026-05-12 (`NEXT_BLOCK_ID=B2_RERUN_UNBLOCKERS_PROVIDER_COST_DIFF_VERIFIER`). allowed_files = `sage-python/scripts/run_dryrun_arm_d.py` + ses tests + `sage/**/{model,provider,routing}*.py` + tests provider/model + doc postmortem. Forbidden = toucher I-11/ledger/schemas, affaiblir provider_gate, allowlister `unknown`, désactiver les gates verifier, run payant, refonte prompt/topologie. Stop si un fix exige un changement de sémantique policy ou sort du périmètre runner+verifier.

**Tech Stack:** pytest (conventions de `tests/test_run_dryrun_arm_d.py`, 55 tests existants), cards.toml comme unique vérité provider (directive #7), `verify_diff_context_with_reasons` existant de `sage/bench/swebench_diff_verifier.py`.

**Reconnaissance 2026-06-10 (file:line vérifiés):**
- Bug 1 : le canary lit `provider_id`/`provider` depuis les événements (`run_dryrun_arm_d.py:666-675` + `:1074-1080`) ; l'émission runtime passe par `runtime_provider_id_for_model` (`pipeline_v2/runtime_events.py:207-224`) → `provider_pool.infer_provider(model_id)` puis fallback `pipeline.llm_config.provider` → `""`. Le littéral `"unknown"` n'est PAS dans le script canary — il vient d'amont. Fait discriminant du N=5 : `gemini-2.5-flash` et `gemini-3-flash-preview` ont résolu `google`, seul `gemini-3.1-pro-preview` (le modèle du tier reasoner) a donné `unknown` ⇒ la résolution n'est pas un simple substring ; suspect = chemin tier-explicite qui contourne le lookup registry.
- Bug 2 : hors timeout, `total_cost_usd` vient exclusivement du payload `cli_complete` (`run_dryrun_arm_d.py:1143-1147` → `:1990-1994`) ; la récupération depuis `_observed_event_cost_usd` (accumulé `:676-678`) n'existe QUE dans `_timeout_task_result` (`:1748-1777`, fix `eeb3a7fb`), de même que `cost_integrity_warning`.
- Bug 3 : architectural — l'env `SAGE_DIFF_VERIFIER_MODE=observe` EST transmis au subprocess (`:962-1007`), mais le verifier vit dans `SWEBenchBench._run_one_instance` (`swebench_bench.py:1315-1350`) que le canary n'instancie jamais ; le canary n'utilise que `_extract_patch` (`:1597-1609`) et écrit `_diff_verifier_outcome: None` en dur (`:1785`, `:2015`).

**Les 9 tests requis (contrat cgpro)** : (1) mapping `gemini-3.1-pro-preview`→`google` ; (2) provider_gate reste NO_GO sur `unknown` ; (3) provider_gate PASS sur exécution google+deepseek-only ; (4) coût d'un échec CLI hors-timeout récupéré depuis l'event audit ; (5) `_total_cost_usd_source` explicite ; (6) `SAGE_DIFF_VERIFIER_MODE` propagé au subprocess ; (7) patch+observe → `_diff_verifier_outcome` non-null ; (8) sans patch → `skipped_no_patch` explicite, PAS null ; (9) les 55 tests run_dryrun existants restent verts.

---

### Task 1: Bug 1 — provider du tier reasoner résolu depuis cards.toml

**Files:**
- Test: `sage-python/tests/test_run_dryrun_arm_d.py` (append) + test provider runtime si le fix touche `providers/*.py`
- Modify (selon localisation exacte du littéral) : `sage-python/src/sage/providers/registry.py` OU `providers/pool*.py` (`infer_provider`) — DANS le glob autorisé `sage/**/provider*.py`

- [ ] **Step 1.1: Localiser l'émetteur du littéral `unknown`**

```bash
cd /c/Code/YGN-SAGE/sage-python
grep -rn '"unknown"' src/sage --include="*.py" | grep -iE "provider" | head -20
grep -rn "def infer_provider" src/sage --include="*.py"
```

Attendu : le(s) site(s) où `provider_id` peut devenir `"unknown"` (ou la chaîne vide convertie en `unknown` côté canary/CLI). Identifier LE site du chemin reasoner (tier explicite via `SAGE_LLM_TIER` → `llm/router.py:25`).

- [ ] **Step 1.2: Écrire le test rouge — mapping reasoner**

Dans `test_run_dryrun_arm_d.py`, suivant la convention de `test_run_sage_cli_extracts_final_model_and_provider_from_events` :

```python
def test_reasoner_tier_model_resolves_google_provider() -> None:
    """B2 bug 1: gemini-3.1-pro-preview (tier reasoner) must resolve provider
    'google' from cards.toml, never 'unknown' (2026-05-12 canary task #3)."""
    resolved = run_dryrun_arm_d._resolve_provider_for_model("gemini-3.1-pro-preview")
    assert resolved == "google"
```

(Si le fix atterrit dans `infer_provider` runtime, ajouter le test jumeau dans le fichier de tests provider correspondant, même assertion via l'API publique du pool/registry.)

- [ ] **Step 1.3: Vérifier l'échec** — `python -m pytest tests/test_run_dryrun_arm_d.py::test_reasoner_tier_model_resolves_google_provider -q` → FAIL (AttributeError ou "unknown").

- [ ] **Step 1.4: Implémenter le fix minimal**

Contrainte directive #7 : la résolution DOIT lire cards.toml (catalog/registry déjà chargé au boot — `ModelCardCatalog`/`RustModelRegistry` via `boot_topology._find_cards_toml`), PAS un substring hardcodé. Forme attendue : dans le résolveur identifié au 1.1, avant le fallback `""`/`"unknown"`, interroger le catalog (`card = catalog.get(model_id)` → `card.provider`). Le canary (`run_dryrun_arm_d.py`) garde sa logique de LECTURE d'événements inchangée — le fix est dans la chaîne d'attribution, pas dans le gate.

- [ ] **Step 1.5: Vérifier le vert + non-régression gates provider** — tests (1)(2)(3) du contrat : le mapping passe, `test_run_provider_gate_blocks_denylisted_provider` + `test_run_provider_gate_uses_all_observed_providers` restent verts, et ajouter si absent :

```python
def test_provider_gate_still_no_go_on_unknown_provider() -> None:
    """B2 contract test 2: fixing attribution must NOT weaken the gate —
    an 'unknown' execution provider still yields NO_GO."""
    gate = run_dryrun_arm_d._provider_gate_result(
        observed_execution_providers={"unknown"},
        allowlist=("google", "deepseek"),
        denylist=(),
        runtime_policy_blocks=[],
    )
    assert gate["status"] == "NO_GO"
    assert gate["execution_outside_allowlist"] == ["unknown"]

def test_provider_gate_pass_on_google_deepseek_only() -> None:
    gate = run_dryrun_arm_d._provider_gate_result(
        observed_execution_providers={"google", "deepseek"},
        allowlist=("google", "deepseek"),
        denylist=(),
        runtime_policy_blocks=[],
    )
    assert gate["status"] == "PASS"
```

(Adapter les signatures aux helpers réels du script — lire `:1557-1642` avant écriture ; si le gate n'est pas factorisé en helper testable, le test passe par le chemin summary existant comme les tests gate actuels.)

- [ ] **Step 1.6: Commit** — `fix(b2): reasoner-tier provider attribution resolves from cards.toml`.

### Task 2: Bug 2 — récupération de coût généralisée hors-timeout + source explicite

**Files:**
- Test: `sage-python/tests/test_run_dryrun_arm_d.py` (append)
- Modify: `sage-python/scripts/run_dryrun_arm_d.py` (`_run_sage_cli` result `:1143-1147`, summary `:1990-1994`; factoriser la logique de `:1748-1777`)

- [ ] **Step 2.1: Tests rouges (contrat 4 + 5)** — modèles : `test_timeout_task_result_recovers_cost_from_node_completed_events` existant.

```python
def test_non_timeout_failure_recovers_cost_from_event_audit() -> None:
    """B2 bug 2: a hard CLI failure (no cli_complete) must surface the
    observed event cost, not $0 (2026-05-12 canary tasks #2/#3)."""
    summary = _summary_for(  # helper existant ou fixture équivalente du fichier
        cli_result={"total_cost_usd": None, "outcome": "failure"},
        event_audit={"_observed_event_cost_usd": 0.158,
                     "_execution_model_ids": ["gemini-2.5-flash"]},
    )
    assert summary["total_cost_usd"] == pytest.approx(0.158)
    assert summary["_total_cost_usd_source"] == "event_audit_observed_event_cost_usd"
    assert summary["cost_integrity_warning"] is not None

def test_success_underreport_prefers_larger_observed_cost() -> None:
    """B2 bug 2: success path where cli_complete under-reports vs event audit
    (tutanota db90ac26: $0.134 reported vs $0.266 observed)."""
    summary = _summary_for(
        cli_result={"total_cost_usd": 0.134, "outcome": "success"},
        event_audit={"_observed_event_cost_usd": 0.266},
    )
    assert summary["total_cost_usd"] == pytest.approx(0.266)
    assert summary["_total_cost_usd_source"] == "event_audit_observed_event_cost_usd"

def test_total_cost_source_is_cli_complete_when_consistent() -> None:
    summary = _summary_for(
        cli_result={"total_cost_usd": 0.099, "outcome": "success"},
        event_audit={"_observed_event_cost_usd": 0.099},
    )
    assert summary["_total_cost_usd_source"] == "cli_complete"
```

(`_summary_for` : si aucun helper n'isole la construction du summary, extraire d'abord un pur helper `_resolve_total_cost(cli_total, observed, had_llm_execution) -> tuple[float, str, warning|None]` et tester CELUI-LÀ — refactor minimal, comportement timeout inchangé.)

- [ ] **Step 2.2: Vérifier l'échec** des 3 tests.

- [ ] **Step 2.3: Implémenter** — un helper unique `_resolve_total_cost` utilisé par les TROIS chemins (success, failure, timeout) : règle = `max(cli_total or 0, observed)` MAIS jamais d'addition (stop-condition cgpro « double-count ») ; source = `cli_complete` si `cli_total >= observed` sinon `event_audit_observed_event_cost_usd` ; `cost_integrity_warning` (reason_code `llm_execution_observed_zero_cost` existant OU nouveau `cli_complete_cost_underreport`) émis quand observed > reported ou quand exécution LLM observée avec coût 0. Le chemin timeout migre sur le helper sans changement de comportement (ses 3 tests existants restent verts tels quels).

- [ ] **Step 2.4: Vert + non-régression** — les 3 nouveaux + `test_timeout_task_result_*` (3 tests) + `test_run_global_budget_*` (le budget global doit consommer le coût RÉSOLU, pas le sous-rapporté — vérifier le câblage à la lecture).

- [ ] **Step 2.5: Commit** — `fix(b2): non-timeout cost recovery from event audit + explicit cost source`.

### Task 3: Bug 3 — diff-verifier câblé dans la boucle canary

**Files:**
- Test: `sage-python/tests/test_run_dryrun_arm_d.py` (append)
- Modify: `sage-python/scripts/run_dryrun_arm_d.py` (point d'extraction patch `:1597-1609` + champs `:1785`/`:2015`)

- [ ] **Step 3.1: Tests rouges (contrat 6 + 7 + 8)**

```python
def test_diff_verifier_env_propagates_to_subprocess(monkeypatch) -> None:
    """B2 contract 6: launch env must carry SAGE_DIFF_VERIFIER_MODE=observe
    into the per-task subprocess env."""
    env = run_dryrun_arm_d._task_subprocess_env(tier="reasoner")  # helper réel :962-987
    assert env["SAGE_DIFF_VERIFIER_MODE"] == "observe"

def test_patch_with_observe_mode_yields_non_null_verifier_outcome(tmp_path) -> None:
    """B2 contract 7: a non-empty patch under observe mode must produce a
    non-null _diff_verifier_outcome in the prediction record."""
    record = run_dryrun_arm_d._annotate_diff_verifier(
        patch=_MINIMAL_VALID_DIFF, repo_dir=tmp_path, mode="observe"
    )
    assert record["_diff_verifier_outcome"] is not None

def test_no_patch_yields_explicit_skipped_no_patch() -> None:
    """B2 contract 8: absence of patch must be an explicit outcome, not null."""
    record = run_dryrun_arm_d._annotate_diff_verifier(
        patch="", repo_dir=None, mode="observe"
    )
    assert record["_diff_verifier_outcome"] == "skipped_no_patch"
```

(`_MINIMAL_VALID_DIFF` : fixture écrite contre un fichier réel créé dans `tmp_path` pour que le verifier matche le contexte — réutiliser le pattern des tests de `test_swebench_diff_verifier_*.py`.)

- [ ] **Step 3.2: Vérifier l'échec** (helpers inexistants).

- [ ] **Step 3.3: Implémenter `_annotate_diff_verifier`** — nouveau helper dans le script canary : entrées `(patch, repo_dir, mode)` ; `mode` lu une fois depuis l'env du LAUNCHER (réutiliser `_get_diff_verifier_mode` importé de `sage.bench.swebench_bench` — import déjà établi pour `_extract_patch`) ; sans patch → `{"_diff_verifier_outcome": "skipped_no_patch"}` ; mode off → `"skipped_mode_off"` ; patch + observe/repair → appel `verify_diff_context_with_reasons(patch, repo_dir)` et mapping `{outcome, mismatches, reasons}` vers les 3 champs `_diff_verifier_*` déjà déclarés dans `_PREDICTION_AUDIT_FIELDS:542`. `repair` reste observe-only ici (forbidden: changer les gates). Câbler au point `:1597-1609` (chemin nominal) ; le chemin timeout `:1785` passe à `"skipped_timeout"`.

- [ ] **Step 3.4: Vert + non-régression** — 3 nouveaux + `test_run_writes_predictions_jsonl_with_canary_audit_fields` (le champ change de `None` à valeur explicite : adapter SON assertion si elle pin `None` — c'est un changement de contrat voulu, le documenter dans le message de commit).

- [ ] **Step 3.5: Commit** — `fix(b2): wire diff-context verifier into canary loop (observe)`.

### Task 4: Clôture — suite complète, dry-run de preuve, postmortem, cgpro VERIFY, push

- [ ] **Step 4.1: Contrat 9** — `python -m pytest tests/test_run_dryrun_arm_d.py -q` → 55 anciens + ~8 nouveaux, 0 échec. Puis gates : `ruff check src/`, `mypy src/sage/ --ignore-missing-imports`, `claims_audit --strict`, `narrative_guard_phase22.py`.

- [ ] **Step 4.2: Dry-run gratuit de preuve** — `--mock` mode du script (existant) sur 1 tâche fixture : vérifier dans le summary produit `provider_final≠unknown`, `_total_cost_usd_source` présent, `_diff_verifier_outcome≠None`. Archiver sous `docs/benchmarks/2026-06-XX-b2-unblockers-dryrun/`.

- [ ] **Step 4.3: Postmortem doc** — `docs/benchmarks/2026-05-12-b2-n5-graded/postmortem-fixes.md` : tableau bug → cause racine → fix → test, lien vers les 3 commits.

- [ ] **Step 4.4: cgpro VERIFY pré-push** (sémantique runner) — nouvelle conv in-project `--new-session --save cgpro_b2_unblockers_verify` (directive 2026-05-06, pas de --resume) : diff summary + évidence TDD (rouge→vert par bug) + 3 points de scrutiny (gate non affaibli ? double-count ? scope verifier observe-only ?) + draft des messages de commit. Appliquer les EDIT_REQUIRED éventuels.

- [ ] **Step 4.5: Push + CI watch + statut** — push des commits B2 (+ roadmap commitée en attente), surveiller Security/CI, puis `status_snapshot.py` → `sync_doc_counters.py` si le compte de tests a bougé (+~8) → commit statut → re-push. Mettre à jour la mémoire projet (nouveau fichier `project_jun10_*`).

**Sortie de phase** : les 3 champs prouvés peuplés sur dry-run gratuit ⇒ demander le GO explicite à Yann pour Phase 2.a (canary N=5 payant ~$1).

---

## Self-Review

1. **Couverture contrat cgpro** : 9/9 tests mappés (1→T1.2, 2-3→T1.5, 4-5→T2.1, 6-8→T3.1, 9→T4.1). allowed_files respectés (script canary + tests + `provider*.py` si le fix runtime y atterrit) ; si le Step 1.1 localise le littéral `unknown` HORS du glob autorisé (ex. `pipeline_v2/runtime_events.py` ou `topology/runner.py`), STOP et soumettre l'amendement de scope à cgpro avant d'éditer (stop-condition « production CLI changes outside runner+verifier boundary »).
2. **No-placeholder** : les signatures `_summary_for`/`_provider_gate_result`/`_task_subprocess_env` sont des hypothèses de factorisation à confirmer à la lecture des lignes citées — le plan l'annonce explicitement à chaque step concerné (lecture avant écriture) ; les assertions et noms de tests sont définitifs.
3. **Cohérence types** : `_resolve_total_cost` retourne `(float, str, dict|None)` partout ; outcomes verifier = chaînes du vocabulaire existant de `swebench_diff_verifier` + 3 nouveaux `skipped_*` explicites.
