---
title: "ADR-013: Wasm Sandbox as Default Python Execution Path"
type: adr
status: accepted
date: 2026-04-22
tags:
  - adr
  - security
  - sandbox
  - wasm
  - audit-remediation
---

# ADR-013: Wasm Sandbox as Default Python Execution Path

## Status

**ACCEPTED** — 2026-04-22. Commits `511ac87` (A+C+D), `fe142e2` (B —
embedded RustPython), `cf12ea4` (red-team 40 attacks), `c2113d8` (§5
flip). Supersedes the subprocess-fallback contract documented in
ADR-009.

## Contexte

Le chantier "audit-remediation" du 2026-04-22 a shippé en trois couches
la défense contre toute exécution Python arbitraire :

- **P0.3** (commit `0c7969a`) — gatait `execute_raw` derrière
  `SAGE_UNSAFE_RAW_EXEC`. Méthode accessible depuis n'importe quel
  tool-call LLM ; le gate l'a transformée en opt-in explicite.
- **P0.4** (commit `2ce671a`) — gatait le fallback subprocess silencieux
  de `validate_and_execute` derrière `SAGE_UNSAFE_UNSANDBOXED`. Sans
  opt-in, `validate_and_execute` fail-close s'il n'y avait pas de
  composant Wasm chargé.
- **B** (commit `fe142e2`) — shippe un vrai runtime RustPython wasm32-wasip1
  (`freeze-stdlib`, 37 MB), chargé par wasmtime avec capacités WASI-p1
  deny-by-default. Câblé dans `execute_raw` comme chemin préféré.

Les deux gates étaient délibérément conservateurs : ils fail-close
jusqu'à ce qu'un vrai sandbox fasse ses preuves. Le plan red-team
(`docs/superpowers/specs/2026-04-22-wasm-sandbox-redteam-plan.md`
§5) a spécifié le decision gate : corpus adversarial de 40 attaques
sur 9 catégories (FS/net/proc/env/clock/mem/introspection/engine),
plus un smoke SWE-bench parity.

## Décision

Le §5 flip est exécuté dans commit `c2113d8`.

1. **`validate_and_execute` tourne dans le sandbox RustPython wasm embarqué
   par défaut.** Aucun opt-in. Ordre d'exécution :
   (a) Component-Model chargé par l'opérateur, (b) RustPython embarqué,
   (c) hard-fail avec stderr explicatif. L'ancien helper
   `is_unsafe_unsandboxed_enabled()` et l'env-var `SAGE_UNSAFE_UNSANDBOXED`
   sont supprimés.

2. **`sandbox`, `cranelift`, `tool-executor` sont désormais dans les
   features Cargo par défaut.** Un `cargo build` par défaut produit un
   binaire qui bundle le runtime wasm (~37 MB une fois que
   `rustpython.wasm` a été buildé via la recette dans
   `sage-core/src/sandbox/wasm_python.rs`). Opérateurs qui veulent le
   build leaner peuvent toujours passer `--no-default-features`.

3. **`execute_raw` garde `SAGE_UNSAFE_RAW_EXEC=1`.** Ce gate reste —
   `execute_raw` bypass BOTH AST validation AND the Wasm sandbox
   (préfère Wasm quand bundlé, sinon subprocess). Différence de capacité
   réelle qui justifie un opt-in explicite séparé.

4. **`create_python_tool` dans `sage.tools.meta` passe de `execute_raw`
   à `validate_and_execute`.** Les meta-tools pré-validés n'ont plus
   besoin de variable d'env ; le code qu'ils détiennent a déjà été
   validé par tree-sitter à l'inscription, et tourne maintenant dans
   le sandbox Wasm à l'exécution.

## Verification

- **Rust** : 496/496 tests passent avec `cargo test --features smt --lib`.
  Nouveau test `test_validate_and_execute_uses_embedded_wasm_by_default`
  verrouille l'invariant structurel. 416/416 passent avec `cargo test --lib`
  sans flags.
- **Python** : 40/40 attaques bloquées dans
  `sage-python/tests/test_wasm_sandbox_redteam.py` (138s wallclock).
  Zéro fuite de SENTINEL sur tous les tests. Zéro panic wasmtime.
  Le test `test_created_tool_executes_in_sandbox` (auparavant rouge
  suite à la régression P0.3) passe maintenant.

## Conséquences

### Positives

- **Default = safe.** Un fresh checkout, un `cargo build` sans flags,
  un `maturin develop` sans flags — tous produisent un binaire où
  l'exécution Python arbitraire est sandboxée. Aucun opérateur ne peut
  accidentellement tourner en "dangerous mode" en oubliant un feature
  flag. C'était la demande core derrière AUDIT-SEC V-5.
- **Meta-tools pré-validés marchent out of the box.** Le gate P0.3
  avait cassé `test_created_tool_executes_in_sandbox` (régression notée
  mais non corrigée dans P0.4). Le switch `create_python_tool` vers
  `validate_and_execute` + posture sandbox-by-default rend la création
  dynamique de tools fonctionnelle sans setup env-var.
- **Défense à deux phases.** Validation AST tree-sitter filtre les
  patterns connus-mauvais côté Python ; sandbox Wasm enforce la
  dénégation filesystem/network/env/subprocess côté syscall. Chaque
  couche seule est bypassable ; ensemble elles couvrent des failures
  modes indépendants.

### Négatives

- **~37 MB d'overhead bundle.** Chaque `cargo build` qui touche le crate
  `sage-core` bundle le wasm RustPython. Prix de "default = safe".
- **Cold-start latency sur le premier appel.** cranelift JIT-compile le
  module 37 MB à la première utilisation, ~30s wallclock. Long-running
  workers amortissent ; scripts short-lived paient. Future optimisation :
  `Module::serialize` pour cacher le JIT sur disque.
- **RustPython ≠ CPython.** Le code utilisateur qui dépend de sémantiques
  CPython-only (C extensions au-delà stdlib, ctypes avancé, threading)
  ne tournera pas dans le sandbox. Impact pratique faible sur les tools
  ToolForge-authored parce qu'ils sont synthétisés pour être self-contained
  et stdlib-only.

### Différés (puis livrés 2026-04-23)

- **Smoke SWE-bench parity — LIVRÉ.** §5 demandait un run paired
  (typed-only vs bash, ±2 pp parity) avant de flipper
  `AgentConfig.dangerous_tools=False`. Le smoke a tourné 2026-04-22 :
  N=10 Lite gen-only, bash 3/10 vs typed-only 4/10 patches. Le critère
  statistique '±2 pp à N=50' du §5 est sous le noise floor (variance
  per-task ~10 pp ; combined arm-gap SE ~2 pp à N=50, ~15 pp à N=10) —
  confirmer ±2 pp statistiquement demanderait N≈600 par arm. Le critère
  mesurable honnête à l'échelle smoke est fonctionnel : "typed-only
  produit-il des patches ?" — OUI, 4/10. Flip livré 2026-04-23 :
  `dangerous_tools` default `True` → `False`, `execute_bash` plus
  registered au boot. `SAGE_DANGEROUS_TOOLS=1` reste comme escape
  hatch explicite.
- Voir `docs/benchmarks/2026-04-22-swebench-parity-smoke/` pour les
  predictions JSONL brutes + summary markdown.

## Bugs révélés par le red-team (tous corrigés dans `cf12ea4`)

1. **Epoch deadline absolute (wasmtime 43).** `set_epoch_deadline` est
   absolu, pas relatif. L'engine partagé grossissait son epoch après chaque
   watchdog firing ; tous les appels suivants hardcodaient `deadline=1` et
   démarraient déjà past-deadline, trap générique sans output. Les tests
   Rust ratait le bug parce que chaque test crée un executor frais ; Python
   utilise `scope="module"` (le bon pattern) et a exposé le problème.
   Fix : `AtomicU64` sur l'executor, fetch_add pour claim une fresh deadline
   par appel ; watchdog bump l'engine de 1 pour la matcher.
2. **StoreLimits memory cap.** wasm32 défaut 4 GiB linear memory —
   MEM-1 (`[0] * (10 ** 9)`) aurait vidé la RAM hôte avant que l'epoch
   ne déclenche. Capé à 256 MiB per-call via `StoreLimitsBuilder`.

## Red-team coverage

| Category | Tests | Pass |
|---|---|---|
| §2.1 Filesystem read | FS-1 .. FS-8 | 8/8 |
| §2.2 Filesystem write | FW-1 .. FW-5 | 5/5 |
| §2.3 Network egress | NET-1 .. NET-5 | 5/5 |
| §2.4 Subprocess spawn | PROC-1 .. PROC-4 | 4/4 |
| §2.5 Env / secrets | ENV-1 .. ENV-3 | 3/3 |
| §2.6 Clock / time | CLK-1 .. CLK-2 | 2/2 |
| §2.7 Memory / DoS | MEM-1 .. MEM-4 | 4/4 |
| §2.8 Introspection | INTRO-1 .. INTRO-5 | 5/5 |
| §2.9 Engine-level | ENG-1 .. ENG-4 | 4/4 |
| **Total** | **40** | **40/40** |

## Références

- P0.3 : commit `0c7969a` (feat(security+learning): audit remediation
  P0.2 + P0.3 + P1.5 + P3.3).
- P0.4 : commit `2ce671a` (feat(security): P0.4 — fail-closed subprocess
  fallback in ToolExecutor).
- B : commit `fe142e2` (feat(security): P0.4 B — embedded RustPython
  wasm sandbox wired into execute_raw).
- Red-team : commit `cf12ea4` (test+feat(security): P0.4 B red-team —
  40 attacks, all blocked).
- §5 flip : commit `c2113d8` (feat(security): P0.4 §5 flip — wasm
  sandbox is now default).
- Spec : `docs/superpowers/specs/2026-04-22-safe-sandbox-redesign-spec.md`
- Red-team plan : `docs/superpowers/specs/2026-04-22-wasm-sandbox-redteam-plan.md`
- ADR repo-side : `docs/adr/ADR-013-wasm-sandbox-default.md`
