Tu es Codex 5.5 en reasoning xhigh. Objectif : intégrer OpenAI Symphony dans le développement du dépôt https://github.com/yannabadie/YGN-SAGE, branche main, sans casser le runtime SAGE.

Contexte non négociable :
- Symphony est une couche d’orchestration de développement : issue tracker → workspace isolé → Codex app-server → PR/revue/preuves.
- Ne remplace pas le pipeline SAGE `CLASSIFY → DECOMPOSE → TOPOLOGY → ASSIGN → EXECUTE → LEARN`.
- Ne modifie pas le runtime SAGE sauf nécessité démontrée.
- Respecte CLAUDE.md :
  - Rust first, Python tolerant.
  - Evidence before assertions.
  - No `verify=False`.
  - kNN/SystemRouter restent prioritaires.
  - Toute nouvelle étiquette qui autorise un side-effect doit être liée à une preuve vérifiée et documentée dans `docs/contracts/runtime-integrity-ledger.md`.
- Branche de travail : partir de `main`, créer une branche courte du type `feat/symphony-dev-orchestration`.

Mission :
1. Ajouter une intégration Symphony repo-owned pour YGN-SAGE.
2. Ajouter un `WORKFLOW.md` racine exploitable par Symphony.
3. Ajouter une documentation opérationnelle `ops/symphony/README.md`.
4. Ajouter un preflight local qui vérifie `codex`, `cgpro`, GitHub/Linear, env vars, workspaces, et l’absence de fuite de secrets.
5. Ajouter une intégration `cgpro` locale :
   - `cgpro` est trouvé localement sur le poste de l’utilisateur.
   - Source locale attendue : `C:\Code\CGPro4Code`, mais ne hardcode pas cette valeur comme unique vérité.
   - Détecte d’abord `CGPRO_BIN`, puis `command -v cgpro` / `where cgpro`, puis le chemin connu Windows.
   - Ne copie pas le plugin dans le repo.
   - Ne commit aucun secret ni profil navigateur.
6. Préparer le repo pour lancer Symphony contre YGN-SAGE `main`.

Livrables attendus :
- `WORKFLOW.md`
- `ops/symphony/README.md`
- `ops/symphony/env.example`
- `scripts/symphony_preflight.py`
- `scripts/symphony_cgpro.py`
- Optionnel si utile : `ops/symphony/run_symphony.sh` et `ops/symphony/run_symphony.ps1`
- Tests unitaires légers pour les scripts, si la structure existante le permet.
- Pas de gros vendoring de `openai/symphony` dans le repo au premier passage. Préférer :
  - soit un clone externe documenté,
  - soit un submodule `external/symphony` uniquement si tu prouves que c’est nécessaire et proprement piné.

Phase 0 — Audit initial :
- Lire `CLAUDE.md`, `README.md`, `SECURITY.md`, `docs/contracts/runtime-integrity-ledger.md`, `docs/contracts/rust-python-boundary.md`.
- Lire la spec Symphony et le README Elixir upstream.
- Vérifier si `WORKFLOW.md` existe déjà.
- Vérifier si `.agents/skills`, `.codex`, ou des scripts de dev-orchestration existent déjà.
- Produire dans ta réponse un court “integration surface map” :
  - fichiers à ajouter,
  - fichiers à ne pas toucher,
  - risques,
  - commandes de validation.

Phase 1 — WORKFLOW.md YGN-SAGE :
Créer un `WORKFLOW.md` racine avec YAML front matter + prompt Markdown.

Configuration cible de départ :
- Tracker : Linear par défaut, car Symphony upstream cible Linear.
- Ne pas bloquer le projet si Linear n’est pas configuré : documenter clairement l’option future “GitHub Issues adapter”.
- Concurrence initiale faible : `max_concurrent_agents: 1` ou `2`, car YGN-SAGE est lourd et mélange Rust/Python.
- Workspaces isolés sous `$SYMPHONY_WORKSPACE_ROOT` ou `~/code/ygn-sage-symphony-workspaces`.
- `codex.command` doit lancer Codex app-server avec GPT-5.5 xhigh.

Squelette attendu :

---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: $LINEAR_PROJECT_SLUG
  active_states:
    - Todo
    - In Progress
    - Rework
    - Merging
  terminal_states:
    - Done
    - Closed
    - Cancelled
    - Canceled
    - Duplicate

polling:
  interval_ms: 10000

workspace:
  root: $SYMPHONY_WORKSPACE_ROOT

hooks:
  timeout_ms: 180000
  after_create: |
    git clone --branch main https://github.com/yannabadie/YGN-SAGE.git .
    git status --short
  before_run: |
    git fetch origin main --prune
    git status --short
    test -f CLAUDE.md || exit 1
    python --version || true
    rustc --version || true
    cargo --version || true
  after_run: |
    git status --short || true

agent:
  max_concurrent_agents: 1
  max_turns: 20
  max_retry_backoff_ms: 300000
  max_concurrent_agents_by_state:
    merging: 1
    rework: 1

codex:
  command: codex --config shell_environment_policy.inherit=all --config 'model="gpt-5.5"' --config 'model_reasoning_effort="xhigh"' app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
---

Prompt body :
- Rappeler que l’agent travaille sur YGN-SAGE branche main.
- Rappeler les directives CLAUDE.md.
- Rappeler que toute livraison doit inclure preuve : tests, logs, ou raison explicite si un test est impossible.
- Rappeler que `cgpro` est requis pour DESIGN et VERIFY sur les changements substantiels.
- Forcer une boucle :
  1. Lire issue + repo context.
  2. Mettre à jour workpad.
  3. Si changement substantiel : lancer `cgpro DESIGN`.
  4. Implémenter.
  5. Valider localement.
  6. Si changement substantiel : lancer `cgpro VERIFY`.
  7. Corriger si pushback.
  8. Préparer PR/handoff.
- Définir “substantiel” :
  - runtime SAGE,
  - routing,
  - topology,
  - provider logic,
  - memory/evolution,
  - sandbox/security,
  - benchmarks,
  - side-effect gating,
  - workflow Symphony lui-même.

Phase 2 — cgpro local :
Créer `scripts/symphony_cgpro.py`.

Fonction :
- Construire un prompt Markdown temporaire dans `.tmp/cgpro/`.
- Inclure :
  - repo URL,
  - branche,
  - commit SHA,
  - résumé de ticket,
  - diff résumé ou patch ciblé,
  - questions précises.
- Appeler :
  `cgpro ask --json --background --timeout 1800 "$(cat .tmp/cgpro/<file>.md)"`
- Ne jamais utiliser `--no-stream`.
- Capturer NDJSON dans `.tmp/cgpro/<ticket>-<phase>.jsonl`.
- Capturer une synthèse lisible dans `.tmp/cgpro/<ticket>-<phase>.md`.
- Retourner un code non zéro uniquement si :
  - `cgpro` absent,
  - timeout,
  - aucune sortie exploitable,
  - verdict explicite “blocker/pushback” si le mode est `VERIFY --strict`.

CLI proposée :
- `python scripts/symphony_cgpro.py design --ticket YGN-123 --title "..." --summary-file .tmp/summary.md`
- `python scripts/symphony_cgpro.py verify --ticket YGN-123 --diff-file .tmp/diff.patch --evidence-file .tmp/evidence.md --strict`

Important :
- Ne pas exposer secrets/env.
- Ne pas logger tokens.
- Ne pas tuer un process `cgpro` encore susceptible de produire une réponse.
- Si profil Chromium verrouillé, afficher diagnostic : fermer ChatGPT Desktop ou démarrer le daemon cgpro si disponible.

Phase 3 — preflight :
Créer `scripts/symphony_preflight.py`.

Vérifier :
- Python >= version repo attendue.
- Git disponible.
- Repo propre ou signaler les fichiers modifiés.
- `codex` disponible.
- `codex app-server generate-json-schema --out .tmp/codex-schema` fonctionne ou documenter l’échec.
- `cgpro` disponible via :
  1. `$CGPRO_BIN`,
  2. PATH,
  3. chemin Windows connu `C:\Code\CGPro4Code`.
- `LINEAR_API_KEY` présent si tracker Linear actif.
- `LINEAR_PROJECT_SLUG` présent si utilisé dans `WORKFLOW.md`.
- `SYMPHONY_WORKSPACE_ROOT` défini ou fallback documenté.
- Le workspace root est hors repo et inscriptible.
- `WORKFLOW.md` parse YAML correctement.
- Les hooks ne contiennent pas de secrets littéraux.
- Afficher un rapport clair, sans secrets.

Phase 4 — docs ops :
Créer `ops/symphony/README.md` avec :
- But : orchestration de développement, pas runtime SAGE.
- Prérequis :
  - Codex CLI,
  - Symphony upstream,
  - mise/Elixir si implémentation Elixir,
  - Linear API key,
  - `cgpro` local.
- Installation recommandée :
  - cloner Symphony à côté du repo ou dans `external/symphony` si submodule décidé.
  - lancer `mise install`, `mix setup`, `mix build`.
- Commandes :
  - `python scripts/symphony_preflight.py --workflow WORKFLOW.md`
  - `cd ../symphony/elixir && mise exec -- ./bin/symphony /path/to/YGN-SAGE/WORKFLOW.md --logs-root /path/to/logs --port 4000`
- Windows PowerShell équivalent.
- Explique comment `cgpro` est utilisé :
  - DESIGN avant implémentation substantielle,
  - VERIFY après diff + preuves,
  - logs dans `.tmp/cgpro/`,
  - jamais de secrets.
- Explique les états Linear attendus :
  - Todo,
  - In Progress,
  - Rework,
  - Human Review,
  - Merging,
  - Done.
- Ajouter section “Fallback sans Linear” :
  - pour l’instant utiliser Linear,
  - futur : adapter GitHub Issues normalisant les issues au domain model Symphony.
- Ajouter section “Risques” :
  - Symphony preview,
  - app-server schemas dépendants de la version Codex,
  - `approval_policy: never` seulement en environnement local trusted,
  - `cgpro` dépend du profil navigateur local.

Phase 5 — éventuelle extension dynamique cgpro :
Ne l’implémente que si raisonnable dans le temps.
Sinon documente précisément comme follow-up.

Objectif extension :
- Exposer à Codex app-server un dynamic tool `cgpro_review`.
- Input schema :
  - `phase`: design | verify | audit
  - `ticket_id`
  - `title`
  - `prompt_md`
  - `repo_url`
  - `branch`
  - `commit_sha`
  - `strict`
- Output :
  - `status`: approved | pushback | blocked | unavailable
  - `summary`
  - `transcript_path`
  - `jsonl_path`
- L’implémentation appelle `scripts/symphony_cgpro.py`.
- Ne pas exposer cette tool si `cgpro` absent.
- Ne pas laisser Codex contourner la sandbox vers des chemins arbitraires ; le wrapper écrit uniquement sous workspace `.tmp/cgpro/`.

Phase 6 — validation :
Exécuter au minimum :
- `python scripts/symphony_preflight.py --workflow WORKFLOW.md`
- Test parsing YAML du `WORKFLOW.md`.
- Test dry-run `scripts/symphony_cgpro.py --help`.
- Si `cgpro` disponible : faire un probe court sans contenu secret.
- Si Codex disponible : générer schema app-server dans `.tmp/codex-schema`.
- Ne pas lancer tout le benchmark YGN-SAGE sauf nécessaire.
- Pour changements Python :
  - `python -m pytest <tests ciblés> -q`
  - `ruff check <fichiers>`
  - `mypy <fichiers>` si applicable.
- Pour changements Rust : éviter si aucun Rust modifié ; sinon test ciblé.

Definition of done :
- `WORKFLOW.md` présent, parseable, adapté YGN-SAGE.
- Preflight vert ou failures clairement classées “missing local credential/tool”.
- Documentation ops suffisante pour lancer Symphony.
- `cgpro` local détecté sans hardcode fragile.
- Aucun secret commité.
- Pas de modification invasive du runtime SAGE.
- Diff propre et PR-ready.
- Résumé final avec :
  - fichiers ajoutés/modifiés,
  - commandes exécutées,
  - preuves,
  - limites,
  - follow-ups.

  NOTE: Tu peux aussi appeler kimi 2.6 via powershell, un agent est deja installé en local