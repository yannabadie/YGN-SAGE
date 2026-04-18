---
title: "ADR-009: Telemetry wire-up + per-model routing + TTL provider exclusion"
type: adr
status: accepte
date: 2026-04-18
tags:
  - adr
  - telemetry
  - routing
  - plumbing
---

# ADR-009: Telemetry wire-up, per-model routing, TTL provider exclusion

## Contexte

Session autonome du 18 avril 2026, déclenchée par un review advisor +
Codex gpt-5.4 (reasoning_effort=high) du plan post-smoke-v5a. Codex a
mis en lumière **quatre bugs silencieux** dans le plumbing Python
qui ont invalidé rétrospectivement tout le diagnostic `_tool_call_count=0`
des smokes v3/v4/v5a.

## Bugs identifiés

### 1. `tool_call_count` compteur mort

`sage-python/src/sage/pipeline.py:54` déclarait
`tool_call_count: int = 0` sur `PipelineContext`, **jamais incrémenté**.
Le bench `predictions_meta.json` lisait `ctx.tool_call_count` → toujours 0.

Conséquence : le fix `da839dc` (`tool_choice="required"` sur coder/actor)
était fondé sur l'hypothèse fausse "les modèles n'utilisent jamais les
outils". En réalité : jusqu'à 62 tool_calls par tâche.

### 2. TopologyRunner perdait les compteurs per-node

Les runs SWE-bench passent par `TopologyRunner` qui crée une `AgentLoop`
fraîche par node. Les compteurs per-loop étaient jetés à la fin du node.
Aucun roll-up vers `ctx.tool_call_count`.

### 3. `LiteLLMProvider.generate()` ignorait `config.model`

`litellm_provider.py:218` envoyait `self.model_string` (adapter default)
à `litellm.acompletion`, ignorant `config.model`. Toutes les décisions
de `ModelAssigner` depuis la création de `cards.toml` étaient
silencieusement droppées au dernier moment.

### 4. `health_check` coroutine never-awaited + permanent exclusion

- `boot_pipeline.py:165` utilisait `new_event_loop() + run_until_complete()` ;
  quand ça levait RuntimeError (event loop déjà actif), l'`except` swallowait
  sans await → `RuntimeWarning: coroutine was never awaited`. Health check
  silencieusement skippé. Tous les providers restaient "alive".
- Même quand le health check tournait, 429 `insufficient_quota` était
  classifié ALIVE ("l'API répond, c'est juste nos paramètres de probe
  qui sont mauvais"). OpenAI en quota-out restait dans le pool.
- L'exclusion résultante était **permanente** pour tout le process lifetime.
  Un provider qui recover 5 min plus tard restait exclu.

## Décisions

### A. Telemetry wire-up (988aa99 + 0677376)

1. `AgentLoop` gagne `tool_call_count`, `tool_turn_count`,
   `executed_commands` — reset à chaque `run()`/`stream()`.
2. `phases/act.py` incrémente dans le for-loop `for tc in response.tool_calls:`.
   Récupère aussi la commande `execute_bash` (tronquée 120 chars) pour forensics.
3. `pipeline.py` bypass path : `ctx.tool_call_count = agent_loop.tool_call_count`.
4. `TopologyRunner` : aggregate sur runner-level + push vers `ctx` après
   `runner.run()` (primary path et re-route).
5. `SWEBenchBench` : lit `ctx.tool_call_count` et l'expose dans predictions_meta.

### B. Per-model routing (c9ff902 + 4a2c038 + f754535)

`LiteLLMProvider.generate()` compute `effective_model` :
1. Si `config.model` contient `/` ET premier segment ∈ `{openai, gemini,
   deepseek, xai, minimax, openrouter, vertex_ai}` → pre-formaté, pass-through.
2. Sinon, détermine provider via `config.provider` (traite "unknown" comme vide).
3. Si toujours vide, `_infer_provider_from_model_id()` pattern-match
   (`gemini-*` → google, `gpt-*`/`o1`/`o3` → openai, etc.).
4. Dernier recours : adapter default (`self.model_string.split("/", 1)[0]`).

### C. Health check classifier (fe66d52)

```python
if is_connection_error or is_quota_exhaustion:
    results[name] = False              # DEAD
    self._dead_at[name] = time.time()
    for _ in range(3):
        self.record_failure(name, exc)
else:
    results[name] = True                # API-layer error, reachable
    self.record_success(name)
    self._dead_at.pop(name, None)
```

`is_quota_exhaustion` cherche `429` + `(quota|billing|credit|payment|
insufficient_quota|exceeded your current quota)`. Pure 429 sans quota
wording = probe noise, reste ALIVE.

### D. TTL + re-probe (3148667)

- `DEFAULT_EXCLUSION_TTL_SEC = 300` (5 min)
- `ProviderPool._dead_at: dict[str, float]` — timestamp par provider mort
- `reprobe_excluded_providers(timeout, ttl_sec)` : pour chaque entrée
  dont l'âge ≥ TTL, re-teste. Si OK → retire. Si toujours dead → refresh TTL.
- `refresh_exclusion_list(model_assigner)` : helper qui reprobe + push la
  nouvelle liste vers le Rust `ModelAssigner`.
- Appelé depuis `SWEBenchBench.generate_patches` au début de chaque batch
  (hook standard, autres benches peuvent faire pareil).

### E. Boot resilience (fe66d52)

```python
# Au lieu de asyncio.new_event_loop() + run_until_complete()
try:
    asyncio.get_running_loop()
    _running_in_async = True
except RuntimeError:
    _running_in_async = False

if _running_in_async:
    # Thread dédié pour ne pas dead-lock
    t = threading.Thread(target=lambda: result.append(asyncio.run(probe())))
    t.start(); t.join(timeout=30)
else:
    health = asyncio.run(probe())
```

Jamais de coroutine qui reste pending.

## Évidence empirique

| Run | offset | limit | Real | Sentinel | Empty | tool_calls (REELS) |
|-----|--------|-------|------|----------|-------|--------------------|
| v4 (avant fixes) | 0 | 10 | 1 | 5 | 4 | dead counter (0) |
| v5a (revert + strip) | 3 | 5 | 1 | 3 | 1 | dead counter (0) |
| **v5c** (+telemetry + per-model) | 3 | 5 | **3** | **0** | 2 | **27-62** réels |
| **v5d** (+"unknown" infer) | 3 | 5 | **4** | **0** | 1 | 31-62 |
| v5e (+openrouter prefix) | 3 | 5 | 3 | 1 | 1 | 19-33 |

**Average v5d+v5e = 70% real vs baseline v4 10%.** Sentinels quasi-éliminés
après wire-up telemetry + routing (étaient causés par cascade detectable
uniquement avec le compteur réel).

Patches réels (vrais diffs, pas mémorisation) :
- astropy-14995 : 567 chars, 62 bash (`find`/`grep`/`sed` sur astropy/nddata)
- astropy-6938 : 451 chars, 27 bash
- astropy-7746 : 512 chars, 62 bash (NEW en v5d, échec en v5a)
- django-10924 : 1386 chars, 33 bash (`FilePathField(callable)` fix)

## Conséquences

- **Tests Python** : 1896 → 1906 (+10 nouveaux)
- **Tests Rust** : 441 stable
- **Observability** : bench manifests ont finalement du vrai signal pour
  distinguer "agent n'utilise pas les outils" de "agent utilise les outils
  mais hit step budget"
- **Recovery automatique** : outages transitoires (Gemini brown-out 5-30
  min, quota OpenAI 24h) ne nécessitent plus de restart du pipeline
- **Direction suivante** : avec telemetry réelle, le nouveau signal est que
  F1 S3=20 max_steps est trop tight quand planner fait 20+ tool_calls. ADR
  futur : dynamic step budget via plateau detection (non implémenté Apr 18)

## Alternatives rejetées

- **Tool_choice="required" sur steps 1-2** (le `da839dc` reverté) : coercif,
  overfitté aux rôles coder/actor, générait des sentinels quand le modèle
  était forcé d'appeler l'outil sans pouvoir finaliser. Advisor + generalist
  constraint.
- **submit_final_answer tool (SWE-agent style)** : overfit SWE-bench,
  change la surface d'outils (15e tool). Deferred.
- **Boot-only health probe, manual restart pour recovery** : ne passe pas
  la exigence UX "verify before launch, not permanent".

## Tests

- `tests/test_litellm_provider.py` : 21 tests (routing, inference, prefix check)
- `tests/test_provider_pool.py` : 15 tests (health classifier, TTL, re-probe, sync with assigner)
- `tests/test_topology_runner.py` : 14 tests (telemetry aggregation, sentinel strip, planner injection)
- `tests/test_swebench_bench.py` : 5 tests (classifier real/sentinel/empty, cross-module sync)
- `tests/test_bench_main.py` : argparse defaults (`--offset` SimpleNamespace compat)
