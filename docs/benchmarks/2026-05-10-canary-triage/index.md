# Cycle-13 Canary Triage — 2026-05-10

**Status**: index documenté des artefacts canary cycle-13 conservés pour forensique. **Aucun de ces artefacts n'est gate-quality**, même les "evidence". Voir `../cycle-13-canary-manifest.md` pour les acceptance gates qu'un canary gate-quality doit passer.

**HEAD au triage**: `dbc76813536ac8919ec59d52bf8ec8f00e8a0bfd`
**Conv cgpro de référence**: `6a00a5a1-96e8-8396-ad88-46d0c6b46623` (`cgpro_ygn_sage_global_analysis_20260510`)
**Plan applicable**: `docs/superpowers/plans/2026-05-10-handoff-recovery-plan.md` Block 1.

## Pourquoi cet index

Cycle-13 a accumulé 14 dossiers `docs/benchmarks/2026-05-{08,09,10}-*` non suivis pendant le handoff harness MiniMax/DeepSeek → codex → Claude. Le pattern declared≠verified du runtime-integrity-ledger interdit qu'un artefact incomplet ou NO_GO soit confondu avec une preuve gate-quality. Ce fichier classe chaque artefact pour préserver la traçabilité sans induire en erreur les lecteurs futurs (humains ou agents).

## Classification

### keep evidence — artefact complet, utile comme contexte historique

| Path | Date UTC | Mode | Outcome | Notes |
|---|---|---|---|---|
| `2026-05-08-provider-preflight.json` | 2026-05-08 19:29 | provider preflight | 10/10 OK | Antérieur à `2026-05-10-provider-preflight-post-model-catalog.json` (déjà commité) |
| `2026-05-10-provider-preflight.json` | 2026-05-10 08:57 | provider preflight | 10/10 OK | Coûts mesurés ; antérieur immédiat au model catalog refresh `dbc76813` |
| `2026-05-10-grader-preflight-8848714e.json` | 2026-05-10 | grader preflight | NO_GO_GRADER_REPO_DIRTY | host_disk_below_swebench_minimum + grader_repo_dirty au SHA `8848714e` |
| `2026-05-10-canary-harness-mock/` | 2026-05-10 08:56 | mock | 5/5 patches générés | Valide la forme du runner ; **PAS un canary réel** |
| `2026-05-10-canary-n5-real-ec0b775e/` | 2026-05-10 09:01-09:11 | real | 5/5 timeouts, 0 patches, model_id_final=null, provider_final=null | **NO_GO** ; provider_gate=NO_GO `provider_audit_failed` ; preuve historique du bug à reproduire en Block A4 |
| `2026-05-10-canary-n5-real-8844c42e/` | 2026-05-10 09:13-09:23 | real | 5/5 timeouts identiques | **NO_GO** ; confirme la reproductibilité ; second repro pour Block A4 |
| `2026-05-10-canary-n1-preflight-8848714e/` | 2026-05-10 ~13:19 | real | 1 task, exit 1, latency 9301ms, `model_id_final="gpt-5.5-pro"`, `provider_final=null`, `_provider_policy_failure_seen=true` | **BLOCKED** ; provider policy gate a fire correctement (`openai` en denylist) ; **preuve directe que la policy fonctionne, mais l'assigner sélectionne gpt-5.5-pro malgré allowlist `[google, deepseek]`** ; repro target Block A4 |

### quarantine — artefact incomplet, NON gate-quality

| Path | Pourquoi quarantine | À conserver pour |
|---|---|---|
| `2026-05-10-canary-n5-real-8848714e/` | Seulement `input/` + `cycle-13-canary-launch-manifest.md` ; aucun `events.jsonl` / `summary.json` / `predictions.*`. Run avorté APRÈS manifest gate, AVANT exécution. | Forensique : recoupement instances input vs runs effectifs |
| `2026-05-09-canary-n3/` | Itération N=3 du 2026-05-09 ; 1 events.jsonl pour 1 instance ; pas de summary.json. | Forensique provider gate pré-N=5 |
| `2026-05-09-canary-n3-v2/` | Idem n3, version 2 (8930→ taille variable). | Idem |
| `2026-05-09-canary-n3-v4/` | Idem n3, version 4. | Idem |
| `2026-05-09-canary-n3-v5/` | Idem n3, version 5. | Idem |
| `2026-05-09-canary-n3-v6/` | Idem n3, version 6 (plus récente n3 historique). | Idem |

## Implications pour le cycle suivant

- **Block A4 `provider-gate-no-go-forensic-repro`** doit reproduire le pattern observé sur `ec0b775e` (5/5 timeout sans appel provider) ET sur `n1-preflight-8848714e` (assigner sélectionne `gpt-5.5-pro` mais allowlist `[google, deepseek]`). Les deux attestent du même class de bug à des stades différents.
- **Block B2 `first-graded-swebench-pro-n5`** doit produire un artefact qui REMPLACE structurellement les "quarantine" : un summary.json complet avec `instances_resolved` non null et grader résultat ≠ NO_GO.
- **Aucun de ces artefacts n'apparaît dans `docs/CLAIMS.yaml`** ni n'est référencé comme evidence dans un claim public — confirmé par lecture du registry au SHA `dbc76813`.

## Maintenance discipline

- Quand un nouveau canary cycle-13+ ship, mettre à jour ce fichier ou créer `cycle-NN-canary-triage.md` correspondant.
- Quand un artefact "quarantine" devient inutile (ex: le bug repro est passé en test unitaire qui le remplace), le **supprimer** plutôt que le laisser traîner.
- Ce fichier n'est PAS une source-of-truth — c'est un index. La source de vérité reste les `summary.json` / `events.jsonl` individuels et `docs/CLAIMS.yaml`.
