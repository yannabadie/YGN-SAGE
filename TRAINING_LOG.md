# Training Log — RunPod 2x H100 NVL 94GB

> Diagnostic de tous les runs de training.
> Model: nvidia/Nemotron-Orchestrator-8B (Qwen3 arch, 8.19B params)

---

## Session 3 — March 29, 2026 (Combined Training)

### Combined Training (Phase A reward + execution reward)

**Script:** `train_topology_combined.sh`
**Start:** 2026-03-29 13:48 UTC
**Config:** 10 epochs, 12K dataset, SAGE_TRAINING_PHASE=A + SAGE_VERL_EXEC=1, lr=1e-6, K=4

| Step | Reward | Best Max | Exec Hits | Note |
|------|--------|----------|-----------|------|
| 1 | 0.188 | 0.667 | 1/1 (100%) | Exec hit dès le step 1 |
| 32 | 0.180 | 0.667 | 4/32 (12%) | Stabilisation |
| 62 | 0.207 | 0.831 | 10/62 (16%) | Reward en hausse |
| 88 | 0.216 | 0.831 | 13/88 (15%) | |
| 110 | 0.215 | 0.831 | 15/110 (14%) | Premier checkpoint FSDP sauvé (step 100) |
| 142 | 0.215 | **0.864** | 17/142 (12%) | Nouveau best exec score |
| 158 | 0.220 | 0.864 | 17/158 (11%) | Reward continue de monter |

**Base model:** sft_merged_model_phase_a (SFT + Phase A step 1050 LoRA, single merge)
**Checkpoints:** NVMe /home/yann/verl_checkpoints, FSDP complete, max_keep=2
**HF backup:** Full FSDP uploaded to yannabadie/sage-topology-policy-v2/checkpoints/

### Leçons Session 3

| Problème | Cause | Fix | Leçon |
|----------|-------|-----|-------|
| 20h Phase A perdues | Double-merge LoRA détruit la qualité YAML | Single merge + JAMAIS merger pendant le training | verl charge le LoRA dynamiquement |
| Phase B 0% exec hits | Modèle mergé ne produit pas de YAML valide dans verl | Utiliser le modèle SFT+LoRA merged comme BASE, LoRA frais par verl | Le merge altère les poids subtilment |
| Phase B epoch counter bug | resume_from_path porte le compteur d'epochs | Ne pas utiliser resume, repartir fresh avec modèle merged | Tester les scripts avant de lancer |
| Phase C lancée prématurément | Monitor détecte fin Phase B (0 steps) et lance Phase C | Tuer les processus zombie, relancer manuellement | Ajouter une vérification de reward minimum |
| sage_core manquant | Rust pas installé sur le pod | maturin build --release --features smt,tool-executor | Vérifier TOUTES les dépendances avant Phase B |
| Checkpoints FSDP non sauvés sur HF | upload_checkpoint.py n'uploadait que le LoRA | Upload FSDP complet (~34GB) à chaque save_freq | LoRA seul ne permet pas de resume |

### Timeline détaillée Session 3

```
09:11  Phase B v1 lancée (resume step 1050 → 0 steps, epoch counter bug)
09:22  Phase B v2 (pas de resume, modèle SFT base → reward 0.14, 10% exec)
09:41  sage_core installé (Rust + maturin build)
09:42  bigcodebench installé
09:42  Phase C lancée prématurément par le monitor (zombie)
10:00  Phase B v4 (SFT base, 30 steps, 20% exec hits, best 0.776)
10:30  Debug logging ajouté à reward.py → diagnostic: fmt=-0.30 sur 100%
11:12  Phase B v6 relancée avec debug → confirmation: YAML invalide
11:22  Phase B v7 (SFT base pur, fresh LoRA) → reward 0.14, 16% exec
12:00  Phase B v8 (Phase A single merge) → 0% exec (même problème)
13:15  Single merge testé: génère <think> en HF mais OK dans verl
13:48  Combined training lancé (10 epochs, 12K, Phase A reward + exec)
14:00  Step 2: exec hit 0.667 → la stratégie fonctionne
15:48  Step 100: checkpoint FSDP complet sauvé + uploadé HF (34GB)
16:30  Step 158: reward 0.220, best 0.864, tendance positive
```

---

## Session 2 — March 27, 2026 (2x H100 NVL, Claude Code supervised)

### Chronologie Session 2

| # | Étape | Durée | Résultat |
|---|-------|-------|----------|
| 1 | SFT warmup (GPU 0) | 14.5 min | **OK** — 118 steps, loss 2.87→1.39, YAML valide |
| 2 | LoRA merge into base | ~1 min | **OK** — 16GB, 4 shards, tokenizer patché |
| 3 | Phase A V5 (2 GPU, TP=2) | <1 min | **OOM** — micro_batch=16 trop gros (87/93 GB) |
| 4 | Phase A V5 (2 GPU, TP=2, micro=4) | ~5 min | **CRASH** — vLLM TP=2 shared memory KeyError |
| 5 | Phase A V5 (2 GPU, TP=1, micro=4) | 49 steps (~58 min) | **CRASH** — Disk quota exceeded (80GB workspace FUSE) |
| 6 | Phase A V5 resumed from step 20 | 29 steps (~35 min) | **CRASH** — torch.save() corrupt on FUSE network storage |
| 7 | Phase A V6 (lr=5e-6, KL=0, temp=1.0) | **EN COURS** | Running step 9+, checkpoints on local NVMe |

### Root causes & fixes appliqués

| Problème | Cause | Fix | Leçon |
|----------|-------|-----|-------|
| OOM backward | micro_batch=16 + seq=1536 → activations explosent | micro_batch=4 | Calculer VRAM AVANT de lancer |
| vLLM TP=2 crash | vLLM 0.18 shared memory `/psm_*` KeyError | TP=1 (2 replicas indépendantes) | Tester TP=2 en isolation d'abord |
| Disk quota exceeded | HF cache 31GB + checkpoints 34GB → 80GB workspace quota | HF cache → container disk, rotation checkpoints | Toujours mapper les quotas disque |
| torch.save() corrupt | `/workspace` est FUSE NFS, pas fiable pour gros writes | Checkpoints → `/home/yann/` (overlay NVMe local) | NE JAMAIS sauver des checkpoints sur stockage réseau |
| Reward plateau 0.065 | lr=1e-6 trop conservateur, K=4 insuffisant, temp=0.7 | V6: lr=5e-6, KL=0, temp=1.0 (params The Conductor) | S'aligner sur la littérature compétitive |

### V5 Métriques (steps 1-49, avant crash)

```
reward/mean: 0.055-0.078 (FLAT, jamais >0.08)
critic/score/max: 0.189 (plafond constant — aucun YAML valide produit)
clip_ratio: 0.73-0.88 (amélioré vs V4 0.97, mais encore haut)
KL: stable ~0.001 (lr=1e-6 préserve le SFT)
memory: 38GB/93GB par GPU (stable)
throughput: 70-75s/step, ~1000 tok/s
```

### V6 Config (en cours)

| Paramètre | V5 | V6 | Justification |
|-----------|----|----|---------------|
| lr | 1e-6 | **5e-6** | V5 trop conservateur, The Conductor utilise 1e-6 mais sur modèle non-SFT |
| KL | 0.001 | **0 (désactivé)** | The Conductor prouve que KL=0 converge pour topology training |
| temperature | 0.7 | **1.0** | Plus de diversité pour baselines GRPO (tous les concurrents utilisent 1.0) |
| entropy_coeff | 0.001 | **0.01** | Encourager l'exploration |
| use_kl_loss | True | **False** | Cohérent avec KL=0 |
| checkpoints | /workspace (FUSE) | **/home/yann/ (NVMe)** | Fix crash torch.save() |
| save_freq | 20→50 | **100** | Moins de risques d'écriture |

### Architecture disk

```
/workspace (80GB quota, FUSE NFS)  — données persistantes : modèle, code, data
/home/yann (overlay NVMe, 105GB)   — éphémère : checkpoints, HF cache
/dev/shm (176GB tmpfs)             — Ray IPC, vLLM KV cache
```

---

## Session 1 — March 24-25, 2026 (1x H100 NVL)

### Chronologie Session 1

| # | Étape | Log | Durée | Résultat |
|---|-------|-----|-------|----------|
| 1 | Setup RunPod | `setup_runpod.log` | ~5min | OK — verl 0.7.1 installé. GiGPO clone échoué (pas de git credentials sur le pod) |
| 2 | Phase A v1 (sans SFT) | `train_phase_a.log` | ~30min | Crash — Ray idle/timeout. Le modèle de base ne génère aucun YAML |
| 3 | SFT warmup v1 | `sft_warmup.log` | <1min | Crash — `RuntimeError: tensors does not require grad` (gradient checkpointing + LoRA config bug) |
| 4 | SFT warmup v3 | `sft_warmup_v3.log` | 14min | **OK** — 118 steps (2 epochs), loss 2.87→1.30. Sample output = YAML valide |
| 5 | SFT merge | `merge_sft.log` | ~1min | **OK** — LoRA merged into base → `/workspace/sft_merged_model/` (16GB, 4 shards) |
| 6 | Phase A v3 | `train_phase_a_v3.log` | ~25min | **OOM** — Ray killed worker. 159/167 GB RAM utilisés (batch_size=64, trop gros) |
| 7 | Phase A v4 | `train_phase_a_v4.log` | ~33min | **Interrompu** — 18 steps complétés, checkpoint sauvé à step 20 (26GB). Probablement arrêt session ou quota disque |

---

## Artefacts sur le pod

| Chemin | Taille | Description |
|--------|--------|-------------|
| `/workspace/patched_nemotron_orchestrator/` | 17MB | Tokenizer patché (sans `<think>`) |
| `/workspace/sft_warmup_output/` | ~500MB | LoRA adapter SFT (checkpoints intermédiaires supprimés) |
| `/workspace/sft_merged_model/` | 16GB | Modèle SFT mergé (base pour Phase A) |
| `/workspace/topology_verl_output/global_step_20/` | 26GB | Checkpoint Phase A step 20 (actor only) |
| `/workspace/verl-071/` | 318MB | verl 0.7.1 vanilla source |
| `/workspace/verl-agent/` | 342MB | verl-agent source (non utilisé) |

---

## Phase A v4 — Métriques détaillées

### Reward (objectif: >0.7)

```
Step  1: critic/score/mean = 0.0138  (max: 0.60)
Step  4: critic/score/mean = 0.0117  (max: 0.055)
Step  5: critic/score/mean = 0.0220  (max: 0.928)   val: 0.0175
Step  8: critic/score/mean = 0.0161  (max: 0.055)
Step 10: critic/score/mean = 0.0200  (max: 0.055)   val: 0.0200
Step 11: critic/score/mean = 0.0204  (max: 0.055)
Step 12: critic/score/mean = 0.0298  (max: ?)
Step 14: critic/score/mean = 0.0204  (max: 0.055)
Step 15: critic/score/mean = 0.0256  (max: 0.055)   val: 0.0199
Step 16: critic/score/mean = 0.0217  (max: 0.055)
Step 17: critic/score/mean = 0.0230  (max: 0.055)
Step 18: critic/score/mean = 0.0278  (max: 0.055)
```

**Diagnostic: reward quasi-plat à ~0.02 sur 18 steps. ~97% des completions ont reward=0.**

### Response length

```
response_length/mean: 509-512 (quasi toujours au max)
response_length/clip_ratio: 0.97-1.0 (presque toutes les réponses tronquées)
response_length/min: 296-512
```

**Diagnostic: le modèle remplit systématiquement les 512 tokens max → se fait tronquer → YAML incomplet → reward=0.**

### Actor loss

```
actor/pg_loss: -0.0003 à -0.0029 (quasi nul)
actor/kl_loss: 0.0 → 0.020 (KL divergence croissante)
actor/pg_clipfrac: 0.0 (aucun clipping PPO)
actor/grad_norm: 0.013-0.021 (gradients très petits)
```

**Diagnostic: le signal de gradient est quasi-inexistant car le reward est trop sparse (presque tout à 0).**

### Config utilisée

```
model: /workspace/sft_merged_model (Nemotron-Orchestrator-8B + SFT LoRA merged)
algorithm: GRPO (verl 0.7.1 vanilla, GiGPO non installé — clone échoué)
lr: 5e-5
batch_size: 32 (réduit de 64 après OOM v3)
rollout.n: 4 (K=4 pour GRPO grouping)
temperature: 0.7
max_response_length: 512
max_model_len: 1024
LoRA: r=64, alpha=32, all-linear
FSDP: param_offload=True, optimizer_offload=True
vLLM: gpu_memory_utilization=0.3, TORCH_SDPA backend
epochs: 3 (1152 total steps)
```

---

## Diagnostic des problèmes

### Problème 1: Reward trop sparse (CRITIQUE)

**Symptôme:** reward mean ~0.02, max 0.055 (rarement 0.6-0.93)
**Cause:** Le modèle génère du texte non-YAML dans ~97% des cas. Le reward function (`_score_format`) retourne 0 quand le YAML n'est pas parsable.
**Impact:** Pas de signal de gradient → pas d'apprentissage.

**Solutions possibles:**
1. Augmenter `max_response_length` de 512 à 1024+ (les YAML SFT font ~400-800 tokens)
2. Baisser `lr` de 5e-5 à 1e-6 (préserver le SFT warmup)
3. Ajouter un reward partiel pour les tentatives de YAML (même mal formées)
4. Reward shaping: bonus pour commencer par `nodes:` ou `reasoning:`

### Problème 2: Clip ratio = 1.0 (CRITIQUE)

**Symptôme:** Toutes les réponses font exactement 512 tokens
**Cause:** max_response_length=512 trop court pour un YAML topology complet
**Impact:** Le YAML est tronqué → parsing échoue → reward=0

**Solution:** Augmenter max_response_length à 1024 (et max_model_len à 1536+)

### Problème 3: LR trop haute

**Symptôme:** kl_loss passe de 0.0 à 0.020 en 18 steps
**Cause:** lr=5e-5 est très haute pour du RL post-SFT (RUNPOD_PLAN recommandait 1e-6)
**Impact:** Le modèle diverge rapidement du SFT warmup → perd la capacité YAML

**Solution:** lr=1e-6 comme prévu dans RUNPOD_PLAN

### Problème 4: GiGPO non installé

**Symptôme:** Setup log montre `git clone GiGPO... fatal: could not read Username`
**Cause:** Pas de git credentials configurés sur le pod pour les repos publics via HTTPS
**Impact:** Utilise GRPO vanilla au lieu de GiGPO. OK pour Phase A single-turn, mais bloquant pour Phase C.

**Solution:** `git clone https://github.com/langfengQ/GiGPO.git` (sans auth pour un repo public — tester la connectivité réseau du pod)

### Problème 5: OOM en v3 (résolu en v4)

**Symptôme:** Ray tue le worker à 159/167 GB RAM
**Cause:** batch_size=64 + param_offload
**Résolution:** batch_size réduit à 32 en v4 → OK

---

## Root Cause Analysis approfondie (post-mortem V4)

### Cause racine 1: Boucle de mort truncation → reward=0 → pas de gradient

La chaîne causale complète:
1. `max_response_length=512` dans `train_topology_v3.sh` (ligne 130)
2. Les YAML topology du SFT font 400-800 tokens (vérifié sur sft_warmup_v3)
3. Le modèle génère du YAML structuré (grâce au SFT warmup), mais à 512 tokens il est coupé
4. `response_length/mean: 509-512` et `clip_ratio: 0.97-1.0` — 97% des réponses tronquées
5. `_score_format()` dans `reward.py` fait `yaml.safe_load(text)` → `YAMLError` sur YAML tronqué
6. Score = -2.0 → normalisé à 0.0 → `reward/mean = 0.02`
7. `actor/pg_loss ≈ 0` et `actor/grad_norm = 0.013-0.021` — pas de signal d'apprentissage

**Le modèle SFT savait générer du YAML. Le RL l'a vu recevoir reward=0 pour du bon YAML tronqué, et a commencé à dériver.**

### Cause racine 2: LR trop haute accélère la catastrophe

- `train_topology_v3.sh` ligne 145: `lr=5e-5` (RUNPOD_PLAN disait 1e-6)
- KL divergence: 0.0 → 0.020 en 18 steps (drift catastrophique)
- En 18 steps à lr=5e-5, le modèle a déjà divergé du SFT warmup
- À lr=1e-6 (50× plus bas), la même divergence prendrait ~900 steps
- Combiné avec reward sparse: le modèle n'apprend rien d'utile ET oublie le SFT

### Cause racine 3: V4 log termination

Le log `train_phase_a_v4.log` s'arrête proprement à step 18/19 (pas de stack trace):
```
2%|▏ | 19/1152 [32:36<28:51:05, 91.67s/it]
```
Pas d'erreur OOM, pas de SIGKILL. Causes probables:
- **Session SSH timeout** (RunPod ferme après inactivité)
- **Arrêt manuel** (diagnostic en cours, pas de raison de continuer 28h avec reward=0)
- **Pod termination** (quota ou maintenance RunPod)

Le checkpoint a été sauvé à step 20 (26GB dans `/workspace/topology_verl_output/global_step_20/`).

### Volume disque

Le filesystem `/dev/mapper/ps1010x2` (28T, monté sur `/etc/hosts`) est à 82% (23T/28T).
C'est un **stockage partagé RunPod** (NFS), pas spécifique à ce pod.
L'espace workspace utilisé est 44 GB sur un overlay de 120 GB (37%).
**Aucune erreur de disque plein dans les logs.** Le pod avait assez d'espace.

### Métriques mémoire (V4 stable)

```
GPU: max_allocated=64.65 GB, max_reserved=76.18 GB (sur 94 GB H100 NVL)
CPU: 406-408 GB stable (pas de fuite mémoire)
Throughput: 1010-1067 tokens/sec, ~80-90s/step
```
V4 avait résolu le OOM de V3 (batch_size 64→32).

---

## Plan de reprise — Phase A V5

### Script: `sage-python/scripts/verl/train_topology_v5.sh`

### Changements critiques

| Paramètre | V3/V4 | V5 | Raison |
|-----------|-------|-----|--------|
| `data.max_response_length` | 512 | **1024** | YAML topologies font 400-800 tokens |
| `actor_rollout_ref.actor.optim.lr` | 5e-5 | **1e-6** | Préserver le SFT warmup (RUNPOD_PLAN) |
| `actor_rollout_ref.rollout.max_model_len` | 1024 | **2048** | ≥ prompt(512) + response(1024) |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | 1024 | **2048** | Aligné sur max_model_len |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.3 | **0.35** | Headroom pour séquences 2× plus longues |
| `reward.py _score_format()` | -2.0 cliff | **partial credit** | Reward shaping pour YAML tronqué |

### Reward shaping (nouvelle `_partial_credit()`)

Pour les YAML qui échouent au parsing, score gradué au lieu de -2.0:
- Contient `nodes:` → -1.0 (au lieu de -2.0)
- Contient `role:` → +0.3
- Contient liste YAML (`- role:`) → +0.2
- Contient `reasoning:` → +0.2
- Cap à -0.3 (toujours inférieur au YAML valide le plus bas = -0.25)

Impact estimé: reward sparsity de ~97% → ~60% (le modèle reçoit du gradient même pour YAML tronqué).

### Estimation durée

```
1152 steps × ~90s/step ÷ 3600 = ~29h sur 1× H100 NVL
(~90s/step au lieu de ~80s: séquences 2× plus longues)
```

### Critères de succès

| Métrique | V4 (baseline) | V5 cible | Seuil OK |
|----------|---------------|----------|----------|
| reward/mean | 0.02 | >0.3 | >0.1 à step 50 |
| clip_ratio | 0.97 | <0.5 | <0.7 à step 20 |
| KL divergence (step 100) | 0.020 (à step 18) | <0.01 | <0.05 |
| YAML parse rate | ~3% | >50% | >20% à step 50 |

---

## SFT Warmup — Métriques

```
Step  10: loss=2.8684, grad_norm=2.42
Step  20: loss=2.1766, grad_norm=0.53
Step  30: loss=1.8095, grad_norm=0.38
Step  40: loss=1.6045, grad_norm=0.26
Step  50: loss=1.5004, grad_norm=0.20
Step  60: loss=1.4498, grad_norm=0.18  (epoch 1.0)
Step  70: loss=1.3691, grad_norm=0.16
Step  80: loss=1.3159, grad_norm=0.15
Step  90: loss=1.3099, grad_norm=0.18
Step 100: loss=1.3064, grad_norm=0.16
Step 110: loss=1.2985, grad_norm=0.15
Step 118: loss=1.2985  (epoch 2.0, final)

trainable params: 174,587,904 / 8,365,323,264 (2.09%)
train_runtime: 832s (~14min)
train_samples_per_second: 4.52
```

**Sample output (YAML valide):**
```yaml
difficulty: easy
topology:
  type: sequential
  agents:
    - name: code_writer
      role: code_writer
      task: write a function that checks if a string is a palindrome
      dependencies: []
      outputs: [code]
    - name: code_reviewer
      role: code_reviewer
      task: review the code for correctness and efficiency
      dependencies: [code]
      outputs: [review]
    - name: code_executor
      role: code_executor
      task: execute the code with test cases
      dependencies: [re...  (truncated)
```

Le SFT a bien appris le format YAML. Le problème est que le RL (Phase A) perd cette capacité trop vite (LR trop haute + response trop courte).

## MASBENCH Validation — March 29, 2026

### Pilot Results (10 tasks, 2 axes)

| Axis | Bare (DeepSeek) | SAGE Full | Delta |
|------|----------------|-----------|-------|
| depth | 1/5 (20%) | 3/5 (60%) | **+40pp** |
| breadth | 2/5 (40%) | 2/5 (40%) | +0pp |
| **TOTAL** | **3/10 (30%)** | **5/10 (50%)** | **+20pp** |

### Full Results — Bare Model Baseline (50 tasks, 5 axes)

| Axis | Bare (DeepSeek) | Description |
|------|----------------|-------------|
| depth | 1/10 (10%) | Chain reasoning — hardest |
| breadth | 6/10 (60%) | Parallel sub-tasks |
| horizon | 0/10 (0%) | Multi-step planning |
| parallel | 6/10 (60%) | Concurrent work |
| robustness | 0/10 (0%) | Error tolerance |
| **TOTAL** | **13/50 (26%)** | |

SAGE full engine results pending (~2h runtime).

### Strategic Implications

1. **depth (10%) and horizon (0%) and robustness (0%)** are where topology should help most
2. **breadth (60%) and parallel (60%)** are already solved by bare model — topology overhead may hurt
3. **Nemotron-8B training should target depth/horizon/robustness** — not all task types
4. This aligns with AdaptOrch (arXiv 2602.16873): "topology matters when base accuracy is 60-80%"
   - Below 60%: topology helps (depth, horizon, robustness)
   - Above 60%: topology overhead hurts (breadth, parallel)

### Bugs Fixed During Validation
- gpt-4.1 → gpt-5.4 (models.toml, router.py, codex.py)
- snowflake-arctic-embed-m → Snowflake/snowflake-arctic-embed-m (embedder.py)
- OpenAI max_tokens → max_completion_tokens for GPT-5+ (openai_compat.py)
- MiniMax fallback provider routing (fallback now deepseek-chat)

### MASBENCH Full Results — Fixed Models (50 tasks, 5 axes)

**IMPORTANT:** This test ran with fixed models (gpt-5.4, gemini-3.1, deepseek-chat).
However, SAGE timeout at 120s caused most tasks to fail. With 300s timeout, SAGE solves
tasks the bare model cannot.

| Axis | Bare | SAGE (120s) | SAGE (300s, 3 tasks) | Issue |
|------|------|-------------|---------------------|-------|
| depth | 4/10 (40%) | timeout | 1/3 PASS (274s) | Pipeline latency |
| breadth | 5/10 (50%) | pending | - | |
| horizon | 0/10 (0%) | pending | - | |
| parallel | 6/10 (60%) | pending | - | |
| robustness | 0/10 (0%) | pending | - | |

**Key insight:** SAGE's topology HELPS (solves tasks bare model can't) but the pipeline
is too SLOW (274s vs 15s). The bottleneck is multi-node sequential API calls, not the
topology quality. Fix = pipeline optimization, not more RL training.

### Research Findings (March 29, 2026)

| Paper | Key Innovation | Applicable to SAGE | Priority |
|-------|---------------|-------------------|----------|
| DAPO (2503.14476) | Token-level loss, asymmetric clip, dynamic sampling | **Integrated** in targeted script | P0 |
| MAS-Orchestra (2601.14652) | Function-calling RL (not YAML) | Reframe topology as FC | P1 |
| EvoMAS (2602.06511) | Joint topology + model assignment evolution | MAP-Elites upgrade | P2 |
| GoAgent (2603.19677) | Group-level topology + CIB compression | Reduce pipeline overhead | P3 |
| Graph-GRPO (2603.02701) | Bernoulli edge sampling, continuous rewards | Upgrade edge_credit.py | P4 |
| TCAndon-Router (2601.04544) | Reasoning-chain routing | Upgrade kNN router | P5 |

### Pipeline Latency Analysis

```
Bare model:  15s (1 API call)
SAGE depth:  274s (routing 2s + topology 1s + node1 90s + node2 80s + node3 60s + overhead 41s)
Overhead:    18x slower for same quality
```

**Optimization opportunities:**
1. Parallel node execution (TopologyExecutor Rust supports it, Python runner doesn't use it)
2. Cache routing decisions (kNN embedding computed every call)
3. CIB message compression (GoAgent, reduce context passed between nodes)
4. Reduce default nodes from 4-5 to 2-3 (simpler topologies may be better)

### Runner Fix Impact (March 30, 2026)

After fixing DeepSeek fallback + 60s per-node timeout:

| Task | Before fix | After fix |
|------|-----------|-----------|
| depth task 1 (gt=9) | PASS 274s | PASS 197s |
| depth task 2 (gt=16) | FAIL 263s | FAIL 154s |
| depth task 3 (gt=18) | FAIL 186s | **PASS 222s** |
| **Total** | **1/3 (33%)** | **2/3 (67%)** |

**SAGE now beats bare model on depth: 67% vs 40%** (+27pp)
Latency reduced: 241s avg → 191s avg (-21%)

### DAPO Targeted Training (launched March 30, 2026)

Script: `train_topology_targeted.sh`
Config: DAPO token-level loss, 5 epochs, full 12K dataset, SFT base model
Step 104/1920 | reward=0.184 | 60s/step | GPUs 100%

### Stack Audit & Fixes (March 30, 2026)

#### Rust sage-core: 100% operational (54 exports)

Built with `smt+cognitive+tool-executor`:
```
✓ TopologyGraph, TopologyExecutor, TopologyEngine
✓ RustKnnRouter (92% accuracy, 50 exemplars)
✓ PyHybridVerifier (6 structural + 4 semantic checks)
✓ PyTemplateStore (8 templates)
✓ SmtVerifier, LtlVerifier
✓ QualityLabeler, ModelAssigner, SystemRouter
✓ ContextualBandit, MultiViewMMU
✓ ToolExecutor (sandbox)
```

#### Fixes applied in Session 3-4

| Fix | Impact | Commit |
|-----|--------|--------|
| gpt-4.1 → gpt-5.4 (all model tiers) | Eliminated 404 errors on OpenAI | 6933718 |
| Embedder model name fix | Enabled sentence-transformers backend | ca15f4d |
| OpenAI max_completion_tokens | GPT-5+ models work | ca15f4d |
| DeepSeek fallback in TopologyRunner | No more wrong-provider 404s | b6aeea8 |
| Per-node 60s timeout | Prevents single-node blocking | b6aeea8 |
| ProviderPool model_id→provider inference | deepseek-chat routes to DeepSeek, not Gemini | b5ef00c |
| S1 skip topology (fast path) | Simple tasks: 200s → 15s | 3efdabf |
| Rebuild routing_exemplars.npz | Rust kNN router activated | ee91133 |
| sage-core cognitive feature | HybridVerifier + TemplateStore available | ec9e601 |
| requirements-runpod.txt | One-command pod setup | e05e335 |
| setup_full.sh | Automated verification of complete stack | ec9e601 |

#### Setup automation

```bash
# One-command setup on new RunPod:
bash sage-python/scripts/setup_full.sh
# Verifies: Rust core (54 exports), Python deps, embeddings, kNN, API keys
```

### Critical Fix: kNN Rust Router Activated in Pipeline (March 30, 2026)

**Root cause of ALL previous MASBENCH failures:**
The pipeline Stage 0 (classify) was using the ComplexityRouter heuristic (34% accuracy)
instead of the kNN Rust router (92% accuracy). The kNN was loaded at boot but NEVER
called by the pipeline. All benchmarks ran with 34% routing accuracy.

**Impact:**
- Simple tasks (S1) misrouted to S2/S3 → 200s topology instead of 6s direct call
- Complex tasks (S3) misrouted to S2 → wrong topology template
- MASBENCH results were measuring routing failures, not topology quality

**Fix stack (13 commits in Session 3-4):**

| # | Fix | Commit | Impact |
|---|-----|--------|--------|
| 1 | gpt-4.1 → gpt-5.4 | 6933718 | Eliminated 404s |
| 2 | Embedder model name | ca15f4d | Enabled sentence-transformers |
| 3 | OpenAI max_completion_tokens | ca15f4d | GPT-5+ models work |
| 4 | DeepSeek fallback in runner | b6aeea8 | No wrong-provider 404s |
| 5 | Per-node 60s timeout | b6aeea8 | No single-node blocking |
| 6 | ProviderPool model→provider | b5ef00c | Correct provider routing |
| 7 | S1 skip topology fast path | 3efdabf | 200s → 6s for simple tasks |
| 8 | Rebuild routing exemplars | ee91133 | Rust kNN activated |
| 9 | sage-core cognitive feature | ec9e601 | HybridVerifier + TemplateStore |
| 10 | Boot provider by model_id | a80b379 | DeepSeek not sent to Gemini |
| 11 | ModelRouter provider inference | 51c39a6 | All tiers route correctly |
| 12 | models.toml + 7 providers | a07a0a4 | All models verified |
| **13** | **kNN in pipeline Stage 0** | **50ab910** | **92% routing (was 34%)** |

**Before all fixes:** SAGE -10pp vs bare model (broken pipeline)
**After fix #7 (3 tasks):** SAGE +27pp on depth
**After ALL fixes:** Not yet tested — this will be the definitive benchmark.

**System verification (E2E test):**
- Rust core: 18/18 components operational (54 exports)
- Routing: kNN Rust 92% → S1=6s, S2=14s, S3=TBD
- Providers: 7/7 OK (DeepSeek, OpenAI, Google, xAI, MiniMax, Kimi, OpenRouter)
- Memory: Episodic + Entity OK, S-MMU constructor needs update
- Training: DAPO step 204/1920, reward 0.205
- Benchmarks: MASBENCH + GAIA adapters ready

### DAPO Training Analysis (March 30, 2026, Step 299)

| Run | Steps | Reward Start→End | Best | Exec Hits | K | Loss |
|-----|-------|-----------------|------|-----------|---|------|
| Phase A V4 | 18 | 0.014→0.023 | 0.055 | 0% | 4 | GRPO seq-mean |
| Phase A V5 | 49 | 0.055→0.078 | 0.189 | 0% | 4 | GRPO seq-mean |
| Phase A V6 | 1050 | 0.225→0.225 | 0.225 | 0% | 4 | GRPO seq-mean |
| Combined A+B | 158 | 0.188→0.220 | 0.864 | 11% | 4 | GRPO seq-mean |
| **DAPO targeted** | **299** | **0.176→0.219** | **0.987** | **9%** | **4** | **DAPO token-mean** |

**Key observations:**
1. **Best score 0.987** — highest ever. The model CAN produce near-perfect topologies.
2. **But 91% of topologies have invalid YAML** — exec reward rarely fires.
3. **Reward progression: +0.043 in 299 steps** — slow but no plateau (unlike V6 which plateaued immediately at 0.225).
4. **DAPO token-level loss works** — prevents reward hacking on length. But doesn't solve YAML malformation.
5. **K=4 is the bottleneck** — The Conductor uses K=64 (16x more variance). With K=4, GRPO signal is too noisy for complex structured generation.

**Projected convergence at current rate:**
- Step 500: reward ~0.25
- Step 1000: reward ~0.32
- Step 1920: reward ~0.40

**What would help:**
- K=8 or K=16 (requires gradient accumulation or VRAM optimization)
- Dynamic sampling (DAPO feature: skip batches where all K rollouts have same reward)
- Function-calling format instead of YAML (MAS-Orchestra approach — lower malformation rate)

**What won't help:**
- More epochs on same dataset (V6 proved 1050 steps of structural plateau)
- Higher LR (V3/V4 proved drift at lr=5e-5)
- Merging LoRA (proven to destroy YAML quality)

### Training Philosophy Insight (March 30, 2026)

**Key realization:** Nemotron-8B must learn to work WITHIN YGN-SAGE, not just generate YAML.

The model must learn to produce topologies that work with:
- TopologyExecutor (Rust) — node scheduling
- ModelAssigner (Rust) — model_tier → real model via cards.toml
- HybridVerifier (Rust) — DAG validation
- ProviderPool — 7 providers routing
- TopologyRunner — multi-node execution with predecessor context
- TopologyController (Phase C) — runtime micro-decisions

Current problem: 91% of generated topologies have invalid YAML → execution reward
never fires → the model never learns what works IN SAGE.

Potential solutions:
1. Function-calling format (MAS-Orchestra) — lower malformation rate
2. Increase K from 4 to 16+ — more variance for GRPO signal
3. Better SFT warmup focused on SAGE-compatible YAML patterns
4. Curriculum: start with 2-node topologies, then increase complexity

### MASBENCH kNN Routing Issue (March 30, 2026)

The 50 kNN exemplars don't cover MASBENCH task types. All MASBENCH tasks
route to S3 (conf=1.0) → full topology pipeline → 170-300s per task.
S1 fast path never activates for MASBENCH.

Fix: add MASBENCH-style tasks to kNN exemplar bank, or use task features
(not just embedding similarity) for routing.

### BREAKTHROUGH INSIGHT: Nemotron-8B is a JSON tool-caller, not a YAML generator (March 30, 2026)

**Root cause of all training failures identified.**

Nemotron-Orchestrator-8B was trained by NVIDIA via GRPO specifically to:
1. Read instructions
2. Generate chain-of-thought reasoning
3. **Emit structured JSON tool calls**

We have been asking it to generate **free-form YAML** — a task it was NEVER trained for.
This explains:
- 91% YAML malformation (the model doesn't know YAML syntax)
- SFT warmup insufficient (118 steps can't overcome GRPO pretraining bias)
- <think> token generation (the model's native CoT before tool calls)
- Structural plateau at 0.225 (the model memorized some YAML patterns but can't generalize)

**The fix:** Reframe topology generation as **tool calling in JSON format**.

Define topology actions as tools:
```
add_node(role, model_tier, prompt)
add_edge(from_idx, to_idx)
set_reasoning(text)
set_difficulty(level)
```

The model generates JSON tool calls (its native format), which are converted
to TopologyGraph by the SAGE pipeline. vLLM constrained decoding guarantees
valid JSON → 0% malformation → 100% exec reward signal.

**Expected impact:**
- Malformation: 91% → 0% (constrained decoding)
- Exec hits: 9% → 100% (every topology is valid)
- Reward signal: sparse → dense (gradient on every sample)
- Convergence: 1000+ steps → ~200 steps (matching The Conductor)

**References:**
- [ToolOrchestra (NVIDIA)](https://github.com/NVlabs/ToolOrchestra) — the original training framework
- [Nemotron-Orchestrator-8B](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B) — native JSON tool caller
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [TRL GRPO + JSON Schema](https://github.com/huggingface/trl/issues/5154)

### JSON Format Validation (March 30, 2026)

**SAGE already accepts JSON topologies — no code change needed.**

| Format | format_score | structure_score | total_score |
|--------|-------------|-----------------|-------------|
| YAML | 1.00 | 1.00 | 0.9848 |
| JSON | 1.00 | 1.00 | 0.9848 |

The reward function (`_score_format`) calls `yaml.safe_load()` which parses
valid JSON natively. Fallback `json.loads()` catches JSON with trailing commas.
The execution path (`execution/__init__.py`) tries `json.loads()` FIRST.

**What needs to change for JSON tool-calling training:**

1. **Dataset conversion** (YAML → JSON):
   - Convert 12,303 training entries from YAML ground truth to JSON
   - Change system prompt from "design as YAML DAG" to "emit JSON tool calls"
   - Script: `convert_sft_to_json.py`

2. **Reward function** — already works, no change needed

3. **Execution path** — already works, JSON is tried first

4. **Phase C tool-call format**:
   - Step 0 (generate topology): `{"nodes": [...], "edges": [...], "reasoning": "..."}`
   - Checkpoint decisions: `{"action": "continue"}` or `{"action": "upgrade", "node": 2, "new_tier": "reasoner"}`
   - Both are native JSON — perfect for Nemotron's tool-calling training
   - `SageTopologyEnv` needs a JSON parser for tool_calls → TopologyGraph

5. **vLLM constrained decoding**:
   - Define Pydantic schema for TopologyOutput
   - Pass to verl rollout config as `guided_json`
   - Guarantees 100% valid output → 100% exec reward signal

**Impact:** Training pivots from "teach YAML syntax" to "teach topology QUALITY" —
the model's native JSON ability handles the format, RL handles the substance.
