# Training Log — RunPod 2x H100 NVL 94GB

> Diagnostic de tous les runs de training.
> Model: nvidia/Nemotron-Orchestrator-8B (Qwen3 arch, 8.19B params)

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
