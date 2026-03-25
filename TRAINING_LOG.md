# Training Log — RunPod H100 NVL 94GB (March 24-25, 2026)

> Diagnostic des runs de training sur le pod RunPod.
> Model: nvidia/Nemotron-Orchestrator-8B (Qwen3 arch)

---

## Chronologie

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

## Plan de reprise (Phase A v5)

```bash
# Changements critiques pour v5:
max_response_length: 512 → 1024
max_model_len: 1024 → 1536
lr: 5e-5 → 1e-6
# Optionnel: reward shaping pour réduire la sparsity
```

**Estimation:** 1152 steps × ~80s/step ÷ 3600 = ~25h sur 1× H100 NVL
Avec batch_size=32 et 3 epochs sur 12303 entries: 12303/32 × 3 = 1153 steps (correct)

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
