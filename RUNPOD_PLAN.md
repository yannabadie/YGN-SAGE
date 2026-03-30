# YGN-SAGE V2 — Plan d'entraînement topology (RunPod H100)

> **Ce document est la référence pour l'opérateur (humain ou Claude Code) sur le pod.**
> Mis à jour: 2026-03-30 — Post-mortem V1 + pivot JSON tool-call.

---

## Contexte

**YGN-SAGE** (Self-Adaptive Generation Engine) est un Agent Development Kit piloté par un moteur multi-agents **apprenant**. Le modèle Nemotron-Orchestrator-8B apprend à orchestrer TOUTES les fonctionnalités SAGE (Rust + Python) via RL.

### Découverte critique (2026-03-29)

**Nemotron-Orchestrator-8B est un tool-caller JSON natif**, pas un générateur YAML.
- Token `<tool_call>` id=151657, `</tool_call>` id=151658
- Entraîné par NVIDIA via GRPO pour l'orchestration d'outils (ToolOrchestra, arXiv 2511.21689)
- **L'entraînement YAML précédent causait 91% de malformation** car le format ne correspond pas au pretraining

### Problème `<think>` (Qwen3)

Nemotron est basé sur l'architecture Qwen3 qui a un biais fort vers `<think>` en début de réponse.
**Sans tools définis dans le prompt**, 100% des complétions commencent par `<think>` → reward=0 → pas d'apprentissage.
**Solution**: Bake les définitions de tools dans le system prompt pour déclencher le mode `<tool_call>`.

---

## Infrastructure

| Composant | Spécification |
|-----------|---------------|
| GPU | 2× H100 NVL 94GB |
| RAM | 251 GB |
| Stockage | NVMe `/home/yann/` (checkpoints), FUSE `/workspace/` (code) |
| Framework | verl 0.7.1 (GRPO/DAPO) |
| Modèle | nvidia/Nemotron-Orchestrator-8B (8B params, Qwen3 arch) |
| Poids | `/home/yann/nemotron_original` (poids NVIDIA originaux, NVMe) |
| verl | `/workspace/verl-071` |

### Fixes infrastructure appliqués

| Fix | Description |
|-----|-------------|
| `use_shm=False` | Symlinks cross-filesystem → copytree échoue |
| `<think>` ban | `logit_bias={151667: -100}` dans verl vllm_async_server.py |
| Model path | `/home/yann/nemotron_original` (NVMe, pas FUSE) |
| max_model_len=4096 | Prompts avec tools = ~1800 tokens |
| batch_size=16 | Réduit pour séquences plus longues |

---

## Phase A : Apprentissage du format tool_call (EN COURS)

### Objectif
Le modèle apprend à générer `<tool_call>JSON</tool_call>` au lieu de `<think>` ou YAML.

### Dataset V2 (généré from scratch)
- **13,000 exemples** : 5K simple + 5K moderate + 3K complex
- **7 tools SAGE** : `create_topology`, `route_task`, `assign_models`, `verify_topology`, `adapt_topology`, `execute_code`, `manage_memory`
- **System prompt** : Contient `# Tools` + `<tools>` XML (déclenche le mode natif)
- Fichiers : `verl_topology_train_toolcall.parquet`, `verl_topology_curated_toolcall.parquet`

### Script
```bash
cd /workspace/YGN-SAGE/sage-python
bash scripts/verl/train_topology_json.sh
```

### Configuration
```yaml
model: /home/yann/nemotron_original (poids NVIDIA originaux)
dataset: data/verl_topology_train_toolcall.parquet (13K, 7 tools)
batch_size: 16 (train) / 8 (val)
max_prompt_length: 2048 (tools = ~1800 tokens)
max_response_length: 1024
max_model_len: 4096
lr: 1e-6
lora_rank: 64
epochs: 5
loss: DAPO token-mean
```

### Reward V8
```
<tool_call> + valid topology    → 0.99 (maximum)
JSON/YAML + valid topology      → 0.92
<tool_call> partial (truncated) → 0.22 (gradient signal)
<think>                         → 0.00 (eliminated)
```

### Critères de succès
- [ ] `critic/score/mean > 0.5` avant step 50 (format tool_call appris)
- [ ] `critic/score/max > 0.9` avant step 100 (topologies valides)
- [ ] Entropy stable > 0.1 (pas de collapse)
- [ ] `response_length/mean` entre 200-800 tokens

### Commande monitoring
```bash
grep "step:" /tmp/ray/session_latest/logs/worker-*-01000000-*.out | tail -5
```

---

## Phase B : Entraînement complet (10 epochs, dataset full)

### Objectif
Affiner la qualité des topologies sur le dataset complet avec plus d'epochs.

### Pré-requis
- Phase A converge (`critic/score/mean > 0.5`)
- Checkpoint Phase A sauvegardé

### Configuration
```yaml
resume_from: checkpoint Phase A
epochs: 10
batch_size: 16
lr: 5e-7 (réduit)
dataset: même que Phase A (ou augmenté)
```

### Critères de succès
- [ ] `critic/score/mean > 0.7`
- [ ] 90%+ des réponses sont des `<tool_call>` valides
- [ ] Topologies variées (pas de mode collapse sur un seul template)

---

## Phase C : GiGPO multi-step avec ancres

### Objectif
Le modèle apprend l'orchestration **dynamique** : observation de l'exécution, adaptation des topologies en temps réel (upgrade modèle, prune nœud, reroute).

### Pré-requis
- Phase B converge (`critic/score/mean > 0.7`)
- Checkpoint Phase B sauvegardé

### SageTopologyEnv (4 états)
```
awaiting_topology → executing → awaiting_decision → terminal
```

### Micro-décisions JSON
```json
{"action": "upgrade_model", "target_node": 2, "new_model_id": "gemini-3.1-pro", "quality_score": 0.25}
{"action": "continue", "target_node": 3, "quality_score": 0.85}
{"action": "prune_node", "target_node": 1, "reason": "quality below THETA_CRITICAL"}
```

### Récompense par nœud
- **Graph-GRPO** (arXiv 2603.02701) : crédit au niveau des edges
- **Per-node quality** via QualityLabeler Rust (Z3 formel)
- **Anchor states** : checkpoints intermédiaires pour GiGPO

### Script
```bash
python3 scripts/verl/train_phase_c_custom.py
```

### Critères de succès
- [ ] Adaptations correctes > 70% du temps
- [ ] Topologies adaptées > topologies statiques sur BigCodeBench Hard
- [ ] Reward multi-step > reward single-step

---

## Post-Training Pipeline

### 1. Export & Merge LoRA
```bash
python3 scripts/verl/post_training_pipeline.py merge
```

### 2. Quantification (GGUF pour inférence locale)
```bash
python3 scripts/verl/post_training_pipeline.py quantize --formats q4_k_m,q8_0
```

### 3. Push HuggingFace
```bash
# Modèle complet
python3 scripts/verl/post_training_pipeline.py push \
  --repo yannabadie/sage-topology-orchestrator

# Dataset
python3 scripts/verl/post_training_pipeline.py push-dataset \
  --repo yannabadie/sage-topology-dataset
```

### 4. Benchmarks finaux
```bash
# BigCodeBench Hard Instruct (benchmark principal)
python -m sage.bench --type bigcodebench --subset hard --split instruct --limit 50

# Ablation (preuve de valeur du framework)
python -m sage.bench --type ablation --limit 50

# MASBENCH (benchmark multi-agent)
python -m sage.bench --type masbench --timeout 600
```

---

## Historique des runs

| Date | Phase | Steps | Reward | Problème |
|------|-------|-------|--------|----------|
| 2026-03-20 | SFT warmup | 118 | loss 1.30 | OK (mais SFT a endommagé les poids) |
| 2026-03-22 | Phase A V3 | 0 | OOM | batch=64, 159/167 GB RAM |
| 2026-03-23 | Phase A V4 | 18/1152 | 0.02 | 97% reward=0, max_resp=512 trop court |
| 2026-03-28 | Phase A V5 | 1063 | 0.225 | Plafond structurel YAML, modèle bloqué |
| 2026-03-29 | JSON V1 | 8 | 0.087 | `<think>` au lieu de JSON (pas de tools) |
| 2026-03-30 | Tool-call V2 | EN COURS | ? | Dataset V2 avec tools, reward V8 |

### Leçons apprises
1. **Nemotron = JSON tool-caller** : ne jamais entraîner en YAML
2. **`<think>` ban obligatoire** : Qwen3 génère `<think>` par défaut
3. **Tools dans le prompt** : sans `<tools>` XML, pas de `<tool_call>` en sortie
4. **NVMe pour checkpoints** : FUSE storage corrompt torch.save
5. **Poids originaux** : l'étape SFT a détruit la capacité tool-calling
6. **max_prompt_length=2048** : les définitions de tools font ~1800 tokens

---

## Références

| Papier | Usage |
|--------|-------|
| ToolOrchestra (arXiv 2511.21689) | Framework NVIDIA pour Nemotron |
| DAPO (arXiv 2503.14476) | Token-level loss, asymmetric clipping |
| AgentConductor (arXiv 2602.17100) | RL topology evolution, S_complex density |
| Graph-GRPO (arXiv 2603.02701) | Edge-level credit pour topologies |
| OpenSage (arXiv 2602.16891) | Benchmark concurrent |
