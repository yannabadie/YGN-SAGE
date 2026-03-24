# Briefing technique pour Claude Code sur le pod — 26 mars 2026

> **Ce document explique les choix technologiques, les pièges connus, et ce qui n'a PAS été mis en place.**
> Lis-le AVANT de lancer quoi que ce soit.

---

## 1. Le modèle : nvidia/Nemotron-Orchestrator-8B

**Pourquoi ce modèle :**
- Entraîné par NVIDIA avec **GRPO** (Group Relative Policy Optimization) pour **orchestrer des outils et modèles**
- Architecture **Qwen3-8B** (transformer standard, PAS de GDN/Mamba2 = zéro problème flashinfer/causal_conv1d)
- Bat GPT-5 sur HLE (37.1% vs 35.1%) à 30% du coût
- NVIDIA Open Model License (commercial OK, derivatives OK)
- Papier : arXiv 2511.21689, code : github.com/NVlabs/ToolOrchestra

**Ce que Nemotron sait déjà faire :**
- Décider quel outil/modèle utiliser à chaque step (= nos décisions checkpoint continue/upgrade/reroute)
- Multi-turn reasoning + tool-calling (jusqu'à 50 turns)
- Optimiser accuracy vs cost vs latency (= notre 5-signal reward)

**Ce que Nemotron ne sait PAS faire (et qu'on lui apprend) :**
- Générer du YAML topology (nodes, edges, roles, model_tiers, adaptation)
- Placer des checkpoints et fallback_tiers
- Écrire un reasoning expliquant la topologie

**Tokenizer :**
- Base Qwen3 → a un mode `<think>` qui doit être désactivé
- `patch_tokenizer.py` supprime les blocs thinking du chat template
- Alternative : `/no_think` dans le system prompt (moins fiable)
- Vérifier après patch : `assert '<think>' not in tok.apply_chat_template(...)`

---

## 2. Le framework : verl 0.7.1 + GiGPO

**Choix : verl vanilla (pas verl-agent)**

verl-agent (github.com/langfengQ/verl-agent) est le fork qui a GiGPO natif + multi-turn env support. MAIS il a un dispatch d'environnement hardcodé (`make_envs()` avec des `if/elif`) et pas de plugin system.

On utilise **verl 0.7.1 vanilla** (github.com/volcengine/verl) + **GiGPO comme plugin séparé** (github.com/langfengQ/GiGPO).

**Pourquoi :**
- verl 0.7.1 est plus stable et mieux documenté
- GiGPO se registre dans `ADV_ESTIMATOR_REGISTRY` comme plugin
- Phase A/B (single-turn YAML generation) n'a PAS besoin de multi-turn env

**Vérification critique :**
```python
# GiGPO doit être dans le registry
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
assert 'gigpo' in ADV_ESTIMATOR_REGISTRY, "GiGPO pas enregistré!"
```

**Si GiGPO ne s'enregistre pas avec verl vanilla :**
Alternative : installer verl-agent à la place :
```bash
pip uninstall verl -y
git clone https://github.com/langfengQ/verl-agent.git /workspace/verl-agent
cd /workspace/verl-agent && pip install -e .
```
verl-agent a GiGPO natif (pas besoin du plugin séparé).

---

## 3. Ce qui N'A PAS été mis en place (angles morts)

### 3.1 Le fork verl-agent avec env_package propre

**Prévu :** Forker verl-agent et ajouter `agent_system/environments/env_package/sage_topology/` avec un `elif` dans `make_envs()`.

**Réalité :** Jamais fait. On a créé `sage-python/src/sage/verl/env_package/` (wrapper Python) et un `env_register.py` qui **monkey-patche** `make_envs()` au runtime. C'est fragile.

**Impact :** Phase C (multi-step micro-décisions) risque d'échouer avec l'Approche A (verl-agent). **Utilise l'Approche B** (`train_phase_c_custom.py`) qui est un training loop Python standalone sans dépendance à verl-agent.

### 3.2 ToolOrchestra non intégré

**Prévu :** Adapter le code d'entraînement de ToolOrchestra (`training/resume_h100.py`) pour SAGE.

**Réalité :** Jamais fait. Le code de ToolOrchestra utilise son propre framework, pas verl.

**Ce qui est utile dans ToolOrchestra :**
- `training/resume_h100.py` — les hyperparamètres GRPO de Nemotron (learning rate, batch size, KL coef)
- Le reward function à 3 composantes (accuracy, cost, preference) — similaire à notre 5-signal reward
- `data_synthesis/` — pipeline de génération de données synthétiques

**Action recommandée :** Si le training avec nos hyperparamètres ne converge pas, inspecte `resume_h100.py` de ToolOrchestra pour les hyperparams de référence :
```bash
git clone https://github.com/NVlabs/ToolOrchestra.git /workspace/ToolOrchestra
cat /workspace/ToolOrchestra/training/resume_h100.py | head -100
```

### 3.3 RewardFlow et edge_credit — câblés seulement dans Phase C custom

`rewardflow.py` (PageRank per-node credit) et `edge_credit.py` (Graph-GRPO per-edge advantage) sont câblés dans `train_phase_c_custom.py` UNIQUEMENT. Ils ne sont PAS dans le reward function de Phase A/B (`reward.py compute_score()`).

Phase A/B utilise : `_score_format + _score_structure + _score_rust_density` (structural) ou `0.3*structural + 0.7*execution` (execution mode).

Phase C custom ajoute : `+0.2 * rewardflow_credit + 0.1 * edge_bonus` aux step rewards.

### 3.4 Persistence bandit/MAP-Elites — feature `cognitive` requis

La persistence (save/load state entre restarts) est derrière le feature flag Rust `cognitive`. Le build standard (`--features smt,onnx,tool-executor`) ne l'inclut PAS.

**Pour activer :** `maturin develop --features smt,onnx,cognitive,tool-executor --release`

Le boot.py a déjà le code `atexit` pour save_state, mais il faut que le feature soit compilé.

### 3.5 Dataset 12,303 entries — pas dans le parquet sur le pod

Le parquet (`verl_topology_train.parquet`) est dans le repo Git et contient 12,303 entries. MAIS il faut vérifier qu'il est correct après clone :
```python
import pandas as pd
df = pd.read_parquet('sage-python/data/verl_topology_train.parquet')
print(f'{len(df)} entries')  # Doit afficher 12303
```

Si le nombre est différent (e.g., 2225 — ancien dataset), relancer la conversion :
```bash
cd sage-python && python scripts/verl/convert_sft_to_verl.py \
    --input data/topology_sft_v2_combined.jsonl \
    --output data/verl_topology_train.parquet
```

---

## 4. Architecture GiGPO multi-step (Phase C)

### Comment ça marche

La `SageTopologyEnv` est une machine à 4 états :
```
AWAITING_YAML → EXECUTING → AWAITING_DECISION → EXECUTING → ... → TERMINAL
```

- **Step 0 :** Le modèle génère un YAML topology
- **Steps 1..N :** L'env exécute les nœuds un par un. Aux nœuds checkpoint, il pause et demande au modèle : "continue", "upgrade", ou "reroute"
- **Terminal :** Le code final est testé en sandbox → reward

GiGPO assigne des advantages par step via des anchor keys :
- `topology_generator:moderate:abc123` — pour la génération YAML
- `decision:coder:moderate:low` — pour une décision de checkpoint quand le coder a produit du mauvais code
- `upgrade:coder:moderate:` — pour un upgrade
- `terminal:PASSED` — pour le résultat final

**Masking (CRITIQUE) :**
verl-agent masque automatiquement les tokens d'observation (mask=0) — seuls les tokens du modèle (YAML, "upgrade", "continue") reçoivent des gradients. Si tu utilises le custom loop (`train_phase_c_custom.py`), cette distinction est gérée au niveau de l'advantage computation, pas du masking.

### Scripts disponibles

| Script | Approche | Dépendance |
|--------|----------|------------|
| `train_topology_v3.sh` | Phase A/B single-turn | verl 0.7.1 + GiGPO |
| `train_topology_phase_c.sh` | Phase C Approche A | verl-agent (env registration) |
| `train_phase_c_custom.py` | Phase C Approche B | PyTorch seul (pas de verl) |

**Recommandation :** Utilise `train_phase_c_custom.py` (Approche B) pour Phase C. C'est un training loop Python standalone qui :
1. Charge le modèle + LoRA depuis le checkpoint Phase A/B
2. Pour chaque prompt : K=4 rollouts via SageTopologyEnv
3. Compute GiGPO advantages (group by anchor, normalize within-group)
4. REINFORCE gradient update
5. Intègre RewardFlow + edge_credit au niveau batch

---

## 5. Hyperparamètres recommandés

### Phase A (structural, $0 API)
```
model: nvidia/Nemotron-Orchestrator-8B
algorithm: GiGPO (adv_estimator=gigpo)
LoRA: r=64, alpha=32, all-linear
lr: 1e-6
epochs: 5
batch: 64
rollout.n: 4 (K=4 pour GiGPO grouping)
temperature: 0.7 (diversité pour GiGPO)
max_response_length: 768
SAGE_VERL_EXEC: 0
```

### Phase B (execution, ~$50-80 API)
```
Hérite Phase A checkpoint
lr: 5e-7 (plus bas pour préserver Phase A)
epochs: 3
batch: 32
SAGE_VERL_EXEC: 1
```

### Phase C (micro-décisions)
```
Hérite Phase B checkpoint
lr: 5e-7
epochs: 3
k: 4
batch: 8 (plus petit car multi-step = plus de mémoire)
memory_db: /workspace/training_memory.db
```

### Référence ToolOrchestra (si besoin d'ajuster)
Les hyperparamètres originaux de Nemotron-Orchestrator-8B sont dans :
```
https://github.com/NVlabs/ToolOrchestra/blob/main/training/resume_h100.py
```

---

## 6. Post-training

Après Phase C (ou Phase B si Phase C échoue) :
```bash
python3 scripts/verl/post_training_pipeline.py all
```

Cela fait :
1. **export** — LoRA depuis checkpoint veRL
2. **merge** — Merge LoRA + Nemotron-Orchestrator-8B base → float16 (~16GB)
3. **push** — Upload vers `yannabadie/sage-topology-policy-v2` sur HuggingFace
4. **quantize** — Q8_0 GGUF (~8.7GB) pour local RTX 3500 Ada 12GB

Les 2 artifacts sur HuggingFace :
- `yannabadie/sage-topology-policy-v2` — merged float16
- `yannabadie/sage-topology-policy-v2/gguf/sage-topology-v2-Q8_0.gguf` — quantifié

---

## 7. Troubleshooting

### GiGPO pas enregistré
```bash
# Vérifier
python3 -c "from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY; print(ADV_ESTIMATOR_REGISTRY.keys())"
# Si gigpo absent : installer verl-agent au lieu de verl vanilla
```

### Tokenizer génère <think>
```bash
# Re-patcher
python3 scripts/verl/patch_tokenizer.py --model nvidia/Nemotron-Orchestrator-8B --output /workspace/patched_nemotron_orchestrator
# Vérifier
python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/workspace/patched_nemotron_orchestrator'); print('<think>' not in t.apply_chat_template([{'role':'user','content':'test'}], tokenize=False, add_generation_prompt=True))"
```

### vLLM crash
```bash
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# Si toujours crash, ajouter --enforce-eager dans les params vLLM
```

### OOM
```
gpu_memory_utilization=0.5 → 0.4
train_batch_size: 64 → 32 → 16
rollout.n: 4 → 3 → 2
```

### Phase C Approche A échoue (env registration)
```bash
# Utiliser Approche B (custom loop, pas de verl)
python3 scripts/verl/train_phase_c_custom.py \
    --model /workspace/patched_nemotron_orchestrator \
    --checkpoint /workspace/topology_verl_output \
    --data sage-python/data/verl_topology_curated.parquet \
    --output /workspace/topology_verl_phase_c \
    --epochs 3 --k 4 --memory-db /workspace/training_memory.db
```

### Reward = 0 pour toutes les completions
Le modèle génère probablement du texte non-YAML. Vérifier :
1. Le tokenizer est-il patché ? (pas de `<think>` flooding)
2. Le system prompt demande-t-il du YAML ? (vérifier convert_sft_to_verl.py SYSTEM_PROMPT)
3. `_score_format()` accepte YAML et JSON (fallback `json.loads`)
4. Le modèle génère-t-il dans un code fence ` ```yaml ``` ` ? (`_strip_code_fence()` le gère)
