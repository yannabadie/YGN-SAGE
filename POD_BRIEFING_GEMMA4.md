# Briefing technique pour Claude Code sur le pod — Gemma4 (3 avril 2026)

> **Tu es un Claude Code opérant sur un RunPod H200 141GB.**
> **Ta mission : superviser l'entraînement de Gemma4-26B-A4B comme modèle de topology policy pour YGN-SAGE.**
> Lis ce document AVANT de toucher à quoi que ce soit.

---

## 1. Le modèle : google/gemma-4-26B-A4B-it

**Pourquoi ce modèle :**
- MoE : 25.2B params total, **3.8B actifs** par token (128 experts, top-8 + 1 partagé)
- 97% de la qualité du 31B dense avec ~7x moins de compute par token
- 256K context natif, Apache 2.0
- Benchmarks base : 77.1% LiveCodeBench, 82.6% MMLU Pro, 88.3% AIME 2026
- **Hypothèse** : ces capacités brutes → meilleures topologies que Qwen3-4B (3.8B dense)

**Ce que Gemma4 sait déjà faire :**
- Raisonnement complexe (AIME 88.3%)
- Code generation (LiveCodeBench 77.1%)
- Tool calling natif (format `<|tool_call>call:name{...}<tool_call|>`)

**Ce que Gemma4 ne sait PAS faire (et qu'on lui apprend) :**
- Générer du `<tool_call>` JSON dans le format SAGE (pas son format natif)
- Concevoir des topologies multi-agents (nodes, edges, roles, model_tiers)
- Faire des décisions d'adaptation runtime (continue/upgrade/reroute)

**Baseline à battre :**
- Qwen3-4B Phase C : N1 avg = 0.922, MASBENCH depth = 40% (4/10)

---

## 2. Le framework : TRL (SFTTrainer + GRPOTrainer) + PEFT LoRA

**PAS de verl, PAS de GiGPO.** On utilise TRL vanilla car :
- GRPO standard suffit pour single-turn topology generation
- TRL SFTTrainer/GRPOTrainer sont battle-tested
- Pas besoin de multi-turn env pour Phase 1-2

**Scripts :**
| Script | Usage |
|--------|-------|
| `scripts/train_gemma4_topology.py` | Training Python (SFT + GRPO) |
| `scripts/train_gemma4_pod.sh` | Pipeline complète pod (setup→train→eval→upload) |
| `scripts/validate_gemma4_data.py` | Validation tokenizer |
| `scripts/eval_reward_holdout_gemma4.py` | N1 évaluation |

---

## 3. GOTCHAS CRITIQUES — Lis ça AVANT de lancer

### 3.1 mm_token_type_ids (BLOQUANT)

Gemma4 EXIGE `mm_token_type_ids` dans **chaque forward pass**, même text-only.
- Le `Gemma4DataCollator` dans `train_gemma4_topology.py` le gère pour SFT
- Pour eval/generate : `inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])`
- Si tu oublies → RuntimeError ou résultats silencieusement corrompus

**Vérification :**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-26B-A4B-it", torch_dtype=torch.bfloat16, device_map="auto")
inputs = tok("test", return_tensors="pt").to(model.device)
inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
with torch.no_grad():
    out = model(**inputs)
print(f"Forward pass OK: loss shape={out.logits.shape}")
```

### 3.2 remove_unused_columns=False (BLOQUANT)

Le Trainer de HuggingFace drop par défaut les colonnes non standard.
`mm_token_type_ids` est non standard → il est droppé → crash.
**TOUJOURS** mettre `remove_unused_columns=False` dans TrainingArguments.

### 3.3 BF16 obligatoire

FP16 cause des NaN sur Gemma4. **Toujours BF16.**
```python
torch_dtype=torch.bfloat16  # PAS torch.float16
bf16=True                    # Dans TrainingArguments
```

### 3.4 Ghost thought channels

Gemma4 grands modèles peuvent générer des "ghost thoughts" même sans `<|think|>` :
```
<|channel>thought
Some reasoning...
<channel|>
```
Si tu vois ça dans les outputs, ajoute ces tokens aux banned tokens :
```python
# Trouver les IDs
bad_ids = tok.encode("<|channel>", add_special_tokens=False)
# Ajouter à generate()
model.generate(..., suppress_tokens=bad_ids)
```

### 3.5 Format natif vs format SAGE

Gemma4 natif : `<|tool_call>call:create_topology{...}<tool_call|>` (PAS JSON)
SAGE format : `<tool_call>{"name": "create_topology", "arguments": {...}}</tool_call>`

On entraîne Gemma4 à utiliser le format SAGE via SFT. Si le format natif leak pendant GRPO :
1. Vérifier avec le callback format drift (logs toutes les 50 steps)
2. Si P(format natif) > 5% : augmenter le bonus format_reward dans reward.py
3. Si P(format natif) > 20% : réduire temperature à 0.8 et augmenter SFT epochs

### 3.6 Chat template

Gemma4 utilise `<|turn>` / `<turn|>` (PAS ChatML `<|im_start|>`).
Le script `train_gemma4_topology.py` utilise `tokenizer.apply_chat_template()` qui gère ça automatiquement.
**Ne jamais construire le prompt manuellement** — laisse le tokenizer faire.

---

## 4. TON RÔLE (monitoring actif)

### Phase 1 : SFT

**Lance :**
```bash
cd /workspace/YGN-SAGE/sage-python
python3 scripts/train_gemma4_topology.py \
    --model google/gemma-4-26B-A4B-it \
    --sft-data data/v2_final.jsonl \
    --sft-only --sft-epochs 2 --sft-lr 1e-5 \
    --lora-rank 16 --output models/gemma4_topology
```

**Surveille :**
```bash
# Loss en temps réel
tail -f models/gemma4_topology/sft_metrics.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    print(f'step={d[\"step\"]:4d}  loss={d[\"loss\"]:.4f}  grad={d.get(\"grad_norm\",0):.2f}  lr={d.get(\"lr\",0):.2e}')
"
```

**Attend :**
| Métrique | Seuil | Action si échoue |
|----------|-------|-------------------|
| loss < 1.0 | Avant epoch 1 | Vérifier data loading, chat template |
| loss < 0.5 | Avant fin epoch 1 | OK, continue |
| loss < 0.3 | Fin epoch 2 | Succès → passe à eval |
| NaN | N'importe quand | STOP. Vérifier BF16 partout |
| grad_norm > 50 | N'importe quand | Réduire lr à 5e-6 |

**Après SFT, lance l'eval N1 :**
```bash
python3 scripts/eval_reward_holdout_gemma4.py \
    --adapter models/gemma4_topology/sft_checkpoint \
    --model google/gemma-4-26B-A4B-it
```

**Critères N1 :**
- avg >= 0.85 → PASSE, go GRPO
- avg 0.70-0.85 → OK mais en-dessous de la baseline Qwen3 (0.922) → continue GRPO quand même
- avg < 0.70 → PROBLÈME. Inspecte les outputs manuellement :
  ```python
  # Voir ce que le modèle génère
  tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
  # ... load model + adapter ...
  output = model.generate(...)
  print(tok.decode(output[0], skip_special_tokens=False))  # skip_special_tokens=FALSE pour voir les tokens spéciaux
  ```

### Phase 2 : GRPO

**Lance :**
```bash
SAGE_VERL_EXEC=0 SAGE_TRAINING_PHASE=C \
python3 scripts/train_gemma4_topology.py \
    --model google/gemma-4-26B-A4B-it \
    --adapter models/gemma4_topology/sft_checkpoint \
    --data data/v2_final.jsonl \
    --output models/gemma4_topology \
    --lr 3e-6 --num-generations 4 --max-completion-length 1024
```

**Surveille :**
```bash
tail -f models/gemma4_topology/grpo_metrics.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    print(f'step={d[\"step\"]:4d}  reward={d.get(\"reward\",0):.4f}  loss={d.get(\"loss\",0):.4f}  grad={d.get(\"grad_norm\",0):.2f}')
"
```

**Attend :**
| Métrique | Seuil | Action si échoue |
|----------|-------|-------------------|
| reward augmente | Steps 1-50 | Si plat → vérifier que SFT a bien marché |
| reward > 0.5 | Step 100 | Si non → augmenter temperature à 1.2 |
| format drift < 5% | Callback log | Si > 5% natif → voir section 3.5 |
| grad_norm < 10 | Continu | Si explosion → lr 1e-6 + grad clip 1.0 |

### Phase 3 : MASBENCH

**Seulement si N1 GRPO >= 0.85.** Lance :
```bash
SAGE_ENABLE_PATH6=1 \
SAGE_PATH6_ADAPTER=models/gemma4_topology/grpo_checkpoint \
SAGE_PATH6_MODEL=google/gemma-4-26B-A4B-it \
SAGE_PATH6_MODEL_TYPE=gemma4 \
python3 -m sage.bench --type masbench --limit 50 --timeout 600
```

Baseline Qwen3-4B : 40% depth (4/10). Si Gemma4 > 40% → succès.

---

## 5. Upload HuggingFace

Après la meilleure phase (SFT ou GRPO, selon les résultats) :

```bash
# Upload adapter
huggingface-cli upload yannabadie/sage-topology-policy-gemma4 \
    models/gemma4_topology/grpo_checkpoint . \
    --repo-type model

# Upload métriques
huggingface-cli upload yannabadie/sage-topology-policy-gemma4 \
    models/gemma4_topology/sft_metrics.jsonl training_data/ \
    --repo-type model
huggingface-cli upload yannabadie/sage-topology-policy-gemma4 \
    models/gemma4_topology/grpo_metrics.jsonl training_data/ \
    --repo-type model
```

---

## 6. Si tout échoue

1. **SFT ne converge pas** : le modèle MoE ne s'adapte peut-être pas bien au format tool_call. Essaie avec Gemma4-E4B (8B dense, même famille) → plus simple mais moins puissant.

2. **GRPO format drift incontrôlable** : le pretraining MoE est trop fort. Options :
   - Plus d'epochs SFT (3-5 au lieu de 2)
   - Negative reward explicite pour `<|tool_call>` natif dans reward.py
   - Freeze les couches d'expert MLP, entraîne seulement l'attention

3. **VRAM OOM** : reduce batch, enable gradient_checkpointing (déjà activé), reduce max_length.

4. **Aucune amélioration vs Qwen3-4B** : c'est un résultat de recherche valide. Documente les métriques, sauvegarde les checkpoints, et on garde Qwen3-4B.

---

## 7. Fichiers de référence

| Fichier | Description |
|---------|-------------|
| `RUNPOD_PLAN_GEMMA4.md` | Plan opérateur complet |
| `CLAUDE.md` | Directives du projet |
| `SESSION_SUMMARY.md` | Historique V2 (contexte) |
| `sage-python/src/sage/verl/reward.py` | Reward function (model-agnostic) |
| `sage-python/scripts/sage_tool_schemas.py` | Tool definitions (2 tools) |
| `sage-python/data/v2_final.jsonl` | 8633 training examples |
| `sage-python/experiments/holdout_50_toolcall.json` | 50 holdout prompts for N1 eval |
