# local — Training local sur RTX 3500 Ada 12 GB

## Scope
Training Qwen3-4B avec Unsloth GRPO sur GPU local.
Itération rapide pour tester reward/data avant déploiement pod.

## Key Files
- `sage-python/scripts/train_local_qwen3_4b.py` — Script principal
- `sage-python/src/sage/verl/reward.py` — Reward function (SAGE_TRAINING_PHASE=A)
- `sage-python/data/verl_topology_train.parquet` — 12303 entries

## Commands
```bash
pip install unsloth vllm
python sage-python/scripts/train_local_qwen3_4b.py --smoke  # test rapide
python sage-python/scripts/train_local_qwen3_4b.py           # training complet
```

## Hardware
- GPU: NVIDIA RTX 3500 Ada (12 GB VRAM)
- Model: Qwen3-4B en 4-bit (Unsloth)
- LoRA rank 32, GRPO K=4
