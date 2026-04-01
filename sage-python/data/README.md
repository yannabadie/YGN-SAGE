# Training Data

Large training files (SFT, GRPO, phase C) have been moved to HuggingFace:

**https://huggingface.co/datasets/yannabadie/sage-training-data**

Files include:
- `topology_sft_*.jsonl` — SFT topology generation data
- `verl_topology_*.parquet` — veRL GRPO/DAPO training data
- `code_contests_test.parquet` — Code contest benchmark
- `synthetic_*.jsonl` — Synthetic topology and recovery scenarios

To download:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("yannabadie/sage-training-data", "topology_sft_clean.jsonl", repo_type="dataset")
```

Small reference files (benchmark results, error analysis) remain in this directory.
