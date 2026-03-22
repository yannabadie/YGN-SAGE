"""Validate RunPod environment before veRL training.

Run after setup_runpod.sh to verify GPU, veRL, sage-core, data, API keys,
patched tokenizer, and flash-linear-attention.
"""
import subprocess
import sys


def check(name: str, cmd: str) -> bool:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        msg = result.stdout.strip()[:80] if ok else result.stderr.strip()[:200]
        print(f"{'OK' if ok else 'FAIL'} {name}: {msg}")
        return ok
    except Exception as e:
        print(f"FAIL {name}: {e}")
        return False


checks = [
    ("GPU", 'python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"'),
    ("veRL", 'python3 -c "import verl; print(\'OK\')"'),
    ("vLLM", 'python3 -c "import vllm; print(vllm.__version__)"'),
    ("flash-linear-attention", 'python3 -c "import fla; print(\'OK — Qwen3.5 GDN fast path enabled\')"'),
    ("sage-core", 'python3 -c "from sage_core import TopologyGraph, TopologyReward, PyHybridVerifier; print(\'OK\')"'),
    ("SAGE SDK", 'python3 -c "from sage.topology.runner import TopologyRunner; print(\'OK\')"'),
    ("Reward", 'python3 -c "from sage.verl.reward import compute_score; print(compute_score(\'t\',\'nodes:\\n- role: coder\',\'\',{}))"'),
    ("Data", 'python3 -c "import pandas as pd; df=pd.read_parquet(\'data/verl_topology_train.parquet\'); print(f\'{len(df)} entries\')"'),
    ("API keys", 'python3 -c "import os; keys=[k for k in [\'GOOGLE_API_KEY\',\'DEEPSEEK_API_KEY\'] if os.environ.get(k)]; print(f\'{len(keys)} API keys set\')"'),
    ("Patched tokenizer", """python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/workspace/patched_model')
test = tok.apply_chat_template([{'role':'user','content':'test'}], tokenize=False, add_generation_prompt=True)
assert '<think>' not in test, 'THINKING MODE STILL ACTIVE'
print('OK — no thinking mode')
" """),
]

if __name__ == "__main__":
    results = [check(n, c) for n, c in checks]
    passed = sum(results)
    print(f"\n{'=' * 40}")
    print(f"Validation: {passed}/{len(results)} passed")
    if passed < len(results):
        print("FIX the failures above before training!")
        sys.exit(1)
    print("Ready to train!")
