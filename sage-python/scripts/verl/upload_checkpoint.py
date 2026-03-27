#!/usr/bin/env python3
"""Upload Phase A checkpoint to HuggingFace and push metrics to GitHub.

Usage:
    python3 scripts/verl/upload_checkpoint.py [--step N]

    Without --step, finds the latest checkpoint automatically.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

def get_hf_token():
    env_path = "/workspace/YGN-SAGE/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("HF_TOKEN", "")

def find_latest_checkpoint(output_dir="/workspace/topology_verl_output"):
    """Find the latest global_step_N directory."""
    p = Path(output_dir)
    checkpoints = sorted(p.glob("global_step_*"), key=lambda x: int(x.name.split("_")[-1]))
    return checkpoints[-1] if checkpoints else None

def parse_training_metrics(log_path="/workspace/train_v5.log"):
    """Parse latest metrics from training log."""
    with open(log_path) as f:
        lines = f.read()

    gsteps = re.findall(r'training/global_step:(\d+)', lines)
    rewards = re.findall(r'critic/score/mean:([\d.]+)', lines)
    clips = re.findall(r'response_length/clip_ratio:([\d.]+)', lines)
    kls = re.findall(r'actor/kl_loss:np\.float64\(([\d.e+-]+)\)', lines)

    n = min(len(gsteps), len(rewards))
    metrics = []
    for i in range(n):
        metrics.append({
            "step": int(gsteps[i]),
            "reward": float(rewards[i]),
            "clip_ratio": float(clips[i]) if i < len(clips) else None,
            "kl_loss": float(kls[i]) if i < len(kls) else None,
        })
    return metrics

def upload_checkpoint(checkpoint_path, step, token):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = "yannabadie/sage-topology-policy-v2"

    print(f"Uploading checkpoint step {step} to {repo_id}...")
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        commit_message=f"Phase A checkpoint step {step}",
        path_in_repo=f"phase_a_step_{step}",
        ignore_patterns=["*.bin", "*.pt"],
    )
    print(f"Checkpoint step {step} uploaded")

def update_github_metrics(metrics):
    """Commit latest metrics snapshot to GitHub."""
    snapshot_path = "/workspace/YGN-SAGE/sage-python/scripts/verl/logs/training_v5_snapshot.json"

    import datetime
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": "nvidia/Nemotron-Orchestrator-8B",
        "hardware": "2x H100 NVL 94GB",
        "config": "V5: lr=1e-6, max_response_length=1024, reward_shaping, n_gpus=2, TP=2",
        "total_steps": 1152,
        "steps_logged": len(metrics),
        "metrics": metrics,
    }

    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    last = metrics[-1] if metrics else {}
    step = last.get("step", "?")
    reward = last.get("reward", "?")

    subprocess.run(["git", "-C", "/workspace/YGN-SAGE", "add", snapshot_path], check=True)
    subprocess.run([
        "git", "-C", "/workspace/YGN-SAGE", "commit", "-m",
        f"metrics: Phase A V5 step {step}, reward {reward}"
    ], check=True)
    subprocess.run(["git", "-C", "/workspace/YGN-SAGE", "push", "origin", "main"], check=True)
    print(f"GitHub updated: step {step}, reward {reward}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    token = get_hf_token()

    # Parse metrics
    metrics = parse_training_metrics()
    if metrics:
        print(f"Latest metric: step {metrics[-1]['step']}, reward {metrics[-1]['reward']:.4f}")

    if not args.metrics_only:
        # Find checkpoint
        if args.step:
            ckpt = Path(f"/workspace/topology_verl_output/global_step_{args.step}")
        else:
            ckpt = find_latest_checkpoint()

        if ckpt and ckpt.exists():
            step = int(ckpt.name.split("_")[-1])
            upload_checkpoint(ckpt, step, token)
        else:
            print(f"No checkpoint found at {ckpt}")

    # Update GitHub
    if metrics:
        try:
            update_github_metrics(metrics)
        except Exception as e:
            print(f"GitHub update failed: {e}")

if __name__ == "__main__":
    main()
