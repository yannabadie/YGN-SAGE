#!/usr/bin/env python3
"""Upload LoRA adapter checkpoint to HuggingFace and push metrics to GitHub.

Uploads ONLY the LoRA adapter (~667MB) + tokenizer (~16MB), NOT full FSDP weights.
Manages HF repo: keeps only latest checkpoint, deletes previous ones.

Usage:
    python3 scripts/verl/upload_checkpoint.py [--step N]
    python3 scripts/verl/upload_checkpoint.py --metrics-only
    python3 scripts/verl/upload_checkpoint.py --rotate-only
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ID = "yannabadie/sage-topology-policy-v2"
CHECKPOINT_DIRS = [
    "/home/yann/verl_checkpoints",
    "/workspace/topology_verl_output",
]
RAY_LOG_DIR = "/tmp/ray"
GIT_DIR = "/workspace/YGN-SAGE"
SNAPSHOT_PATH = f"{GIT_DIR}/sage-python/scripts/verl/logs/training_v5_snapshot.json"


def get_hf_token():
    env_path = f"{GIT_DIR}/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("HF_TOKEN", "")


def find_all_checkpoints():
    """Find all global_step_N directories across both checkpoint locations."""
    checkpoints = []
    for d in CHECKPOINT_DIRS:
        p = Path(d)
        if p.exists():
            checkpoints.extend(p.glob("global_step_*"))
    return sorted(checkpoints, key=lambda x: int(x.name.split("_")[-1]))


def find_latest_checkpoint():
    """Find the latest checkpoint across all locations."""
    all_ckpts = find_all_checkpoints()
    return all_ckpts[-1] if all_ckpts else None


def rotate_local_checkpoints(keep=1):
    """Delete old local checkpoints, keeping only the N most recent."""
    all_ckpts = find_all_checkpoints()
    if len(all_ckpts) <= keep:
        print(f"[rotate] {len(all_ckpts)} checkpoint(s), nothing to delete")
        return

    to_delete = all_ckpts[:-keep]
    for ckpt in to_delete:
        step = ckpt.name.split("_")[-1]
        size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
        print(f"[rotate] Deleting step {step} ({size / 1024**3:.1f} GB) from {ckpt.parent}")
        shutil.rmtree(ckpt)

    kept = all_ckpts[-keep:]
    print(f"[rotate] Kept: {[c.name for c in kept]}")


def parse_training_metrics_from_ray():
    """Parse metrics from Ray worker logs (more reliable than main log)."""
    ray_sessions = sorted(Path(RAY_LOG_DIR).glob("session_*"), key=lambda x: x.stat().st_mtime)
    if not ray_sessions:
        return []

    log_dir = ray_sessions[-1] / "logs"
    if not log_dir.exists():
        return []

    # Find TaskRunner worker logs (metrics in .out, progress in .err)
    worker_logs = sorted(
        list(log_dir.glob("worker-*.out")) + list(log_dir.glob("worker-*.err")),
        key=lambda x: x.stat().st_mtime, reverse=True,
    )

    metrics = []
    for wlog in worker_logs[:5]:
        try:
            text = wlog.read_text(errors="replace")
        except Exception:
            continue

        steps = re.findall(r'training/global_step:(\d+)', text)
        rewards = re.findall(r'critic/score/mean:([\d.e+-]+)', text)
        clips = re.findall(r'response_length/clip_ratio:([\d.e+-]+)', text)
        kls = re.findall(r'actor/kl_loss:np\.float64\(([\d.e+-]+)\)', text)
        grad_norms = re.findall(r'actor/grad_norm:np\.float64\(([\d.e+-]+)\)', text)

        n = min(len(steps), len(rewards))
        for i in range(n):
            step = int(steps[i])
            if any(m["step"] == step for m in metrics):
                continue
            metrics.append({
                "step": step,
                "reward": float(rewards[i]),
                "clip_ratio": float(clips[i]) if i < len(clips) else None,
                "kl_loss": float(kls[i]) if i < len(kls) else None,
                "grad_norm": float(grad_norms[i]) if i < len(grad_norms) else None,
            })

    return sorted(metrics, key=lambda m: m["step"])


def upload_lora_checkpoint(checkpoint_path, step, token):
    """Upload ONLY LoRA adapter + tokenizer to HF. Delete previous checkpoint from HF."""
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi(token=token)

    lora_dir = checkpoint_path / "actor" / "lora_adapter"
    hf_dir = checkpoint_path / "actor" / "huggingface"

    if not lora_dir.exists():
        print(f"[upload] No lora_adapter/ in {checkpoint_path}")
        return False

    # Delete previous checkpoints from HF repo
    try:
        existing = list(api.list_repo_tree(REPO_ID, repo_type="model", recursive=True))
        old_files = [
            f.rfilename for f in existing
            if hasattr(f, "rfilename")
            and f.rfilename.startswith("phase_a_step_")
            and not f.rfilename.startswith(f"phase_a_step_{step}/")
        ]
        if old_files:
            ops = [CommitOperationDelete(path_in_repo=p) for p in old_files]
            api.create_commit(
                repo_id=REPO_ID,
                operations=ops,
                commit_message=f"cleanup: remove old checkpoints (replaced by step {step})",
            )
            print(f"[upload] Deleted {len(old_files)} old files from HF")
    except Exception as e:
        print(f"[upload] Warning: cleanup failed: {e}")

    # Upload LoRA adapter
    print(f"[upload] Uploading LoRA adapter step {step} ({lora_dir.stat().st_size if lora_dir.is_file() else 'dir'})...")
    api.upload_folder(
        folder_path=str(lora_dir),
        repo_id=REPO_ID,
        path_in_repo=f"phase_a_step_{step}/actor/lora_adapter",
        commit_message=f"checkpoint: LoRA adapter step {step}",
    )

    # Upload tokenizer/config
    if hf_dir.exists():
        api.upload_folder(
            folder_path=str(hf_dir),
            repo_id=REPO_ID,
            path_in_repo=f"phase_a_step_{step}/actor/huggingface",
            commit_message=f"checkpoint: tokenizer step {step}",
        )

    print(f"[upload] Step {step} uploaded to {REPO_ID}")
    return True


def update_github_metrics(metrics):
    """Commit latest metrics snapshot to GitHub."""
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)

    last = metrics[-1] if metrics else {}
    step = last.get("step", "?")
    reward = last.get("reward", "?")

    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": "nvidia/Nemotron-Orchestrator-8B",
        "hardware": "2x H100 NVL 94GB",
        "config": "V5+V6: lr=5e-6, KL=0, temp=1.0, max_response_length=1024",
        "total_steps": 1152,
        "latest_step": step,
        "latest_reward": reward,
        "metrics": metrics[-20:],  # Keep only last 20 for readability
    }

    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

    subprocess.run(["git", "-C", GIT_DIR, "add", SNAPSHOT_PATH], check=True)
    result = subprocess.run(
        ["git", "-C", GIT_DIR, "diff", "--cached", "--quiet"],
        capture_output=True,
    )
    if result.returncode != 0:  # There are staged changes
        subprocess.run(
            ["git", "-C", GIT_DIR, "commit", "-m", f"metrics: Phase A step {step}, reward {reward}"],
            check=True,
        )
        subprocess.run(["git", "-C", GIT_DIR, "push", "origin", "main"], check=True)
        print(f"[github] Updated: step {step}, reward {reward}")
    else:
        print("[github] No changes to commit")


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA checkpoint + rotate + push metrics")
    parser.add_argument("--step", type=int, default=None, help="Specific step to upload")
    parser.add_argument("--metrics-only", action="store_true", help="Only push metrics to GitHub")
    parser.add_argument("--rotate-only", action="store_true", help="Only rotate local checkpoints")
    parser.add_argument("--keep", type=int, default=1, help="Number of local checkpoints to keep")
    args = parser.parse_args()

    token = get_hf_token()

    # Parse metrics from Ray logs
    metrics = parse_training_metrics_from_ray()
    if metrics:
        last = metrics[-1]
        print(f"[metrics] Latest: step {last['step']}, reward {last['reward']:.4f}")

    # Rotate local checkpoints
    if args.rotate_only:
        rotate_local_checkpoints(keep=args.keep)
        return

    if not args.metrics_only:
        # Find checkpoint
        if args.step:
            candidates = find_all_checkpoints()
            ckpt = next((c for c in candidates if c.name == f"global_step_{args.step}"), None)
        else:
            ckpt = find_latest_checkpoint()

        if ckpt and ckpt.exists():
            step = int(ckpt.name.split("_")[-1])
            upload_lora_checkpoint(ckpt, step, token)
            rotate_local_checkpoints(keep=args.keep)
        else:
            print(f"[upload] No checkpoint found")

    # Update GitHub
    if metrics:
        try:
            update_github_metrics(metrics)
        except Exception as e:
            print(f"[github] Failed: {e}")


if __name__ == "__main__":
    main()
