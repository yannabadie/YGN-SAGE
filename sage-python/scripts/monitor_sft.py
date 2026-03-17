"""Active SFT training monitor — logs metrics, GPU stats, anomaly detection."""
import time
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("data/sft_training_dq.log")
MONITOR_LOG = Path("data/sft_training_monitor.jsonl")

last_step = 0
last_time = time.time()
history = []

print(f"=== SFT Monitor started {datetime.now().isoformat()} ===", flush=True)

while True:
    try:
        content = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        time.sleep(10)
        continue

    # Parse metrics blocks
    matches = re.findall(r"\{'loss'.*?'epoch'.*?\}", content)
    step_matches = re.findall(r"(\d+)/1640", content)

    current_step = int(step_matches[-1]) if step_matches else 0

    if matches and current_step > last_step:
        try:
            metrics = eval(matches[-1])
        except Exception:
            time.sleep(10)
            continue

        now = time.time()
        elapsed = now - last_time
        steps_done = current_step - last_step
        speed = steps_done / elapsed if elapsed > 0 else 0
        eta_min = (1640 - current_step) / speed / 60 if speed > 0 else 999

        # GPU
        gpu_util = gpu_mem = gpu_temp = gpu_power = "?"
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = r.stdout.strip().split(", ")
            gpu_util, gpu_mem, gpu_temp, gpu_power = f"{parts[0]}%", f"{parts[1]}MB", f"{parts[2]}C", f"{parts[3]}W"
        except Exception:
            pass

        entry = {
            "ts": datetime.now().isoformat()[:19],
            "step": current_step,
            "pct": round(current_step / 1640 * 100, 1),
            "loss": float(metrics.get("loss", 0)),
            "acc": float(metrics.get("mean_token_accuracy", 0)),
            "lr": float(metrics.get("learning_rate", 0)),
            "grad": float(metrics.get("grad_norm", 0)),
            "epoch": float(metrics.get("epoch", 0)),
            "speed": round(speed * 60, 1),
            "eta_min": round(eta_min),
            "gpu": gpu_util,
            "mem": gpu_mem,
            "temp": gpu_temp,
            "pwr": gpu_power,
        }

        print(
            f"[{entry['ts']}] {current_step}/1640 ({entry['pct']}%) "
            f"loss={entry['loss']:.3f} acc={entry['acc']:.1%} "
            f"lr={entry['lr']:.1e} epoch={entry['epoch']:.2f} "
            f"speed={entry['speed']}st/min ETA={entry['eta_min']}min "
            f"GPU={gpu_util} {gpu_mem} {gpu_temp} {gpu_power}",
            flush=True,
        )

        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Anomalies
        if entry["loss"] > 5.0:
            print(f"  WARNING: HIGH LOSS {entry['loss']:.3f}", flush=True)
        if entry["grad"] > 10.0:
            print(f"  WARNING: GRAD EXPLOSION {entry['grad']:.3f}", flush=True)
        if history and entry["loss"] > history[-1]["loss"] * 1.5:
            print(f"  WARNING: LOSS SPIKE {history[-1]['loss']:.3f} -> {entry['loss']:.3f}", flush=True)

        history.append(entry)
        last_step = current_step
        last_time = now

    # Check process alive
    try:
        r = subprocess.run(["ps", "-ef"], capture_output=True, text=True, timeout=5)
        if "train_topology" not in r.stdout:
            print(f"TRAINING ENDED at step {last_step}/1640", flush=True)
            if last_step >= 1640:
                print("COMPLETED SUCCESSFULLY", flush=True)
            else:
                # Show last errors
                lines = content.strip().split("\n")[-5:]
                for line in lines:
                    if "Error" in line or "Traceback" in line or "CUDA" in line or "OOM" in line:
                        print(f"  ERROR: {line.strip()}", flush=True)
            break
    except Exception:
        pass

    time.sleep(30)
