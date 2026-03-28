#!/bin/bash
# ============================================================
# Training Pipeline Monitor — autonomous A→B→C→Export
# ============================================================
# Runs in background, survives SSH disconnect.
#
# Phase A (running): monitor, rotate, upload, push metrics
# Phase A done → auto-launch Phase B (execution reward)
# Phase B done → auto-launch Phase C (multi-step GiGPO)
# Phase C done → auto-run post-training (merge + push HF + GGUF)
#
# Usage:
#   nohup bash scripts/verl/monitor_training.sh > /workspace/monitor.log 2>&1 &
#   tail -f /workspace/monitor.log
#
# To stop:
#   kill $(cat /tmp/training_monitor.pid)
# ============================================================

set -uo pipefail

INTERVAL=600  # 10 minutes
CKPT_DIRS=("/home/yann/verl_checkpoints" "/workspace/topology_verl_output")
LAST_UPLOADED_STEP=0
LAST_PUSHED_STEP=0
PID_FILE="/tmp/training_monitor.pid"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="/workspace/YGN-SAGE"
PHASE_FILE="/tmp/training_phase"  # tracks current phase

# Load .env
if [ -f "$SAGE_DIR/.env" ]; then
    set -a && source "$SAGE_DIR/.env" && set +a
fi

# Prevent duplicate monitors
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[$(date)] Monitor already running (PID $(cat "$PID_FILE")). Exiting."
    exit 1
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

# Initialize phase tracker
if [ ! -f "$PHASE_FILE" ]; then
    echo "A" > "$PHASE_FILE"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

is_training_running() {
    pgrep -f "verl.trainer.main_ppo" > /dev/null 2>&1
}

is_phase_c_running() {
    pgrep -f "train_phase_c_custom" > /dev/null 2>&1
}

find_latest_checkpoint() {
    local latest_step=0
    for dir in "${CKPT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            for ckpt in "$dir"/global_step_*/; do
                [ -d "$ckpt" ] || continue
                step=$(basename "$ckpt" | grep -oP '\d+')
                if [ "$step" -gt "$latest_step" ]; then
                    latest_step=$step
                fi
            done
        fi
    done
    echo "$latest_step"
}

find_latest_checkpoint_path() {
    local latest_step=0
    local latest_path=""
    for dir in "${CKPT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            for ckpt in "$dir"/global_step_*/; do
                [ -d "$ckpt" ] || continue
                step=$(basename "$ckpt" | grep -oP '\d+')
                if [ "$step" -gt "$latest_step" ]; then
                    latest_step=$step
                    latest_path="$ckpt"
                fi
            done
        fi
    done
    echo "$latest_path"
}

checkpoint_is_stable() {
    local ckpt_dir="$1"
    local now=$(date +%s)
    local mtime=$(stat --format="%Y" "$ckpt_dir" 2>/dev/null || echo "$now")
    local age=$((now - mtime))
    [ "$age" -gt 120 ]
}

get_current_step() {
    local ray_dir=$(ls -dt /tmp/ray/session_* 2>/dev/null | head -1)
    [ -z "$ray_dir" ] && echo "0" && return
    grep -h 'training/global_step:' "$ray_dir"/logs/worker-*.out "$ray_dir"/logs/worker-*.err 2>/dev/null | \
        grep -oP 'training/global_step:\K\d+' | sort -n | tail -1 || echo "0"
}

get_latest_reward() {
    local ray_dir=$(ls -dt /tmp/ray/session_* 2>/dev/null | head -1)
    [ -z "$ray_dir" ] && echo "?" && return
    grep -h 'critic/score/mean:' "$ray_dir"/logs/worker-*.out "$ray_dir"/logs/worker-*.err 2>/dev/null | \
        grep -oP 'critic/score/mean:\K[\d.e+-]+' | tail -1 || echo "?"
}

do_upload_and_rotate() {
    local step="$1"
    log "New checkpoint step $step — rotating + uploading..."
    bash "$SCRIPT_DIR/rotate_checkpoints.sh" 1 2>&1 | while IFS= read -r line; do log "  $line"; done
    cd "$SAGE_DIR/sage-python"
    python3 scripts/verl/upload_checkpoint.py --step "$step" --keep 1 2>&1 | while IFS= read -r line; do log "  $line"; done
    LAST_UPLOADED_STEP=$step
    log "Checkpoint step $step processed"
}

do_push_metrics() {
    cd "$SAGE_DIR/sage-python"
    python3 scripts/verl/upload_checkpoint.py --metrics-only 2>&1 | while IFS= read -r line; do log "  $line"; done
}

# ── Phase transition handlers ──────────────────────────────

launch_phase_b() {
    log "=================================================="
    log "=== PHASE A COMPLETE — launching Phase B ==="
    log "=================================================="

    # Final upload
    local ckpt_step=$(find_latest_checkpoint)
    if [ "$ckpt_step" -gt "$LAST_UPLOADED_STEP" ]; then
        do_upload_and_rotate "$ckpt_step"
    fi
    do_push_metrics

    echo "B" > "$PHASE_FILE"
    LAST_UPLOADED_STEP=0
    LAST_PUSHED_STEP=0

    log "Starting Phase B: execution reward (SAGE_VERL_EXEC=1)..."
    cd "$SAGE_DIR/sage-python"
    nohup bash scripts/verl/train_topology_phase_b.sh > /workspace/train_phase_b.log 2>&1 &
    local phase_b_pid=$!
    log "Phase B launched (PID $phase_b_pid), log: /workspace/train_phase_b.log"

    # Wait for Phase B to initialize (Ray + vLLM startup)
    sleep 120
}

launch_phase_c() {
    log "=================================================="
    log "=== PHASE B COMPLETE — launching Phase C ==="
    log "=================================================="

    # Final upload
    local ckpt_step=$(find_latest_checkpoint)
    if [ "$ckpt_step" -gt "$LAST_UPLOADED_STEP" ]; then
        do_upload_and_rotate "$ckpt_step"
    fi
    do_push_metrics

    echo "C" > "$PHASE_FILE"
    LAST_UPLOADED_STEP=0
    LAST_PUSHED_STEP=0

    # Find latest checkpoint for Phase C
    local ckpt_path=$(find_latest_checkpoint_path)
    log "Starting Phase C: multi-step GiGPO (checkpoint: $ckpt_path)..."

    cd "$SAGE_DIR/sage-python"
    nohup python3 scripts/verl/train_phase_c_custom.py \
        --model /workspace/sft_merged_model \
        --checkpoint "$ckpt_path" \
        --data data/verl_topology_phase_c.parquet \
        --output /workspace/topology_verl_phase_c \
        --epochs 3 --lr 5e-7 --k 4 --batch-size 4 \
        > /workspace/train_phase_c.log 2>&1 &
    local phase_c_pid=$!
    log "Phase C launched (PID $phase_c_pid), log: /workspace/train_phase_c.log"

    sleep 60
}

run_post_training() {
    log "=================================================="
    log "=== ALL PHASES COMPLETE — post-training ==="
    log "=================================================="

    # Final upload
    local ckpt_step=$(find_latest_checkpoint)
    if [ "$ckpt_step" -gt 0 ]; then
        do_upload_and_rotate "$ckpt_step"
    fi

    echo "POST" > "$PHASE_FILE"

    cd "$SAGE_DIR/sage-python"
    log "Running post-training pipeline: export → merge → push HF → GGUF..."
    python3 scripts/verl/post_training_pipeline.py all 2>&1 | while IFS= read -r line; do log "  $line"; done

    log "=================================================="
    log "=== PIPELINE COMPLETE ==="
    log "=== Model: yannabadie/sage-topology-policy-v2 ==="
    log "=== Enable: SAGE_ENABLE_PATH6=1 ==="
    log "=================================================="

    # Final git push
    cd "$SAGE_DIR"
    git add -A && git commit -m "training: full pipeline complete (Phase A+B+C + export)" && git push origin main 2>/dev/null || true
}

# ── Main loop ──────────────────────────────────────────────

CURRENT_PHASE=$(cat "$PHASE_FILE" 2>/dev/null || echo "A")
log "=== Training Pipeline Monitor started (PID $$) ==="
log "Current phase: $CURRENT_PHASE"
log "Checkpoint dirs: ${CKPT_DIRS[*]}"
log "Interval: ${INTERVAL}s"

while true; do
    CURRENT_PHASE=$(cat "$PHASE_FILE" 2>/dev/null || echo "A")

    case "$CURRENT_PHASE" in
        A|B)
            if ! is_training_running; then
                # Training just finished
                sleep 30  # Grace period for final writes

                if [ "$CURRENT_PHASE" = "A" ]; then
                    launch_phase_b
                elif [ "$CURRENT_PHASE" = "B" ]; then
                    launch_phase_c
                fi
                continue
            fi

            # Normal monitoring: step + reward + checkpoint rotation
            current_step=$(get_current_step)
            reward=$(get_latest_reward)
            latest_ckpt_step=$(find_latest_checkpoint)

            log "[Phase $CURRENT_PHASE] Step $current_step | reward=$reward | ckpt=$latest_ckpt_step | uploaded=$LAST_UPLOADED_STEP"

            # Push metrics
            if [ "$current_step" != "0" ] && [ "$current_step" != "$LAST_PUSHED_STEP" ]; then
                do_push_metrics
                LAST_PUSHED_STEP=$current_step
            fi

            # Handle new checkpoint
            if [ "$latest_ckpt_step" -gt "$LAST_UPLOADED_STEP" ] && [ "$latest_ckpt_step" != "0" ]; then
                for dir in "${CKPT_DIRS[@]}"; do
                    ckpt_path="$dir/global_step_$latest_ckpt_step"
                    if [ -d "$ckpt_path" ] && checkpoint_is_stable "$ckpt_path"; then
                        do_upload_and_rotate "$latest_ckpt_step"
                        break
                    fi
                done
            fi
            ;;

        C)
            if ! is_phase_c_running; then
                sleep 30
                run_post_training
                continue
            fi

            # Monitor Phase C progress (different log format)
            if [ -f /workspace/train_phase_c.log ]; then
                last_line=$(tail -1 /workspace/train_phase_c.log 2>/dev/null || echo "")
                log "[Phase C] $last_line"
            fi
            ;;

        POST)
            log "Pipeline complete. Monitor exiting."
            exit 0
            ;;
    esac

    sleep "$INTERVAL"
done
