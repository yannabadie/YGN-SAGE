#!/bin/bash
# Rotate checkpoints: keep only the latest one to stay within 80GB quota
# Run this after each checkpoint is saved (save_freq=50)

CKPT_DIR="/workspace/topology_verl_output"
MAX_KEEP=1  # Keep only the latest checkpoint

# Find all checkpoint dirs sorted by step number
CKPTS=$(find "$CKPT_DIR" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n)
NUM_CKPTS=$(echo "$CKPTS" | grep -c "global_step_")

if [ "$NUM_CKPTS" -le "$MAX_KEEP" ]; then
    echo "[$(date)] $NUM_CKPTS checkpoints, nothing to rotate"
    exit 0
fi

# Delete all but the latest
TO_DELETE=$(echo "$CKPTS" | head -n -"$MAX_KEEP")
for ckpt in $TO_DELETE; do
    STEP=$(basename "$ckpt" | grep -oP '\d+')
    SIZE=$(du -sh "$ckpt" | cut -f1)
    echo "[$(date)] Deleting old checkpoint step $STEP ($SIZE)"
    rm -rf "$ckpt"
done

echo "[$(date)] Kept: $(find "$CKPT_DIR" -maxdepth 1 -name "global_step_*" -type d | sort -t_ -k3 -n | tail -1)"
echo "[$(date)] Workspace usage: $(du -sh /workspace/ | cut -f1) / 80GB"
