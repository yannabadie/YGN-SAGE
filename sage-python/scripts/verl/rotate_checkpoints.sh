#!/bin/bash
# Rotate checkpoints across both storage locations.
# Keeps only the latest N checkpoints (default: 1) globally.
# Safe to run from cron or manually.
#
# Usage:
#   bash rotate_checkpoints.sh          # keep 1
#   bash rotate_checkpoints.sh 2        # keep 2

set -euo pipefail

MAX_KEEP=${1:-1}
CKPT_DIRS=("/home/yann/verl_checkpoints" "/workspace/topology_verl_output")

# Collect all checkpoints across both dirs, sorted by step number
ALL_CKPTS=""
for dir in "${CKPT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        for ckpt in "$dir"/global_step_*/; do
            [ -d "$ckpt" ] && ALL_CKPTS="$ALL_CKPTS $ckpt"
        done
    fi
done

if [ -z "$ALL_CKPTS" ]; then
    echo "[$(date)] No checkpoints found"
    exit 0
fi

# Sort by step number (extract number from path)
SORTED=$(echo "$ALL_CKPTS" | tr ' ' '\n' | grep -v '^$' | \
    awk -F'global_step_' '{print $2 " " $0}' | sed 's|/ | |' | \
    sort -n | awk '{print $2}')

NUM=$(echo "$SORTED" | wc -l)

if [ "$NUM" -le "$MAX_KEEP" ]; then
    echo "[$(date)] $NUM checkpoint(s), keeping all (max=$MAX_KEEP)"
    exit 0
fi

# Delete all but the latest MAX_KEEP
TO_DELETE=$(echo "$SORTED" | head -n -"$MAX_KEEP")
KEPT=$(echo "$SORTED" | tail -n "$MAX_KEEP")

for ckpt in $TO_DELETE; do
    STEP=$(basename "$ckpt" | grep -oP '\d+')
    SIZE=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
    echo "[$(date)] Deleting step $STEP ($SIZE) from $(dirname "$ckpt")"
    rm -rf "$ckpt"
done

echo "[$(date)] Kept: $KEPT"
echo "[$(date)] Disk: NVMe=$(df -h / | tail -1 | awk '{print $3"/"$2}') | Workspace=$(df -h /workspace | tail -1 | awk '{print $3"/"$2}')"
