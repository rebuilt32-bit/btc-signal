#!/bin/bash
set -e
source /root/.restic-env

echo "=== Backup started: $(date -u) ==="

restic backup \
    /var/lib/btc-signal \
    /root/btc-signal/data/settled.jsonl \
    /root/btc-signal/data/closing_gap_history \
    --tag daily

echo "=== Applying retention policy ==="
restic forget \
    --keep-daily 30 \
    --keep-weekly 12 \
    --keep-monthly 24 \
    --prune

echo "=== Backup finished: $(date -u) ==="
