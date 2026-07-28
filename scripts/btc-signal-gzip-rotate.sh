#!/bin/bash
# Gzip yesterday-and-older daily JSONL in predictions + history to bound disk.
# Skips the current UTC day (still being written). Live path only reads today's raw file.
set -uo pipefail
TODAY=$(date -u +%Y-%m-%d)
for d in /var/lib/btc-signal/predictions /var/lib/btc-signal/history /root/btc-signal/data/closing_gap_history; do
  [ -d "$d" ] || continue
  for f in "$d"/*.jsonl; do
    [ -e "$f" ] || continue
    base=$(basename "$f" .jsonl)
    [ "$base" = "$TODAY" ] && continue
    gzip -f "$f"
  done
done
