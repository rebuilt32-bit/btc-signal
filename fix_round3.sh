#!/bin/bash
set -e
cd /root/btc-signal

echo "=== Round 3: Funding rate scale + env var mode ==="

# 1. Funding rate normalization
echo "[1/3] Normalizing funding rate..."
python3 << 'FIX1'
p = '/root/btc-signal/collect_continuous.py'
c = open(p).read()
old = 'return float(t.get("fundingRate", 0))'
new = '''fr = float(t.get("fundingRate", 0))
                mark = float(t.get("markPrice", 0))
                return fr / mark if mark > 0 else None'''
if old in c:
    c = c.replace(old, new)
    open(p, 'w').write(c)
    print("  Patched: funding rate divided by markPrice")
else:
    print("  Already patched or pattern moved")
FIX1

# 2. Pass LIVE_ONLY env var to closing_gap_analysis call
echo "[2/3] Passing LIVE_ONLY env var..."
python3 << 'FIX2'
import re
p = '/root/btc-signal/collect_continuous.py'
c = open(p).read()
if 'import os' not in c:
    c = 'import os\n' + c
m = re.search(r'subprocess\.run\(\s*\[VENV_PYTHON,\s*s\][^)]*\)', c)
if m:
    old = m.group(0)
    if 'CLOSING_GAP_LIVE_ONLY' in old:
        print("  Already patched")
    else:
        new = old[:-1] + ', env={**os.environ, "CLOSING_GAP_LIVE_ONLY": "1"})'
        c = c.replace(old, new, 1)
        open(p, 'w').write(c)
        print("  Patched: LIVE_ONLY env var set")
else:
    print("  WARNING: subprocess.run pattern not found")
FIX2

echo "[3/3] Restarting service..."
systemctl restart btc-signal.service
sleep 15
echo ""
echo "=== Recent logs ==="
tail -20 /var/log/btc-signal.log
echo ""
echo "=== closing_gap_live.json ==="
head -25 /root/btc-signal/data/closing_gap_live.json 2>/dev/null || echo "Not yet written"
