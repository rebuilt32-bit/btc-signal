#!/bin/bash
set -e
cd /root/btc-signal

echo "=== Applying round 1 fixes ==="

# Fix 1: Speed up closing_gap_analysis by limiting history scan to last 14 days
echo "[1/2] Patching closing_gap_analysis.py..."
python3 << 'FIX1'
import re
p = '/root/btc-signal/closing_gap_analysis.py'
c = open(p).read()
pattern = r'def load_all_history\(\):.*?return all_history'
new_func = '''def load_all_history():
    if not os.path.exists(HIST_DIR):
        return []
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    all_history = []
    for fname in sorted(os.listdir(HIST_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        date_part = fname.replace(".jsonl", "")
        if date_part < cutoff:
            continue
        all_history.extend(load_jsonl(os.path.join(HIST_DIR, fname)))
    return all_history'''
c2 = re.sub(pattern, new_func, c, count=1, flags=re.DOTALL)
if c2 != c:
    open(p, 'w').write(c2)
    print("  closing_gap_analysis.py patched")
else:
    print("  WARNING: closing_gap_analysis.py — pattern not matched")
FIX1

# Fix 2: Switch funding_rate to Kraken Futures (Binance Futures is geo-blocked)
echo "[2/2] Switching funding_rate to Kraken Futures..."
python3 << 'FIX2'
import re
p = '/root/btc-signal/collect_continuous.py'
c = open(p).read()
pattern = r'def funding_rate\(sym\):.*?return float\(j\["lastFundingRate"\]\) if j else None'
new_func = '''def funding_rate(asset):
    """Fetch funding rate from Kraken Futures perpetuals."""
    try:
        j = get("https://futures.kraken.com/derivatives/api/v3/tickers")
        if not j or "tickers" not in j:
            return None
        sym_map = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD", "XRP": "PF_XRPUSD", "DOGE": "PF_XDGUSD"}
        target = sym_map.get(asset)
        if not target:
            return None
        for t in j["tickers"]:
            if t.get("symbol") == target:
                return float(t.get("fundingRate", 0))
    except Exception:
        return None
    return None'''
c2 = re.sub(pattern, new_func, c, count=1, flags=re.DOTALL)
c2 = c2.replace("fr = funding_rate(BINANCE_FUT[a])", "fr = funding_rate(a)")
if c2 != c:
    open(p, 'w').write(c2)
    print("  collect_continuous.py patched")
else:
    print("  WARNING: collect_continuous.py — pattern not matched")
FIX2

echo ""
echo "Restarting service..."
systemctl restart btc-signal.service
sleep 20
echo ""
echo "=== Recent logs ==="
tail -30 /var/log/btc-signal.log
echo ""
echo "=== Kraken Futures sample (verify field name) ==="
curl -s "https://futures.kraken.com/derivatives/api/v3/tickers" | python3 -c "import sys,json; d=json.load(sys.stdin); t=[x for x in d.get('tickers',[]) if 'PF_XBTUSD' in x.get('symbol','')]; print(json.dumps(t[0] if t else {}, indent=2))"
