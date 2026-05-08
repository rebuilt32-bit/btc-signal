#!/bin/bash
# One-shot VPS setup for btc-signal.
# Run from /root/btc-signal: `bash setup.sh`
set -e

echo "=== BTC Signal VPS Setup ==="

# 1. Install system packages
echo ""
echo "[1/6] Installing packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip python3-venv git nginx ufw curl

# 2. Firewall
echo "[2/6] Configuring firewall..."
ufw allow 22/tcp || true
ufw allow 80/tcp || true
ufw --force enable

# 3. Python venv
echo "[3/6] Python virtual environment..."
cd /root/btc-signal
if [ ! -d venv ]; then
    python3 -m venv venv
fi
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet requests

# 4. Create the collector script
echo "[4/6] Creating collector..."
cat > /root/btc-signal/collect_continuous.py << 'COLLECTOR_PY'
#!/usr/bin/env python3
"""Continuous collector. REST polling every 1s, analysis every 5s."""
import json, os, time, subprocess
from datetime import datetime, timezone
import requests

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
COINBASE = {"BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD","DOGE":"DOGE-USD"}
KRAKEN = {"BTC":"XXBTZUSD","ETH":"XETHZUSD","SOL":"SOLUSD","XRP":"XXRPZUSD","DOGE":"XDGUSD"}
BINANCE_US = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT","DOGE":"DOGEUSDT"}
BINANCE_FUT = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT","DOGE":"DOGEUSDT"}
KALSHI = {"BTC":"KXBTC15M","ETH":"KXETH15M","SOL":"KXSOL15M","XRP":"KXXRP15M","DOGE":"KXDOGE15M"}

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HISTORY_DIR = "data/history"
SNAPSHOT_SEC = 1.0
ANALYSIS_EVERY = 5
S = requests.Session()
S.headers["User-Agent"] = "btc-signal-vps/1.0"

def get(url, **kw):
    try:
        r = S.get(url, timeout=3, **kw)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def coinbase(sym):
    j = get(f"https://api.exchange.coinbase.com/products/{sym}/ticker")
    return float(j["price"]) if j else None

def kraken(sym):
    j = get("https://api.kraken.com/0/public/Ticker", params={"pair": sym})
    if not j or not j.get("result"):
        return None
    for v in j["result"].values():
        return float(v["c"][0])
    return None

def binance_us(sym):
    j = get("https://api.binance.us/api/v3/ticker/price", params={"symbol": sym})
    return float(j["price"]) if j else None

def funding_rate(sym):
    j = get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": sym})
    return float(j["lastFundingRate"]) if j else None

def kalshi_markets(et):
    j = get(f"{KALSHI_BASE}/markets", params={"event_ticker": et, "status": "open", "limit": 200})
    if not j:
        return []
    out = []
    for m in j.get("markets", []):
        if m.get("status") != "active":
            continue
        out.append({
            "ticker": m.get("ticker"),
            "strike": m.get("strike_price") or m.get("floor_strike") or m.get("cap_strike"),
            "close_time": m.get("close_time"),
            "yes_bid": str(m.get("yes_bid", "")),
            "yes_ask": str(m.get("yes_ask", "")),
            "no_bid": str(m.get("no_bid", "")),
            "no_ask": str(m.get("no_ask", "")),
            "last_price": str(m.get("last_price", "")),
            "volume": str(m.get("volume", "")),
            "yes_bid_size": str(m.get("yes_bid_size", "")),
            "yes_ask_size": str(m.get("yes_ask_size", "")),
            "status": m.get("status", "active"),
        })
    return out

def collect():
    now = datetime.now(timezone.utc)
    snap = {"ts": now.isoformat(), "assets": {}}
    for a in ASSETS:
        k = kraken(KRAKEN[a])
        c = coinbase(COINBASE[a])
        bu = binance_us(BINANCE_US[a])
        fr = funding_rate(BINANCE_FUT[a])
        mk = kalshi_markets(KALSHI[a])
        prices = [p for p in [k, c, bu] if p is not None]
        mark = sum(prices) / len(prices) if prices else None
        snap["assets"][a] = {
            "kraken": k, "coinbase": c, "binance_us": bu,
            "funding_rate": fr, "mark_price": mark, "markets": mk,
        }
    return snap

def write_snap(snap):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(f"{HISTORY_DIR}/{date}.jsonl", "a") as f:
        f.write(json.dumps(snap) + "\n")

def run_analysis():
    for s in ["analyze.py", "calibrate.py", "signal_attribution.py", "closing_gap_analysis.py"]:
        try:
            subprocess.run(["python3", s], timeout=30, capture_output=True)
        except Exception as e:
            print(f"  {s} failed: {e}", flush=True)

def main():
    print(f"Collector starting. Snapshot every {SNAPSHOT_SEC}s.", flush=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    n = 0
    while True:
        try:
            t0 = time.time()
            snap = collect()
            write_snap(snap)
            n += 1
            n_data = sum(1 for d in snap["assets"].values() if d["mark_price"])
            n_mkt = sum(len(d["markets"]) for d in snap["assets"].values())
            elapsed = time.time() - t0
            print(f"[{snap['ts']}] {n_data}/{len(ASSETS)} prices, {n_mkt} mkts, {elapsed:.2f}s", flush=True)
            if n % ANALYSIS_EVERY == 0:
                run_analysis()
            time.sleep(max(0, SNAPSHOT_SEC - elapsed))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(2)

if __name__ == "__main__":
    main()
COLLECTOR_PY

# 5. Update index.html to read from local files instead of GitHub
echo "[5/6] Updating frontend URLs..."
python3 << 'HTML_FIX_PY'
import re
path = '/root/btc-signal/index.html'
with open(path, 'r') as f:
    c = f.read()
c = re.sub(r'const DATA_URL = .*?;', 'const DATA_URL = "/data/prediction.json";', c)
c = re.sub(r'const GAP_URL = .*?;', 'const GAP_URL = "/data/closing_gap_analysis.json";', c)
c = c.replace("url + '&t=' + Date.now()", "url + '?t=' + Date.now()")
with open(path, 'w') as f:
    f.write(c)
print("index.html configured for VPS")
HTML_FIX_PY

# 6. nginx + systemd
echo "[6/6] nginx + systemd..."
cat > /etc/nginx/sites-available/btc-signal << 'NGINX_CFG'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    root /root/btc-signal;
    index index.html;
    location ~ \.json$ {
        add_header Access-Control-Allow-Origin *;
        add_header Cache-Control "no-cache";
        try_files $uri =404;
    }
    location / {
        try_files $uri $uri/ =404;
    }
}
NGINX_CFG
ln -sf /etc/nginx/sites-available/btc-signal /etc/nginx/sites-enabled/btc-signal
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

cat > /etc/systemd/system/btc-signal.service << 'SYSTEMD_SVC'
[Unit]
Description=BTC Signal Continuous Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/btc-signal
ExecStart=/root/btc-signal/venv/bin/python3 /root/btc-signal/collect_continuous.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/btc-signal.log
StandardError=append:/var/log/btc-signal.log

[Install]
WantedBy=multi-user.target
SYSTEMD_SVC
systemctl daemon-reload
systemctl enable btc-signal.service
systemctl restart btc-signal.service

echo ""
echo "=== Setup complete ==="
sleep 5
systemctl status btc-signal.service --no-pager | head -10
IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
echo ""
echo "Frontend: http://$IP/"
echo "Logs:     tail -f /var/log/btc-signal.log"
echo "Service:  systemctl status btc-signal.service"
