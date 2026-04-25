import requests
import json
import os
from datetime import datetime, timezone

OUT_DIR = "data"
HIST_DIR = "data/history"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

now = datetime.now(timezone.utc)

result = {
    "timestamp_utc": now.isoformat(),
    "kalshi": {"markets": [], "error": None},
    "kraken": {"price": None, "error": None},
    "coinbase": {"price": None, "error": None},
}

# Kalshi: open KXBTC15M markets
try:
    r = requests.get(
        "https://api.elections.kalshi.com/trade-api/v2/markets",
        params={"series_ticker": "KXBTC15M", "status": "open", "limit": 20},
        timeout=15,
    )
    r.raise_for_status()
    markets = r.json().get("markets", [])
    enriched = []
    for m in markets[:5]:
        ticker = m.get("ticker")
        try:
            ob = requests.get(
                f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook",
                timeout=10,
            )
            ob.raise_for_status()
            orderbook = ob.json().get("orderbook")
        except Exception as e:
            orderbook = {"error": str(e)}
        enriched.append({"market": m, "orderbook": orderbook})
    result["kalshi"]["markets"] = enriched
except Exception as e:
    result["kalshi"]["error"] = str(e)

# Coinbase spot
try:
    r = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=10)
    r.raise_for_status()
    result["coinbase"]["price"] = float(r.json()["data"]["amount"])
except Exception as e:
    result["coinbase"]["error"] = str(e)

# Kraken spot
try:
    r = requests.get(
        "https://api.kraken.com/0/public/Ticker",
        params={"pair": "XBTUSD"},
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    pair = list(j["result"].keys())[0]
    result["kraken"]["price"] = float(j["result"][pair]["c"][0])
except Exception as e:
    result["kraken"]["error"] = str(e)

# Build a slim record for the history log (one line per snapshot)
slim = {
    "ts": result["timestamp_utc"],
    "kraken": result["kraken"]["price"],
    "coinbase": result["coinbase"]["price"],
    "markets": [],
}
for entry in result["kalshi"]["markets"]:
    m = entry.get("market") or {}
    slim["markets"].append({
        "ticker": m.get("ticker"),
        "strike": m.get("floor_strike"),
        "close_time": m.get("close_time"),
        "yes_bid": m.get("yes_bid_dollars"),
        "yes_ask": m.get("yes_ask_dollars"),
        "no_bid": m.get("no_bid_dollars"),
        "no_ask": m.get("no_ask_dollars"),
        "last_price": m.get("last_price_dollars"),
        "volume": m.get("volume_fp"),
        "yes_bid_size": m.get("yes_bid_size_fp"),
        "yes_ask_size": m.get("yes_ask_size_fp"),
        "status": m.get("status"),
    })

# Write latest snapshot (overwrites each minute)
with open(os.path.join(OUT_DIR, "latest.json"), "w") as f:
    json.dump(result, f, indent=2)

# Append slim record to today's history file (one JSON object per line, "JSONL")
date_str = now.strftime("%Y-%m-%d")
hist_path = os.path.join(HIST_DIR, f"{date_str}.jsonl")
with open(hist_path, "a") as f:
    f.write(json.dumps(slim) + "\n")

print(f"Collected at {result['timestamp_utc']}")
print(f"  Kalshi markets: {len(result['kalshi']['markets'])}")
print(f"  Coinbase: {result['coinbase']['price']}")
print(f"  Kraken: {result['kraken']['price']}")
print(f"  Appended to: {hist_path}")
