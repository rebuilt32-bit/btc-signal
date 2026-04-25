import requests
import json
import os
from datetime import datetime, timezone

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

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

with open(os.path.join(OUT_DIR, "latest.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"Collected at {result['timestamp_utc']}")
print(f"  Kalshi markets: {len(result['kalshi']['markets'])}")
print(f"  Coinbase: {result['coinbase']['price']}")
print(f"  Kraken: {result['kraken']['price']}")
