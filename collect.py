import requests
import json
import os
import time
from datetime import datetime, timezone

OUT_DIR = "data"
HIST_DIR = "data/history"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

SNAPSHOTS_PER_RUN = 10
INTERVAL_SECONDS = 30

# Each asset: Kalshi series ticker, Kraken pair, Coinbase pair
ASSETS = {
    "BTC":  {"kalshi": "KXBTC15M",  "kraken": "XBTUSD", "coinbase": "BTC-USD"},
    "ETH":  {"kalshi": "KXETH15M",  "kraken": "ETHUSD", "coinbase": "ETH-USD"},
    "SOL":  {"kalshi": "KXSOL15M",  "kraken": "SOLUSD", "coinbase": "SOL-USD"},
    "XRP":  {"kalshi": "KXXRP15M",  "kraken": "XRPUSD", "coinbase": "XRP-USD"},
    "DOGE": {"kalshi": "KXDOGE15M", "kraken": "XDGUSD", "coinbase": "DOGE-USD"},
}


def fetch_kalshi(series_ticker):
    try:
        r = requests.get(
            "https://api.elections.kalshi.com/trade-api/v2/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 20},
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
        return {"markets": enriched, "error": None}
    except Exception as e:
        return {"markets": [], "error": str(e)}


def fetch_coinbase(pair):
    try:
        r = requests.get(f"https://api.coinbase.com/v2/prices/{pair}/spot", timeout=10)
        r.raise_for_status()
        return {"price": float(r.json()["data"]["amount"]), "error": None}
    except Exception as e:
        return {"price": None, "error": str(e)}


def fetch_kraken(pair):
    try:
        r = requests.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": pair},
            timeout=10,
        )
        r.raise_for_status()
        j = r.json()
        result_keys = list(j.get("result", {}).keys())
        if not result_keys:
            return {"price": None, "error": "no result key"}
        actual_key = result_keys[0]
        return {"price": float(j["result"][actual_key]["c"][0]), "error": None}
    except Exception as e:
        return {"price": None, "error": str(e)}


def collect_one():
    now = datetime.now(timezone.utc)
    result = {"timestamp_utc": now.isoformat(), "assets": {}}

    for asset_name, cfg in ASSETS.items():
        result["assets"][asset_name] = {
            "kalshi": fetch_kalshi(cfg["kalshi"]),
            "kraken": fetch_kraken(cfg["kraken"]),
            "coinbase": fetch_coinbase(cfg["coinbase"]),
        }
    return result


def write_outputs(result):
    now_iso = result["timestamp_utc"]
    now = datetime.fromisoformat(now_iso)

    # Slim record for history log
    slim = {"ts": now_iso, "assets": {}}
    for asset_name, asset_data in result["assets"].items():
        markets_slim = []
        for entry in asset_data["kalshi"]["markets"]:
            m = entry.get("market") or {}
            markets_slim.append({
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
        slim["assets"][asset_name] = {
            "kraken": asset_data["kraken"]["price"],
            "coinbase": asset_data["coinbase"]["price"],
            "markets": markets_slim,
        }

    with open(os.path.join(OUT_DIR, "latest.json"), "w") as f:
        json.dump(result, f, indent=2)

    date_str = now.strftime("%Y-%m-%d")
    hist_path = os.path.join(HIST_DIR, f"{date_str}.jsonl")
    with open(hist_path, "a") as f:
        f.write(json.dumps(slim) + "\n")

    return hist_path


# Main loop
for i in range(SNAPSHOTS_PER_RUN):
    try:
        result = collect_one()
        write_outputs(result)
        line = f"[{i+1}/{SNAPSHOTS_PER_RUN}] {result['timestamp_utc']}"
        for asset, data in result["assets"].items():
            kr = data["kraken"]["price"]
            cb = data["coinbase"]["price"]
            mk = len(data["kalshi"]["markets"])
            line += f" | {asset}: kr={kr} cb={cb} mkts={mk}"
        print(line)
    except Exception as e:
        print(f"[{i+1}/{SNAPSHOTS_PER_RUN}] FAILED: {e}")

    if i < SNAPSHOTS_PER_RUN - 1:
        time.sleep(INTERVAL_SECONDS)

print("Run complete.")
