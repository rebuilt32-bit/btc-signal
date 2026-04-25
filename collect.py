import requests
import json
import os
import time
from datetime import datetime, timezone

OUT_DIR = "data"
HIST_DIR = "data/history"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

# How many snapshots per workflow run, and seconds between them.
# 10 snapshots * 30 sec = 5 minutes per run.
SNAPSHOTS_PER_RUN = 10
INTERVAL_SECONDS = 30


def collect_one():
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

    return result


def write_outputs(result):
    now_iso = result["timestamp_utc"]
    now = datetime.fromisoformat(now_iso)

    # Slim record for history log
    slim = {
        "ts": now_iso,
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

    # Overwrite latest snapshot
    with open(os.path.join(OUT_DIR, "latest.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Append to today's history
    date_str = now.strftime("%Y-%m-%d")
    hist_path = os.path.join(HIST_DIR, f"{date_str}.jsonl")
    with open(hist_path, "a") as f:
        f.write(json.dumps(slim) + "\n")

    return hist_path


# Main loop: collect SNAPSHOTS_PER_RUN times, INTERVAL_SECONDS apart
for i in range(SNAPSHOTS_PER_RUN):
    try:
        result = collect_one()
        hist_path = write_outputs(result)
        print(
            f"[{i+1}/{SNAPSHOTS_PER_RUN}] {result['timestamp_utc']} "
            f"kraken={result['kraken']['price']} "
            f"coinbase={result['coinbase']['price']} "
            f"markets={len(result['kalshi']['markets'])}"
        )
    except Exception as e:
        print(f"[{i+1}/{SNAPSHOTS_PER_RUN}] FAILED: {e}")

    # Don't sleep after the last collection
    if i < SNAPSHOTS_PER_RUN - 1:
        time.sleep(INTERVAL_SECONDS)

print("Run complete.")
