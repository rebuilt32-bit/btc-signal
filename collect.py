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

ASSETS = {
    "BTC": {
        "kalshi": "KXBTC15M",
        "kraken": "XBTUSD",
        "coinbase": "BTC-USD",
        "binance_us": "BTCUSDT",
        "kraken_perp": "PF_XBTUSD",
    },
    "ETH": {
        "kalshi": "KXETH15M",
        "kraken": "ETHUSD",
        "coinbase": "ETH-USD",
        "binance_us": "ETHUSDT",
        "kraken_perp": "PF_ETHUSD",
    },
    "SOL": {
        "kalshi": "KXSOL15M",
        "kraken": "SOLUSD",
        "coinbase": "SOL-USD",
        "binance_us": "SOLUSDT",
        "kraken_perp": "PF_SOLUSD",
    },
    "XRP": {
        "kalshi": "KXXRP15M",
        "kraken": "XRPUSD",
        "coinbase": "XRP-USD",
        "binance_us": "XRPUSDT",
        "kraken_perp": "PF_XRPUSD",
    },
    "DOGE": {
        "kalshi": "KXDOGE15M",
        "kraken": "XDGUSD",
        "coinbase": "DOGE-USD",
        "binance_us": "DOGEUSDT",
        "kraken_perp": "PF_DOGEUSD",
    },
}

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"


def safe_float(v):
    try:
        if v in [None, "", "null"]:
            return None
        return float(v)
    except Exception:
        return None


def fetch_kalshi(series_ticker):
    try:
        r = requests.get(
            f"{KALSHI_BASE}/markets",
            params={
                "series_ticker": series_ticker,
                "limit": 20,
            },
            timeout=15,
        )

        r.raise_for_status()
        data = r.json()

        markets = []

        for m in data.get("markets", []):

            status = (m.get("status") or "").lower()

            # Skip unusable markets
            if status in [
                "initialized",
                "closed",
                "settled",
                "finalized",
                "expired",
            ]:
                continue

            strike = (
                m.get("floor_strike")
                or m.get("strike")
                or m.get("functional_strike")
            )

            # New Kalshi API sometimes stores strike in subtitle
            if strike is None:
                yes_sub = m.get("yes_sub_title", "")

                if "Target price:" in yes_sub:
                    try:
                        strike = float(
                            yes_sub.split("Target price:")[1]
                            .replace(",", "")
                            .strip()
                        )
                    except Exception:
                        strike = None

            markets.append({
                "ticker": m.get("ticker"),
                "strike": safe_float(strike),
                "close_time": m.get("close_time"),

                "yes_bid": safe_float(m.get("yes_bid_dollars")),
                "yes_ask": safe_float(m.get("yes_ask_dollars")),
                "no_bid": safe_float(m.get("no_bid_dollars")),
                "no_ask": safe_float(m.get("no_ask_dollars")),
                "last_price": safe_float(m.get("last_price_dollars")),

                "volume": safe_float(m.get("volume_fp")),
                "yes_bid_size": safe_float(m.get("yes_bid_size_fp")),
                "yes_ask_size": safe_float(m.get("yes_ask_size_fp")),

                "status": status,
                "title": m.get("title"),
            })

        return {"markets": markets}

    except Exception as e:
        print(f"Kalshi fetch error for {series_ticker}: {e}")
        return {"markets": []}


def fetch_coinbase(pair):
    try:
        r = requests.get(
            f"https://api.coinbase.com/v2/prices/{pair}/spot",
            timeout=10,
        )

        r.raise_for_status()

        return {
            "price": float(r.json()["data"]["amount"]),
            "error": None,
        }

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

        return {
            "price": float(j["result"][actual_key]["c"][0]),
            "error": None,
        }

    except Exception as e:
        return {"price": None, "error": str(e)}


def fetch_binance_us(symbol):
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=10,
        )

        r.raise_for_status()

        j = r.json()

        return {
            "price": float(j["price"]),
            "error": None,
        }

    except Exception as e:
        return {"price": None, "error": str(e)}


def fetch_kraken_futures_all_tickers():
    try:
        r = requests.get(
            "https://futures.kraken.com/derivatives/api/v3/tickers",
            timeout=10,
        )

        r.raise_for_status()

        j = r.json()

        tickers = j.get("tickers", [])

        symbol_map = {}

        for t in tickers:
            sym = t.get("symbol", "").upper()

            if sym:
                symbol_map[sym] = t

        return {
            "tickers": symbol_map,
            "error": None,
        }

    except Exception as e:
        return {
            "tickers": {},
            "error": str(e),
        }


def extract_funding_for_symbol(all_tickers_result, symbol):

    if all_tickers_result.get("error"):
        return {
            "funding_rate": None,
            "mark_price": None,
            "index_price": None,
            "error": all_tickers_result["error"],
        }

    tickers = all_tickers_result.get("tickers", {})

    t = tickers.get(symbol.upper())

    if not t:
        return {
            "funding_rate": None,
            "mark_price": None,
            "index_price": None,
            "error": f"{symbol} not found",
        }

    return {
        "funding_rate": safe_float(t.get("fundingRate")),
        "funding_rate_prediction": safe_float(
            t.get("fundingRatePrediction")
        ),
        "mark_price": safe_float(t.get("markPrice")),
        "index_price": safe_float(t.get("indexPrice")),
        "error": None,
    }


def collect_one():

    now = datetime.now(timezone.utc)

    result = {
        "timestamp_utc": now.isoformat(),
        "assets": {},
    }

    futures_data = fetch_kraken_futures_all_tickers()

    for asset_name, cfg in ASSETS.items():

        funding = extract_funding_for_symbol(
            futures_data,
            cfg["kraken_perp"],
        )

        result["assets"][asset_name] = {
            "kalshi": fetch_kalshi(cfg["kalshi"]),
            "kraken": fetch_kraken(cfg["kraken"]),
            "coinbase": fetch_coinbase(cfg["coinbase"]),
            "binance_us": fetch_binance_us(cfg["binance_us"]),
            "kraken_funding": funding,
        }

    return result


def write_outputs(result):

    now_iso = result["timestamp_utc"]

    now = datetime.fromisoformat(now_iso)

    slim = {
        "ts": now_iso,
        "assets": {},
    }

    for asset_name, asset_data in result["assets"].items():

        markets_slim = []

        for m in asset_data["kalshi"]["markets"]:

            markets_slim.append({
                "ticker": m.get("ticker"),
                "strike": m.get("strike"),
                "close_time": m.get("close_time"),

                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "no_bid": m.get("no_bid"),
                "no_ask": m.get("no_ask"),
                "last_price": m.get("last_price"),

                "volume": m.get("volume"),
                "yes_bid_size": m.get("yes_bid_size"),
                "yes_ask_size": m.get("yes_ask_size"),

                "status": m.get("status"),
            })

        funding_data = asset_data.get("kraken_funding", {})

        slim["assets"][asset_name] = {
            "kraken": asset_data["kraken"]["price"],
            "coinbase": asset_data["coinbase"]["price"],
            "binance_us": asset_data["binance_us"]["price"],

            "funding_rate": funding_data.get("funding_rate"),
            "mark_price": funding_data.get("mark_price"),

            "markets": markets_slim,
        }

    with open(os.path.join(OUT_DIR, "latest.json"), "w") as f:
        json.dump(result, f, indent=2)

    date_str = now.strftime("%Y-%m-%d")

    hist_path = os.path.join(
        HIST_DIR,
        f"{date_str}.jsonl",
    )

    with open(hist_path, "a") as f:
        f.write(json.dumps(slim) + "\n")

    return hist_path


for i in range(SNAPSHOTS_PER_RUN):

    try:

        result = collect_one()

        write_outputs(result)

        line = f"[{i+1}/{SNAPSHOTS_PER_RUN}] {result['timestamp_utc']}"

        for asset, data in result["assets"].items():

            kr = data["kraken"]["price"]
            cb = data["coinbase"]["price"]
            bn = data["binance_us"]["price"]

            funding_data = data.get("kraken_funding", {})

            fr = funding_data.get("funding_rate")

            mk = len(data["kalshi"]["markets"])

            line += (
                f" | {asset}: "
                f"kr={kr} "
                f"cb={cb} "
                f"bn={bn} "
                f"fr={fr} "
                f"mkts={mk}"
            )

        print(line)

    except Exception as e:

        print(f"[{i+1}/{SNAPSHOTS_PER_RUN}] FAILED: {e}")

    if i < SNAPSHOTS_PER_RUN - 1:
        time.sleep(INTERVAL_SECONDS)

print("Run complete.")
