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

# Each asset: Kalshi series ticker, Kraken pair, Coinbase pair, Binance.US pair, Kraken Futures perp symbol
# Funding rates from Kraken Futures (US-licensed, geo-open from US IPs).
# Kraken Futures uses "PF_" prefix for linear perpetuals (e.g., PF_XBTUSD).
# Note: DOGE on Kraken uses "XDG" not "DOGE".
ASSETS = {
    "BTC":  {"kalshi": "KXBTC15M",  "kraken": "XBTUSD", "coinbase": "BTC-USD",
             "binance_us": "BTCUSDT", "kraken_perp": "PF_XBTUSD"},
    "ETH":  {"kalshi": "KXETH15M",  "kraken": "ETHUSD", "coinbase": "ETH-USD",
             "binance_us": "ETHUSDT", "kraken_perp": "PF_ETHUSD"},
    "SOL":  {"kalshi": "KXSOL15M",  "kraken": "SOLUSD", "coinbase": "SOL-USD",
             "binance_us": "SOLUSDT", "kraken_perp": "PF_SOLUSD"},
    "XRP":  {"kalshi": "KXXRP15M",  "kraken": "XRPUSD", "coinbase": "XRP-USD",
             "binance_us": "XRPUSDT", "kraken_perp": "PF_XRPUSD"},
    "DOGE": {"kalshi": "KXDOGE15M", "kraken": "XDGUSD", "coinbase": "DOGE-USD",
             "binance_us": "DOGEUSDT", "kraken_perp": "PF_XDGUSD"},
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


def fetch_binance_us(symbol):
    """Fetch spot price from Binance.US (US-licensed, geo-open from US IPs)."""
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=10,
        )
        r.raise_for_status()
        j = r.json()
        return {"price": float(j["price"]), "error": None}
    except Exception as e:
        return {"price": None, "error": str(e)}


def fetch_kraken_futures_all_tickers():
    """
    Fetch all Kraken Futures tickers in one call.
    Returns dict mapping symbol -> ticker data, plus an error key if it fails.
    """
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
        return {"tickers": symbol_map, "error": None}
    except Exception as e:
        return {"tickers": {}, "error": str(e)}


def extract_funding_for_symbol(all_tickers_result, symbol):
    """Pull funding rate + mark/index from the bulk tickers response for a given symbol."""
    if all_tickers_result.get("error"):
        return {"funding_rate": None, "mark_price": None, "index_price": None,
                "error": all_tickers_result["error"]}
    tickers = all_tickers_result.get("tickers", {})
    t = tickers.get(symbol.upper())
    if not t:
        return {"funding_rate": None, "mark_price": None, "index_price": None,
                "error": f"symbol {symbol} not found in {len(tickers)} tickers"}
    return {
        "funding_rate": t.get("fundingRate"),
        "funding_rate_prediction": t.get("fundingRatePrediction"),
        "mark_price": t.get("markPrice"),
        "index_price": t.get("indexPrice"),
        "error": None,
    }


def collect_one():
    now = datetime.now(timezone.utc)
    result = {"timestamp_utc": now.isoformat(), "assets": {}}

    # Fetch Kraken Futures bulk tickers ONCE for this snapshot, then look up each asset
    futures_data = fetch_kraken_futures_all_tickers()

    for asset_name, cfg in ASSETS.items():
        funding = extract_funding_for_symbol(futures_data, cfg["kraken_perp"])
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
        funding_data = asset_data.get("kraken_funding", {})
        slim["assets"][asset_name] = {
            "kraken": asset_data["kraken"]["price"],
            "coinbase": asset_data["coinbase"]["price"],
            "binance_us": asset_data.get("binance_us", {}).get("price"),
            "funding_rate": funding_data.get("funding_rate"),
            "mark_price": funding_data.get("mark_price"),
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
            bn = data.get("binance_us", {}).get("price")
            funding_data = data.get("kraken_funding", {})
            fr = funding_data.get("funding_rate")
            fr_err = funding_data.get("error")
            mk = len(data["kalshi"]["markets"])
            if fr is None and fr_err:
                line += f" | {asset}: kr={kr} cb={cb} bn={bn} fr=ERR({fr_err[:40]}) mkts={mk}"
            else:
                line += f" | {asset}: kr={kr} cb={cb} bn={bn} fr={fr} mkts={mk}"
        print(line)
    except Exception as e:
        print(f"[{i+1}/{SNAPSHOTS_PER_RUN}] FAILED: {e}")

    if i < SNAPSHOTS_PER_RUN - 1:
        time.sleep(INTERVAL_SECONDS)

print("Run complete.")
