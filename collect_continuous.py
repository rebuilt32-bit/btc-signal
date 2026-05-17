#!/usr/bin/env python3
"""Continuous collector for VPS. REST polling, snapshot every 1s."""
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
    j = get(f"{KALSHI_BASE}/markets", params={"series_ticker": et, "status": "open", "limit": 200})
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
            "yes_bid": str(m.get("yes_bid_dollars", "")),
            "yes_ask": str(m.get("yes_ask_dollars", "")),
            "no_bid": str(m.get("no_bid_dollars", "")),
            "no_ask": str(m.get("no_ask_dollars", "")),
            "last_price": str(m.get("last_price_dollars", "")),
            "volume": str(m.get("volume_fp", "")),
            "yes_bid_size": str(m.get("yes_bid_size_fp", "")),
            "yes_ask_size": str(m.get("yes_ask_size_fp", "")),
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
            subprocess.run(["python3", s], timeout=180, capture_output=True)
        except Exception as e:
            print(f"  {s} failed: {e}")


def main():
    print(f"Collector starting. Snapshot every {SNAPSHOT_SEC}s.")
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
            print(f"[{snap['ts']}] {n_data}/{len(ASSETS)} prices, {n_mkt} mkts, {elapsed:.2f}s")
            if n % ANALYSIS_EVERY == 0:
                run_analysis()
            time.sleep(max(0, SNAPSHOT_SEC - elapsed))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
