#!/usr/bin/env python3
"""Continuous collector for VPS. REST polling, snapshot every 1s."""
import time
import json, os, time, subprocess
from statistics import median
from datetime import datetime, timezone
import requests

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "BNB"]
COINBASE = {"BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD","DOGE":"DOGE-USD","HYPE":"HYPE-USD","BNB":"BNB-USD"}
KRAKEN = {"BTC":"XXBTZUSD","ETH":"XETHZUSD","SOL":"SOLUSD","XRP":"XXRPZUSD","DOGE":"XDGUSD","HYPE":"HYPEUSD","BNB":"BNBUSD"}
BINANCE_US = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT","DOGE":"DOGEUSDT","HYPE":"HYPEUSDT","BNB":"BNBUSDT"}
BINANCE_FUT = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT","DOGE":"DOGEUSDT","HYPE":"HYPEUSDT","BNB":"BNBUSDT"}
BITSTAMP = {"BTC":"btcusd","ETH":"ethusd","SOL":"solusd","XRP":"xrpusd","DOGE":"dogeusd"}
GEMINI = {"BTC":"btcusd","ETH":"ethusd","SOL":"solusd","XRP":"xrpusd","DOGE":"dogeusd"}
BULLISH = {"BTC":"BTCUSDC","ETH":"ETHUSDC"}  # CFB constituent for BTC/ETH only
CRYPTO_COM = {"BTC":"BTC_USD","ETH":"ETH_USDT","SOL":"SOL_USD","XRP":"XRP_USD"}

# CFB RTI constituents per asset (authoritative, May 2026). mark_price = median over these.
# itBit/LMAX omitted (no free API). Raw prices from all venues still logged regardless.
CONSTITUENTS = {
    "BTC":  ["coinbase", "kraken", "bitstamp", "gemini", "bullish", "crypto_com"],
    "ETH":  ["coinbase", "kraken", "bitstamp", "gemini", "bullish", "crypto_com"],
    "SOL":  ["coinbase", "kraken", "gemini", "bitstamp", "crypto_com"],
    "XRP":  ["coinbase", "kraken", "bitstamp", "crypto_com"],
    "DOGE": ["coinbase", "gemini", "kraken"],
    "BNB":  ["coinbase", "kraken"],
    "HYPE": ["bitstamp", "coinbase", "kraken"],
}
KALSHI = {"BTC":"KXBTC15M","ETH":"KXETH15M","SOL":"KXSOL15M","XRP":"KXXRP15M","DOGE":"KXDOGE15M","HYPE":"KXHYPE15M","BNB":"KXBNB15M"}

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HISTORY_DIR = os.environ.get("BTC_HISTORY_DIR", "data/history")
SNAPSHOT_SEC = 1.0
ANALYSIS_EVERY = 2  # was 5; bumped for ~2s prediction freshness
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


def bitstamp(sym):
    if not sym: return None
    try:
        j = get(f"https://www.bitstamp.net/api/v2/ticker/{sym}/")
        if j and "last" in j: return float(j["last"])
    except Exception: pass
    return None

def gemini(sym):
    if not sym: return None
    try:
        j = get(f"https://api.gemini.com/v1/pubticker/{sym}")
        if j and "last" in j: return float(j["last"])
    except Exception: pass
    return None

BULLISH_TTL = 20  # s; Bullish rate-limits hard
CC_TTL = 10       # s; Crypto.com plural ticker cache
_BULLISH_CACHE = {}              # sym -> (ts, price_or_None)
_CC_CACHE = {"ts": 0.0, "map": {}}

def bullish(sym):
    if not sym: return None
    now = time.time()
    hit = _BULLISH_CACHE.get(sym)
    if hit and now - hit[0] < BULLISH_TTL:
        return hit[1]
    price = None
    try:
        j = get(f"https://api.exchange.bullish.com/trading-api/v1/markets/{sym}/tick")
        if j and "last" in j:
            price = float(j["last"])
    except Exception:
        pass
    _BULLISH_CACHE[sym] = (now, price)  # cache even None to throttle
    return price

def crypto_com_map():
    now = time.time()
    if now - _CC_CACHE["ts"] < CC_TTL:
        return _CC_CACHE["map"]
    _CC_CACHE["ts"] = now  # back off regardless of outcome
    try:
        j = get("https://api.crypto.com/exchange/v1/public/get-tickers")
        if j and j.get("code") == 0:
            out = {}
            for d in j.get("result", {}).get("data", []):
                inst = d.get("i"); a = d.get("a")
                if inst and a is not None:
                    try: out[inst] = float(a)
                    except Exception: pass
            if out:
                _CC_CACHE["map"] = out
    except Exception:
        pass
    return _CC_CACHE["map"]

def crypto_com(sym):
    if not sym: return None
    return crypto_com_map().get(sym)

def funding_rate(sym):
    return None  # fapi.binance.com is 451 from cloud IPs; skip
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
        bs = bitstamp(BITSTAMP.get(a))
        gm = gemini(GEMINI.get(a))
        bl = bullish(BULLISH.get(a))
        cc = crypto_com(CRYPTO_COM.get(a))
        fr = funding_rate(BINANCE_FUT[a])
        mk = kalshi_markets(KALSHI[a])
        all_prices = {"kraken": k, "coinbase": c, "bitstamp": bs, "gemini": gm, "bullish": bl, "crypto_com": cc}
        constituents = CONSTITUENTS.get(a, ["coinbase", "kraken"])
        prices = [all_prices[n] for n in constituents if all_prices.get(n) is not None]
        mark = median(prices) if prices else None
        snap["assets"][a] = {
            "kraken": k, "coinbase": c, "binance_us": bu, "bitstamp": bs, "gemini": gm,
            "bullish": bl, "crypto_com": cc,
            "funding_rate": fr, "mark_price": mark, "markets": mk,
        }
    snap["ts"] = datetime.now(timezone.utc).isoformat()  # restamp: current as of cycle end
    return snap


def write_snap(snap):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(f"{HISTORY_DIR}/{date}.jsonl", "a") as f:
        f.write(json.dumps(snap) + "\n")


def run_analysis():
    return  # Disabled; btc-analyze-loop.service runs analyze.py at 2s cadence
    for s in ["analyze.py"]:  # calibrate + attribution moved to cron
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
