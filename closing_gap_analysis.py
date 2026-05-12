import os
import json
import math
from datetime import datetime, timezone

LIVE_ONLY = os.getenv("CLOSING_GAP_LIVE_ONLY", "0") == "1"

HISTORY_PATH = "data/history"
LIVE_OUTPUT = "data/closing_gap_live.json"


# ---------------------------
# Helpers
# ---------------------------

def parse_time(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except:
        return None


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except:
        return None


def normalize_market(m):
    """
    Supports any past/future format safely.
    Current collector uses FLAT structure.
    """
    if not m:
        return {}

    if isinstance(m.get("market"), dict):
        return m["market"]

    return m


# ---------------------------
# History loader
# ---------------------------

def load_history():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(HISTORY_PATH, f"{today}.jsonl")

    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                continue

    return rows


def build_series(history_rows):
    """
    Build per-asset mark price time series.
    """
    series = {}

    for row in history_rows:
        ts = parse_time(row.get("ts"))
        if not ts:
            continue

        assets = row.get("assets", {})

        for asset, data in assets.items():
            price = safe_float(data.get("mark_price"))
            if price is None:
                continue

            series.setdefault(asset, []).append({
                "t": ts,
                "price": price
            })

    return series


def velocity(series, ref_time, lookback=60):
    """
    absolute price/sec over last window
    """
    if not series:
        return None

    ref_ts = ref_time.timestamp()
    cutoff = ref_ts - lookback

    window = [
        p for p in series
        if p.get("t")
        and cutoff <= p["t"].timestamp() <= ref_ts
    ]

    if len(window) < 2:
        return None

    window.sort(key=lambda x: x["t"])

    dt = (window[-1]["t"] - window[0]["t"]).total_seconds()
    if dt <= 0:
        return None

    move = abs(window[-1]["price"] - window[0]["price"])
    if move == 0:
        return None

    return move / dt


# ---------------------------
# Core logic
# ---------------------------

def compute_live():
    history = load_history()
    history = history[-3000:]
    series = build_series(history)

    if not history:
        return {"error": "no history"}

    latest = history[-1]
    now = parse_time(latest.get("ts"))

    results = []

    for asset, data in latest.get("assets", {}).items():

        markets = data.get("markets", [])
        mark_price = safe_float(data.get("mark_price"))

        for m in markets:

            m = normalize_market(m)

            ticker = m.get("ticker")
            strike = safe_float(m.get("strike"))
            yes_bid = safe_float(m.get("yes_bid"))
            yes_ask = safe_float(m.get("yes_ask"))
            close_time = parse_time(m.get("close_time"))

            if not ticker or strike is None or not close_time or mark_price is None:
                continue

            gap = mark_price - strike
            seconds_left = (close_time - now).total_seconds()

            if seconds_left <= 0:
                continue

            v = velocity(series.get(asset, []), now, 60)

            if v is None or v == 0:
                seconds_to_cross = None
                margin = "infinite"
                bucket = "no_velocity"
            else:
                seconds_to_cross = abs(gap) / v
                margin = seconds_to_cross / seconds_left

                if margin > 1.5:
                    bucket = "safe"
                elif margin > 0.8:
                    bucket = "neutral"
                else:
                    bucket = "risky"

            safe_side = "YES" if gap > 0 else "NO"

            results.append({
                "ticker": ticker,
                "asset": asset,
                "strike": strike,
                "current_price": mark_price,
                "gap": round(gap, 6),
                "seconds_left": int(seconds_left),

                "velocity_per_sec": v,
                "seconds_to_cross": seconds_to_cross,
                "margin_ratio": margin,
                "bucket": bucket,

                "safe_side": safe_side,
                "safe_explanation": f"{asset} {'above' if gap > 0 else 'below'} strike",

                "market_yes_bid": yes_bid,
                "market_yes_ask": yes_ask,
            })

    return {
        "generated_at": now.isoformat(),
        "config": {
            "window_seconds": 240,
            "velocity_lookback_seconds": 60,
            "live_only": LIVE_ONLY
        },
        "live_calls": results
    }


def save(result):
    os.makedirs("data", exist_ok=True)

    with open(LIVE_OUTPUT, "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":

    result = compute_live()
    save(result)

    print(f"Closing-gap analysis complete: {len(result.get('live_calls', []))} markets")

    for x in result.get("live_calls", []):
        print(
            f"{x['asset']} {x['ticker']} "
            f"gap={x['gap']} "
            f"margin={x['margin_ratio']} "
            f"bucket={x['bucket']}"
        )
