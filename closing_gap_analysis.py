import os
import json
import math
import time
from datetime import datetime, timezone

LIVE_ONLY = os.getenv("CLOSING_GAP_LIVE_ONLY", "0") == "1"

HISTORY_PATH = "data/history"
LIVE_OUTPUT = "data/closing_gap_live.json"

REFRESH_SECONDS = 5
MAX_HISTORY = 3000


# ---------------------------
# Helpers
# ---------------------------

def parse_time(ts):
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


# ---------------------------
# History
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


def build_series(rows):
    series = {}

    for r in rows:
        ts = parse_time(r.get("ts"))
        if not ts:
            continue

        for asset, data in r.get("assets", {}).items():
            price = safe_float(data.get("mark_price"))
            if price is None:
                continue

            series.setdefault(asset, []).append({
                "t": ts,
                "price": price
            })

    return series


def velocity(series, ref_time, lookback=60):
    if not series:
        return None

    ref = ref_time.timestamp()
    cutoff = ref - lookback

    window = [
        p for p in series
        if p.get("t")
        and cutoff <= p["t"].timestamp() <= ref
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
# Core computation
# ---------------------------

def compute_live():
    history = load_history()[-MAX_HISTORY:]

    now = datetime.now(timezone.utc)

    if not history:
        return {
            "generated_at": now.isoformat(),
            "live_calls": [],
            "note": "no history yet"
        }

    series = build_series(history)

    latest = history[-1]

    results = []

    for asset, data in latest.get("assets", {}).items():

        mark = safe_float(data.get("mark_price"))
        markets = data.get("markets", [])

        for m in markets:
            ticker = m.get("ticker")
            strike = safe_float(m.get("strike"))
            close_time = parse_time(m.get("close_time"))

            if not ticker or strike is None or not close_time or mark is None:
                continue

            gap = mark - strike
            seconds_left = (close_time - now).total_seconds()

            if seconds_left <= 0:
                continue

            v = velocity(series.get(asset, []), now, 60)

            if not v:
                margin = "infinite"
                seconds_to_cross = None
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
                "current_price": mark,
                "gap": round(gap, 6),
                "seconds_left": int(seconds_left),
                "velocity_per_sec": v,
                "seconds_to_cross": seconds_to_cross,
                "margin_ratio": margin,
                "bucket": bucket,
                "safe_side": safe_side,
                "safe_explanation": f"{asset} {'above' if gap > 0 else 'below'} strike",
                "market_yes_bid": m.get("yes_bid"),
                "market_yes_ask": m.get("yes_ask"),
            })

    return {
        "generated_at": now.isoformat(),
        "config": {
            "live_only": LIVE_ONLY,
            "window_seconds": 240,
            "velocity_lookback_seconds": 60,
        },
        "live_calls": results
    }


def save(result):
    os.makedirs(os.path.dirname(LIVE_OUTPUT), exist_ok=True)
    with open(LIVE_OUTPUT, "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------
# MAIN LOOP (THIS FIXES YOUR FRONTEND ISSUE)
# ---------------------------

if __name__ == "__main__":
    print("Starting closing-gap live service...")

    while True:
        try:
            result = compute_live()
            save(result)

            print(
                f"[{result['generated_at']}] "
                f"live_calls={len(result.get('live_calls', []))}"
            )

        except Exception as e:
            print("ERROR:", str(e))

        time.sleep(REFRESH_SECONDS)
