"""Closing-gap analysis for Kalshi 15-min crypto markets.

Modes:
  - Default (full): live calls + per-asset backtest + recent outcomes
  - Live-only (env CLOSING_GAP_LIVE_ONLY=1): live calls only
"""
import json, os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
LIVE_ONLY = os.environ.get("CLOSING_GAP_LIVE_ONLY") == "1"
OUT_PATH = "data/closing_gap_live.json" if LIVE_ONLY else "data/closing_gap_analysis.json"

WINDOW_SECONDS = 240
VELOCITY_LOOKBACK_SECONDS = 60
HISTORY_DAYS = 14
RECENT_OUTCOMES_LIMIT = 10

BUCKETS = [
    ("extreme", 3.0, float("inf")), ("exceptional", 2.5, 3.0),
    ("very_high", 2.0, 2.5), ("high", 1.5, 2.0),
    ("moderate", 1.0, 1.5), ("low", 0.5, 1.0),
    ("very_low", 0.0, 0.5), ("losing_side", float("-inf"), 0.0),
]
CHECKPOINTS = [("3min", 130, 240), ("2min", 70, 130), ("1min", 20, 70)]


def parse_iso(s):
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except: return None


def load_jsonl(path):
    if not os.path.exists(path): return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out


def load_all_history():
    if not os.path.exists(HIST_DIR): return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    all_history = []
    for fname in sorted(os.listdir(HIST_DIR)):
        if not fname.endswith(".jsonl"): continue
        if fname.replace(".jsonl", "") < cutoff: continue
        all_history.extend(load_jsonl(os.path.join(HIST_DIR, fname)))
    return all_history


def load_settled():
    settled = {}
    for row in load_jsonl(SETTLED_PATH):
        t = row.get("ticker")
        if t: settled[t] = row
    return settled


def get_bucket(margin_ratio):
    for label, lo, hi in BUCKETS:
        if margin_ratio >= lo and (hi == float("inf") or margin_ratio < hi):
            return label
    return "unknown"


def get_checkpoint(seconds_left):
    for name, lo, hi in CHECKPOINTS:
        if lo <= seconds_left <= hi: return name
    return None


def compute_margin(composite, strike, velocity_per_sec, seconds_left):
    if composite is None or strike is None: return None
    distance = composite - strike
    expected_drift = abs(velocity_per_sec or 0.0) * max(seconds_left, 1)
    noise_floor = max(abs(strike) * 0.0005, 0.01)
    return abs(distance) / max(expected_drift, noise_floor)


def compute_live(history):
    if not history: return [], None
    latest = history[-1]
    snap_time = parse_iso(latest.get("ts"))
    if not snap_time: return [], None
    cutoff_v = snap_time - timedelta(seconds=VELOCITY_LOOKBACK_SECONDS)
    velocities = {}
    for asset_name in latest.get("assets", {}):
        prices = []
        for s in history[-VELOCITY_LOOKBACK_SECONDS - 5:]:
            t = parse_iso(s.get("ts"))
            if not t or t < cutoff_v: continue
            p = s.get("assets", {}).get(asset_name, {}).get("composite_price")
            if p is not None: prices.append((t, float(p)))
        if len(prices) >= 2:
            dur = (prices[-1][0] - prices[0][0]).total_seconds()
            if dur > 0: velocities[asset_name] = (prices[-1][1] - prices[0][1]) / dur
    calls = []
    for asset_name, asset_data in latest.get("assets", {}).items():
        composite = asset_data.get("composite_price")
        if composite is None: continue
        composite = float(composite)
        velocity = velocities.get(asset_name)
        for market in asset_data.get("markets", []):
            close_time = parse_iso(market.get("close_time"))
            if not close_time: continue
            seconds_left = (close_time - snap_time).total_seconds()
            if seconds_left < 0 or seconds_left > WINDOW_SECONDS: continue
            strike = market.get("strike")
            if strike is None: continue
            try: strike = float(strike)
            except: continue
            margin = compute_margin(composite, strike, velocity, seconds_left)
            if margin is None: continue
            distance = composite - strike
            calls.append({
                "asset": asset_name, "ticker": market.get("ticker"),
                "seconds_left": round(seconds_left, 1),
                "composite_price": round(composite, 4), "strike": strike,
                "distance_dollars": round(distance, 4),
                "velocity_per_second": round(velocity, 6) if velocity is not None else None,
                "margin_ratio": round(margin, 3), "bucket": get_bucket(margin),
                "side_holding": "yes" if distance >= 0 else "no",
                "yes_bid": market.get("yes_bid"), "yes_ask": market.get("yes_ask"),
            })
    return calls, snap_time


def compute_backtest(history):
    settled = load_settled()
    if not settled: return None
    ticker_data = defaultdict(list)
    ticker_meta = {}
    for snap in history:
        snap_time = parse_iso(snap.get("ts"))
        if not snap_time: continue
        for asset_name, asset_data in snap.get("assets", {}).items():
            composite = asset_data.get("composite_price")
            if composite is None: continue
            try: composite = float(composite)
            except: continue
            for market in asset_data.get("markets", []):
                ticker = market.get("ticker")
                if not ticker or ticker not in settled: continue
                close_time = parse_iso(market.get("close_time"))
                if not close_time: continue
                seconds_left = (close_time - snap_time).total_seconds()
                if seconds_left < 0 or seconds_left > 240: continue
                strike = market.get("strike")
                if strike is None: continue
                try: strike = float(strike)
                except: continue
                if ticker not in ticker_meta:
                    ticker_meta[ticker] = {"asset": asset_name, "close_time": market.get("close_time"), "strike": strike}
                ticker_data[ticker].append({"seconds_left": seconds_left, "composite": composite, "strike": strike, "snap_time": snap_time})
    by_bucket = defaultdict(lambda: {"calls": 0, "wins": 0})
    by_bucket_x_cp = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "wins": 0}))
    by_asset = defaultdict(lambda: {"by_bucket": defaultdict(lambda: {"calls": 0, "wins": 0}),
                                     "by_bucket_x_cp": defaultdict(lambda: defaultdict(lambda: {"calls": 0, "wins": 0}))})
    recent = defaultdict(list)
    for ticker, snaps in ticker_data.items():
        meta = ticker_meta.get(ticker)
        if not meta: continue
        asset = meta["asset"]
        result_yes = settled[ticker].get("result") == "yes"
        snaps.sort(key=lambda x: -x["seconds_left"])
        for i in range(len(snaps)):
            if i > 0:
                dt = (snaps[i-1]["snap_time"] - snaps[i]["snap_time"]).total_seconds()
                snaps[i]["velocity"] = (snaps[i-1]["composite"] - snaps[i]["composite"]) / dt if dt > 0 else None
            else:
                snaps[i]["velocity"] = None
        strongest = None
        for cp_name, cp_lo, cp_hi in CHECKPOINTS:
            cp_snaps = [s for s in snaps if cp_lo <= s["seconds_left"] <= cp_hi]
            if not cp_snaps: continue
            s = cp_snaps[len(cp_snaps) // 2]
            margin = compute_margin(s["composite"], s["strike"], s.get("velocity"), s["seconds_left"])
            if margin is None: continue
            bucket = get_bucket(margin)
            distance = s["composite"] - s["strike"]
            side = "yes" if distance >= 0 else "no"
            won = (side == "yes" and result_yes) or (side == "no" and not result_yes)
            by_bucket[bucket]["calls"] += 1
            if won: by_bucket[bucket]["wins"] += 1
            by_bucket_x_cp[bucket][cp_name]["calls"] += 1
            if won: by_bucket_x_cp[bucket][cp_name]["wins"] += 1
            by_asset[asset]["by_bucket"][bucket]["calls"] += 1
            if won: by_asset[asset]["by_bucket"][bucket]["wins"] += 1
            by_asset[asset]["by_bucket_x_cp"][bucket][cp_name]["calls"] += 1
            if won: by_asset[asset]["by_bucket_x_cp"][bucket][cp_name]["wins"] += 1
            strength = next((idx for idx, b in enumerate(BUCKETS) if b[0] == bucket), 99)
            if strongest is None or strength < strongest["_strength"]:
                strongest = {"ticker": ticker, "asset": asset, "close_time": meta["close_time"],
                             "checkpoint": cp_name, "bucket": bucket, "side_holding": side,
                             "won": won, "_strength": strength}
        if strongest:
            strongest.pop("_strength", None)
            recent[asset].append(strongest)
    for a in list(recent.keys()):
        recent[a].sort(key=lambda x: x["close_time"], reverse=True)
        recent[a] = recent[a][:RECENT_OUTCOMES_LIMIT]
    return {
        "by_bucket": dict(by_bucket),
        "by_bucket_x_cp": {k: dict(v) for k, v in by_bucket_x_cp.items()},
        "by_asset": {a: {"by_bucket": dict(d["by_bucket"]),
                          "by_bucket_x_cp": {k: dict(v) for k, v in d["by_bucket_x_cp"].items()}}
                      for a, d in by_asset.items()},
        "recent_outcomes": dict(recent),
    }


def main():
    print(f"Closing-gap analysis (live_only={LIVE_ONLY})")
    history = load_all_history()
    print(f"  Loaded {len(history)} snapshots")
    calls, snap_time = compute_live(history)
    print(f"  {len(calls)} live calls in final {WINDOW_SECONDS}s window")
    for c in calls:
        print(f"    {c['asset']:5} {c['ticker']:30} {c['seconds_left']:5.0f}s  "
              f"gap=${c['distance_dollars']:.2f}  margin={c['margin_ratio']:6.2f}x ({c['bucket']:11})")
    backtest = None if LIVE_ONLY else compute_backtest(history)
    if backtest:
        n = sum(b["calls"] for b in backtest.get("by_bucket", {}).values())
        print(f"  Backtest: {n} calls across {len(backtest.get('by_asset', {}))} assets")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_snapshot_time": snap_time.isoformat() if snap_time else None,
                "config": {"window_seconds": WINDOW_SECONDS, "live_only": LIVE_ONLY},
        "live_calls": calls, "backtest": backtest,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
