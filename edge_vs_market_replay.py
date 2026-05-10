"""Edge vs market study, replay-style.

Same approach as closing_gap_replay: iterate tickers seen in snaps with market
data, at each checkpoint compute bucket AND read yes_ask/yes_bid from the same
snap. Aggregate edge and PnL per (bucket, checkpoint, side).
"""
import json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/edge_vs_market_replay.json"

VELOCITY_LOOKBACK_SECONDS = 60

BACKTEST_CHECKPOINTS = [
    {"label": "3min_left", "target_seconds": 165, "min": 130, "max": 200},
    {"label": "2min_left", "target_seconds": 110, "min": 80,  "max": 130},
    {"label": "1min_left", "target_seconds": 50,  "min": 20,  "max": 80},
]

BUCKETS = [
    {"label": "extreme",     "min": 3.0,  "max": float("inf")},
    {"label": "exceptional", "min": 2.5,  "max": 3.0},
    {"label": "very_high",   "min": 2.0,  "max": 2.5},
    {"label": "high",        "min": 1.75, "max": 2.0},
    {"label": "moderate",    "min": 1.5,  "max": 1.75},
    {"label": "narrow",      "min": 1.25, "max": 1.5},
    {"label": "coinflip",    "min": 1.0,  "max": 1.25},
    {"label": "losing_side", "min": -1.0, "max": 1.0},
]
BUCKET_LABELS = [b["label"] for b in BUCKETS]


def classify(margin):
    if margin is None: return None
    for b in BUCKETS:
        if b["min"] <= margin < b["max"]:
            return b["label"]
    return "losing_side"


def parse_iso(s):
    if not s: return None
    try: return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except: return None


def composite_price(ad):
    prices = [ad.get(k) for k in ("kraken", "coinbase", "binance_us") if ad.get(k) is not None]
    return sum(prices) / len(prices) if prices else None


def normalize_price(v):
    """Normalize Kalshi price to cents (0-100). Handles dollars by multiplying."""
    if v is None: return None
    try: f = float(v)
    except: return None
    if f <= 1.0: return f * 100
    return f


def margin_and_bucket(price, strike, seconds_left, velocity):
    gap = abs(price - strike)
    if seconds_left <= 0: return None, "expired"
    if velocity is None: return None, "no_velocity"
    if velocity <= 0:
        if gap > 0: return float("inf"), "extreme"
        return 0.0, "losing_side"
    return (gap / velocity) / seconds_left, None  # bucket assigned below


def main():
    settled_by_ticker = {}
    if os.path.exists(SETTLED_PATH):
        with open(SETTLED_PATH) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: s = json.loads(line)
                except: continue
                tk = s.get("ticker"); out = s.get("outcome")
                if tk and out in ("YES", "NO"):
                    settled_by_ticker[tk] = out
    print(f"Settled tickers: {len(settled_by_ticker)}", file=sys.stderr)

    asset_series = defaultdict(list)
    ticker_info = {}
    sample_prices = []

    files = sorted(f for f in os.listdir(HIST_DIR) if f.endswith(".jsonl"))
    for fname in files:
        path = os.path.join(HIST_DIR, fname)
        print(f"  processing {fname}...", file=sys.stderr)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: snap = json.loads(line)
                except: continue
                ts = parse_iso(snap.get("ts"))
                if ts is None: continue
                t_epoch = ts.timestamp()
                for asset_name, ad in snap.get("assets", {}).items():
                    cp = composite_price(ad)
                    if cp is None: continue
                    asset_series[asset_name].append((t_epoch, cp))
                    for market in ad.get("markets", []):
                        tk = market.get("ticker")
                        if tk not in settled_by_ticker: continue
                        try: strike = float(market.get("strike"))
                        except: continue
                        ct = parse_iso(market.get("close_time"))
                        if ct is None: continue
                        ya = normalize_price(market.get("yes_ask"))
                        yb = normalize_price(market.get("yes_bid"))
                        if ya is None or yb is None: continue
                        if len(sample_prices) < 5:
                            sample_prices.append((market.get("yes_ask"), market.get("yes_bid"), ya, yb))
                        secs_left = (ct - ts).total_seconds()
                        if tk not in ticker_info:
                            ticker_info[tk] = {"asset": asset_name, "strike": strike, "snapshots": []}
                        ticker_info[tk]["snapshots"].append((t_epoch, cp, secs_left, ya, yb))

    for asset in asset_series:
        asset_series[asset].sort()
    print(f"Tickers with snaps + market data: {len(ticker_info)}", file=sys.stderr)
    if sample_prices:
        print("Sample (raw_ya, raw_yb, cents_ya, cents_yb):", file=sys.stderr)
        for sp in sample_prices:
            print(f"  {sp}", file=sys.stderr)

    agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0, "sum_cost": 0.0, "sum_pnl": 0.0}))
    n_calls = 0

    for ticker, info in ticker_info.items():
        outcome = settled_by_ticker[ticker]
        asset = info["asset"]
        strike = info["strike"]
        series = asset_series[asset]
        series_times = [t for t, _ in series]

        for cp_def in BACKTEST_CHECKPOINTS:
            in_range = [s for s in info["snapshots"]
                        if cp_def["min"] <= s[2] <= cp_def["max"]]
            if not in_range: continue
            best = min(in_range, key=lambda s: abs(s[2] - cp_def["target_seconds"]))
            t, p, sl, ya, yb = best

            cutoff_lo = t - VELOCITY_LOOKBACK_SECONDS
            lo_idx = bisect_left(series_times, cutoff_lo)
            hi_idx = bisect_right(series_times, t)
            pts = series[lo_idx:hi_idx]
            if len(pts) < 2:
                velocity = None
            else:
                elapsed = pts[-1][0] - pts[0][0]
                velocity = abs(pts[-1][1] - pts[0][1]) / elapsed if elapsed > 0 else None

            margin, special_bucket = margin_and_bucket(p, strike, sl, velocity)
            if special_bucket:
                bucket = special_bucket
            else:
                bucket = classify(margin)
            if bucket not in BUCKET_LABELS: continue

            if p > strike: side = "YES"
            elif p < strike: side = "NO"
            else: continue

            won = (side == outcome)
            if side == "YES":
                cost = ya
                pnl = (100 - ya) if won else -ya
            else:
                cost = 100 - yb
                pnl = yb if won else -(100 - yb)

            key = (cp_def["label"], side)
            agg[bucket][key]["n"] += 1
            if won: agg[bucket][key]["wins"] += 1
            agg[bucket][key]["sum_cost"] += cost
            agg[bucket][key]["sum_pnl"] += pnl
            n_calls += 1

    results = {}
    for cp_def in BACKTEST_CHECKPOINTS:
        cp_label = cp_def["label"]
        print(f"\n[{cp_label}]")
        print(f"  {'bucket':<14} {'side':>4} {'n':>5} {'win%':>6} {'mkt¢':>6} {'edge':>7} {'PnL¢':>7}")
        results[cp_label] = {}
        for b in BUCKETS:
            bl = b["label"]
            results[cp_label][bl] = {}
            for side in ("YES", "NO"):
                stats = agg[bl].get((cp_label, side))
                if not stats or stats["n"] < 5: continue
                n = stats["n"]
                wr = stats["wins"] / n
                mean_cost = stats["sum_cost"] / n
                mean_pnl = stats["sum_pnl"] / n
                edge = wr - mean_cost / 100
                print(f"  {bl:<14} {side:>4} {n:>5} {wr*100:>5.1f}% {mean_cost:>5.1f}c {edge*100:>+6.1f}% {mean_pnl:>+6.2f}")
                results[cp_label][bl][side] = {
                    "n": n, "win_rate": round(wr, 3),
                    "mean_market_cents": round(mean_cost, 2),
                    "edge": round(edge, 4),
                    "mean_pnl_cents": round(mean_pnl, 2),
                }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(ticker_info),
        "n_total_calls": n_calls,
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}", file=sys.stderr)
    print(f"n_tickers={len(ticker_info)} n_calls={n_calls}", file=sys.stderr)
    print("\nInterpretation:")
    print("  win% = actual win rate of that side")
    print("  mkt¢ = mean Kalshi cost in cents (50c = 50% market-implied probability)")
    print("  edge = win% - mkt% (positive = model beats market)")
    print("  PnL¢ = mean cents per contract; need >~2c to beat fees")


if __name__ == "__main__":
    main()
