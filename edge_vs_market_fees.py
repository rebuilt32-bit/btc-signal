"""Edge vs market with Kalshi taker fees applied per trade.

Same logic as edge_vs_market_replay.py but tracks Kalshi taker fees:
  fee_cents = ceil(7 * P * (1-P)) where P = price in dollars (0-1)

Output: data/edge_vs_market_fees.json with gross AND net PnL per cell.
"""
import json, os, sys, math
from collections import defaultdict
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/edge_vs_market_fees.json"

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


def kalshi_taker_fee_cents(price_cents):
    """Kalshi taker fee per contract. price_cents in 0-100 range."""
    p = max(0.0, min(1.0, price_cents / 100.0))
    return math.ceil(7 * p * (1 - p))


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
    if v is None: return None
    try: f = float(v)
    except: return None
    if f <= 1.0: return f * 100
    return f


def margin_and_special_bucket(price, strike, seconds_left, velocity):
    gap = abs(price - strike)
    if seconds_left <= 0: return None, "expired"
    if velocity is None: return None, "no_velocity"
    if velocity <= 0:
        if gap > 0: return float("inf"), "extreme"
        return 0.0, "losing_side"
    return (gap / velocity) / seconds_left, None


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
                        secs_left = (ct - ts).total_seconds()
                        if tk not in ticker_info:
                            ticker_info[tk] = {"asset": asset_name, "strike": strike, "snapshots": []}
                        ticker_info[tk]["snapshots"].append((t_epoch, cp, secs_left, ya, yb))

    for asset in asset_series:
        asset_series[asset].sort()
    print(f"Tickers tracked: {len(ticker_info)}", file=sys.stderr)

    agg = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "wins": 0, "sum_cost": 0.0,
        "sum_pnl_gross": 0.0, "sum_fee": 0.0, "sum_pnl_net": 0.0,
    }))
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

            margin, special = margin_and_special_bucket(p, strike, sl, velocity)
            bucket = special if special else classify(margin)
            if bucket not in BUCKET_LABELS: continue

            if p > strike: side = "YES"
            elif p < strike: side = "NO"
            else: continue

            won = (side == outcome)
            if side == "YES":
                cost = ya
                pnl_gross = (100 - ya) if won else -ya
            else:
                cost = 100 - yb
                pnl_gross = yb if won else -(100 - yb)

            fee = kalshi_taker_fee_cents(cost)
            pnl_net = pnl_gross - fee

            key = (cp_def["label"], side)
            agg[bucket][key]["n"] += 1
            if won: agg[bucket][key]["wins"] += 1
            agg[bucket][key]["sum_cost"] += cost
            agg[bucket][key]["sum_pnl_gross"] += pnl_gross
            agg[bucket][key]["sum_fee"] += fee
            agg[bucket][key]["sum_pnl_net"] += pnl_net
            n_calls += 1

    results = {}
    for cp_def in BACKTEST_CHECKPOINTS:
        cp_label = cp_def["label"]
        print(f"\n[{cp_label}]  (¢/contract; net = after Kalshi taker fee)")
        print(f"  {'bucket':<14} {'side':>4} {'n':>5} {'win%':>6} {'cost':>5} {'fee':>4} {'gross':>6} {'net':>6}")
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
                mean_fee = stats["sum_fee"] / n
                mean_gross = stats["sum_pnl_gross"] / n
                mean_net = stats["sum_pnl_net"] / n
                print(f"  {bl:<14} {side:>4} {n:>5} {wr*100:>5.1f}% {mean_cost:>4.1f}c {mean_fee:>3.1f}c {mean_gross:>+5.2f} {mean_net:>+5.2f}")
                results[cp_label][bl][side] = {
                    "n": n, "win_rate": round(wr, 3),
                    "mean_cost_cents": round(mean_cost, 2),
                    "mean_fee_cents": round(mean_fee, 2),
                    "mean_pnl_gross_cents": round(mean_gross, 2),
                    "mean_pnl_net_cents": round(mean_net, 2),
                }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(ticker_info),
        "n_total_calls": n_calls,
        "fee_model": "Kalshi taker: ceil(7 * P * (1-P)) cents, P in dollars",
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}", file=sys.stderr)
    print("\nNet PnL is per-contract profit after fees, hold-to-settlement.")
    print("Cells with net > 0 are profitable; net > 2c is comfortable margin.")


if __name__ == "__main__":
    main()
