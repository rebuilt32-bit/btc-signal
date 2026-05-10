"""Memory-efficient replay of closing_gap_analysis.compute_backtest().

Streams history day-by-day, keeps only essentials (per-asset price series,
per-ticker observations). Replicates production bucket/margin/velocity math
exactly. Does NOT modify production code.
"""
import json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/closing_gap_replay.json"

WINDOW_SECONDS = 240
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


def margin_and_bucket(price, strike, seconds_left, velocity):
    gap = abs(price - strike)
    if seconds_left <= 0: return None, "expired"
    if velocity is None: return None, "no_velocity"
    if velocity <= 0:
        if gap > 0: return float("inf"), "extreme"
        return 0.0, "losing_side"
    secs_to_cross = gap / velocity
    margin = secs_to_cross / seconds_left
    return margin, classify(margin)


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
                        secs_left = (ct - ts).total_seconds()
                        if tk not in ticker_info:
                            ticker_info[tk] = {"asset": asset_name, "strike": strike, "snapshots": []}
                        ticker_info[tk]["snapshots"].append((t_epoch, cp, secs_left))

    for asset in asset_series:
        asset_series[asset].sort()
    print(f"Asset series sizes: {[(a, len(s)) for a, s in asset_series.items()]}", file=sys.stderr)
    print(f"Tickers with snaps: {len(ticker_info)}", file=sys.stderr)

    by_bucket = defaultdict(lambda: {"n": 0, "wins": 0, "n_uncalled": 0})
    by_bucket_x_cp = defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0}))
    by_asset = defaultdict(lambda: {
        "by_bucket": defaultdict(lambda: {"n": 0, "wins": 0, "n_uncalled": 0}),
        "by_bucket_x_cp": defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0})),
    })
    n_calls = 0

    for ticker, info in ticker_info.items():
        outcome = settled_by_ticker[ticker]
        asset = info["asset"]
        strike = info["strike"]
        series = asset_series[asset]
        series_times = [t for t, _ in series]

        for cp_def in BACKTEST_CHECKPOINTS:
            in_range = [(t, p, sl) for (t, p, sl) in info["snapshots"]
                        if cp_def["min"] <= sl <= cp_def["max"]]
            if not in_range: continue
            best = min(in_range, key=lambda s: abs(s[2] - cp_def["target_seconds"]))
            t, p, sl = best

            cutoff_lo = t - VELOCITY_LOOKBACK_SECONDS
            lo_idx = bisect_left(series_times, cutoff_lo)
            hi_idx = bisect_right(series_times, t)
            pts = series[lo_idx:hi_idx]
            if len(pts) < 2:
                velocity = None
            else:
                elapsed = pts[-1][0] - pts[0][0]
                velocity = abs(pts[-1][1] - pts[0][1]) / elapsed if elapsed > 0 else None

            margin, bucket = margin_and_bucket(p, strike, sl, velocity)

            if p > strike: side = "YES"
            elif p < strike: side = "NO"
            else: side = None

            if side is None:
                if bucket in (b["label"] for b in BUCKETS):
                    by_bucket[bucket]["n_uncalled"] += 1
                    by_asset[asset]["by_bucket"][bucket]["n_uncalled"] += 1
                continue
            if bucket not in (b["label"] for b in BUCKETS):
                continue
            won = (side == outcome)
            by_bucket[bucket]["n"] += 1
            by_bucket_x_cp[bucket][cp_def["label"]]["n"] += 1
            by_asset[asset]["by_bucket"][bucket]["n"] += 1
            by_asset[asset]["by_bucket_x_cp"][bucket][cp_def["label"]]["n"] += 1
            n_calls += 1
            if won:
                by_bucket[bucket]["wins"] += 1
                by_bucket_x_cp[bucket][cp_def["label"]]["wins"] += 1
                by_asset[asset]["by_bucket"][bucket]["wins"] += 1
                by_asset[asset]["by_bucket_x_cp"][bucket][cp_def["label"]]["wins"] += 1

    def make_bucket_summary(bb):
        out = {}
        for b in BUCKETS:
            label = b["label"]
            stats = bb.get(label, {"n": 0, "wins": 0, "n_uncalled": 0})
            n = stats["n"]
            out[label] = {
                "n_calls": n,
                "n_wins": stats["wins"],
                "win_rate": round(stats["wins"] / n, 3) if n > 0 else None,
                "n_at_strike_skipped": stats.get("n_uncalled", 0),
            }
        return out

    def make_bucket_x_cp(bxcp):
        out = {}
        for b in BUCKETS:
            label = b["label"]
            out[label] = {}
            for cp_def in BACKTEST_CHECKPOINTS:
                cp_label = cp_def["label"]
                stats = bxcp.get(label, {}).get(cp_label, {"n": 0, "wins": 0})
                n = stats["n"]
                out[label][cp_label] = {
                    "n": n,
                    "win_rate": round(stats["wins"] / n, 3) if n > 0 else None,
                }
        return out

    by_asset_out = {}
    for asset, ad in by_asset.items():
        by_asset_out[asset] = {
            "by_bucket": make_bucket_summary(ad["by_bucket"]),
            "by_bucket_x_checkpoint": make_bucket_x_cp(ad["by_bucket_x_cp"]),
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_settled_tickers_evaluated": len(ticker_info),
        "n_total_calls": n_calls,
        "by_bucket": make_bucket_summary(by_bucket),
        "by_bucket_x_checkpoint": make_bucket_x_cp(by_bucket_x_cp),
        "by_asset": by_asset_out,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}", file=sys.stderr)

    print("\n=== OVERALL by_bucket ===")
    print(json.dumps(result["by_bucket"], indent=2))
    print("\n=== BY CHECKPOINT ===")
    print(json.dumps(result["by_bucket_x_checkpoint"], indent=2))
    print(f"\nn_tickers={result['n_settled_tickers_evaluated']} n_calls={result['n_total_calls']}")


if __name__ == "__main__":
    main()
