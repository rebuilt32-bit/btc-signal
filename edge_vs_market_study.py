"""Edge vs market study — does our bucket signal beat Kalshi's price?"""
import json, os, re, bisect
from datetime import datetime, timezone, timedelta
from collections import defaultdict

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/edge_vs_market_study.json"
CHECKPOINTS_SEC = [180, 120, 60]
LOOKBACK_SEC = 240  # match production WINDOW_SECONDS

BUCKETS = [
    ("extreme",     3.0,  float("inf")),
    ("exceptional", 2.5,  3.0),
    ("very_high",   2.0,  2.5),
    ("high",        1.75, 2.0),
    ("moderate",    1.5,  1.75),
    ("narrow",      1.25, 1.5),
    ("coinflip",    1.0,  1.25),
    ("losing_side", -1.0, 1.0),
]

TICKER_RE = re.compile(r'^KX([A-Z]+)15M-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})-')
MONTHS = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}


def parse_ticker(ticker):
    m = TICKER_RE.match(ticker)
    if not m: return None, None
    asset, yy, mon, dd, hh, mm = m.groups()
    try:
        ct = datetime(2000 + int(yy), MONTHS[mon], int(dd), int(hh), int(mm), tzinfo=timezone.utc)
        return asset, ct
    except: return None, None


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


def iter_jsonl(path):
    if not os.path.exists(path): return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except: pass


def composite(ad):
    prices = [ad.get(k) for k in ("kraken", "coinbase", "binance_us") if ad.get(k) is not None]
    return sum(prices) / len(prices) if prices else None


def velocity_endpoint(pts):
    if len(pts) < 2: return None
    elapsed = (pts[-1][0] - pts[0][0]).total_seconds()
    if elapsed <= 0: return None
    return abs(pts[-1][1] - pts[0][1]) / elapsed


def get_bucket(margin):
    if margin is None: return None
    for label, lo, hi in BUCKETS:
        if margin >= lo and (hi == float("inf") or margin < hi):
            return label
    return "losing_side"


def compute_margin(gap, velocity, seconds_left):
    if velocity is None or seconds_left <= 0: return None
    if velocity <= 0: return float("inf") if gap > 0 else 0.0
    return (gap / velocity) / seconds_left


def get_market(ad, ticker):
    """Find market in snap matching ticker."""
    for m in ad.get("markets", []):
        if m.get("ticker") == ticker:
            return m
    return None


def normalize_price(v):
    """Kalshi yes_ask/yes_bid may be in cents (0-100) or dollars (0-1). Normalize to cents."""
    if v is None: return None
    try: f = float(v)
    except: return None
    if f <= 1.0: return f * 100
    return f


def main():
    settled_rows = load_jsonl(SETTLED_PATH)
    settled_map = {}
    for s in settled_rows:
        tk, outcome = s.get("ticker"), s.get("outcome")
        if tk and outcome in ("YES", "NO"):
            asset, ct = parse_ticker(tk)
            if not (asset and ct): continue
            try: strike = float(s.get("strike"))
            except: continue
            settled_map[tk] = {"outcome": outcome, "asset": asset, "close_time": ct, "strike": strike}
    print(f"Settled with strike: {len(settled_map)}")

    by_date = {}
    for tk, meta in settled_map.items():
        d = meta["close_time"].strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append(tk)

    samples = []
    no_market_match = 0
    sample_yes_ask_values = []  # for debugging price scale

    for date_str in sorted(by_date.keys()):
        path = os.path.join(HIST_DIR, date_str + ".jsonl")
        if not os.path.exists(path): continue

        targets = []
        for tk in by_date[date_str]:
            ct = settled_map[tk]["close_time"]
            for cs in CHECKPOINTS_SEC:
                tgt = ct - timedelta(seconds=cs)
                if (tgt - timedelta(seconds=LOOKBACK_SEC)).strftime("%Y-%m-%d") != date_str:
                    continue
                targets.append((tgt, tk, cs))
        targets.sort()
        target_times = [t[0] for t in targets]

        windows = defaultdict(list)
        market_at_cp = {}  # (tk, cs) -> (diff, yes_ask_cents, yes_bid_cents)

        for snap in iter_jsonl(path):
            snap_t = parse_iso(snap.get("ts"))
            if not snap_t: continue
            lo = bisect.bisect_left(target_times, snap_t)
            hi = bisect.bisect_right(target_times, snap_t + timedelta(seconds=LOOKBACK_SEC))
            for i in range(lo, hi):
                tgt, tk, cs = targets[i]
                if not (tgt - timedelta(seconds=LOOKBACK_SEC) <= snap_t <= tgt):
                    continue
                meta = settled_map[tk]
                ad = snap.get("assets", {}).get(meta["asset"], {})
                if not ad: continue
                cp = composite(ad)
                if cp is None: continue
                windows[(tk, cs)].append((snap_t, cp))

                diff = (tgt - snap_t).total_seconds()
                if 0 <= diff <= 30:
                    m = get_market(ad, tk)
                    if m:
                        ya = normalize_price(m.get("yes_ask"))
                        yb = normalize_price(m.get("yes_bid"))
                        if ya is not None and yb is not None:
                            if len(sample_yes_ask_values) < 5:
                                sample_yes_ask_values.append((m.get("yes_ask"), ya))
                            cur = market_at_cp.get((tk, cs))
                            if cur is None or diff < cur[0]:
                                market_at_cp[(tk, cs)] = (diff, ya, yb)

        for (tk, cs), pts in windows.items():
            if len(pts) < 3: continue
            mkt = market_at_cp.get((tk, cs))
            if mkt is None:
                no_market_match += 1
                continue
            _, yes_ask, yes_bid = mkt
            meta = settled_map[tk]
            pts.sort(key=lambda x: x[0])
            cp_now = pts[-1][1]
            strike = meta["strike"]
            gap = abs(cp_now - strike)
            vel = velocity_endpoint(pts)
            bucket = get_bucket(compute_margin(gap, vel, cs))
            outcome_yes = meta["outcome"] == "YES"
            side = "YES" if cp_now > strike else ("NO" if cp_now < strike else None)
            if side is None: continue
            won = (side == "YES") == outcome_yes
            samples.append({
                "asset": meta["asset"], "checkpoint_sec": cs, "bucket": bucket, "side": side,
                "yes_ask": yes_ask, "yes_bid": yes_bid, "won": won,
            })
        print(f"  {date_str}: {len(by_date[date_str])} tickers")

    print(f"\nBuilt {len(samples)} samples ({no_market_match} skipped — no market match)")
    if sample_yes_ask_values:
        print(f"Sample yes_ask values (raw, normalized cents): {sample_yes_ask_values}")
    if not samples:
        return

    print("\n=== Edge vs market by bucket × checkpoint × side ===")
    print("PnL = mean cents per contract bet (positive = profitable, need >~2c after fees)")

    bucket_order = [b[0] for b in BUCKETS]
    results = {}

    for cs in CHECKPOINTS_SEC:
        print(f"\n[{cs}s before close]")
        print(f"  {'bucket':<14} {'side':>4} {'n':>5} {'win%':>6} {'mkt%':>6} {'edge':>+7} {'PnL¢':>+7}")
        results[f"{cs}s"] = {}
        for bucket in bucket_order:
            for side in ("YES", "NO"):
                ss = [s for s in samples if s["checkpoint_sec"] == cs and s["bucket"] == bucket and s["side"] == side]
                if len(ss) < 5: continue
                n = len(ss)
                wins = sum(1 for s in ss if s["won"])
                wr = wins / n
                if side == "YES":
                    cost = sum(s["yes_ask"] for s in ss) / n
                    pnls = [(100 - s["yes_ask"]) if s["won"] else (-s["yes_ask"]) for s in ss]
                else:
                    cost = sum(100 - s["yes_bid"] for s in ss) / n
                    pnls = [s["yes_bid"] if s["won"] else (-(100 - s["yes_bid"])) for s in ss]
                mean_pnl = sum(pnls) / n
                mkt_p = cost / 100
                edge = wr - mkt_p
                print(f"  {bucket:<14} {side:>4} {n:>5} {wr*100:>5.1f}% {mkt_p*100:>5.1f}% {edge*100:>+6.1f}% {mean_pnl:>+6.2f}")
                results[f"{cs}s"].setdefault(bucket, {})[side] = {
                    "n": n, "win_rate": wr, "mean_market_cents": cost,
                    "edge": edge, "mean_pnl_cents": mean_pnl,
                }

    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "total_samples": len(samples),
                   "no_market_match_skipped": no_market_match,
                   "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nInterpretation:")
    print("  win% = how often that side actually won")
    print("  mkt% = what the market priced (mean cost / 100)")
    print("  edge = win% - mkt% (model's advantage over market price)")
    print("  PnL¢ = mean cents per contract; needs >~2c to beat fees")


if __name__ == "__main__":
    main()
