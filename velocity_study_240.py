"""Velocity formula study with 240s lookback — production-matching window."""
import json, os, re, bisect
from datetime import datetime, timezone, timedelta
from statistics import stdev as pstdev
from collections import defaultdict

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/velocity_study_240.json"
CHECKPOINTS_SEC = [180, 120, 60]
LOOKBACK_SEC = 240  # production window length

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


def velocity_stdev(pts):
    if len(pts) < 3: return None
    changes = []
    for i in range(1, len(pts)):
        dt = (pts[i][0] - pts[i-1][0]).total_seconds()
        if dt > 0:
            changes.append((pts[i][1] - pts[i-1][1]) / dt)
    if len(changes) < 2: return None
    try: return pstdev(changes)
    except: return None


def velocity_pathlen(pts):
    if len(pts) < 2: return None
    total = sum(abs(pts[i][1] - pts[i-1][1]) for i in range(1, len(pts)))
    elapsed = (pts[-1][0] - pts[0][0]).total_seconds()
    if elapsed <= 0: return None
    return total / elapsed


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


def fmt(x, places=3):
    return f"{x:.{places}f}" if x is not None else "n/a"


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
    for date_str in sorted(by_date.keys()):
        path = os.path.join(HIST_DIR, date_str + ".jsonl")
        if not os.path.exists(path): continue

        targets = []
        for tk in by_date[date_str]:
            ct = settled_map[tk]["close_time"]
            for cs in CHECKPOINTS_SEC:
                tgt = ct - timedelta(seconds=cs)
                if (tgt - timedelta(seconds=LOOKBACK_SEC)).strftime("%Y-%m-%d") != date_str: continue
                targets.append((tgt, tk, cs))
        targets.sort()
        target_times = [t[0] for t in targets]

        windows = defaultdict(list)
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

        for (tk, cs), pts in windows.items():
            if len(pts) < 3: continue
            meta = settled_map[tk]
            strike = meta["strike"]
            pts.sort(key=lambda x: x[0])
            cp_now = pts[-1][1]
            gap = abs(cp_now - strike)
            outcome_yes = meta["outcome"] == "YES"
            side = "YES" if cp_now > strike else ("NO" if cp_now < strike else None)
            if side is None: continue
            won = (side == "YES") == outcome_yes
            samples.append({
                "asset": meta["asset"], "checkpoint_sec": cs, "won": won,
                "buckets": {
                    "endpoint": get_bucket(compute_margin(gap, velocity_endpoint(pts), cs)),
                    "stdev": get_bucket(compute_margin(gap, velocity_stdev(pts), cs)),
                    "pathlen": get_bucket(compute_margin(gap, velocity_pathlen(pts), cs)),
                }
            })
        print(f"  {date_str}: {len(by_date[date_str])} tickers")

    print(f"\nBuilt {len(samples)} samples")
    if not samples:
        print("No samples")
        return

    bucket_order = [b[0] for b in BUCKETS]
    results = {}
    for formula in ("endpoint", "stdev", "pathlen"):
        results[formula] = {}
        print(f"\n[{formula}]")
        print(f"  {'bucket':<14} | " + " | ".join(f"  {cs}s   " for cs in CHECKPOINTS_SEC))
        for bucket in bucket_order:
            row = []
            results[formula][bucket] = {}
            for cs in CHECKPOINTS_SEC:
                ss = [s for s in samples if s["checkpoint_sec"] == cs and s["buckets"][formula] == bucket]
                if not ss:
                    row.append("    -    ")
                    continue
                wins = sum(1 for s in ss if s["won"])
                rate = wins / len(ss)
                row.append(f"{fmt(rate)}({len(ss):>3})")
                results[formula][bucket][f"{cs}s"] = {"n": len(ss), "win_rate": rate}
            print(f"  {bucket:<14} | " + " | ".join(row))

    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "total_samples": len(samples), "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nResearch only: do not change closing_gap_analysis.py from this output.")


if __name__ == "__main__":
    main()
