"""Funding rate study — memory-safe version (processes day by day)."""
import json, os, re, bisect
from datetime import datetime, timezone, timedelta
from statistics import mean

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/funding_rate_study.json"
CHECK_POINTS_SEC = [60, 300, 900]

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


def correlation(xs, ys):
    if len(xs) < 2: return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    den = (sx2 * sy2) ** 0.5
    return num / den if den > 0 else None


def fmt(x, places=4):
    return f"{x:.{places}f}" if x is not None else "n/a"


def main():
    settled_rows = load_jsonl(SETTLED_PATH)
    settled_map = {}
    for s in settled_rows:
        tk, outcome = s.get("ticker"), s.get("outcome")
        if tk and outcome in ("YES", "NO"):
            asset, ct = parse_ticker(tk)
            if asset and ct:
                settled_map[tk] = {"outcome": outcome, "asset": asset, "close_time": ct}
    print(f"Settled tickers parsed: {len(settled_map)}")

    # Group by close date
    by_date = {}
    for tk, meta in settled_map.items():
        d = meta["close_time"].strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append(tk)

    samples = []
    for date_str in sorted(by_date.keys()):
        path = os.path.join(HIST_DIR, date_str + ".jsonl")
        if not os.path.exists(path):
            continue

        # Build target list for this day's tickers
        targets = []  # list of (target_time, tk, cs)
        for tk in by_date[date_str]:
            ct = settled_map[tk]["close_time"]
            for cs in CHECK_POINTS_SEC:
                tgt = ct - timedelta(seconds=cs)
                if tgt.strftime("%Y-%m-%d") != date_str: continue
                targets.append((tgt, tk, cs))
        targets.sort()
        target_times = [t[0] for t in targets]

        # Per-ticker best matches
        best = {}  # (tk, cs) -> (diff, fr, cp)
        strikes = {}  # tk -> strike

        for snap in iter_jsonl(path):
            snap_t = parse_iso(snap.get("ts"))
            if not snap_t: continue
            lo = bisect.bisect_left(target_times, snap_t - timedelta(seconds=60))
            hi = bisect.bisect_right(target_times, snap_t + timedelta(seconds=60))
            if lo >= hi: continue

            # Cache this snap's relevant data
            for i in range(lo, hi):
                tgt, tk, cs = targets[i]
                asset = settled_map[tk]["asset"]
                ad = snap.get("assets", {}).get(asset, {})
                if not ad: continue
                # Cache strike on first sighting
                if tk not in strikes:
                    for m in ad.get("markets", []):
                        if m.get("ticker") == tk:
                            try: strikes[tk] = float(m.get("strike"))
                            except: pass
                            break
                diff = abs((snap_t - tgt).total_seconds())
                if diff > 60: continue
                key = (tk, cs)
                if key not in best or diff < best[key][0]:
                    prices = [ad.get(k) for k in ("kraken", "coinbase", "binance_us") if ad.get(k) is not None]
                    cp = sum(prices) / len(prices) if prices else None
                    best[key] = (diff, ad.get("funding_rate"), cp)

        # Build samples
        for tk in by_date[date_str]:
            meta = settled_map[tk]
            sample = {"asset": meta["asset"], "outcome_yes": meta["outcome"] == "YES"}
            strike = strikes.get(tk)
            for cs in CHECK_POINTS_SEC:
                b = best.get((tk, cs))
                if not b: continue
                _, fr, cp = b
                if fr is not None:
                    sample[f"fr_{cs}"] = fr
                    if strike is not None and cp is not None:
                        sample[f"dist_{cs}"] = cp - strike
            if "fr_60" in sample or "fr_300" in sample or "fr_900" in sample:
                samples.append(sample)

        print(f"  {date_str}: {len(by_date[date_str])} tickers")

    print(f"\nBuilt {len(samples)} samples")
    if not samples:
        print("No samples — bailing.")
        return

    results = {}

    print("\n=== 1. Raw funding_rate vs outcome ===")
    raw = {}
    for cs in CHECK_POINTS_SEC:
        valid = [(s[f"fr_{cs}"], 1 if s["outcome_yes"] else 0) for s in samples if s.get(f"fr_{cs}") is not None]
        if valid:
            xs, ys = zip(*valid)
            c = correlation(list(xs), list(ys))
            print(f"  T-{cs}s: n={len(valid)} corr={fmt(c)}")
            raw[f"{cs}s"] = {"n": len(valid), "corr": c}
    results["raw_correlation"] = raw

    print("\n=== 2. Sign split (yes_rate: pos vs neg funding) ===")
    sign = {}
    for cs in CHECK_POINTS_SEC:
        key = f"fr_{cs}"
        pos = [s for s in samples if s.get(key, 0) > 0]
        neg = [s for s in samples if s.get(key, 0) < 0]
        pos_yr = sum(1 for s in pos if s["outcome_yes"]) / len(pos) if pos else None
        neg_yr = sum(1 for s in neg if s["outcome_yes"]) / len(neg) if neg else None
        diff = (pos_yr - neg_yr) if (pos_yr is not None and neg_yr is not None) else None
        print(f"  T-{cs}s: pos n={len(pos)} yr={fmt(pos_yr, 3)} | neg n={len(neg)} yr={fmt(neg_yr, 3)} | diff={fmt(diff, 3)}")
        sign[f"{cs}s"] = {"pos": {"n": len(pos), "yes_rate": pos_yr}, "neg": {"n": len(neg), "yes_rate": neg_yr}, "diff": diff}
    results["sign_split"] = sign

    print("\n=== 3. Per-asset correlation (T-60s) ===")
    asset = {}
    for a in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        as_samples = [s for s in samples if s["asset"] == a]
        valid = [(s["fr_60"], 1 if s["outcome_yes"] else 0) for s in as_samples if s.get("fr_60") is not None]
        if len(valid) >= 50:
            xs, ys = zip(*valid)
            c = correlation(list(xs), list(ys))
            print(f"  {a}: n={len(valid)} corr={fmt(c)}")
            asset[a] = {"n": len(valid), "corr": c}
    results["per_asset"] = asset

    print("\n=== 4. |funding_rate| vs price-crossed-strike ===")
    mag = {}
    for cs in CHECK_POINTS_SEC:
        valid = []
        for s in samples:
            fr, dist = s.get(f"fr_{cs}"), s.get(f"dist_{cs}")
            if fr is None or dist is None: continue
            crossed = (dist > 0 and not s["outcome_yes"]) or (dist < 0 and s["outcome_yes"])
            valid.append((abs(fr), 1 if crossed else 0))
        if len(valid) >= 50:
            xs, ys = zip(*valid)
            c = correlation(list(xs), list(ys))
            cross_rate = sum(ys) / len(ys)
            print(f"  T-{cs}s: n={len(valid)} corr={fmt(c)} (cross_rate={fmt(cross_rate, 3)})")
            mag[f"{cs}s"] = {"n": len(valid), "corr_with_crossing": c, "overall_cross_rate": cross_rate}
    results["magnitude_vs_crossing"] = mag

    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "total_samples": len(samples), "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nInterpretation: |corr| < 0.05 = noise; 0.05-0.1 = weak; > 0.1 = notable")


if __name__ == "__main__":
    main()
