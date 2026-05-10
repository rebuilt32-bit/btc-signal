"""Funding rate study — tests if this signal carries any predictive value."""
import json, os, bisect
from datetime import datetime, timezone, timedelta
from statistics import mean

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/funding_rate_study.json"
CHECK_POINTS_SEC = [60, 300, 900]


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


def load_history():
    if not os.path.exists(HIST_DIR): return []
    h = []
    for fname in sorted(os.listdir(HIST_DIR)):
        if fname.endswith(".jsonl"):
            h.extend(load_jsonl(os.path.join(HIST_DIR, fname)))
    return h


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
    history = load_history()
    settled_rows = load_jsonl(SETTLED_PATH)
    settled_map = {s["ticker"]: s for s in settled_rows if s.get("ticker") and s.get("outcome") in ("YES", "NO")}
    print(f"History: {len(history)} snapshots, Settled: {len(settled_map)} tickers")

    indexed = []
    for snap in history:
        t = parse_iso(snap.get("ts"))
        if t: indexed.append((t, snap))
    indexed.sort(key=lambda x: x[0])
    times = [t for t, _ in indexed]

    ticker_meta = {}
    for _, snap in indexed:
        for asset_name, asset_data in snap.get("assets", {}).items():
            for m in asset_data.get("markets", []):
                tk = m.get("ticker")
                if tk in settled_map and tk not in ticker_meta:
                    ct = parse_iso(m.get("close_time"))
                    try: strike = float(m.get("strike"))
                    except: strike = None
                    if ct:
                        ticker_meta[tk] = {"asset": asset_name, "close_time": ct, "strike": strike}
    print(f"Meta for {len(ticker_meta)} tickers")

    samples = []
    for tk, meta in ticker_meta.items():
        sample = {"asset": meta["asset"], "outcome_yes": settled_map[tk]["outcome"] == "YES"}
        for cs in CHECK_POINTS_SEC:
            target = meta["close_time"] - timedelta(seconds=cs)
            idx = bisect.bisect_left(times, target)
            candidates = []
            for i in range(max(0, idx-2), min(len(times), idx+3)):
                diff = abs((times[i] - target).total_seconds())
                if diff <= 60:
                    candidates.append((diff, i))
            if not candidates: continue
            candidates.sort()
            best = indexed[candidates[0][1]][1]
            ad = best.get("assets", {}).get(meta["asset"], {})
            fr = ad.get("funding_rate")
            if fr is not None:
                sample[f"fr_{cs}"] = fr
                if meta["strike"] is not None:
                    prices = [ad.get(k) for k in ("kraken", "coinbase", "binance_us") if ad.get(k) is not None]
                    if prices:
                        sample[f"dist_{cs}"] = sum(prices) / len(prices) - meta["strike"]
        samples.append(sample)
    print(f"Built {len(samples)} samples")

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

    print("\n=== 3. Per-asset correlation (at T-60s) ===")
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
            print(f"  T-{cs}s: n={len(valid)} corr={fmt(c)} (overall cross_rate={fmt(cross_rate, 3)})")
            mag[f"{cs}s"] = {"n": len(valid), "corr_with_crossing": c, "overall_cross_rate": cross_rate}
    results["magnitude_vs_crossing"] = mag

    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "total_samples": len(samples), "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nInterpretation key:")
    print("- |corr| < 0.05: noise")
    print("- 0.05-0.1: weak, possibly real")
    print("- > 0.1: notable signal worth modeling")


if __name__ == "__main__":
    main()
