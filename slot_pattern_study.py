"""Slot pattern study — do markets in certain minute-slots decide earlier?"""
import json, os, re, bisect
from datetime import datetime, timezone, timedelta

HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/slot_pattern_study.json"
CHECKPOINTS_SEC = [600, 300, 120, 60]

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


def fmt(x, places=3):
    return f"{x:.{places}f}" if x is not None else "n/a"


def main():
    settled_rows = load_jsonl(SETTLED_PATH)
    settled_map = {}
    for s in settled_rows:
        tk, outcome = s.get("ticker"), s.get("outcome")
        if tk and outcome in ("YES", "NO"):
            asset, ct = parse_ticker(tk)
            if asset and ct:
                settled_map[tk] = {"outcome": outcome, "asset": asset, "close_time": ct, "slot": ct.minute}
    print(f"Settled: {len(settled_map)}")

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
                if tgt.strftime("%Y-%m-%d") != date_str: continue
                targets.append((tgt, tk, cs))
        targets.sort()
        target_times = [t[0] for t in targets]

        best = {}
        strikes = {}

        for snap in iter_jsonl(path):
            snap_t = parse_iso(snap.get("ts"))
            if not snap_t: continue
            lo = bisect.bisect_left(target_times, snap_t - timedelta(seconds=60))
            hi = bisect.bisect_right(target_times, snap_t + timedelta(seconds=60))
            if lo >= hi: continue
            for i in range(lo, hi):
                tgt, tk, cs = targets[i]
                asset = settled_map[tk]["asset"]
                ad = snap.get("assets", {}).get(asset, {})
                if not ad: continue
                if tk not in strikes:
                    for m in ad.get("markets", []):
                        if m.get("ticker") == tk:
                            try: strikes[tk] = float(m.get("strike"))
                            except: pass
                            break
                strike = strikes.get(tk)
                if strike is None: continue
                prices = [ad.get(k) for k in ("kraken", "coinbase", "binance_us") if ad.get(k) is not None]
                if not prices: continue
                cp = sum(prices) / len(prices)
                diff = abs((snap_t - tgt).total_seconds())
                if diff > 60: continue
                lead = "YES" if cp > strike else ("NO" if cp < strike else None)
                key = (tk, cs)
                if key not in best or diff < best[key][0]:
                    best[key] = (diff, lead)

        for tk in by_date[date_str]:
            meta = settled_map[tk]
            sample = {"asset": meta["asset"], "slot": meta["slot"], "outcome_yes": meta["outcome"] == "YES", "leads": {}}
            for cs in CHECKPOINTS_SEC:
                b = best.get((tk, cs))
                sample["leads"][cs] = b[1] if b else None
            samples.append(sample)
        print(f"  {date_str}: {len(by_date[date_str])} tickers")

    print(f"\nBuilt {len(samples)} samples")

    print("\n=== Lead-held rate by slot × checkpoint ===")
    print(f"  {'slot':>6} | " + " | ".join(f"   {cs}s   " for cs in CHECKPOINTS_SEC))
    results = {}
    for slot in (0, 15, 30, 45):
        ss = [s for s in samples if s["slot"] == slot]
        if not ss: continue
        row, slot_results = [], {}
        for cs in CHECKPOINTS_SEC:
            held, total = 0, 0
            for s in ss:
                lead = s["leads"].get(cs)
                if lead is None: continue
                total += 1
                if (lead == "YES") == s["outcome_yes"]:
                    held += 1
            rate = held / total if total > 0 else None
            row.append(f"{fmt(rate)}({total})")
            slot_results[f"{cs}s"] = {"n": total, "held_rate": rate}
        print(f"  :{slot:02d}    | " + " | ".join(row))
        results[f":{slot:02d}"] = slot_results

    print("\n=== Per asset × slot at 5min before close ===")
    asset_slot = {}
    for a in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        for slot in (0, 15, 30, 45):
            ss = [s for s in samples if s["asset"] == a and s["slot"] == slot]
            held, total = 0, 0
            for s in ss:
                lead = s["leads"].get(300)
                if lead is None: continue
                total += 1
                if (lead == "YES") == s["outcome_yes"]:
                    held += 1
            rate = held / total if total > 0 else None
            if total >= 30:
                print(f"  {a} :{slot:02d}: n={total} held={fmt(rate)}")
            asset_slot.setdefault(a, {})[f":{slot:02d}"] = {"n": total, "held_rate": rate}
    results["per_asset_at_5min"] = asset_slot

    with open(OUT_PATH, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "total_samples": len(samples), "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nIf one slot's held-rate is consistently higher across checkpoints,")
    print("the 'decides earlier' hypothesis is supported.")


if __name__ == "__main__":
    main()
