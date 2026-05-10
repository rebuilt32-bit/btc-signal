"""Diagnose strike field + structure in history snapshots."""
import json, os
from glob import glob

HIST_DIR = "data/history"
files = sorted(glob(os.path.join(HIST_DIR, "*.jsonl")))
if not files:
    print("No files"); exit(1)

# Sample three files: oldest, middle, newest
sample_files = [files[0], files[len(files)//2], files[-1]]

for path in sample_files:
    print(f"\n=== {os.path.basename(path)} ===")
    snaps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: snaps.append(json.loads(line))
            except: pass
    if not snaps:
        print("  no parseable lines"); continue
    print(f"  snaps in file: {len(snaps)}")

    snap = snaps[-1]  # latest snap of the day
    print(f"  top keys: {list(snap.keys())}")
    assets = snap.get("assets", {})
    print(f"  assets present: {list(assets.keys())}")

    for an, ad in assets.items():
        if not isinstance(ad, dict): continue
        markets = ad.get("markets", [])
        print(f"  [{an}] keys={list(ad.keys())[:8]} markets_count={len(markets)}")
        if markets:
            m = markets[0]
            print(f"    first market keys: {list(m.keys())}")
            strike_keys = {k: v for k, v in m.items() if 'strike' in k.lower()}
            print(f"    strike-like fields: {strike_keys}")
            ticker_v = m.get("ticker")
            print(f"    ticker: {ticker_v}")
        break  # one asset per file is enough
