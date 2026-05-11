"""Rebuild settled.jsonl using Kalshi's actual outcomes and expiration_values.

Backs up original. Replaces our derived outcome+settle with Kalshi's authoritative
result+expiration_value. Preserves original values as new fields for reference.
"""
import json, os, sys, time, shutil
import requests

SETTLED_PATH = "data/settled.jsonl"
BACKUP_PATH = "data/settled.jsonl.backup"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKERS = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"]


def fetch_kalshi_settled(series_ticker, max_pages=20):
    out = {}
    cursor = None
    for page in range(max_pages):
        params = {"series_ticker": series_ticker, "status": "settled", "limit": 1000}
        if cursor: params["cursor"] = cursor
        try:
            r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ERROR {series_ticker} p{page}: {e}", file=sys.stderr)
            break
        for m in data.get("markets", []):
            tk = m.get("ticker")
            res = (m.get("result") or "").lower()
            if not tk: continue
            try: ev = float(m.get("expiration_value") or 0)
            except: ev = None
            try: fs = float(m.get("floor_strike") or 0)
            except: fs = None
            outcome = "YES" if res == "yes" else ("NO" if res == "no" else None)
            out[tk] = {"outcome": outcome, "settle": ev, "floor_strike": fs}
        cursor = data.get("cursor") or None
        print(f"  {series_ticker} p{page+1}: +{len(data.get('markets', []))} (more: {bool(cursor)})", file=sys.stderr)
        if not cursor: break
        time.sleep(0.3)
    return out


def main():
    if not os.path.exists(SETTLED_PATH):
        print("No settled.jsonl found", file=sys.stderr)
        return 1

    print("Fetching all Kalshi settled outcomes...", file=sys.stderr)
    kalshi = {}
    for st in SERIES_TICKERS:
        print(f"  {st}:", file=sys.stderr)
        kalshi.update(fetch_kalshi_settled(st))
        time.sleep(0.5)
    print(f"Total Kalshi outcomes fetched: {len(kalshi)}", file=sys.stderr)

    shutil.copy(SETTLED_PATH, BACKUP_PATH)
    print(f"\nBacked up original to {BACKUP_PATH}", file=sys.stderr)

    rows = []
    with open(SETTLED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    print(f"Loaded {len(rows)} entries", file=sys.stderr)

    n_flipped = 0
    n_unknown_resolved = 0
    n_missing_kalshi = 0
    n_unchanged = 0
    new_rows = []

    for row in rows:
        tk = row.get("ticker")
        k = kalshi.get(tk)
        if not k or k["outcome"] is None:
            new_rows.append(row)
            n_missing_kalshi += 1
            continue

        old_outcome = row.get("outcome")
        old_settle = row.get("settle_avg_price")
        new_outcome = k["outcome"]
        new_settle = k["settle"]

        if old_outcome == "unknown":
            n_unknown_resolved += 1
        elif old_outcome != new_outcome:
            n_flipped += 1
        else:
            n_unchanged += 1

        new_row = dict(row)
        new_row["outcome"] = new_outcome
        new_row["settle_avg_price"] = new_settle
        new_row["our_outcome_original"] = old_outcome
        new_row["our_settle_original"] = old_settle
        new_row["outcome_source"] = "kalshi"
        new_rows.append(new_row)

    print(f"\n=== Stats ===", file=sys.stderr)
    print(f"  Total: {len(rows)}", file=sys.stderr)
    print(f"  Unchanged: {n_unchanged}", file=sys.stderr)
    print(f"  YES/NO flipped: {n_flipped}", file=sys.stderr)
    print(f"  Unknown resolved: {n_unknown_resolved}", file=sys.stderr)
    print(f"  Not in Kalshi (kept as-is): {n_missing_kalshi}", file=sys.stderr)

    with open(SETTLED_PATH, "w") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nRewrote {SETTLED_PATH}", file=sys.stderr)
    print("\nNext: re-run backtests to see corrected numbers:", file=sys.stderr)
    print("  /root/btc-signal/venv/bin/python3 closing_gap_replay.py", file=sys.stderr)
    print("  /root/btc-signal/venv/bin/python3 edge_vs_market_fees.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
