"""Compare our derived outcomes to Kalshi's actual outcomes.

Pulls all settled markets per series_ticker from Kalshi, builds ticker -> outcome map,
compares to settled.jsonl. Reports disagreement rate and lists every disagreement
with our settle vs Kalshi's expiration_value.

Does NOT modify settled.jsonl. Read-only diagnostic.
"""
import json, os, sys, time
import requests

SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/outcome_verification.json"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKERS = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"]


def fetch_kalshi_settled(series_ticker, max_pages=20):
    outcomes = {}  # ticker -> (result, expiration_value, floor_strike)
    cursor = None
    for page in range(max_pages):
        params = {"series_ticker": series_ticker, "status": "settled", "limit": 1000}
        if cursor: params["cursor"] = cursor
        try:
            r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ERROR {series_ticker} page {page}: {e}", file=sys.stderr)
            break
        markets = data.get("markets", [])
        for m in markets:
            tk = m.get("ticker")
            res = (m.get("result") or "").lower()
            if not tk: continue
            try: ev = float(m.get("expiration_value") or 0)
            except: ev = None
            try: fs = float(m.get("floor_strike") or 0)
            except: fs = None
            outcome = "YES" if res == "yes" else ("NO" if res == "no" else None)
            outcomes[tk] = (outcome, ev, fs)
        cursor = data.get("cursor") or None
        print(f"  {series_ticker} page {page+1}: +{len(markets)} (more: {bool(cursor)})", file=sys.stderr)
        if not cursor: break
        time.sleep(0.3)
    return outcomes


def load_settled():
    rows = []
    if not os.path.exists(SETTLED_PATH): return rows
    with open(SETTLED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    return rows


def main():
    our = load_settled()
    print(f"Our settled entries: {len(our)}", file=sys.stderr)

    print("\nFetching Kalshi outcomes per series...", file=sys.stderr)
    kalshi = {}
    for st in SERIES_TICKERS:
        print(f"  {st}:", file=sys.stderr)
        kalshi.update(fetch_kalshi_settled(st))
        time.sleep(0.5)
    print(f"\nFetched {len(kalshi)} Kalshi outcomes total", file=sys.stderr)

    matches = 0
    disagreements = []
    not_found = []
    settle_diffs = []  # (our_settle - kalshi_expiration_value) for those we can compare

    for row in our:
        tk = row.get("ticker")
        our_out = row.get("outcome")
        our_settle = row.get("settle_avg_price")
        k = kalshi.get(tk)
        if not k or k[0] is None:
            not_found.append(tk)
            continue
        kalshi_out, kalshi_ev, kalshi_fs = k
        if our_out == kalshi_out:
            matches += 1
        else:
            disagreements.append({
                "ticker": tk,
                "our_outcome": our_out,
                "kalshi_outcome": kalshi_out,
                "strike": row.get("strike"),
                "our_settle": our_settle,
                "kalshi_settle": kalshi_ev,
                "gap_at_strike": (abs(our_settle - row.get("strike")) if our_settle and row.get("strike") else None),
            })
        if our_settle is not None and kalshi_ev is not None:
            settle_diffs.append(our_settle - kalshi_ev)

    n_compared = matches + len(disagreements)
    rate = matches / n_compared if n_compared else 0
    print(f"\n=== Results ===", file=sys.stderr)
    print(f"  Compared: {n_compared}", file=sys.stderr)
    print(f"  Matches: {matches}", file=sys.stderr)
    print(f"  Disagreements: {len(disagreements)}", file=sys.stderr)
    print(f"  Not in Kalshi: {len(not_found)}", file=sys.stderr)
    print(f"  Agreement: {rate*100:.2f}%", file=sys.stderr)

    if settle_diffs:
        import statistics
        sd_abs = [abs(d) for d in settle_diffs]
        print(f"\nSettle diff (ours - Kalshi):", file=sys.stderr)
        print(f"  n={len(settle_diffs)} mean_abs={statistics.mean(sd_abs):.6f} max_abs={max(sd_abs):.6f}", file=sys.stderr)

    if disagreements:
        print(f"\nFirst 20 disagreements:", file=sys.stderr)
        for d in disagreements[:20]:
            gap = d.get("gap_at_strike")
            gap_str = f"{gap:.5f}" if gap else "n/a"
            print(f"  {d['ticker'][:35]:35s} our={d['our_outcome']}/{d['our_settle']} kalshi={d['kalshi_outcome']}/{d['kalshi_settle']} strike={d['strike']} gap_to_strike={gap_str}", file=sys.stderr)

    result = {
        "n_our_settled": len(our),
        "n_kalshi_outcomes": len(kalshi),
        "n_compared": n_compared,
        "n_matches": matches,
        "n_disagreements": len(disagreements),
        "n_not_found": len(not_found),
        "agreement_rate": round(rate, 4),
        "settle_diff_mean_abs": round(sum(abs(d) for d in settle_diffs)/len(settle_diffs), 6) if settle_diffs else None,
        "disagreements": disagreements,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
