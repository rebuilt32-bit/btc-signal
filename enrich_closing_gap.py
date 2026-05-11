"""Enrich closing_gap_live.json with live edge and trade recommendations.

Per-call enrichment:
- Maps seconds_left to checkpoint label (3min_left/2min_left/1min_left)
- Looks up historical win_rate and n from edge_vs_market_fees.json
- Computes LIVE expected PnL using current yes_ask/yes_bid + historical win_rate
- Applies Kalshi taker fee to compute net expected PnL
- Adds trade_flag based on net edge

Reads:
  data/closing_gap_live.json (produced by closing_gap_analysis.py LIVE_ONLY=1)
  data/edge_vs_market_fees.json (produced by edge_vs_market_fees.py)

Writes:
  data/closing_gap_live_enriched.json (same structure + enrichment per call)
"""
import json, os, math, sys

LIVE_PATH = "data/closing_gap_live.json"
FEES_PATH = "data/edge_vs_market_fees.json"
OUT_PATH = "data/closing_gap_live_enriched.json"

# Thresholds for trade flag (cents/contract net PnL)
THRESHOLD_BUY_STRONG = 5.0
THRESHOLD_BUY = 2.0
MIN_HISTORICAL_N = 30  # need at least this many historical samples to trust a cell


def kalshi_taker_fee_cents(price_cents):
    """Kalshi taker fee per contract. price_cents in 0-100."""
    p = max(0.0, min(1.0, price_cents / 100.0))
    return math.ceil(7 * p * (1 - p))


def normalize_price(v):
    if v is None: return None
    try: f = float(v)
    except: return None
    if f <= 1.0: return f * 100
    return f


def checkpoint_for(seconds_left):
    if seconds_left is None: return None
    if 20 <= seconds_left <= 80: return "1min_left"
    if 80 < seconds_left <= 130: return "2min_left"
    if 130 < seconds_left <= 200: return "3min_left"
    return None


def trade_flag_for(net_pnl_cents, historical_n):
    if historical_n < MIN_HISTORICAL_N:
        return "low_sample"
    if net_pnl_cents is None:
        return "no_data"
    if net_pnl_cents < 0:
        return "avoid"
    if net_pnl_cents < THRESHOLD_BUY:
        return "skip"
    if net_pnl_cents < THRESHOLD_BUY_STRONG:
        return "buy"
    return "buy_strong"


def main():
    if not os.path.exists(LIVE_PATH):
        print(f"Missing {LIVE_PATH}. Aborting.", file=sys.stderr)
        return 1
    if not os.path.exists(FEES_PATH):
        print(f"Missing {FEES_PATH}. Run edge_vs_market_fees.py first.", file=sys.stderr)
        return 1

    with open(LIVE_PATH) as f:
        live = json.load(f)
    with open(FEES_PATH) as f:
        fees_data = json.load(f)

    fees_results = fees_data.get("results", {})
    calls = live.get("live_calls", []) or []

    enriched_calls = []
    n_enriched = 0
    n_total = len(calls)

    for call in calls:
        c = dict(call)  # copy so we don't mutate input
        secs_left = c.get("seconds_left")
        bucket = c.get("bucket")
        side = c.get("safe_side")
        yes_ask_raw = c.get("market_yes_ask")
        yes_bid_raw = c.get("market_yes_bid")

        cp = checkpoint_for(secs_left)
        ya = normalize_price(yes_ask_raw)
        yb = normalize_price(yes_bid_raw)

        enrichment = {
            "checkpoint": cp,
            "historical_win_rate": None,
            "historical_n": 0,
            "live_cost_cents": None,
            "live_fee_cents": None,
            "expected_pnl_gross_cents": None,
            "expected_pnl_net_cents": None,
            "trade_flag": "no_data",
            "reason": None,
        }

        if cp is None:
            enrichment["reason"] = f"seconds_left {secs_left} outside checkpoint ranges"
        elif side not in ("YES", "NO"):
            enrichment["reason"] = f"side is {side}"
        elif bucket is None:
            enrichment["reason"] = "bucket is None"
        elif ya is None or yb is None:
            enrichment["reason"] = "missing yes_ask or yes_bid"
        else:
            cell = (
                fees_results.get(cp, {})
                .get(bucket, {})
                .get(side, None)
            )
            if cell is None:
                enrichment["reason"] = f"no historical data for {bucket}/{side}@{cp}"
            else:
                wr = cell.get("win_rate")
                n_hist = cell.get("n", 0)
                if side == "YES":
                    cost = ya
                else:
                    cost = 100 - yb
                fee = kalshi_taker_fee_cents(cost)
                expected_gross = wr * 100 - cost
                expected_net = expected_gross - fee

                enrichment["historical_win_rate"] = wr
                enrichment["historical_n"] = n_hist
                enrichment["live_cost_cents"] = round(cost, 2)
                enrichment["live_fee_cents"] = fee
                enrichment["expected_pnl_gross_cents"] = round(expected_gross, 2)
                enrichment["expected_pnl_net_cents"] = round(expected_net, 2)
                enrichment["trade_flag"] = trade_flag_for(expected_net, n_hist)
                n_enriched += 1

        c["enrichment"] = enrichment
        enriched_calls.append(c)

    out = dict(live)
    out["live_calls"] = enriched_calls
    out["enrichment_meta"] = {
        "fees_source": FEES_PATH,
        "fees_generated_at": fees_data.get("generated_at"),
        "thresholds_cents": {"buy_strong": THRESHOLD_BUY_STRONG, "buy": THRESHOLD_BUY},
        "min_historical_n": MIN_HISTORICAL_N,
        "n_enriched": n_enriched,
        "n_total": n_total,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Enriched {n_enriched}/{n_total} calls. Wrote {OUT_PATH}")

    # Print summary for any calls with trade_flag in (buy, buy_strong)
    actionable = [c for c in enriched_calls
                  if c["enrichment"]["trade_flag"] in ("buy", "buy_strong")]
    if actionable:
        print(f"\n{len(actionable)} actionable call(s):")
        for c in actionable:
            e = c["enrichment"]
            print(f"  {c['ticker'][:30]:30s} {c['asset']:5s} {c['safe_side']} "
                  f"bucket={c['bucket']:<11s} {c['seconds_left']:>3}s left "
                  f"cost={e['live_cost_cents']:.1f}c "
                  f"net=+{e['expected_pnl_net_cents']:.2f}c "
                  f"flag={e['trade_flag']} (n={e['historical_n']})")
    else:
        if n_total == 0:
            print("\nNo live calls (between rounds or outside 240s window).")
        else:
            print(f"\nNo actionable calls right now ({n_total} live but none meeting buy threshold).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
