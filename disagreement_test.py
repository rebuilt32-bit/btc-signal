import json
import os
from collections import defaultdict
from datetime import datetime

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/disagreement_test.json"

# Realistic Kalshi cost assumption per trade.
# Their fee structure is roughly 7% of contracts traded for retail.
# Plus bid-ask spread eats some on entry/exit.
# 0.05 (5%) is conservative for total cost.
TRADING_COST_PER_TRADE = 0.05

# Disagreement thresholds to test
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_all_predictions():
    rows = []
    if not os.path.exists(PRED_DIR):
        return rows
    for fname in sorted(os.listdir(PRED_DIR)):
        if fname.endswith(".jsonl"):
            rows.extend(load_jsonl(os.path.join(PRED_DIR, fname)))
    return rows


def simulate_trade(direction, yes_bid, yes_ask, outcome_yes, cost=TRADING_COST_PER_TRADE):
    """
    direction: "YES" or "NO"
    yes_bid, yes_ask: market quotes for YES side, in 0-1 range
    outcome_yes: 1 if market settled YES, else 0
    Returns gross profit on a 1-unit bet.
    """
    if direction == "YES":
        # We pay ask to buy YES. If outcome YES, we receive 1.
        if yes_ask is None or yes_ask <= 0 or yes_ask >= 1:
            return None
        cost_paid = yes_ask
        payout = 1.0 if outcome_yes == 1 else 0.0
        gross = payout - cost_paid
    else:  # NO
        # NO contract costs (1 - yes_bid). If outcome NO, we receive 1.
        if yes_bid is None or yes_bid <= 0 or yes_bid >= 1:
            return None
        cost_paid = 1.0 - yes_bid
        payout = 1.0 if outcome_yes == 0 else 0.0
        gross = payout - cost_paid

    net = gross - cost
    return {"gross": gross, "net": net}


def evaluate_strategy(predictions_with_outcomes, threshold, flip=False):
    """
    Iterate through predictions. Skip if disagreement < threshold.
    For each remaining prediction, place a trade.
    flip=False: trade WITH our disagreement (if we say more YES, bet YES)
    flip=True: trade AGAINST our disagreement (if we say more YES, bet NO)
    """
    trades = []
    for p in predictions_with_outcomes:
        our_prob = p.get("our_prob")
        market_mid = p.get("market_mid")
        yes_bid = p.get("yes_bid")
        yes_ask = p.get("yes_ask")
        outcome_yes = p.get("outcome_yes")

        if our_prob is None or market_mid is None or yes_bid is None or yes_ask is None:
            continue

        disagreement = our_prob - market_mid
        if abs(disagreement) < threshold:
            continue

        # Direction we'd take: based on our model's lean vs market
        if flip:
            # Flipped: trade opposite of our disagreement
            direction = "NO" if disagreement > 0 else "YES"
        else:
            direction = "YES" if disagreement > 0 else "NO"

        result = simulate_trade(direction, yes_bid, yes_ask, outcome_yes)
        if result is None:
            continue
        trades.append({
            "ticker": p.get("ticker"),
            "asset": p.get("asset"),
            "seconds_left": p.get("seconds_left"),
            "our_prob": our_prob,
            "market_mid": market_mid,
            "disagreement": disagreement,
            "direction": direction,
            "outcome_yes": outcome_yes,
            "won": (
                (direction == "YES" and outcome_yes == 1)
                or (direction == "NO" and outcome_yes == 0)
            ),
            "gross_profit": round(result["gross"], 4),
            "net_profit": round(result["net"], 4),
        })

    if not trades:
        return None

    n = len(trades)
    n_wins = sum(1 for t in trades if t["won"])
    total_gross = sum(t["gross_profit"] for t in trades)
    total_net = sum(t["net_profit"] for t in trades)

    return {
        "n_trades": n,
        "n_wins": n_wins,
        "win_rate": round(n_wins / n, 3),
        "total_gross_profit_per_unit_bet": round(total_gross, 3),
        "total_net_profit_per_unit_bet_after_fees": round(total_net, 3),
        "mean_gross_per_trade": round(total_gross / n, 4),
        "mean_net_per_trade": round(total_net / n, 4),
        "expectancy_pct": round((total_net / n) * 100, 2),
        "_trades_sample_first_5": trades[:5],
    }


def main():
    predictions = load_all_predictions()
    settlements = load_jsonl(SETTLED_PATH)

    settled_by_ticker = {}
    for s in settlements:
        ticker = s.get("ticker")
        outcome = s.get("outcome")
        if ticker and outcome and outcome != "unknown":
            settled_by_ticker[ticker] = outcome

    joined = []
    for p in predictions:
        ticker = p.get("ticker")
        if ticker not in settled_by_ticker:
            continue

        our_prob = p.get("prob_yes_estimate")
        market_mid = p.get("market_mid")
        yes_bid = p.get("market_yes_bid")
        yes_ask = p.get("market_yes_ask")

        if our_prob is None:
            continue

        joined.append({
            "ticker": ticker,
            "asset": p.get("asset"),
            "seconds_left": p.get("seconds_left"),
            "our_prob": our_prob,
            "market_mid": market_mid,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "outcome_yes": 1 if settled_by_ticker[ticker] == "YES" else 0,
        })

    if not joined:
        result = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "note": "No matched data yet."
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("No data.")
        return

    # Run both strategies at each threshold
    results_by_threshold = []
    for thresh in THRESHOLDS:
        with_disagreement = evaluate_strategy(joined, thresh, flip=False)
        against_disagreement = evaluate_strategy(joined, thresh, flip=True)

        results_by_threshold.append({
            "threshold": thresh,
            "trade_with_disagreement": with_disagreement,
            "trade_against_disagreement": against_disagreement,
        })

    # Phase breakdown for the most interesting threshold (10%)
    phase_buckets = {
        "very_early_10min+": [j for j in joined if j["seconds_left"] is not None and j["seconds_left"] >= 600],
        "early_5_10min": [j for j in joined if j["seconds_left"] is not None and 300 <= j["seconds_left"] < 600],
        "mid_2_5min": [j for j in joined if j["seconds_left"] is not None and 120 <= j["seconds_left"] < 300],
        "late_1_2min": [j for j in joined if j["seconds_left"] is not None and 60 <= j["seconds_left"] < 120],
        "final_minute": [j for j in joined if j["seconds_left"] is not None and j["seconds_left"] < 60],
    }

    phase_results = {}
    for phase_name, items in phase_buckets.items():
        if not items:
            continue
        with_disagree = evaluate_strategy(items, 0.10, flip=False)
        against_disagree = evaluate_strategy(items, 0.10, flip=True)
        phase_results[phase_name] = {
            "trade_with": with_disagree,
            "trade_against": against_disagree,
        }

    # By asset breakdown at 10% threshold
    asset_buckets = defaultdict(list)
    for j in joined:
        if j.get("asset"):
            asset_buckets[j["asset"]].append(j)

    asset_results = {}
    for asset, items in asset_buckets.items():
        with_disagree = evaluate_strategy(items, 0.10, flip=False)
        against_disagree = evaluate_strategy(items, 0.10, flip=True)
        asset_results[asset] = {
            "trade_with": with_disagree,
            "trade_against": against_disagree,
        }

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_matched_predictions": len(joined),
        "trading_cost_assumption_per_trade": TRADING_COST_PER_TRADE,
        "interpretation": (
            "We tested two strategies on the data: "
            "'trade_with_disagreement' = bet the side our model leans more than the market. "
            "'trade_against_disagreement' = bet the opposite of our model's lean (the flip hypothesis). "
            "Look at total_net_profit_per_unit_bet_after_fees. Positive = strategy made money. "
            "Negative = strategy lost money. Compare the two strategies to see if flipping helps."
        ),
        "results_by_threshold": results_by_threshold,
        "results_by_phase_at_10pct_threshold": phase_results,
        "results_by_asset_at_10pct_threshold": asset_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Disagreement test on {len(joined)} matched predictions:")
    for r in results_by_threshold:
        thresh = r["threshold"]
        with_d = r["trade_with_disagreement"]
        against_d = r["trade_against_disagreement"]
        if with_d:
            print(f"  Threshold >={int(thresh*100)}%: WITH disagreement: "
                  f"{with_d['n_trades']} trades, "
                  f"net P&L per trade = {with_d['mean_net_per_trade']:+.4f}")
        if against_d:
            print(f"  Threshold >={int(thresh*100)}%: AGAINST disagreement: "
                  f"{against_d['n_trades']} trades, "
                  f"net P&L per trade = {against_d['mean_net_per_trade']:+.4f}")


if __name__ == "__main__":
    main()
