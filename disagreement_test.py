import json
import os
from collections import defaultdict
from datetime import datetime

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/disagreement_test.json"

TRADING_COST_PER_TRADE = 0.05
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]

# Define phases as (label, target_seconds_left, tolerance)
# We'll pick the prediction closest to the target that falls within the tolerance window
PHASE_CHECKPOINTS = [
    {"label": "early", "target_seconds": 600, "min": 360, "max": 900},   # ~10 min, range 6-15 min
    {"label": "mid", "target_seconds": 180, "min": 90, "max": 360},      # ~3 min, range 1.5-6 min
    {"label": "final_minute", "target_seconds": 30, "min": 0, "max": 90},  # ~30s, range 0-90s
]


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


def select_phase_predictions(predictions_by_ticker):
    """
    For each ticker, pick at most one prediction per phase.
    Returns a list of (ticker, phase_label, prediction) tuples.
    """
    selected = []
    for ticker, preds in predictions_by_ticker.items():
        for phase in PHASE_CHECKPOINTS:
            # Find predictions in this phase's seconds_left range
            in_phase = [
                p for p in preds
                if p.get("seconds_left") is not None
                and phase["min"] <= p["seconds_left"] <= phase["max"]
            ]
            if not in_phase:
                continue
            # Pick the one closest to target
            best = min(in_phase, key=lambda p: abs(p["seconds_left"] - phase["target_seconds"]))
            selected.append((ticker, phase["label"], best))
    return selected


def simulate_trade(direction, yes_bid, yes_ask, outcome_yes, cost=TRADING_COST_PER_TRADE):
    if direction == "YES":
        if yes_ask is None or yes_ask <= 0 or yes_ask >= 1:
            return None
        cost_paid = yes_ask
        payout = 1.0 if outcome_yes == 1 else 0.0
        gross = payout - cost_paid
    else:
        if yes_bid is None or yes_bid <= 0 or yes_bid >= 1:
            return None
        cost_paid = 1.0 - yes_bid
        payout = 1.0 if outcome_yes == 0 else 0.0
        gross = payout - cost_paid

    net = gross - cost
    return {"gross": gross, "net": net}


def evaluate_strategy(decisions, threshold, flip=False):
    """
    decisions = list of dicts with keys: our_prob, market_mid, yes_bid, yes_ask, outcome_yes
    """
    trades = []
    for d in decisions:
        our_prob = d.get("our_prob")
        market_mid = d.get("market_mid")
        yes_bid = d.get("yes_bid")
        yes_ask = d.get("yes_ask")
        outcome_yes = d.get("outcome_yes")

        if our_prob is None or market_mid is None or yes_bid is None or yes_ask is None:
            continue

        disagreement = our_prob - market_mid
        if abs(disagreement) < threshold:
            continue

        if flip:
            direction = "NO" if disagreement > 0 else "YES"
        else:
            direction = "YES" if disagreement > 0 else "NO"

        result = simulate_trade(direction, yes_bid, yes_ask, outcome_yes)
        if result is None:
            continue
        trades.append({
            "ticker": d.get("ticker"),
            "asset": d.get("asset"),
            "phase": d.get("phase"),
            "seconds_left": d.get("seconds_left"),
            "our_prob": our_prob,
            "market_mid": market_mid,
            "disagreement": round(disagreement, 3),
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
        "total_gross": round(total_gross, 3),
        "total_net_after_fees": round(total_net, 3),
        "mean_gross_per_trade": round(total_gross / n, 4),
        "mean_net_per_trade": round(total_net / n, 4),
        "expectancy_pct": round((total_net / n) * 100, 2),
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

    # Group predictions by ticker for phase selection
    by_ticker = defaultdict(list)
    for p in predictions:
        ticker = p.get("ticker")
        if ticker not in settled_by_ticker:
            continue
        by_ticker[ticker].append(p)

    # Pick at most 3 predictions per ticker (one per phase)
    selected = select_phase_predictions(by_ticker)

    # Build decision records
    decisions = []
    for ticker, phase_label, pred in selected:
        outcome = settled_by_ticker[ticker]
        decisions.append({
            "ticker": ticker,
            "asset": pred.get("asset"),
            "phase": phase_label,
            "seconds_left": pred.get("seconds_left"),
            "our_prob": pred.get("prob_yes_estimate"),
            "market_mid": pred.get("market_mid"),
            "yes_bid": pred.get("market_yes_bid"),
            "yes_ask": pred.get("market_yes_ask"),
            "outcome_yes": 1 if outcome == "YES" else 0,
        })

    if not decisions:
        result = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "note": "No matched data yet."
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("No data.")
        return

    n_unique_tickers = len(set(d["ticker"] for d in decisions))
    n_decisions = len(decisions)

    # Strategy results at each threshold (overall)
    results_by_threshold = []
    for thresh in THRESHOLDS:
        with_d = evaluate_strategy(decisions, thresh, flip=False)
        against_d = evaluate_strategy(decisions, thresh, flip=True)
        results_by_threshold.append({
            "threshold": thresh,
            "trade_with_disagreement": with_d,
            "trade_against_disagreement": against_d,
        })

    # Phase breakdown at 10% threshold
    phase_results = {}
    for phase in PHASE_CHECKPOINTS:
        items = [d for d in decisions if d["phase"] == phase["label"]]
        if not items:
            continue
        with_d = evaluate_strategy(items, 0.10, flip=False)
        against_d = evaluate_strategy(items, 0.10, flip=True)
        # Also test higher thresholds within this phase
        with_d_15 = evaluate_strategy(items, 0.15, flip=False)
        with_d_20 = evaluate_strategy(items, 0.20, flip=False)
        phase_results[phase["label"]] = {
            "n_decisions": len(items),
            "trade_with_at_10pct": with_d,
            "trade_against_at_10pct": against_d,
            "trade_with_at_15pct": with_d_15,
            "trade_with_at_20pct": with_d_20,
        }

    # Asset breakdown at 10% threshold
    asset_results = {}
    by_asset = defaultdict(list)
    for d in decisions:
        if d.get("asset"):
            by_asset[d["asset"]].append(d)

    for asset, items in by_asset.items():
        with_d = evaluate_strategy(items, 0.10, flip=False)
        against_d = evaluate_strategy(items, 0.10, flip=True)
        asset_results[asset] = {
            "n_decisions": len(items),
            "trade_with_at_10pct": with_d,
            "trade_against_at_10pct": against_d,
        }

    # Best zone: 5-10 min phase + asset cross
    best_zone_results = {}
    early_decisions = [d for d in decisions if d["phase"] == "early"]
    by_asset_early = defaultdict(list)
    for d in early_decisions:
        if d.get("asset"):
            by_asset_early[d["asset"]].append(d)
    for asset, items in by_asset_early.items():
        with_d = evaluate_strategy(items, 0.10, flip=False)
        with_d_15 = evaluate_strategy(items, 0.15, flip=False)
        best_zone_results[asset] = {
            "n_decisions": len(items),
            "trade_with_at_10pct": with_d,
            "trade_with_at_15pct": with_d_15,
        }

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_unique_tickers_settled": n_unique_tickers,
        "n_decision_points": n_decisions,
        "decisions_per_ticker_avg": round(n_decisions / max(1, n_unique_tickers), 2),
        "trading_cost_assumption": TRADING_COST_PER_TRADE,
        "phase_definitions": [
            {"label": p["label"], "target_seconds": p["target_seconds"],
             "range_seconds": [p["min"], p["max"]]}
            for p in PHASE_CHECKPOINTS
        ],
        "interpretation": (
            "Each ticker contributes at most 3 decisions: one per phase. "
            "Tests trading with the disagreement vs against it. "
            "Look at total_net_after_fees: positive = profit, negative = loss."
        ),
        "results_by_threshold_overall": results_by_threshold,
        "results_by_phase": phase_results,
        "results_by_asset": asset_results,
        "results_by_asset_in_early_phase_only": best_zone_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Disagreement test on {n_unique_tickers} unique tickers, {n_decisions} decision points")
    for r in results_by_threshold:
        thresh = r["threshold"]
        with_d = r["trade_with_disagreement"]
        if with_d:
            print(f"  Threshold >={int(thresh*100)}%: with-disagreement {with_d['n_trades']} trades, "
                  f"net per trade = {with_d['mean_net_per_trade']:+.4f}")


if __name__ == "__main__":
    main()
