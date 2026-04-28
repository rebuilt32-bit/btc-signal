import json
import os
from collections import defaultdict
from datetime import datetime
import math

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/signal_attribution.json"

# All 6 signals — must match what analyze.py logs as signal_<name>_raw
SIGNAL_NAMES = [
    "momentum_short",
    "momentum_medium",
    "trend_slope",
    "exchange_alignment",
    "distance_from_strike",
    "funding_rate",
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


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def correlation(xs, ys):
    """Pearson correlation between two lists."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = sum((xs[i] - mx) ** 2 for i in range(n))
    den_y = sum((ys[i] - my) ** 2 for i in range(n))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def threshold_predict_accuracy(signal_values, outcomes, threshold=0.0):
    """For each row, predict YES if signal > threshold, else NO. Return hit rate."""
    n = len(signal_values)
    if n == 0:
        return None
    hits = 0
    for i in range(n):
        pred_yes = 1 if signal_values[i] > threshold else 0
        if pred_yes == outcomes[i]:
            hits += 1
    return hits / n


def confusion_matrix(signal_values, outcomes, threshold=0.0):
    """Return TP, FP, TN, FN counts when using threshold to predict YES."""
    tp = fp = tn = fn = 0
    for i in range(len(signal_values)):
        pred = signal_values[i] > threshold
        actual = outcomes[i] == 1
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return {"true_yes": tp, "false_yes": fp, "true_no": tn, "false_no": fn}


def analyze_signal(signal_values, outcomes, name):
    """Compute attribution metrics for a single signal."""
    n = len(signal_values)
    if n < 10:
        return {"signal": name, "n": n, "note": "insufficient data"}

    yes_vals = [signal_values[i] for i in range(n) if outcomes[i] == 1]
    no_vals = [signal_values[i] for i in range(n) if outcomes[i] == 0]

    mean_when_yes = mean(yes_vals)
    mean_when_no = mean(no_vals)
    diff = (mean_when_yes - mean_when_no) if (mean_when_yes is not None and mean_when_no is not None) else None

    corr = correlation(signal_values, outcomes)

    accuracy_at_zero = threshold_predict_accuracy(signal_values, outcomes, threshold=0.0)
    cm = confusion_matrix(signal_values, outcomes, threshold=0.0)

    sorted_vals = sorted(signal_values)
    candidate_thresholds = []
    if len(sorted_vals) > 0:
        for pct in [0.25, 0.5, 0.75]:
            idx = int(pct * len(sorted_vals))
            if 0 <= idx < len(sorted_vals):
                candidate_thresholds.append(sorted_vals[idx])

    best_threshold = 0.0
    best_accuracy = accuracy_at_zero or 0
    for t in candidate_thresholds:
        acc = threshold_predict_accuracy(signal_values, outcomes, threshold=t)
        if acc is not None and acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    sig_stdev = stdev(signal_values)
    sig_mean = mean(signal_values)

    return {
        "signal": name,
        "n": n,
        "signal_mean": round(sig_mean, 4) if sig_mean is not None else None,
        "signal_stdev": round(sig_stdev, 4),
        "mean_when_yes": round(mean_when_yes, 4) if mean_when_yes is not None else None,
        "mean_when_no": round(mean_when_no, 4) if mean_when_no is not None else None,
        "separation": round(diff, 4) if diff is not None else None,
        "correlation_with_outcome": round(corr, 4) if corr is not None else None,
        "predict_alone_accuracy_threshold_0": round(accuracy_at_zero, 4) if accuracy_at_zero else None,
        "best_threshold_found": round(best_threshold, 4),
        "best_accuracy_with_threshold": round(best_accuracy, 4),
        "confusion_at_zero": cm,
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
        outcome = settled_by_ticker[ticker]
        outcome_yes = 1 if outcome == "YES" else 0

        our_prob = p.get("prob_yes_estimate")
        market_mid = p.get("market_mid")
        confidence = p.get("confidence")

        if our_prob is None:
            continue

        disagreement = (our_prob - market_mid) if market_mid is not None else None
        prob_strength = abs(our_prob - 0.5)

        # Pull raw signal values that analyze.py logged
        signal_raw = {}
        for sig_name in SIGNAL_NAMES:
            signal_raw[sig_name] = p.get(f"signal_{sig_name}_raw")

        joined.append({
            "ticker": ticker,
            "asset": p.get("asset"),
            "seconds_left": p.get("seconds_left"),
            "outcome_yes": outcome_yes,
            "our_prob": our_prob,
            "market_mid": market_mid,
            "disagreement": disagreement,
            "prob_strength": prob_strength,
            "confidence": confidence,
            "signal_raw": signal_raw,
        })

    if not joined:
        result = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "note": "No matched data yet.",
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("No data.")
        return

    outcomes = [j["outcome_yes"] for j in joined]

    # ----- Analyze each RAW signal individually -----
    raw_signal_results = {}
    for sig_name in SIGNAL_NAMES:
        # Pair (signal_value, outcome) and drop rows where signal is missing
        # (older predictions logged before signals were captured)
        pairs = [
            (j["signal_raw"][sig_name], j["outcome_yes"])
            for j in joined
            if j["signal_raw"].get(sig_name) is not None
        ]
        if len(pairs) < 10:
            raw_signal_results[sig_name] = {
                "signal": sig_name,
                "n": len(pairs),
                "note": "insufficient data — needs more predictions logged with this signal",
            }
            continue
        sig_vals = [p[0] for p in pairs]
        sig_outcomes = [p[1] for p in pairs]
        raw_signal_results[sig_name] = analyze_signal(sig_vals, sig_outcomes, sig_name)

    # ----- Analyze derived signals (unchanged from before) -----
    derived_signals = {
        "our_prob_minus_0.5": [j["our_prob"] - 0.5 for j in joined],
        "market_mid_minus_0.5": [
            (j["market_mid"] - 0.5) if j["market_mid"] is not None else 0.0
            for j in joined
        ],
        "disagreement": [
            j["disagreement"] if j["disagreement"] is not None else 0.0
            for j in joined
        ],
        "prob_strength": [j["prob_strength"] for j in joined],
        "confidence": [
            j["confidence"] if j["confidence"] is not None else 0.5
            for j in joined
        ],
    }

    derived_signal_results = {}
    for name, values in derived_signals.items():
        derived_signal_results[name] = analyze_signal(values, outcomes, name)

    # ----- Where do losses concentrate? -----
    losses = []
    wins = []
    for j in joined:
        if abs(j["our_prob"] - 0.5) > 0.2:
            predicted_yes = j["our_prob"] > 0.5
            actual_yes = j["outcome_yes"] == 1
            if predicted_yes != actual_yes:
                losses.append(j)
            else:
                wins.append(j)

    losses_analysis = {
        "n_confident_predictions": len(wins) + len(losses),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_when_confident": (
            round(len(wins) / (len(wins) + len(losses)), 3)
            if (len(wins) + len(losses)) > 0 else None
        ),
    }

    if losses:
        loss_disagrees = [l["disagreement"] for l in losses if l["disagreement"] is not None]
        win_disagrees = [w["disagreement"] for w in wins if w["disagreement"] is not None]
        losses_analysis["mean_disagreement_in_losses"] = (
            round(mean(loss_disagrees), 4) if loss_disagrees else None
        )
        losses_analysis["mean_disagreement_in_wins"] = (
            round(mean(win_disagrees), 4) if win_disagrees else None
        )

    if losses:
        loss_seconds = [l["seconds_left"] for l in losses if l["seconds_left"] is not None]
        win_seconds = [w["seconds_left"] for w in wins if w["seconds_left"] is not None]
        losses_analysis["mean_seconds_left_in_losses"] = (
            round(mean(loss_seconds), 1) if loss_seconds else None
        )
        losses_analysis["mean_seconds_left_in_wins"] = (
            round(mean(win_seconds), 1) if win_seconds else None
        )

    # ----- Phase analysis -----
    phases = {
        "very_early (>=10min)": [j for j in joined if j["seconds_left"] is not None and j["seconds_left"] >= 600],
        "early (5-10min)": [j for j in joined if j["seconds_left"] is not None and 300 <= j["seconds_left"] < 600],
        "mid (2-5min)": [j for j in joined if j["seconds_left"] is not None and 120 <= j["seconds_left"] < 300],
        "late (1-2min)": [j for j in joined if j["seconds_left"] is not None and 60 <= j["seconds_left"] < 120],
        "final_minute (<60s)": [j for j in joined if j["seconds_left"] is not None and j["seconds_left"] < 60],
    }

    phase_results = {}
    for phase_name, items in phases.items():
        if not items:
            continue
        confident = [j for j in items if abs(j["our_prob"] - 0.5) > 0.2]
        if not confident:
            phase_results[phase_name] = {"n_total": len(items), "n_confident": 0}
            continue
        wins_in_phase = sum(
            1 for j in confident
            if (j["our_prob"] > 0.5) == (j["outcome_yes"] == 1)
        )
        our_b = sum(
            (j["our_prob"] - j["outcome_yes"]) ** 2 for j in items
        ) / len(items)
        market_b = sum(
            (j["market_mid"] - j["outcome_yes"]) ** 2
            for j in items if j["market_mid"] is not None
        ) / max(1, sum(1 for j in items if j["market_mid"] is not None))
        phase_results[phase_name] = {
            "n_total": len(items),
            "n_confident": len(confident),
            "win_rate_when_confident": round(wins_in_phase / len(confident), 3),
            "our_brier": round(our_b, 4),
            "market_brier": round(market_b, 4),
            "we_beat_market": our_b < market_b,
        }

    # ----- By asset -----
    asset_results = {}
    by_asset = defaultdict(list)
    for j in joined:
        if j.get("asset"):
            by_asset[j["asset"]].append(j)

    for asset, items in by_asset.items():
        confident = [j for j in items if abs(j["our_prob"] - 0.5) > 0.2]
        if not confident:
            asset_results[asset] = {"n_total": len(items), "n_confident": 0}
            continue
        wins_in_asset = sum(
            1 for j in confident
            if (j["our_prob"] > 0.5) == (j["outcome_yes"] == 1)
        )
        our_b = sum(
            (j["our_prob"] - j["outcome_yes"]) ** 2 for j in items
        ) / len(items)
        market_b = sum(
            (j["market_mid"] - j["outcome_yes"]) ** 2
            for j in items if j["market_mid"] is not None
        ) / max(1, sum(1 for j in items if j["market_mid"] is not None))
        asset_results[asset] = {
            "n_total": len(items),
            "n_confident": len(confident),
            "win_rate_when_confident": round(wins_in_asset / len(confident), 3),
            "our_brier": round(our_b, 4),
            "market_brier": round(market_b, 4),
            "we_beat_market": our_b < market_b,
        }

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_matched_predictions": len(joined),
        "note": (
            "Raw signal analysis tests each individual signal's predictive power. "
            "Older predictions logged before raw signals were captured will show "
            "lower n. funding_rate signal is brand new — needs significant data "
            "accumulation before its results are meaningful."
        ),
        "raw_signals": raw_signal_results,
        "derived_signals": derived_signal_results,
        "losses_vs_wins_when_confident": losses_analysis,
        "by_phase": phase_results,
        "by_asset": asset_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Signal attribution: analyzed {len(joined)} matched predictions")
    print(f"  Confident win rate overall: {losses_analysis.get('win_rate_when_confident')}")
    print("  Raw signal predictive power (correlation with outcome):")
    for sig_name, stats in raw_signal_results.items():
        if "correlation_with_outcome" in stats:
            print(f"    {sig_name:25s} n={stats['n']:5d} corr={stats['correlation_with_outcome']:+.4f} "
                  f"alone_acc={stats.get('predict_alone_accuracy_threshold_0')}")
        else:
            print(f"    {sig_name:25s} {stats.get('note', 'no data')}")
    for phase, stats in phase_results.items():
        print(f"  {phase}: confident win rate {stats.get('win_rate_when_confident')}, "
              f"beat market: {stats.get('we_beat_market')}")
    for asset, stats in asset_results.items():
        print(f"  {asset}: confident win rate {stats.get('win_rate_when_confident')}, "
              f"beat market: {stats.get('we_beat_market')}")


if __name__ == "__main__":
    main()
