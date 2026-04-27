import json
import os
from collections import defaultdict
from datetime import datetime

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/calibration.json"


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
    """Load all prediction snapshots from all daily files."""
    rows = []
    if not os.path.exists(PRED_DIR):
        return rows
    for fname in sorted(os.listdir(PRED_DIR)):
        if fname.endswith(".jsonl"):
            rows.extend(load_jsonl(os.path.join(PRED_DIR, fname)))
    return rows


def bucket_label(prob):
    """Group probabilities into buckets for calibration."""
    if prob is None:
        return None
    if prob < 0.1:
        return "0-10%"
    if prob < 0.2:
        return "10-20%"
    if prob < 0.3:
        return "20-30%"
    if prob < 0.4:
        return "30-40%"
    if prob < 0.5:
        return "40-50%"
    if prob < 0.6:
        return "50-60%"
    if prob < 0.7:
        return "60-70%"
    if prob < 0.8:
        return "70-80%"
    if prob < 0.9:
        return "80-90%"
    return "90-100%"


# Order buckets for sorted output
BUCKET_ORDER = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
]


def bucket_midpoint(label):
    """Approximate center of a bucket, for comparing to actual hit rate."""
    starts = {
        "0-10%": 0.05, "10-20%": 0.15, "20-30%": 0.25, "30-40%": 0.35,
        "40-50%": 0.45, "50-60%": 0.55, "60-70%": 0.65, "70-80%": 0.75,
        "80-90%": 0.85, "90-100%": 0.95,
    }
    return starts.get(label, 0.5)


def calibrate_bucket(predictions_in_bucket):
    """Given a list of predictions all in one bucket, compute hit rate."""
    n = len(predictions_in_bucket)
    yes_count = sum(1 for p in predictions_in_bucket if p["outcome"] == "YES")
    hit_rate = yes_count / n if n > 0 else None
    return {"n": n, "yes_count": yes_count, "hit_rate": hit_rate}


def main():
    predictions = load_all_predictions()
    settlements = load_jsonl(SETTLED_PATH)

    # Build map: ticker -> outcome (YES/NO/unknown)
    settled_by_ticker = {}
    for s in settlements:
        ticker = s.get("ticker")
        outcome = s.get("outcome")
        if ticker and outcome and outcome != "unknown":
            settled_by_ticker[ticker] = outcome

    # Join predictions to settlements
    joined = []
    for p in predictions:
        ticker = p.get("ticker")
        if ticker not in settled_by_ticker:
            continue
        outcome = settled_by_ticker[ticker]
        joined.append({
            "ticker": ticker,
            "asset": p.get("asset"),
            "snapshot_time": p.get("snapshot_time"),
            "seconds_left": p.get("seconds_left"),
            "our_prob": p.get("prob_yes_estimate"),
            "market_mid": p.get("market_mid"),
            "outcome": outcome,
            "outcome_yes": 1 if outcome == "YES" else 0,
            "confidence": p.get("confidence"),
        })

    if not joined:
        result = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_predictions": 0,
            "total_settled_predictions": 0,
            "note": "No matched predictions and settlements yet.",
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("No data to calibrate yet.")
        return

    # ----- LAYER 1: Calibration of OUR model -----
    our_buckets = defaultdict(list)
    for j in joined:
        if j["our_prob"] is None:
            continue
        b = bucket_label(j["our_prob"])
        our_buckets[b].append(j)

    our_calibration = []
    for label in BUCKET_ORDER:
        if label in our_buckets:
            stats = calibrate_bucket(our_buckets[label])
            stats["bucket"] = label
            stats["expected_rate"] = bucket_midpoint(label)
            stats["calibration_error"] = (
                round(stats["hit_rate"] - stats["expected_rate"], 3)
                if stats["hit_rate"] is not None else None
            )
            our_calibration.append(stats)

    # Calibration of MARKET predictions for comparison
    market_buckets = defaultdict(list)
    for j in joined:
        if j["market_mid"] is None:
            continue
        b = bucket_label(j["market_mid"])
        market_buckets[b].append(j)

    market_calibration = []
    for label in BUCKET_ORDER:
        if label in market_buckets:
            stats = calibrate_bucket(market_buckets[label])
            stats["bucket"] = label
            stats["expected_rate"] = bucket_midpoint(label)
            stats["calibration_error"] = (
                round(stats["hit_rate"] - stats["expected_rate"], 3)
                if stats["hit_rate"] is not None else None
            )
            market_calibration.append(stats)

    # ----- LAYER 2: Disagreement analysis -----
    # When we and market disagreed by >= threshold, who was right more often?
    DISAGREEMENT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
    disagreement_results = []

    for thresh in DISAGREEMENT_THRESHOLDS:
        # Predictions where we disagreed by at least this much
        rows = [
            j for j in joined
            if j["market_mid"] is not None and j["our_prob"] is not None
            and abs(j["our_prob"] - j["market_mid"]) >= thresh
        ]
        n = len(rows)
        if n == 0:
            continue

        # When we and market both leaned the same direction, no real disagreement
        # We care about cases where we and market lean opposite directions
        # OR where one is confident and the other isn't
        # Define "we were right" as: the side closer to actual outcome wins
        we_won = 0
        market_won = 0
        tied = 0
        for j in rows:
            our_dist = abs(j["our_prob"] - j["outcome_yes"])
            mkt_dist = abs(j["market_mid"] - j["outcome_yes"])
            if our_dist < mkt_dist:
                we_won += 1
            elif mkt_dist < our_dist:
                market_won += 1
            else:
                tied += 1

        disagreement_results.append({
            "threshold": thresh,
            "n_predictions": n,
            "we_were_closer": we_won,
            "market_was_closer": market_won,
            "tied": tied,
            "we_win_rate": round(we_won / n, 3) if n > 0 else None,
        })

    # Overall accuracy: who is closer to outcomes on average across all predictions?
    if joined:
        our_brier = sum(
            (j["our_prob"] - j["outcome_yes"]) ** 2
            for j in joined if j["our_prob"] is not None
        ) / max(1, sum(1 for j in joined if j["our_prob"] is not None))
        market_brier = sum(
            (j["market_mid"] - j["outcome_yes"]) ** 2
            for j in joined if j["market_mid"] is not None
        ) / max(1, sum(1 for j in joined if j["market_mid"] is not None))
    else:
        our_brier = None
        market_brier = None

    # Group by phase (early vs late in window) to see if model is better at one
    phase_stats = {"early": [], "late": []}
    for j in joined:
        if j["seconds_left"] is None:
            continue
        if j["seconds_left"] >= 450:
            phase_stats["early"].append(j)
        else:
            phase_stats["late"].append(j)

    phase_results = {}
    for phase, items in phase_stats.items():
        if not items:
            continue
        n = len(items)
        our_b = sum(
            (j["our_prob"] - j["outcome_yes"]) ** 2
            for j in items if j["our_prob"] is not None
        ) / max(1, sum(1 for j in items if j["our_prob"] is not None))
        mkt_b = sum(
            (j["market_mid"] - j["outcome_yes"]) ** 2
            for j in items if j["market_mid"] is not None
        ) / max(1, sum(1 for j in items if j["market_mid"] is not None))
        phase_results[phase] = {
            "n": n,
            "our_brier": round(our_b, 4),
            "market_brier": round(mkt_b, 4),
        }

    # By asset, who's more accurate?
    asset_stats = defaultdict(list)
    for j in joined:
        if j["asset"]:
            asset_stats[j["asset"]].append(j)

    asset_results = {}
    for asset, items in asset_stats.items():
        n = len(items)
        if n == 0:
            continue
        our_b = sum(
            (j["our_prob"] - j["outcome_yes"]) ** 2
            for j in items if j["our_prob"] is not None
        ) / max(1, sum(1 for j in items if j["our_prob"] is not None))
        mkt_b = sum(
            (j["market_mid"] - j["outcome_yes"]) ** 2
            for j in items if j["market_mid"] is not None
        ) / max(1, sum(1 for j in items if j["market_mid"] is not None))
        asset_results[asset] = {
            "n": n,
            "our_brier": round(our_b, 4),
            "market_brier": round(mkt_b, 4),
        }

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_predictions_logged": len(predictions),
        "total_settlements_with_outcomes": len(settled_by_ticker),
        "total_matched_predictions": len(joined),
        "overall_brier_score": {
            "ours": round(our_brier, 4) if our_brier is not None else None,
            "market": round(market_brier, 4) if market_brier is not None else None,
            "lower_is_better": True,
            "interpretation": (
                "Brier score measures prediction accuracy: 0 is perfect, "
                "0.25 is random coinflip predictions, 1 is maximally wrong. "
                "Compare ours vs market — if our score is lower, our predictions "
                "are on average closer to true outcomes."
            ),
        },
        "our_calibration_by_bucket": our_calibration,
        "market_calibration_by_bucket": market_calibration,
        "disagreement_analysis": disagreement_results,
        "by_phase": phase_results,
        "by_asset": asset_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Calibration analysis: {len(joined)} matched predictions")
    print(f"  Our Brier: {result['overall_brier_score']['ours']}")
    print(f"  Market Brier: {result['overall_brier_score']['market']}")
    for d in disagreement_results:
        print(f"  Disagreement >= {int(d['threshold']*100)}%: "
              f"we win {d['we_were_closer']}/{d['n_predictions']} "
              f"({d['we_win_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
