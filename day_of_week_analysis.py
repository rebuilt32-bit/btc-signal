"""
Analyze model performance by day of week and weekday-vs-weekend.

Reads predictions/*.jsonl and settled.jsonl, joins by ticker, groups by
day-of-week of the snapshot time, and computes per-group metrics.

Output: data/day_of_week_analysis.json
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/day_of_week_analysis.json"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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


def parse_iso(s):
    """Parse ISO 8601 string, handling Z suffix."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def compute_metrics(items):
    """Compute Brier scores, accuracy, win rate when confident for a group of decisions."""
    n = len(items)
    if n == 0:
        return None

    our_brier_sum = 0.0
    market_brier_sum = 0.0
    market_n = 0
    for j in items:
        our_brier_sum += (j["our_prob"] - j["outcome_yes"]) ** 2
        if j.get("market_mid") is not None:
            market_brier_sum += (j["market_mid"] - j["outcome_yes"]) ** 2
            market_n += 1
    our_brier = our_brier_sum / n
    market_brier = market_brier_sum / market_n if market_n > 0 else None

    confident = [j for j in items if abs(j["our_prob"] - 0.5) > 0.2]
    confident_wins = sum(
        1 for j in confident
        if (j["our_prob"] > 0.5) == (j["outcome_yes"] == 1)
    )
    confident_win_rate = (
        round(confident_wins / len(confident), 3) if confident else None
    )

    disagreement_15 = []
    for j in items:
        if j.get("market_mid") is None:
            continue
        d = j["our_prob"] - j["market_mid"]
        if abs(d) >= 0.15:
            direction_yes = d > 0
            actual_yes = j["outcome_yes"] == 1
            disagreement_15.append({
                "won": direction_yes == actual_yes,
                "abs_disagreement": abs(d),
            })

    disagreement_win_rate = None
    if disagreement_15:
        disagreement_win_rate = round(
            sum(1 for d in disagreement_15 if d["won"]) / len(disagreement_15), 3
        )

    return {
        "n_total": n,
        "our_brier": round(our_brier, 4),
        "market_brier": round(market_brier, 4) if market_brier is not None else None,
        "brier_gap": round(our_brier - market_brier, 4) if market_brier is not None else None,
        "we_beat_market": (our_brier < market_brier) if market_brier is not None else None,
        "n_confident": len(confident),
        "confident_win_rate": confident_win_rate,
        "n_disagreement_15pct": len(disagreement_15),
        "disagreement_15pct_win_rate": disagreement_win_rate,
    }


def main():
    predictions = load_all_predictions()
    settlements = load_jsonl(SETTLED_PATH)

    settled_by_ticker = {}
    for s in settlements:
        ticker = s.get("ticker")
        outcome = s.get("outcome")
        if ticker and outcome and outcome != "unknown":
            settled_by_ticker[ticker] = 1 if outcome == "YES" else 0

    joined = []
    for p in predictions:
        ticker = p.get("ticker")
        if ticker not in settled_by_ticker:
            continue
        our_prob = p.get("prob_yes_estimate")
        if our_prob is None:
            continue
        snap_time = parse_iso(p.get("snapshot_time"))
        if snap_time is None:
            continue
        dow = snap_time.weekday()  # 0=Monday, 6=Sunday
        is_weekend = dow >= 5

        joined.append({
            "ticker": ticker,
            "asset": p.get("asset"),
            "seconds_left": p.get("seconds_left"),
            "our_prob": our_prob,
            "market_mid": p.get("market_mid"),
            "outcome_yes": settled_by_ticker[ticker],
            "day_of_week": dow,
            "day_name": DAY_NAMES[dow],
            "is_weekend": is_weekend,
            "snapshot_time": p.get("snapshot_time"),
        })

    if not joined:
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "No matched data yet.",
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("No data.")
        return

    # By day of week
    by_dow = defaultdict(list)
    for j in joined:
        by_dow[j["day_of_week"]].append(j)

    by_day_results = {}
    for dow in range(7):
        items = by_dow.get(dow, [])
        metrics = compute_metrics(items)
        by_day_results[DAY_NAMES[dow]] = metrics if metrics else {"n_total": 0}

    # Weekday vs weekend
    weekdays = [j for j in joined if not j["is_weekend"]]
    weekends = [j for j in joined if j["is_weekend"]]

    weekday_vs_weekend = {
        "weekdays": compute_metrics(weekdays) or {"n_total": 0},
        "weekends": compute_metrics(weekends) or {"n_total": 0},
    }

    # Day x phase (early_5_10min only — our edge zone from disagreement_test)
    early_phase = [j for j in joined if j["seconds_left"] is not None
                   and 300 <= j["seconds_left"] < 600]
    by_dow_early = defaultdict(list)
    for j in early_phase:
        by_dow_early[j["day_of_week"]].append(j)

    early_phase_by_day = {}
    for dow in range(7):
        items = by_dow_early.get(dow, [])
        metrics = compute_metrics(items)
        early_phase_by_day[DAY_NAMES[dow]] = metrics if metrics else {"n_total": 0}

    # Asset x day cross-cut
    by_asset_day = defaultdict(lambda: defaultdict(list))
    for j in joined:
        if j.get("asset"):
            by_asset_day[j["asset"]][j["day_name"]].append(j)

    asset_x_day = {}
    for asset, by_day in by_asset_day.items():
        asset_x_day[asset] = {}
        for day_name in DAY_NAMES:
            items = by_day.get(day_name, [])
            if len(items) < 20:
                asset_x_day[asset][day_name] = {"n_total": len(items), "note": "thin sample"}
                continue
            metrics = compute_metrics(items)
            asset_x_day[asset][day_name] = metrics if metrics else {"n_total": 0}

    times = [parse_iso(j["snapshot_time"]) for j in joined]
    times = [t for t in times if t is not None]
    earliest = min(times).isoformat() if times else None
    latest = max(times).isoformat() if times else None

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_matched_predictions": len(joined),
        "data_range": {"earliest": earliest, "latest": latest},
        "note": (
            "Day-of-week analysis. Day inferred from snapshot_time UTC. "
            "Sample sizes per day-of-week are thin until 2-3 weeks of data — "
            "treat results as suggestive, not conclusive."
        ),
        "by_day_of_week": by_day_results,
        "weekday_vs_weekend": weekday_vs_weekend,
        "early_phase_5_10min_by_day": early_phase_by_day,
        "asset_x_day": asset_x_day,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Day-of-week analysis: {len(joined)} matched predictions across "
          f"{len(set(j['day_of_week'] for j in joined))} unique days-of-week")
    print()
    print("By day of week:")
    print(f"  {'Day':12s} {'n':>6s} {'our_brier':>10s} {'mkt_brier':>10s} "
          f"{'gap':>8s} {'conf_win':>10s} {'disagree15_win':>15s}")
    for day_name in DAY_NAMES:
        m = by_day_results[day_name]
        if m.get("n_total", 0) == 0:
            print(f"  {day_name:12s} no data")
            continue
        gap = m.get("brier_gap")
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        cw = m.get("confident_win_rate")
        cw_str = f"{cw:.3f}" if cw is not None else "n/a"
        dw = m.get("disagreement_15pct_win_rate")
        dw_str = f"{dw:.3f}({m.get('n_disagreement_15pct')})" if dw is not None else "n/a"
        print(f"  {day_name:12s} {m['n_total']:>6d} {m.get('our_brier', 'n/a'):>10} "
              f"{m.get('market_brier', 'n/a'):>10} {gap_str:>8s} {cw_str:>10s} {dw_str:>15s}")
    print()
    print("Weekday vs Weekend:")
    for label in ["weekdays", "weekends"]:
        m = weekday_vs_weekend[label]
        if m.get("n_total", 0) == 0:
            continue
        print(f"  {label:12s} n={m['n_total']:>5d}  our_brier={m.get('our_brier')} "
              f"market_brier={m.get('market_brier')} gap={m.get('brier_gap'):+.4f} "
              f"confident_win={m.get('confident_win_rate')}")


if __name__ == "__main__":
    main()
