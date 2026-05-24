"""
Analyze model performance by day of week, time of day, and combinations.

Reads predictions/*.jsonl and settled.jsonl, joins by ticker, groups by
day-of-week and hour-of-day (UTC) of the snapshot time, and computes
per-group metrics.

Cuts produced:
  1. By day of week (Mon..Sun)
  2. Weekday vs weekend
  3. By hour of day (0..23 UTC)
  4. By session bucket (Asia / EU / US / Off-hours)
  5. Day x hour heatmap (with thin-sample flag)
  6. Day x session bucket
  7. Asset x day cross-cut
  8. Early-phase (5-10 min left) by day

Output: data/time_analysis.json
"""
import json
import gzip
import os
from collections import defaultdict
from datetime import datetime, timezone

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/time_analysis.json"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Session buckets defined in UTC.
# Asia: 00:00-06:59 UTC (covers Tokyo/Shanghai trading hours)
# EU:   07:00-12:59 UTC (London/Frankfurt morning to early afternoon)
# US:   13:00-20:59 UTC (US trading day, ET morning through close)
# Off:  21:00-23:59 UTC (low-volume hours between US close and Asia open)
SESSION_BUCKETS = {
    "Asia (00-06 UTC)": (0, 6),
    "EU (07-12 UTC)":   (7, 12),
    "US (13-20 UTC)":   (13, 20),
    "Off (21-23 UTC)":  (21, 23),
}


def hour_to_session(hour):
    for name, (lo, hi) in SESSION_BUCKETS.items():
        if lo <= hour <= hi:
            return name
    return "Off (21-23 UTC)"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
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
        if fname.endswith(".jsonl") or fname.endswith(".jsonl.gz"):
            rows.extend(load_jsonl(os.path.join(PRED_DIR, fname)))
    return rows


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def compute_metrics(items, thin_threshold=20):
    """Compute Brier scores, accuracy, win rate when confident for a group."""
    n = len(items)
    if n == 0:
        return {"n_total": 0}

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
            disagreement_15.append(direction_yes == actual_yes)

    disagreement_win_rate = None
    if disagreement_15:
        disagreement_win_rate = round(
            sum(1 for w in disagreement_15 if w) / len(disagreement_15), 3
        )

    result = {
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
    if n < thin_threshold:
        result["thin_sample"] = True
    return result


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
        dow = snap_time.weekday()
        hour = snap_time.hour
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
            "hour_utc": hour,
            "session": hour_to_session(hour),
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

    # 1. By day of week
    by_dow = defaultdict(list)
    for j in joined:
        by_dow[j["day_of_week"]].append(j)
    by_day_results = {
        DAY_NAMES[dow]: compute_metrics(by_dow.get(dow, []))
        for dow in range(7)
    }

    # 2. Weekday vs weekend
    weekdays = [j for j in joined if not j["is_weekend"]]
    weekends = [j for j in joined if j["is_weekend"]]
    weekday_vs_weekend = {
        "weekdays": compute_metrics(weekdays),
        "weekends": compute_metrics(weekends),
    }

    # 3. By hour of day (UTC)
    by_hour = defaultdict(list)
    for j in joined:
        by_hour[j["hour_utc"]].append(j)
    by_hour_results = {
        f"{h:02d}:00 UTC": compute_metrics(by_hour.get(h, []))
        for h in range(24)
    }

    # 4. By session bucket
    by_session = defaultdict(list)
    for j in joined:
        by_session[j["session"]].append(j)
    by_session_results = {
        name: compute_metrics(by_session.get(name, []))
        for name in SESSION_BUCKETS.keys()
    }

    # 5. Day x hour heatmap
    day_x_hour = {}
    for dow in range(7):
        day_x_hour[DAY_NAMES[dow]] = {}
        for h in range(24):
            cell = [j for j in joined if j["day_of_week"] == dow and j["hour_utc"] == h]
            day_x_hour[DAY_NAMES[dow]][f"{h:02d}:00 UTC"] = compute_metrics(cell)

    # 6. Day x session bucket
    day_x_session = {}
    for dow in range(7):
        day_x_session[DAY_NAMES[dow]] = {}
        for session_name in SESSION_BUCKETS.keys():
            cell = [
                j for j in joined
                if j["day_of_week"] == dow and j["session"] == session_name
            ]
            day_x_session[DAY_NAMES[dow]][session_name] = compute_metrics(cell)

    # 7. Asset x day
    by_asset_day = defaultdict(lambda: defaultdict(list))
    for j in joined:
        if j.get("asset"):
            by_asset_day[j["asset"]][j["day_name"]].append(j)
    asset_x_day = {}
    for asset, by_day in by_asset_day.items():
        asset_x_day[asset] = {
            day_name: compute_metrics(by_day.get(day_name, []))
            for day_name in DAY_NAMES
        }

    # 8. Early phase (5-10 min) by day
    early_phase = [
        j for j in joined
        if j["seconds_left"] is not None and 300 <= j["seconds_left"] < 600
    ]
    by_dow_early = defaultdict(list)
    for j in early_phase:
        by_dow_early[j["day_of_week"]].append(j)
    early_phase_by_day = {
        DAY_NAMES[dow]: compute_metrics(by_dow_early.get(dow, []))
        for dow in range(7)
    }

    times = [parse_iso(j["snapshot_time"]) for j in joined]
    times = [t for t in times if t is not None]
    earliest = min(times).isoformat() if times else None
    latest = max(times).isoformat() if times else None

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_matched_predictions": len(joined),
        "data_range": {"earliest": earliest, "latest": latest},
        "note": (
            "Time-based analysis. Day and hour inferred from snapshot_time UTC. "
            "Cells with n<20 are flagged thin_sample. Sample sizes per "
            "(day x hour) cell will be very thin until 3-4 weeks of data."
        ),
        "session_definitions_utc": {
            name: f"{lo:02d}:00-{hi:02d}:59 UTC"
            for name, (lo, hi) in SESSION_BUCKETS.items()
        },
        "by_day_of_week": by_day_results,
        "weekday_vs_weekend": weekday_vs_weekend,
        "by_hour_of_day_utc": by_hour_results,
        "by_session_bucket": by_session_results,
        "day_x_hour_heatmap": day_x_hour,
        "day_x_session_bucket": day_x_session,
        "asset_x_day": asset_x_day,
        "early_phase_5_10min_by_day": early_phase_by_day,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Time analysis: {len(joined)} matched predictions")
    print(f"  Range: {earliest} to {latest}")
    print()
    print("By day of week:")
    for day_name in DAY_NAMES:
        m = by_day_results[day_name]
        if m.get("n_total", 0) == 0:
            continue
        gap = m.get("brier_gap")
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        cw = m.get("confident_win_rate")
        cw_str = f"{cw:.3f}" if cw is not None else "n/a"
        print(f"  {day_name:12s} n={m['n_total']:>5d}  gap={gap_str}  conf_win={cw_str}")
    print()
    print("Weekday vs Weekend:")
    for label in ["weekdays", "weekends"]:
        m = weekday_vs_weekend[label]
        if m.get("n_total", 0) == 0:
            continue
        gap = m.get("brier_gap")
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        print(f"  {label:12s} n={m['n_total']:>5d}  gap={gap_str}  "
              f"conf_win={m.get('confident_win_rate')}")
    print()
    print("By session bucket:")
    for session_name in SESSION_BUCKETS.keys():
        m = by_session_results[session_name]
        if m.get("n_total", 0) == 0:
            continue
        gap = m.get("brier_gap")
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        cw = m.get("confident_win_rate")
        cw_str = f"{cw:.3f}" if cw is not None else "n/a"
        print(f"  {session_name:20s} n={m['n_total']:>5d}  gap={gap_str}  conf_win={cw_str}")


if __name__ == "__main__":
    main()
