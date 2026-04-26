import json
import os
import math
from datetime import datetime, timezone

OUT_DIR = "data"
HIST_DIR = "data/history"

# Signal weights — fixed for now, learning comes later.
# Note: time_decay merged into distance_from_strike (was double-counting).
WEIGHTS = {
    "momentum_short": 0.20,
    "momentum_medium": 0.18,
    "trend_slope": 0.14,
    "exchange_alignment": 0.10,
    "distance_from_strike": 0.38,  # absorbed time_decay's role
}

# Wider clip so extreme situations (5+ stdev with little time left)
# can express themselves.
SIGNAL_CLIP = 6.0

# Threshold below which exchange-disagreement warning is suppressed
# (small noise differences shouldn't trigger it).
ALIGNMENT_WARN_THRESHOLD = -0.5


def parse_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_today_history():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(HIST_DIR, f"{today}.jsonl")
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


def composite_price(snap_asset):
    prices = []
    if snap_asset.get("kraken") is not None:
        prices.append(snap_asset["kraken"])
    if snap_asset.get("coinbase") is not None:
        prices.append(snap_asset["coinbase"])
    if not prices:
        return None
    return sum(prices) / len(prices)


def get_asset_series(history, asset_name):
    series = []
    for snap in history:
        a = snap.get("assets", {}).get(asset_name)
        if not a:
            continue
        cp = composite_price(a)
        if cp is None:
            continue
        ts = snap.get("ts")
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue
        series.append({
            "t": t,
            "cp": cp,
            "kr": a.get("kraken"),
            "cb": a.get("coinbase"),
        })
    series.sort(key=lambda x: x["t"])
    return series


def price_n_seconds_ago(series, now, seconds):
    cutoff = now.timestamp() - seconds
    best = None
    for pt in series:
        if pt["t"].timestamp() <= cutoff:
            best = pt
        else:
            break
    return best


def linear_slope(points):
    n = len(points)
    if n < 2:
        return 0.0
    t0 = points[0]["t"].timestamp()
    xs = [p["t"].timestamp() - t0 for p in points]
    ys = [p["cp"] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def sigmoid(x):
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def analyze_market(market, asset_name, series, now):
    strike = parse_float(market.get("strike"))
    close_time_str = market.get("close_time")
    if strike is None or close_time_str is None or not series:
        return None

    try:
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    except Exception:
        return None

    seconds_left = (close_time - now).total_seconds()
    if seconds_left <= 0:
        return None

    current = series[-1]
    current_price = current["cp"]
    distance = current_price - strike
    distance_pct = distance / strike

    # Recent volatility
    five_min_cutoff = now.timestamp() - 300
    recent = [p["cp"] for p in series if p["t"].timestamp() >= five_min_cutoff]
    vol = stdev(recent) if len(recent) >= 3 else max(strike * 0.0005, 1e-9)
    if vol <= 0:
        vol = max(strike * 0.0005, 1e-9)

    # ----- Signals -----
    p_60s = price_n_seconds_ago(series, now, 60)
    p_300s = price_n_seconds_ago(series, now, 300)

    momentum_short = 0.0
    if p_60s:
        momentum_short = (current_price - p_60s["cp"]) / vol

    momentum_medium = 0.0
    if p_300s:
        momentum_medium = (current_price - p_300s["cp"]) / vol

    three_min_cutoff = now.timestamp() - 180
    slope_pts = [p for p in series if p["t"].timestamp() >= three_min_cutoff]
    slope_per_sec = linear_slope(slope_pts) if len(slope_pts) >= 3 else 0.0
    trend_slope = (slope_per_sec * 60) / vol

    exchange_alignment = 0.0
    if p_60s and p_60s.get("kr") is not None and p_60s.get("cb") is not None \
            and current.get("kr") is not None and current.get("cb") is not None:
        kr_change = current["kr"] - p_60s["kr"]
        cb_change = current["cb"] - p_60s["cb"]
        if kr_change * cb_change > 0:
            exchange_alignment = math.copysign(1.0, kr_change) * \
                min(abs(kr_change), abs(cb_change)) / vol
        elif abs(kr_change) > vol * 0.2 or abs(cb_change) > vol * 0.2:
            # Only penalize disagreement if at least one exchange moved meaningfully
            exchange_alignment = -0.5
        # else: noise, no penalty

    # Distance from strike, scaled UP by phase elapsed.
    # Earlier in the window: raw distance signal.
    # Later in the window: same distance matters more because less time to reverse.
    phase = max(0.0, min(1.0, 1.0 - (seconds_left / 900.0)))
    base_distance = distance / vol
    # Phase multiplier: 1.0x early, up to 2.5x at expiration
    distance_from_strike = base_distance * (1.0 + phase * 1.5)

    signals = {
        "momentum_short": momentum_short,
        "momentum_medium": momentum_medium,
        "trend_slope": trend_slope,
        "exchange_alignment": exchange_alignment,
        "distance_from_strike": distance_from_strike,
    }

    log_odds = 0.0
    contributions = {}
    for name, value in signals.items():
        clipped = max(-SIGNAL_CLIP, min(SIGNAL_CLIP, value))
        contrib = WEIGHTS[name] * clipped
        log_odds += contrib
        contributions[name] = {
            "raw": round(value, 3),
            "clipped": round(clipped, 3),
            "weight": WEIGHTS[name],
            "contribution": round(contrib, 3),
        }

    prob_yes = sigmoid(log_odds)

    history_score = min(1.0, len(series) / 10)
    alignment_score = 1.0 if exchange_alignment >= ALIGNMENT_WARN_THRESHOLD else 0.7
    confidence = round(history_score * alignment_score, 2)

    yes_bid = parse_float(market.get("yes_bid"))
    yes_ask = parse_float(market.get("yes_ask"))
    market_mid = None
    if yes_bid is not None and yes_ask is not None:
        market_mid = (yes_bid + yes_ask) / 2

    disagreement = None
    if market_mid is not None:
        disagreement = round(prob_yes - market_mid, 3)

    summary_parts = []
    if abs(distance_pct) < 0.0005:
        summary_parts.append(f"{asset_name} essentially at strike")
    elif distance > 0:
        summary_parts.append(f"{asset_name} {abs(distance_pct)*100:.2f}% above strike")
    else:
        summary_parts.append(f"{asset_name} {abs(distance_pct)*100:.2f}% below strike")

    if momentum_short > 0.5:
        summary_parts.append("rising")
    elif momentum_short < -0.5:
        summary_parts.append("falling")

    if exchange_alignment <= ALIGNMENT_WARN_THRESHOLD:
        summary_parts.append("exchanges diverging")

    minutes_left = seconds_left / 60
    summary_parts.append(f"{minutes_left:.1f} min left")

    summary = "; ".join(summary_parts) + "."

    return {
        "ticker": market.get("ticker"),
        "asset": asset_name,
        "strike": strike,
        "current_price": round(current_price, 6),
        "seconds_left": int(seconds_left),
        "minutes_left": round(minutes_left, 1),
        "prob_yes_estimate": round(prob_yes, 3),
        "market_yes_bid": yes_bid,
        "market_yes_ask": yes_ask,
        "market_mid": round(market_mid, 3) if market_mid is not None else None,
        "disagreement": disagreement,
        "confidence": confidence,
        "summary": summary,
        "signals": contributions,
        "history_points_used": len(series),
    }


def main():
    history = load_today_history()
    if not history:
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error": "No history file for today yet.",
            "predictions": [],
        }
        with open(os.path.join(OUT_DIR, "prediction.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("No history. Wrote empty prediction.json.")
        return

    latest = history[-1]
    now = datetime.fromisoformat(latest["ts"])

    predictions = []
    for asset_name, asset_data in latest.get("assets", {}).items():
        series = get_asset_series(history, asset_name)
        if not series:
            continue
        for market in asset_data.get("markets", []):
            if market.get("status") != "active":
                continue
            pred = analyze_market(market, asset_name, series, now)
            if pred:
                predictions.append(pred)

    predictions.sort(
        key=lambda p: abs(p["disagreement"]) if p.get("disagreement") is not None else -1,
        reverse=True,
    )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_time": latest["ts"],
        "history_points_today": len(history),
        "predictions": predictions,
    }

    with open(os.path.join(OUT_DIR, "prediction.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(predictions)} predictions to prediction.json")
    for p in predictions:
        print(f"  {p['asset']:5s} prob={p['prob_yes_estimate']:.3f} "
              f"market={p['market_mid']} disagree={p['disagreement']}")


if __name__ == "__main__":
    main()
