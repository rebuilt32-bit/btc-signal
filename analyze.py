import json
import os
import math
from datetime import datetime, timezone

OUT_DIR = "data"
HIST_DIR = "data/history"
PRED_LOG_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"

# Weights now include funding_rate. Total = 1.00.
# Reduced distance_from_strike (0.38 -> 0.34), momentum_medium (0.18 -> 0.16),
# momentum_short (0.20 -> 0.18) to make room for funding_rate (0.08).
WEIGHTS = {
    "momentum_short": 0.18,
    "momentum_medium": 0.16,
    "trend_slope": 0.14,
    "exchange_alignment": 0.10,
    "distance_from_strike": 0.34,
    "funding_rate": 0.08,
}

SIGNAL_CLIP = 6.0
ALIGNMENT_WARN_THRESHOLD = -0.5
MIN_HISTORY_SECONDS = 300

# Funding rate scaling: typical 0.0001 (0.01%), extreme 0.005-0.01.
# Multiply by 1000 so 0.001 -> 1.0 signal value (comparable to other signals).
FUNDING_SCALE = 1000.0


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
    """Average across available exchanges. Falls back gracefully if any are missing."""
    prices = []
    if snap_asset.get("kraken") is not None:
        prices.append(snap_asset["kraken"])
    if snap_asset.get("coinbase") is not None:
        prices.append(snap_asset["coinbase"])
    if snap_asset.get("binance_us") is not None:
        prices.append(snap_asset["binance_us"])
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
            "bn": a.get("binance_us"),
            "funding": a.get("funding_rate"),
        })
    series.sort(key=lambda x: x["t"])
    return series


def price_n_seconds_ago(series, now, seconds):
    cutoff = now.timestamp() - seconds
    if not series or series[0]["t"].timestamp() > cutoff:
        return None
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

    history_seconds = (current["t"] - series[0]["t"]).total_seconds() if len(series) > 1 else 0
    history_thin = history_seconds < MIN_HISTORY_SECONDS

    five_min_cutoff = now.timestamp() - 300
    recent = [p["cp"] for p in series if p["t"].timestamp() >= five_min_cutoff]
    if len(recent) >= 3 and history_seconds >= 120:
        vol = stdev(recent)
    else:
        vol = max(strike * 0.001, 1e-9)
    vol = max(vol, strike * 0.0003)

    p_60s = price_n_seconds_ago(series, now, 60)
    p_300s = price_n_seconds_ago(series, now, 300)

    momentum_short = 0.0
    if p_60s:
        momentum_short = (current_price - p_60s["cp"]) / vol

    momentum_medium = 0.0
    if p_300s and not history_thin:
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
            exchange_alignment = -0.5

    phase = max(0.0, min(1.0, 1.0 - (seconds_left / 900.0)))
    base_distance = distance / vol
    distance_from_strike = base_distance * (1.0 + (phase ** 2) * 1.0)

    # NEW: Funding rate signal
    # Positive funding = longs paying shorts = bullish positioning -> supports prob_yes up
    # (since YES = price ends >= strike, bullish bias makes price more likely to go up)
    # Funding is a slow-moving variable so we use the latest value, scaled and clipped.
    funding_rate_value = current.get("funding")
    if funding_rate_value is not None:
        funding_rate_signal = funding_rate_value * FUNDING_SCALE
    else:
        funding_rate_signal = 0.0

    signals = {
        "momentum_short": momentum_short,
        "momentum_medium": momentum_medium,
        "trend_slope": trend_slope,
        "exchange_alignment": exchange_alignment,
        "distance_from_strike": distance_from_strike,
        "funding_rate": funding_rate_signal,
    }

    log_odds = 0.0
    contributions = {}
    for name, value in signals.items():
        clipped = max(-SIGNAL_CLIP, min(SIGNAL_CLIP, value))
        contrib = WEIGHTS[name] * clipped
        log_odds += contrib
        contributions[name] = {
            "raw": round(value, 4),
            "clipped": round(clipped, 4),
            "weight": WEIGHTS[name],
            "contribution": round(contrib, 4),
        }

    prob_yes = sigmoid(log_odds)

    if history_thin:
        history_score = 0.3
    else:
        history_score = min(1.0, history_seconds / 600)
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

    if history_thin:
        summary_parts.append("warming up")

    minutes_left = seconds_left / 60
    summary_parts.append(f"{minutes_left:.1f} min left")
    summary = "; ".join(summary_parts) + "."

    return {
        "ticker": market.get("ticker"),
        "asset": asset_name,
        "strike": strike,
        "current_price": round(current_price, 6),
        "binance_us_price": current.get("bn"),
        "funding_rate": funding_rate_value,
        "close_time": close_time_str,
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
        "history_seconds": int(history_seconds),
    }


def log_predictions(predictions, snapshot_time):
    """Append each prediction to a daily prediction log file with raw signals."""
    if not predictions:
        return
    os.makedirs(PRED_LOG_DIR, exist_ok=True)
    try:
        ts = datetime.fromisoformat(snapshot_time)
    except Exception:
        ts = datetime.now(timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    path = os.path.join(PRED_LOG_DIR, f"{date_str}.jsonl")
    with open(path, "a") as f:
        for p in predictions:
            row = {
                "snapshot_time": snapshot_time,
                "ticker": p["ticker"],
                "asset": p["asset"],
                "strike": p["strike"],
                "current_price": p["current_price"],
                "binance_us_price": p.get("binance_us_price"),
                "funding_rate": p.get("funding_rate"),
                "close_time": p["close_time"],
                "seconds_left": p["seconds_left"],
                "prob_yes_estimate": p["prob_yes_estimate"],
                "market_yes_bid": p["market_yes_bid"],
                "market_yes_ask": p["market_yes_ask"],
                "market_mid": p["market_mid"],
                "confidence": p["confidence"],
                "history_seconds": p["history_seconds"],
            }
            # Add raw signal values to each row, flattened for easy analysis later
            sigs = p.get("signals", {})
            for sig_name, sig_data in sigs.items():
                row[f"signal_{sig_name}_raw"] = sig_data.get("raw")
                row[f"signal_{sig_name}_clipped"] = sig_data.get("clipped")
                row[f"signal_{sig_name}_contribution"] = sig_data.get("contribution")
            f.write(json.dumps(row) + "\n")


def load_already_settled():
    if not os.path.exists(SETTLED_PATH):
        return set()
    seen = set()
    with open(SETTLED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                t = row.get("ticker")
                if t:
                    seen.add(t)
            except json.JSONDecodeError:
                continue
    return seen


def detect_and_log_settlements(history, current_open_tickers):
    already = load_already_settled()

    all_seen = {}
    for snap in history:
        for asset_name, asset_data in snap.get("assets", {}).items():
            for m in asset_data.get("markets", []):
                t = m.get("ticker")
                if not t:
                    continue
                if t not in all_seen:
                    all_seen[t] = {
                        "asset": asset_name,
                        "strike": parse_float(m.get("strike")),
                        "close_time": m.get("close_time"),
                    }

    new_settlements = []
    for ticker, info in all_seen.items():
        if ticker in already:
            continue
        if ticker in current_open_tickers:
            continue
        if info["close_time"] is None or info["strike"] is None:
            continue

        try:
            close_time = datetime.fromisoformat(
                info["close_time"].replace("Z", "+00:00")
            )
        except Exception:
            continue

        now = datetime.now(timezone.utc)
        if now < close_time:
            continue

        sixty_before = close_time.timestamp() - 60
        close_ts = close_time.timestamp()
        prices_in_window = []
        for snap in history:
            try:
                snap_ts = datetime.fromisoformat(snap["ts"]).timestamp()
            except Exception:
                continue
            if sixty_before <= snap_ts <= close_ts:
                a = snap.get("assets", {}).get(info["asset"])
                if a:
                    cp = composite_price(a)
                    if cp is not None:
                        prices_in_window.append(cp)

        if not prices_in_window:
            outcome_data = {
                "ticker": ticker,
                "asset": info["asset"],
                "strike": info["strike"],
                "close_time": info["close_time"],
                "settle_avg_price": None,
                "outcome": "unknown",
                "note": "no price data in 60s window before close",
                "logged_at": now.isoformat(),
            }
        else:
            avg = sum(prices_in_window) / len(prices_in_window)
            outcome = "YES" if avg >= info["strike"] else "NO"
            outcome_data = {
                "ticker": ticker,
                "asset": info["asset"],
                "strike": info["strike"],
                "close_time": info["close_time"],
                "settle_avg_price": round(avg, 6),
                "settle_samples": len(prices_in_window),
                "outcome": outcome,
                "logged_at": now.isoformat(),
            }
        new_settlements.append(outcome_data)

    if new_settlements:
        with open(SETTLED_PATH, "a") as f:
            for s in new_settlements:
                f.write(json.dumps(s) + "\n")
        print(f"Logged {len(new_settlements)} new settlements")
        for s in new_settlements:
            print(f"  {s['ticker']:35s} -> {s['outcome']}")


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
    current_open_tickers = set()
    for asset_name, asset_data in latest.get("assets", {}).items():
        series = get_asset_series(history, asset_name)
        if not series:
            continue
        for market in asset_data.get("markets", []):
            t = market.get("ticker")
            if t:
                current_open_tickers.add(t)
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

    log_predictions(predictions, latest["ts"])
    detect_and_log_settlements(history, current_open_tickers)

    print(f"Wrote {len(predictions)} predictions to prediction.json")
    for p in predictions:
        print(f"  {p['asset']:5s} prob={p['prob_yes_estimate']:.3f} "
              f"market={p['market_mid']} disagree={p['disagreement']} "
              f"conf={p['confidence']}")


if __name__ == "__main__":
    main()
