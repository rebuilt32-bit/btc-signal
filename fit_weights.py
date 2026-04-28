"""
Fit weights for the 6 prediction signals via logistic regression with k-fold
cross-validation and L2 regularization sweep.

Reads predictions/*.jsonl (with raw signal values) and settled.jsonl, joins by
ticker, fits logistic regression with k-fold CV across multiple L2 strengths,
picks the best, and reports both the cross-validated metrics and the final fit
on all data.

Output: data/fitted_weights.json
This script does NOT modify analyze.py — the deployment decision is separate.
"""
import json
import os
import math
import random
from datetime import datetime, timezone

PRED_DIR = "data/predictions"
SETTLED_PATH = "data/settled.jsonl"
OUT_PATH = "data/fitted_weights.json"

SIGNAL_NAMES = [
    "momentum_short",
    "momentum_medium",
    "trend_slope",
    "exchange_alignment",
    "distance_from_strike",
    "funding_rate",
]

SIGNAL_CLIP = 6.0
RANDOM_SEED = 42
K_FOLDS = 5

# Hyperparameters
LEARNING_RATE = 0.05
N_EPOCHS = 2000

# L2 regularization strengths to try
L2_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0]


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


def sigmoid(x):
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def clip(v, limit=SIGNAL_CLIP):
    return max(-limit, min(limit, v))


def build_dataset():
    """Returns (X, y, tickers) where X is feature vectors, y is 0/1 outcomes."""
    predictions = load_all_predictions()
    settlements = load_jsonl(SETTLED_PATH)

    settled_by_ticker = {}
    for s in settlements:
        ticker = s.get("ticker")
        outcome = s.get("outcome")
        if ticker and outcome and outcome != "unknown":
            settled_by_ticker[ticker] = 1 if outcome == "YES" else 0

    X = []
    y = []
    tickers = []
    for p in predictions:
        ticker = p.get("ticker")
        if ticker not in settled_by_ticker:
            continue
        features = []
        valid = True
        for sig in SIGNAL_NAMES:
            v = p.get(f"signal_{sig}_raw")
            if v is None:
                valid = False
                break
            features.append(clip(v))
        if not valid:
            continue
        X.append(features)
        y.append(settled_by_ticker[ticker])
        tickers.append(ticker)
    return X, y, tickers


def train_logistic(X_train, y_train, n_features, l2_reg, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train via gradient descent with L2 regularization."""
    weights = [0.0] * n_features
    intercept = 0.0
    n = len(X_train)

    for epoch in range(n_epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for i in range(n):
            z = intercept + sum(weights[j] * X_train[i][j] for j in range(n_features))
            p = sigmoid(z)
            error = p - y_train[i]
            for j in range(n_features):
                grad_w[j] += error * X_train[i][j]
            grad_b += error
        for j in range(n_features):
            grad_w[j] = grad_w[j] / n + l2_reg * weights[j]
        grad_b /= n
        for j in range(n_features):
            weights[j] -= lr * grad_w[j]
        intercept -= lr * grad_b
    return weights, intercept


def evaluate(weights, intercept, X, y):
    if not X:
        return None
    n = len(X)
    correct = 0
    brier_sum = 0.0
    logloss_sum = 0.0
    for i in range(n):
        z = intercept + sum(weights[j] * X[i][j] for j in range(len(weights)))
        p = sigmoid(z)
        pred = 1 if p >= 0.5 else 0
        if pred == y[i]:
            correct += 1
        brier_sum += (p - y[i]) ** 2
        p_clamped = max(1e-9, min(1 - 1e-9, p))
        if y[i] == 1:
            logloss_sum -= math.log(p_clamped)
        else:
            logloss_sum -= math.log(1 - p_clamped)
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "brier_score": round(brier_sum / n, 4),
        "log_loss": round(logloss_sum / n, 4),
    }


def evaluate_with_dict(weight_dict, intercept_val, X, y):
    """Same as evaluate but using a dict of named weights — for current_weights comparison."""
    if not X:
        return None
    n = len(X)
    correct = 0
    brier_sum = 0.0
    for i in range(n):
        z = intercept_val + sum(
            weight_dict[SIGNAL_NAMES[j]] * X[i][j]
            for j in range(len(SIGNAL_NAMES))
        )
        p = sigmoid(z)
        pred = 1 if p >= 0.5 else 0
        if pred == y[i]:
            correct += 1
        brier_sum += (p - y[i]) ** 2
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "brier_score": round(brier_sum / n, 4),
    }


def k_fold_indices(n, k, seed):
    """Generate k folds of indices for cross-validation."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    fold_size = n // k
    folds = []
    for f in range(k):
        start = f * fold_size
        end = start + fold_size if f < k - 1 else n
        folds.append(indices[start:end])
    return folds


def cross_validate(X, y, l2_reg, k=K_FOLDS, seed=RANDOM_SEED):
    """
    Run k-fold CV. For each fold, train on remaining data and evaluate on the fold.
    Returns list of fold results and aggregated stats.
    """
    n = len(X)
    folds = k_fold_indices(n, k, seed)
    fold_results = []
    all_weights = []
    all_intercepts = []

    for f_idx, test_indices in enumerate(folds):
        test_set = set(test_indices)
        X_train = [X[i] for i in range(n) if i not in test_set]
        y_train = [y[i] for i in range(n) if i not in test_set]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        weights, intercept = train_logistic(X_train, y_train, len(SIGNAL_NAMES), l2_reg)
        train_metrics = evaluate(weights, intercept, X_train, y_train)
        test_metrics = evaluate(weights, intercept, X_test, y_test)

        fold_results.append({
            "fold": f_idx + 1,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "weights": {SIGNAL_NAMES[j]: round(weights[j], 4) for j in range(len(SIGNAL_NAMES))},
            "intercept": round(intercept, 4),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        })
        all_weights.append(weights)
        all_intercepts.append(intercept)

    mean_test_brier = sum(f["test_metrics"]["brier_score"] for f in fold_results) / k
    mean_test_acc = sum(f["test_metrics"]["accuracy"] for f in fold_results) / k
    mean_test_logloss = sum(f["test_metrics"]["log_loss"] for f in fold_results) / k

    weight_stability = {}
    for j, sig_name in enumerate(SIGNAL_NAMES):
        vals = [w[j] for w in all_weights]
        mean_w = sum(vals) / k
        var_w = sum((v - mean_w) ** 2 for v in vals) / k
        std_w = math.sqrt(var_w)
        n_positive = sum(1 for v in vals if v > 0)
        sign_consistent = n_positive == k or n_positive == 0
        weight_stability[sig_name] = {
            "mean": round(mean_w, 4),
            "stdev": round(std_w, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n_positive_folds": n_positive,
            "sign_consistent": sign_consistent,
        }

    intercept_mean = sum(all_intercepts) / k
    intercept_std = math.sqrt(sum((v - intercept_mean) ** 2 for v in all_intercepts) / k)

    return {
        "l2_reg": l2_reg,
        "k_folds": k,
        "mean_test_brier": round(mean_test_brier, 4),
        "mean_test_accuracy": round(mean_test_acc, 4),
        "mean_test_log_loss": round(mean_test_logloss, 4),
        "weight_stability": weight_stability,
        "intercept_mean": round(intercept_mean, 4),
        "intercept_stdev": round(intercept_std, 4),
        "fold_results": fold_results,
    }


def main():
    X, y, tickers = build_dataset()
    print(f"Built dataset: {len(X)} examples with all 6 signals + outcomes")

    if len(X) < 50:
        print("Insufficient data for fit.")
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_total": len(X),
            "note": "insufficient data",
        }
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return

    yes_rate = sum(y) / len(y)
    print(f"Overall YES rate: {yes_rate:.3f}")

    current_weights = {
        "momentum_short": 0.18,
        "momentum_medium": 0.16,
        "trend_slope": 0.14,
        "exchange_alignment": 0.10,
        "distance_from_strike": 0.34,
        "funding_rate": 0.08,
    }
    current_intercept = 0.0

    folds = k_fold_indices(len(X), K_FOLDS, RANDOM_SEED)
    n = len(X)
    current_fold_briers = []
    for test_indices in folds:
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]
        m = evaluate_with_dict(current_weights, current_intercept, X_test, y_test)
        if m:
            current_fold_briers.append(m["brier_score"])
    current_mean_brier = sum(current_fold_briers) / len(current_fold_briers)
    print(f"Current weights mean test Brier across {K_FOLDS} folds: {current_mean_brier:.4f}")

    print()
    print(f"Running L2 sweep across {len(L2_CANDIDATES)} candidates...")
    l2_results = []
    for l2 in L2_CANDIDATES:
        print(f"  L2={l2}...", end=" ", flush=True)
        cv_result = cross_validate(X, y, l2, k=K_FOLDS, seed=RANDOM_SEED)
        l2_results.append(cv_result)
        print(f"mean test brier={cv_result['mean_test_brier']}")

    best = min(l2_results, key=lambda r: r["mean_test_brier"])
    print()
    print(f"Best L2: {best['l2_reg']} (mean test brier={best['mean_test_brier']})")

    print(f"Final fit on all {len(X)} samples with L2={best['l2_reg']}...")
    final_weights, final_intercept = train_logistic(X, y, len(SIGNAL_NAMES), best["l2_reg"])
    final_train_metrics = evaluate(final_weights, final_intercept, X, y)
    final_weights_dict = {
        SIGNAL_NAMES[j]: round(final_weights[j], 4) for j in range(len(SIGNAL_NAMES))
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(X),
        "yes_rate": round(yes_rate, 4),
        "k_folds": K_FOLDS,
        "random_seed": RANDOM_SEED,
        "epochs": N_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2_candidates_tried": L2_CANDIDATES,
        "best_l2": best["l2_reg"],
        "current_weights_mean_test_brier": round(current_mean_brier, 4),
        "best_fitted_mean_test_brier": best["mean_test_brier"],
        "best_fitted_mean_test_accuracy": best["mean_test_accuracy"],
        "improves_over_current": best["mean_test_brier"] < current_mean_brier,
        "brier_improvement": round(current_mean_brier - best["mean_test_brier"], 4),
        "best_l2_weight_stability": best["weight_stability"],
        "final_weights_full_data": final_weights_dict,
        "final_intercept_full_data": round(final_intercept, 4),
        "final_metrics_full_data": final_train_metrics,
        "current_weights": current_weights,
        "current_intercept": current_intercept,
        "all_l2_results": [
            {
                "l2": r["l2_reg"],
                "mean_test_brier": r["mean_test_brier"],
                "mean_test_accuracy": r["mean_test_accuracy"],
                "weight_stability": r["weight_stability"],
            }
            for r in l2_results
        ],
        "best_l2_fold_details": best["fold_results"],
    }

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 60)
    print("FINAL WEIGHTS (fit on all data with best L2):")
    for sig, w in final_weights_dict.items():
        old = current_weights[sig]
        print(f"  {sig:25s}: {w:+.4f}  (was {old:+.4f}, change {w - old:+.4f})")
    print(f"  intercept                : {round(final_intercept, 4):+.4f}")
    print()
    print("WEIGHT STABILITY ACROSS FOLDS (best L2):")
    for sig, stats in best["weight_stability"].items():
        sign_note = "OK" if stats["sign_consistent"] else "FLIPS"
        print(f"  {sig:25s}: mean={stats['mean']:+.4f} stdev={stats['stdev']:.4f} "
              f"range=[{stats['min']:+.4f}, {stats['max']:+.4f}] {sign_note}")
    print()
    print(f"Current weights brier: {current_mean_brier:.4f}")
    print(f"Best fitted brier:     {best['mean_test_brier']:.4f}")
    if best["mean_test_brier"] < current_mean_brier:
        improvement = current_mean_brier - best["mean_test_brier"]
        improvement_pct = (improvement / current_mean_brier) * 100
        print(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}% better)")
    else:
        print("Fitted weights do NOT improve over current.")


if __name__ == "__main__":
    main()
