"""
Fit weights for the 6 prediction signals via logistic regression.

Reads predictions/*.jsonl (with raw signal values) and settled.jsonl,
joins by ticker, then fits a logistic regression to find empirical
weights that best predict YES/NO outcomes.

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

SIGNAL_CLIP = 6.0  # Match analyze.py
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42  # Fixed so results are reproducible

# Gradient descent hyperparameters
LEARNING_RATE = 0.05
N_EPOCHS = 2000
L2_REG = 0.001  # Small L2 to prevent any one weight from blowing up


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
    """Returns (X, y) where X is list of feature vectors, y is list of 0/1 outcomes."""
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
    for p in predictions:
        ticker = p.get("ticker")
        if ticker not in settled_by_ticker:
            continue
        # Pull all 6 raw signals; skip if any missing
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
    return X, y


def train_logistic(X_train, y_train, n_features):
    """Train via gradient descent with L2 regularization."""
    weights = [0.0] * n_features
    intercept = 0.0
    n = len(X_train)

    for epoch in range(N_EPOCHS):
        # Compute gradients
        grad_w = [0.0] * n_features
        grad_b = 0.0

        for i in range(n):
            z = intercept + sum(weights[j] * X_train[i][j] for j in range(n_features))
            p = sigmoid(z)
            error = p - y_train[i]
            for j in range(n_features):
                grad_w[j] += error * X_train[i][j]
            grad_b += error

        # Average + L2
        for j in range(n_features):
            grad_w[j] = grad_w[j] / n + L2_REG * weights[j]
        grad_b /= n

        # Update
        for j in range(n_features):
            weights[j] -= LEARNING_RATE * grad_w[j]
        intercept -= LEARNING_RATE * grad_b

    return weights, intercept


def evaluate(weights, intercept, X, y):
    """Return accuracy, Brier score, log-loss on a dataset."""
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
        # Clamp probabilities for log-loss
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


def main():
    X, y = build_dataset()
    print(f"Built dataset: {len(X)} examples with all 6 signals + outcomes")
    if len(X) < 50:
        print("Insufficient data for fit. Need more matched predictions with raw signals.")
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_total": len(X),
            "note": "insufficient data for reliable fit",
        }
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return

    # Shuffle and split
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    split_idx = int(len(X) * TRAIN_FRACTION)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train YES rate: {sum(y_train)/len(y_train):.3f}, Test YES rate: {sum(y_test)/len(y_test):.3f}")

    print("Training...")
    weights, intercept = train_logistic(X_train, y_train, len(SIGNAL_NAMES))

    train_metrics = evaluate(weights, intercept, X_train, y_train)
    test_metrics = evaluate(weights, intercept, X_test, y_test)

    # Compare against current weights from analyze.py
    current_weights = {
        "momentum_short": 0.18,
        "momentum_medium": 0.16,
        "trend_slope": 0.14,
        "exchange_alignment": 0.10,
        "distance_from_strike": 0.34,
        "funding_rate": 0.08,
    }
    current_intercept = 0.0  # analyze.py doesn't have an explicit intercept

    def eval_with_dict(weight_dict, intercept_val, X, y):
        n = len(X)
        if n == 0:
            return None
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

    current_train = eval_with_dict(current_weights, current_intercept, X_train, y_train)
    current_test = eval_with_dict(current_weights, current_intercept, X_test, y_test)

    fitted_weights_dict = {
        SIGNAL_NAMES[j]: round(weights[j], 4) for j in range(len(SIGNAL_NAMES))
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(X),
        "train_fraction": TRAIN_FRACTION,
        "random_seed": RANDOM_SEED,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "epochs": N_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2_reg": L2_REG,
        "fitted_weights": fitted_weights_dict,
        "fitted_intercept": round(intercept, 4),
        "fitted_metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "current_weights": current_weights,
        "current_intercept": current_intercept,
        "current_metrics": {
            "train": current_train,
            "test": current_test,
        },
        "comparison_test": {
            "fitted_brier": test_metrics["brier_score"],
            "current_brier": current_test["brier_score"],
            "fitted_better": test_metrics["brier_score"] < current_test["brier_score"],
            "brier_improvement": round(
                current_test["brier_score"] - test_metrics["brier_score"], 4
            ),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 60)
    print("FITTED WEIGHTS:")
    for sig, w in fitted_weights_dict.items():
        old = current_weights[sig]
        print(f"  {sig:25s}: {w:+.4f}  (was {old:+.4f}, change {w - old:+.4f})")
    print(f"  intercept                : {round(intercept, 4):+.4f}")
    print()
    print(f"TRAIN: fitted brier={train_metrics['brier_score']}, "
          f"current brier={current_train['brier_score']}")
    print(f"TEST:  fitted brier={test_metrics['brier_score']}, "
          f"current brier={current_test['brier_score']}")
    print(f"TEST:  fitted accuracy={test_metrics['accuracy']}, "
          f"current accuracy={current_test['accuracy']}")
    print()
    if result["comparison_test"]["fitted_better"]:
        print(f"Fitted weights improve test Brier by "
              f"{result['comparison_test']['brier_improvement']}")
    else:
        print("Fitted weights do NOT improve over current weights on test set")


if __name__ == "__main__":
    main()
