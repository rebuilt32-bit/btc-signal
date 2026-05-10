"""Slot calibration — produce per-slot log-odds shifts for use in analyze.py later.

Output is data only. Does NOT modify analyze.py or any production code.
"""
import json, os
from math import log
from datetime import datetime, timezone

IN_PATH = "data/slot_pattern_study.json"
OUT_PATH = "data/slot_calibration.json"


def main():
    if not os.path.exists(IN_PATH):
        print(f"Missing {IN_PATH}. Run slot_pattern_study.py first.")
        return

    with open(IN_PATH) as f:
        data = json.load(f)

    results = data.get("results", {})
    cal = {}

    print("=== Slot calibration: log-odds shifts per slot × checkpoint ===")
    print("Positive shift = leading side wins more often than 50% in this slot.")
    print("Use as additive term in log-odds when scoring confident calls.\n")
    print(f"  {'slot':>5} | " + " | ".join(f"  {cs:>4}  " for cs in ("600s", "300s", "120s", "60s")))

    for slot in (":00", ":15", ":30", ":45"):
        if slot not in results: continue
        cal[slot] = {}
        row = []
        for cs in ("600s", "300s", "120s", "60s"):
            entry = results[slot].get(cs, {})
            r = entry.get("held_rate")
            n = entry.get("n", 0)
            if r is None or n < 100 or r <= 0 or r >= 1:
                cal[slot][cs] = {"n": n, "held_rate": r, "log_odds_shift": None}
                row.append("   n/a   ")
                continue
            shift = log(r / (1 - r))
            cal[slot][cs] = {"n": n, "held_rate": r, "log_odds_shift": shift}
            row.append(f"{shift:>+7.4f}")
        print(f"  {slot:>5} | " + " | ".join(row))

    print("\nReference for use:")
    print("  In analyze.py, after computing model_log_odds for the leading side,")
    print("  add slot_calibration[slot][checkpoint]['log_odds_shift'].")
    print("  Convert back: P_adjusted = 1 / (1 + exp(-adjusted_log_odds))")
    print("  Skip if shift is None (insufficient data).")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": IN_PATH,
            "calibration": cal,
            "usage": "Additive log-odds shift per (slot, checkpoint). Apply to leading-side confidence."
        }, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nNo production code modified. Decide if/how to integrate.")


if __name__ == "__main__":
    main()
