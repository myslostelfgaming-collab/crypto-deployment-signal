#!/usr/bin/env python

import csv
import os

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def main():
    if not os.path.isfile(PREDICTIONS_PATH):
        print("Missing predictions file")
        return

    with open(PREDICTIONS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    evaluated = [r for r in rows if r.get("status") == "evaluated"]

    if not evaluated:
        print("No evaluated predictions yet.")
        return

    total = len(evaluated)

    correct = 0
    abs_errors = []
    pct_errors = []

    horizon_stats = {}

    for r in evaluated:
        hit = r.get("hit_direction", "")
        if "correct" in hit:
            correct += 1

        error_abs = safe_float(r.get("error_abs"))
        error_pct = safe_float(r.get("error_pct"))

        if error_abs is not None:
            abs_errors.append(abs(error_abs))
        if error_pct is not None:
            pct_errors.append(abs(error_pct))

        h = r.get("horizon_h")
        if h not in horizon_stats:
            horizon_stats[h] = {
                "total": 0,
                "correct": 0
            }

        horizon_stats[h]["total"] += 1
        if "correct" in hit:
            horizon_stats[h]["correct"] += 1

    accuracy = correct / total * 100.0

    mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else None
    mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else None

    print("\n=== PERFORMANCE SUMMARY v1 ===")
    print(f"Total evaluated: {total}")
    print(f"Directional accuracy: {accuracy:.2f}%")

    if mean_abs_error is not None:
        print(f"Mean absolute error ($): {mean_abs_error:.4f}")
    if mean_pct_error is not None:
        print(f"Mean error (%): {mean_pct_error:.4f}")

    print("\n--- By Horizon ---")
    for h, stats in sorted(horizon_stats.items()):
        acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"{h}h → {acc:.2f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()