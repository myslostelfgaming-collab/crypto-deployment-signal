#!/usr/bin/env python

import csv
import os

LABELS_PATH = os.path.join("data", "labels", "labels_v1.csv")

THR_KEYS = ["0p5", "1", "2", "3", "5"]


def parse_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def main():
    if not os.path.isfile(LABELS_PATH):
        print(f"Labels file not found: {LABELS_PATH}")
        return

    with open(LABELS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        print("Labels file is empty; nothing to repair.")
        return

    fixes = 0
    for r in rows:
        for k in THR_KEYS:
            col = f"mdd_before_hit_up_{k}"
            if col not in r:
                continue
            v = r.get(col, "")
            if v == "":
                continue
            x = parse_float(v)
            if x is None:
                continue
            if x > 0.0:
                r[col] = "0.0"
                fixes += 1

    # Write back (overwrite)
    with open(LABELS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Repaired labels file: {LABELS_PATH}")
    print(f"Positive MDD values clamped to 0.0: {fixes}")


if __name__ == "__main__":
    main()