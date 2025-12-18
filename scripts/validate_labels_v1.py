#!/usr/bin/env python

import csv
import os

LABELS_CSV = os.path.join("data", "labels", "labels_v1.csv")

HORIZONS = [12, 24, 36, 48, 60, 72, 84, 96]
THRESHOLDS = ["0p5", "1", "2", "3", "5"]


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def to_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def monotonic_non_decreasing(values):
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            return False
    return True


def main():
    print("\nLABEL VALIDATION REPORT (v1)")
    print("============================\n")

    if not os.path.isfile(LABELS_CSV):
        print("STATUS: WARN - labels file not found")
        print("Nothing to validate yet.\n")
        return

    rows = load_rows(LABELS_CSV)
    print(f"Total rows: {len(rows)}\n")

    if not rows:
        print("STATUS: WARN - labels file is empty\n")
        return

    # -------------------------------------------------
    # 1) Monotonicity checks
    # -------------------------------------------------
    mono_up_viol = []
    mono_range_viol = []
    checked = 0

    for r in rows:
        up_vals = []
        range_vals = []
        valid = True

        for h in HORIZONS:
            up = to_float(r.get(f"max_up_pct_{h}"))
            rg = to_float(r.get(f"range_pct_{h}"))
            if up is None or rg is None:
                valid = False
                break
            up_vals.append(up)
            range_vals.append(rg)

        if not valid:
            continue

        checked += 1

        if not monotonic_non_decreasing(up_vals):
            mono_up_viol.append(r.get("published_at_utc"))

        if not monotonic_non_decreasing(range_vals):
            mono_range_viol.append(r.get("published_at_utc"))

    # -------------------------------------------------
    # 2) Time-to-hit coherence (0.5%)
    # -------------------------------------------------
    tth_checked = 0
    tth_viol = 0

    for r in rows:
        t_hit = to_int(r.get("t_hit_up_0p5"))
        if t_hit is None:
            continue

        tth_checked += 1

        if t_hit < 1 or t_hit > 96:
            tth_viol += 1
            continue

        # bucket check
        bucket = None
        for h in HORIZONS:
            if t_hit <= h:
                bucket = h
                break

        if bucket is not None:
            max_up = to_float(r.get(f"max_up_pct_{bucket}"))
            if max_up is None or max_up < 0.5:
                tth_viol += 1

    # -------------------------------------------------
    # 3) Drawdown sign checks
    # -------------------------------------------------
    dd_viol = {thr: 0 for thr in THRESHOLDS}
    dd_checked = {thr: 0 for thr in THRESHOLDS}

    for r in rows:
        for thr in THRESHOLDS:
            v = to_float(r.get(f"mdd_before_hit_up_{thr}"))
            if v is None:
                continue
            dd_checked[thr] += 1
            if v > 0:
                dd_viol[thr] += 1

    # -------------------------------------------------
    # 4) Label maturity
    # -------------------------------------------------
    maturity = {h: 0 for h in HORIZONS}
    for r in rows:
        for h in HORIZONS:
            if to_float(r.get(f"max_up_pct_{h}")) is not None:
                maturity[h] += 1

    # -------------------------------------------------
    # Report
    # -------------------------------------------------
    print("Monotonicity:")
    print(f"  Rows checked: {checked}")
    print(f"  max_up_pct : {'PASS' if not mono_up_viol else 'WARN'} ({len(mono_up_viol)} violations)")
    print(f"  range_pct  : {'PASS' if not mono_range_viol else 'WARN'} ({len(mono_range_viol)} violations)\n")

    print("Time-to-hit coherence (0.5%):")
    print(f"  Rows checked: {tth_checked}")
    print(f"  Violations  : {tth_viol}\n")

    print("Drawdown sign (must be <= 0):")
    for thr in THRESHOLDS:
        status = "PASS" if dd_viol[thr] == 0 else "WARN"
        print(f"  {thr.replace('p', '.')}%: {status} ({dd_viol[thr]} violations, checked {dd_checked[thr]})")
    print()

    print("Label maturity:")
    for h in HORIZONS:
        print(f"  {h}h: {maturity[h]}")
    print()

    if not mono_up_viol and not mono_range_viol and tth_viol == 0 and all(dd_viol[t] == 0 for t in THRESHOLDS):
        print("STATUS: DATA SANITY OK\n")
    else:
        print("STATUS: WARN - review issues above\n")


if __name__ == "__main__":
    main()