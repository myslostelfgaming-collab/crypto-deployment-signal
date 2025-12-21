#!/usr/bin/env python

import csv
import os
from typing import Dict, Tuple, Optional

FORECAST_CSV = os.path.join("data", "forecast", "baseline_probs_v1.csv")

HORIZONS = [12, 24, 36, 48, 60, 72, 84, 96]
THRESHOLDS = ["0.5", "1.0", "2.0", "3.0", "5.0"]


def to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def main():
    print("FORECAST VALIDATION REPORT (baseline v1)")
    print("=======================================")
    print()

    if not os.path.isfile(FORECAST_CSV):
        print(f"STATUS: FAIL - missing file: {FORECAST_CSV}")
        return

    with open(FORECAST_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")
    if not rows:
        print("\nSTATUS: WARN - forecast file is empty")
        return

    # Build map: (target, horizon) -> row
    m: Dict[Tuple[str, int], dict] = {}
    for r in rows:
        t = r.get("target_pct", "").strip()
        h = to_int(r.get("horizon_h", "").strip() or "")
        if not t or h is None:
            continue
        m[(t, h)] = r

    # Basic checks
    bad_prob = 0
    bad_counts = 0
    missing_pairs = 0

    for t in THRESHOLDS:
        for h in HORIZONS:
            r = m.get((t, h))
            if r is None:
                missing_pairs += 1
                continue

            n_total = to_int(r.get("n_total", "") or "")
            n_hit = to_int(r.get("n_hit", "") or "")
            p_hit = to_float(r.get("p_hit", "") or "")

            if n_total is None or n_hit is None or p_hit is None:
                bad_counts += 1
                continue

            if n_total < 0 or n_hit < 0 or n_hit > n_total:
                bad_counts += 1

            if p_hit < -1e-9 or p_hit > 1.0 + 1e-9:
                bad_prob += 1

            # Consistency: p_hit ≈ n_hit/n_total
            expected = (n_hit / n_total) if n_total else 0.0
            if abs(p_hit - expected) > 1e-6:
                bad_prob += 1

    print("\nCoverage:")
    print(f"  Missing (target,horizon) pairs: {missing_pairs}")

    print("\nBasic sanity:")
    print(f"  Count issues: {bad_counts}")
    print(f"  Probability issues: {bad_prob}")

    # Monotonicity expectation: for fixed target, p_hit should not decrease as horizon increases
    mono_viol = 0
    mono_checked = 0

    for t in THRESHOLDS:
        prev_p = None
        for h in HORIZONS:
            r = m.get((t, h))
            if r is None:
                continue
            p = to_float(r.get("p_hit", "") or "")
            if p is None:
                continue
            if prev_p is not None:
                mono_checked += 1
                if p + 1e-9 < prev_p:
                    mono_viol += 1
            prev_p = p

    print("\nMonotonicity (p_hit vs horizon):")
    print(f"  Checked: {mono_checked}")
    print(f"  Violations: {mono_viol}")

    if missing_pairs > 0:
        status = "WARN"
    elif bad_counts > 0 or bad_prob > 0 or mono_viol > 0:
        status = "WARN"
    else:
        status = "OK"

    print(f"\nSTATUS: {status} - baseline forecast sanity {'OK' if status == 'OK' else 'check warnings above'}")


if __name__ == "__main__":
    main()