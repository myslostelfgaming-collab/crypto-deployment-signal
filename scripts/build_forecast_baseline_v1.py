#!/usr/bin/env python

import csv
import json
import os
from typing import Dict, List, Tuple, Optional

LABELS_CSV = os.path.join("data", "labels", "labels_v1.csv")
OUT_DIR = os.path.join("data", "forecast")
OUT_CSV = os.path.join(OUT_DIR, "baseline_probs_v1.csv")
OUT_JSON = os.path.join(OUT_DIR, "baseline_summary_v1.json")

HORIZONS = [12, 24, 36, 48, 60, 72, 84, 96]
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]


def fmt_thr(x: float) -> str:
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return str(x).replace(".", "p")


def to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def to_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except Exception:
        return None


def quantile(sorted_vals: List[float], q: float) -> Optional[float]:
    """
    Simple linear interpolation quantile.
    Requires sorted list.
    """
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_hits(hit_times: List[int]) -> Dict[str, Optional[float]]:
    hit_times_sorted = sorted(hit_times)
    return {
        "t_hit_p25": quantile(hit_times_sorted, 0.25),
        "t_hit_median": quantile(hit_times_sorted, 0.50),
        "t_hit_p75": quantile(hit_times_sorted, 0.75),
    }


def summarize_mdd(mdds: List[float]) -> Dict[str, Optional[float]]:
    mdds_sorted = sorted(mdds)
    return {
        "mdd_p25": quantile(mdds_sorted, 0.25),
        "mdd_median": quantile(mdds_sorted, 0.50),
        "mdd_p75": quantile(mdds_sorted, 0.75),
    }


def load_labels_rows() -> List[dict]:
    if not os.path.isfile(LABELS_CSV):
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

    with open(LABELS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    rows = load_labels_rows()
    if not rows:
        print("labels_v1.csv is empty. Nothing to forecast yet.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    out_rows: List[dict] = []
    out_json: Dict[str, dict] = {
        "schema": "baseline_probs_v1",
        "source_labels": LABELS_CSV,
        "n_rows_labels": len(rows),
        "thresholds": THRESHOLDS,
        "horizons": HORIZONS,
        "results": {},
    }

    for thr in THRESHOLDS:
        thr_key = fmt_thr(thr)
        tcol = f"t_hit_up_{thr_key}"
        mddcol = f"mdd_before_hit_up_{thr_key}"

        out_json["results"][str(thr)] = {}

        for h in HORIZONS:
            n_total = 0
            n_hit = 0
            hit_times: List[int] = []
            hit_mdds: List[float] = []

            for r in rows:
                # rows with missing columns are skipped (should not happen, but defensive)
                if tcol not in r:
                    continue

                n_total += 1

                t = to_int(r.get(tcol, ""))
                if t is not None and t <= h:
                    n_hit += 1
                    hit_times.append(t)

                    mdd = to_float(r.get(mddcol, ""))
                    if mdd is not None:
                        # should already be <= 0, but clamp defensively
                        hit_mdds.append(min(0.0, mdd))

            p_hit = (n_hit / n_total) if n_total else 0.0

            t_stats = summarize_hits(hit_times) if hit_times else {
                "t_hit_p25": None,
                "t_hit_median": None,
                "t_hit_p75": None,
            }
            mdd_stats = summarize_mdd(hit_mdds) if hit_mdds else {
                "mdd_p25": None,
                "mdd_median": None,
                "mdd_p75": None,
            }

            row_out = {
                "target_pct": str(thr),
                "horizon_h": str(h),
                "n_total": str(n_total),
                "n_hit": str(n_hit),
                "p_hit": str(round(p_hit, 6)),
                "t_hit_p25": "" if t_stats["t_hit_p25"] is None else str(round(t_stats["t_hit_p25"], 4)),
                "t_hit_median": "" if t_stats["t_hit_median"] is None else str(round(t_stats["t_hit_median"], 4)),
                "t_hit_p75": "" if t_stats["t_hit_p75"] is None else str(round(t_stats["t_hit_p75"], 4)),
                "mdd_p25": "" if mdd_stats["mdd_p25"] is None else str(round(mdd_stats["mdd_p25"], 6)),
                "mdd_median": "" if mdd_stats["mdd_median"] is None else str(round(mdd_stats["mdd_median"], 6)),
                "mdd_p75": "" if mdd_stats["mdd_p75"] is None else str(round(mdd_stats["mdd_p75"], 6)),
            }
            out_rows.append(row_out)

            out_json["results"][str(thr)][str(h)] = {
                "n_total": n_total,
                "n_hit": n_hit,
                "p_hit": round(p_hit, 6),
                **t_stats,
                **mdd_stats,
            }

    # Write CSV
    fieldnames = [
        "target_pct",
        "horizon_h",
        "n_total",
        "n_hit",
        "p_hit",
        "t_hit_p25",
        "t_hit_median",
        "t_hit_p75",
        "mdd_p25",
        "mdd_median",
        "mdd_p75",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"Baseline forecast written to:\n- {OUT_CSV}\n- {OUT_JSON}")
    print(f"Source label rows: {len(rows)}")


if __name__ == "__main__":
    main()