#!/usr/bin/env python

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

LOCAL_TZ = "Africa/Johannesburg"

SIGNAL_PATH = os.path.join("data", "hourly_signal.json")

# Primary forecast source (baseline summary built from labels)
BASELINE_SUMMARY_PATH = os.path.join("data", "forecast", "baseline_summary_v1.json")

# Optional fallback forecast file (if you later generate a per-hour forecast artifact)
FORECAST_FALLBACK_PATH = os.path.join("data", "hourly_forecast.json")

OUT_PATH = os.path.join("data", "hourly_report.json")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def flatten_baseline_summary(summary: dict, as_of_utc: str) -> dict:
    """
    Convert baseline_summary_v1.json (nested results[thr][horizon]) into:
      forecast = {
        schema, as_of, source, thresholds, horizons, targets[], notes[]
      }

    Each item in targets[] is one (target_pct, horizon_h) row with:
      n_total, n_hit, p_hit, t_hit_* and mdd_* stats.
    """
    thresholds = summary.get("thresholds", [])
    horizons = summary.get("horizons", [])
    results = summary.get("results", {})

    n_labels = summary.get("n_rows_labels")
    labels_path = summary.get("source_labels")

    targets: List[Dict[str, Any]] = []

    # results keys are strings like "0.5", "1.0", "2.0"...
    for thr_key, by_h in results.items():
        thr = safe_float(thr_key)
        if thr is None:
            continue
        if not isinstance(by_h, dict):
            continue

        for h_key, row in by_h.items():
            h = None
            try:
                h = int(h_key)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            targets.append(
                {
                    "target_pct": thr,
                    "horizon_h": h,
                    "n_total": row.get("n_total"),
                    "n_hit": row.get("n_hit"),
                    "p_hit": row.get("p_hit"),
                    "t_hit_p25": row.get("t_hit_p25"),
                    "t_hit_median": row.get("t_hit_median"),
                    "t_hit_p75": row.get("t_hit_p75"),
                    "mdd_p25": row.get("mdd_p25"),
                    "mdd_median": row.get("mdd_median"),
                    "mdd_p75": row.get("mdd_p75"),
                }
            )

    # Stable ordering: by target_pct then horizon_h
    targets.sort(key=lambda x: (x.get("target_pct", 0.0), x.get("horizon_h", 0)))

    forecast = {
        "schema": "forecast_baseline_v1",
        "as_of_utc": as_of_utc,
        "source": {
            "baseline_summary": BASELINE_SUMMARY_PATH,
            "labels": labels_path,
            "n_labels": n_labels,
        },
        "thresholds": thresholds,
        "horizons": horizons,
        "targets": targets,
        "notes": [
            "Baseline probabilities are empirical from labels_v1.csv (hit-anytime within horizon).",
            "Sample size may be small early on; interpret probabilities cautiously until n_labels grows.",
        ],
    }

    return forecast


def resolve_report_timestamps(signal: dict) -> Tuple[str, str, str]:
    """
    Prefer timestamps from the signal (authoritative).
    Fallback: current UTC isoformat if missing.
    Returns: (tz, published_at_utc, published_at_local)
    """
    tz = signal.get("timezone", LOCAL_TZ)
    published_at_utc = signal.get("published_at_utc")
    published_at_local = signal.get("published_at_local")

    if not published_at_utc or not published_at_local:
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        published_at_utc = published_at_utc or now_utc
        # If local is missing, keep it simple (you can improve later with zoneinfo if needed)
        published_at_local = published_at_local or now_utc

    return tz, published_at_utc, published_at_local


def main():
    if not os.path.isfile(SIGNAL_PATH):
        print(f"Missing required file: {SIGNAL_PATH}")
        raise SystemExit(1)

    signal = load_json(SIGNAL_PATH)

    tz, published_at_utc, published_at_local = resolve_report_timestamps(signal)

    # Build forecast block
    forecast = None

    if os.path.isfile(BASELINE_SUMMARY_PATH):
        try:
            summary = load_json(BASELINE_SUMMARY_PATH)
            forecast = flatten_baseline_summary(summary, as_of_utc=published_at_utc)
        except Exception as e:
            print(f"WARN: Failed to load/flatten baseline summary: {e}")
            forecast = None

    # Fallback (optional)
    if forecast is None and os.path.isfile(FORECAST_FALLBACK_PATH):
        try:
            forecast = load_json(FORECAST_FALLBACK_PATH)
        except Exception as e:
            print(f"WARN: Failed to load forecast fallback: {e}")
            forecast = None

    report = {
        "schema": "hourly_report_v1",
        "timezone": tz,
        "published_at_utc": published_at_utc,
        "published_at_local": published_at_local,
        "signal": signal,
        "forecast": forecast,
    }

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {OUT_PATH}")
    if forecast is None:
        print("NOTE: forecast is null (baseline summary and fallback forecast not available).")
    else:
        # If this is the baseline format we built, show quick count
        if isinstance(forecast, dict) and forecast.get("schema") == "forecast_baseline_v1":
            n_targets = len(forecast.get("targets", []))
            n_labels = forecast.get("source", {}).get("n_labels")
            print(f"Forecast attached: forecast_baseline_v1 (n_targets={n_targets}, n_labels={n_labels})")
        else:
            print("Forecast attached: (custom/legacy forecast object)")


if __name__ == "__main__":
    main()