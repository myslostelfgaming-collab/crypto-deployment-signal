#!/usr/bin/env python
import csv
import json
import os
from typing import Optional, Dict, Any, List, Tuple

SIGNAL_PATH = os.path.join("data", "hourly_signal.json")
OUT_PATH = os.path.join("data", "hourly_forecast.json")

# We will auto-detect the baseline file in these locations (first match wins)
BASELINE_CANDIDATES = [
    os.path.join("data", "forecast", "baseline_v1.json"),
    os.path.join("data", "forecast", "baseline_v1.csv"),
    os.path.join("data", "forecast_baseline_v1.json"),
    os.path.join("data", "forecast_baseline_v1.csv"),
    os.path.join("data", "baseline_v1.json"),
    os.path.join("data", "baseline_v1.csv"),
    os.path.join("data", "labels", "forecast_baseline_v1.json"),
    os.path.join("data", "labels", "forecast_baseline_v1.csv"),
]

DEFAULT_TARGET_PCT = 0.5
DEFAULT_HORIZON_HOURS = 12


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def find_baseline_file() -> Optional[str]:
    for p in BASELINE_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def load_baseline_json(path: str) -> dict:
    return load_json(path)


def load_baseline_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_probability_from_baseline(
    baseline_path: str, target_pct: float, horizon_hours: int
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Supports either:
      - JSON baseline containing e.g. { "rows": [ { "target_pct":0.5, "horizon_hours":12, "p_hit":0.62, ...}, ... ] }
      - CSV baseline containing columns like: target_pct, horizon_hours, p_hit
    Returns: (p_hit, meta)
    """
    meta: Dict[str, Any] = {"baseline_path": baseline_path}

    if baseline_path.endswith(".json"):
        b = load_baseline_json(baseline_path)

        rows = []
        if isinstance(b, dict) and isinstance(b.get("rows"), list):
            rows = b["rows"]
        elif isinstance(b, list):
            rows = b

        best = None
        for r in rows:
            try:
                t = float(r.get("target_pct"))
                h = int(float(r.get("horizon_hours")))
            except Exception:
                continue
            if abs(t - target_pct) < 1e-9 and h == horizon_hours:
                best = r
                break

        if not best:
            return None, {**meta, "reason": "pair_not_found_in_json_baseline"}

        p_hit = safe_float(best.get("p_hit"))
        meta.update(
            {
                "format": "json",
                "matched_row": {
                    "target_pct": target_pct,
                    "horizon_hours": horizon_hours,
                },
            }
        )
        return p_hit, meta

    # CSV
    rows = load_baseline_csv(baseline_path)
    best = None
    for r in rows:
        t = safe_float(r.get("target_pct"))
        h = safe_float(r.get("horizon_hours"))
        if t is None or h is None:
            continue
        if abs(t - target_pct) < 1e-9 and int(h) == horizon_hours:
            best = r
            break

    if not best:
        return None, {**meta, "reason": "pair_not_found_in_csv_baseline"}

    p_hit = safe_float(best.get("p_hit"))
    meta.update(
        {
            "format": "csv",
            "matched_row": {
                "target_pct": target_pct,
                "horizon_hours": horizon_hours,
            },
        }
    )
    return p_hit, meta


def main():
    if not os.path.isfile(SIGNAL_PATH):
        print(f"Missing required file: {SIGNAL_PATH}")
        raise SystemExit(1)

    signal = load_json(SIGNAL_PATH)

    published_at_utc = signal.get("published_at_utc")
    published_at_local = signal.get("published_at_local")
    tz = signal.get("timezone", "Africa/Johannesburg")

    # For now (v1), forecast defaults are fixed.
    # Later (v4.0), these will come from user inputs in chat/app.
    target_pct = DEFAULT_TARGET_PCT
    horizon_hours = DEFAULT_HORIZON_HOURS

    baseline_path = find_baseline_file()

    p_hit = None
    meta = {"baseline_found": False}
    if baseline_path:
        meta["baseline_found"] = True
        p_hit, meta2 = extract_probability_from_baseline(baseline_path, target_pct, horizon_hours)
        meta.update(meta2)

    forecast = {
        "schema": "hourly_forecast_v1",
        "timezone": tz,
        "published_at_utc": published_at_utc,
        "published_at_local": published_at_local,
        "target_pct": target_pct,
        "horizon_hours": horizon_hours,
        "p_hit": p_hit,  # null until baseline exists/matches
        "meta": meta,
        "notes": (
            "Empirical baseline forecast. If p_hit is null, baseline file not found or "
            "missing (target_pct,horizon_hours) pair."
        ),
    }

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)

    print(f"Wrote: {OUT_PATH}")
    if baseline_path:
        print(f"Baseline used: {baseline_path}")
    else:
        print("No baseline file found yet (ok).")


if __name__ == "__main__":
    main()