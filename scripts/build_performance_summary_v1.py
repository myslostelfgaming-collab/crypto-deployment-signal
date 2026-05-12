#!/usr/bin/env python3

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PREDICTIONS_PATH = Path("data/model/predictions_v1.csv")
OUTPUT_PATH = Path("data/model/performance_summary_v1.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def direction_is_correct(hit_direction: str) -> bool:
    if not hit_direction:
        return False

    hit = hit_direction.strip().lower()

    # Handles values like "correct", "direction_correct", etc.
    if "correct" in hit and "incorrect" not in hit:
        return True

    return False


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def empty_summary(reason: str) -> Dict[str, Any]:
    return {
        "schema": "performance_summary_v1",
        "generated_at_utc": utc_now(),
        "source_file": str(PREDICTIONS_PATH).replace("\\", "/"),
        "available": False,
        "reason": reason,
        "overall": {
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "pending_predictions": 0,
            "direction_correct": 0,
            "direction_accuracy_pct": None,
            "mean_abs_error": None,
            "mean_abs_pct_error": None,
        },
        "by_asset": {},
        "by_horizon": {},
        "by_asset_horizon": {},
    }


def main() -> None:
    if not PREDICTIONS_PATH.is_file():
        payload = empty_summary("Missing predictions file")
        write_json(OUTPUT_PATH, payload)
        print(f"Wrote empty performance summary: {OUTPUT_PATH}")
        return

    with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_predictions = len(rows)
    evaluated = [r for r in rows if (r.get("status") or "").strip().lower() == "evaluated"]
    pending = [r for r in rows if (r.get("status") or "").strip().lower() != "evaluated"]

    if not evaluated:
        payload = empty_summary("No evaluated predictions yet")
        payload["available"] = True
        payload["overall"]["total_predictions"] = total_predictions
        payload["overall"]["pending_predictions"] = len(pending)
        write_json(OUTPUT_PATH, payload)
        print(f"Wrote performance summary with no evaluated rows yet: {OUTPUT_PATH}")
        return

    def bucket_stats(bucket_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(bucket_rows)
        correct = sum(
            1 for r in bucket_rows
            if direction_is_correct(r.get("hit_direction", ""))
        )

        abs_errors = []
        pct_errors = []

        for r in bucket_rows:
            error_abs = safe_float(r.get("error_abs"))
            error_pct = safe_float(r.get("error_pct"))

            if error_abs is not None:
                abs_errors.append(abs(error_abs))

            if error_pct is not None:
                pct_errors.append(abs(error_pct))

        return {
            "evaluated_predictions": count,
            "direction_correct": correct,
            "direction_accuracy_pct": round((correct / count) * 100.0, 2) if count else None,
            "mean_abs_error": round(safe_mean(abs_errors), 6) if abs_errors else None,
            "mean_abs_pct_error": round(safe_mean(pct_errors), 6) if pct_errors else None,
        }

    overall_stats = bucket_stats(evaluated)
    overall_stats["total_predictions"] = total_predictions
    overall_stats["pending_predictions"] = len(pending)

    by_asset: Dict[str, List[Dict[str, Any]]] = {}
    by_horizon: Dict[str, List[Dict[str, Any]]] = {}
    by_asset_horizon: Dict[str, List[Dict[str, Any]]] = {}

    for r in evaluated:
        asset = r.get("asset") or "UNKNOWN"
        horizon = str(r.get("horizon_h") or "UNKNOWN")
        asset_horizon = f"{asset}|{horizon}"

        by_asset.setdefault(asset, []).append(r)
        by_horizon.setdefault(horizon, []).append(r)
        by_asset_horizon.setdefault(asset_horizon, []).append(r)

    payload = {
        "schema": "performance_summary_v1",
        "generated_at_utc": utc_now(),
        "source_file": str(PREDICTIONS_PATH).replace("\\", "/"),
        "available": True,
        "reason": None,
        "overall": overall_stats,
        "by_asset": {
            asset: bucket_stats(bucket_rows)
            for asset, bucket_rows in sorted(by_asset.items())
        },
        "by_horizon": {
            horizon: bucket_stats(bucket_rows)
            for horizon, bucket_rows in sorted(by_horizon.items(), key=lambda x: str(x[0]))
        },
        "by_asset_horizon": {
            key: bucket_stats(bucket_rows)
            for key, bucket_rows in sorted(by_asset_horizon.items())
        },
    }

    write_json(OUTPUT_PATH, payload)

    print("\n=== PERFORMANCE SUMMARY v1 ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total predictions: {total_predictions}")
    print(f"Evaluated predictions: {overall_stats['evaluated_predictions']}")
    print(f"Pending predictions: {overall_stats['pending_predictions']}")
    print(f"Directional accuracy: {overall_stats['direction_accuracy_pct']}%")


if __name__ == "__main__":
    main()