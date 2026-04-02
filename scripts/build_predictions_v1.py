#!/usr/bin/env python

import csv
import json
import os
from datetime import datetime, timezone, timedelta

SIMILARITY_PATH = os.path.join("data", "model", "similarity_forecast_v2.json")
PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")

MODEL_VERSION = "similarity_forecast_v2"
PREDICTION_METHOD = "blend_0p7_median_0p3_mean"

HORIZONS = [24, 48, 168, 336]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str)


def target_ts_utc_from_entry(entry_ts_utc: str, horizon_h: int) -> str:
    ts = int(entry_ts_utc)
    target = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=horizon_h)
    return target.isoformat()


def analogue_quality(best_distance: float | None) -> str:
    if best_distance is None:
        return "weak"
    if best_distance <= 1.75:
        return "strong"
    if best_distance <= 2.5:
        return "medium"
    return "weak"


def blended_prediction_pct(weighted_median: float | None, weighted_mean: float | None) -> float | None:
    if weighted_median is None and weighted_mean is None:
        return None
    if weighted_median is None:
        return weighted_mean
    if weighted_mean is None:
        return weighted_median
    return 0.7 * weighted_median + 0.3 * weighted_mean


def predicted_price(entry_close: float, pred_pct: float | None) -> float | None:
    if pred_pct is None:
        return None
    return entry_close * (1.0 + pred_pct / 100.0)


def existing_prediction_keys(path: str) -> set:
    if not os.path.isfile(path):
        return set()

    keys = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.add((r.get("asset"), r.get("entry_ts_utc"), r.get("horizon_h")))
    return keys


def ensure_predictions_file(path: str) -> None:
    if os.path.isfile(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "asset",
            "created_at_utc",
            "created_at_local",
            "entry_ts_utc",
            "entry_close",
            "horizon_h",
            "target_ts_utc",
            "predicted_close_change_pct",
            "predicted_price",
            "prediction_method",
            "confidence",
            "analogue_quality",
            "neighbors_used",
            "best_distance",
            "weighted_close_mean",
            "weighted_close_median",
            "weighted_upside_mean",
            "weighted_downside_mean",
            "model_version",
            "status",
            "actual_close",
            "actual_close_change_pct",
            "error_abs",
            "error_pct",
            "hit_direction",
            "drift_tag",
            "evaluated_at_utc",
        ])


def main():
    if not os.path.isfile(SIMILARITY_PATH):
        print(f"Missing similarity forecast: {SIMILARITY_PATH}")
        return

    ensure_predictions_file(PREDICTIONS_PATH)
    existing_keys = existing_prediction_keys(PREDICTIONS_PATH)

    sim = load_json(SIMILARITY_PATH)

    asset = sim.get("asset")
    created_at_utc = sim.get("as_of_utc")
    created_at_local = sim.get("as_of_local")

    current_state = sim.get("current_state", {})
    entry_ts_utc = current_state.get("entry_ts_utc")
    entry_close = safe_float(current_state.get("entry_close"))

    overall_confidence = sim.get("overall_confidence", "low")
    best_distance = safe_float(sim.get("similarity", {}).get("best_distance"))
    primary_neighbors_found = sim.get("similarity", {}).get("primary_neighbors_found", 0)

    scorecard = sim.get("directional_scorecard", {})
    primary_summary = sim.get("forecast_summary_primary_top20", {})

    if not asset or not created_at_utc or not entry_ts_utc or entry_close is None:
        print("Similarity forecast missing required fields.")
        return

    rows_to_write = []

    for horizon in HORIZONS:
        horizon_key = f"{horizon}h"
        close_field = f"close_change_pct_{horizon}"
        up_field = f"max_up_pct_{horizon}"
        down_field = f"max_down_pct_{horizon}"

        dedupe_key = (asset, str(entry_ts_utc), str(horizon))
        if dedupe_key in existing_keys:
            continue

        close_summary = primary_summary.get(close_field, {})
        up_summary = primary_summary.get(up_field, {})
        down_summary = primary_summary.get(down_field, {})

        weighted_close_mean = safe_float(close_summary.get("weighted_mean"))
        weighted_close_median = safe_float(close_summary.get("weighted_median"))
        weighted_upside_mean = safe_float(up_summary.get("weighted_mean"))
        weighted_downside_mean = safe_float(down_summary.get("weighted_mean"))

        pred_pct = blended_prediction_pct(weighted_close_median, weighted_close_mean)
        pred_price = predicted_price(entry_close, pred_pct)

        row = {
            "asset": asset,
            "created_at_utc": created_at_utc,
            "created_at_local": created_at_local,
            "entry_ts_utc": str(entry_ts_utc),
            "entry_close": str(entry_close),
            "horizon_h": str(horizon),
            "target_ts_utc": target_ts_utc_from_entry(str(entry_ts_utc), horizon),
            "predicted_close_change_pct": "" if pred_pct is None else str(round(pred_pct, 4)),
            "predicted_price": "" if pred_price is None else str(round(pred_price, 4)),
            "prediction_method": PREDICTION_METHOD,
            "confidence": scorecard.get(horizon_key, {}).get("confidence", overall_confidence),
            "analogue_quality": analogue_quality(best_distance),
            "neighbors_used": str(primary_neighbors_found),
            "best_distance": "" if best_distance is None else str(round(best_distance, 6)),
            "weighted_close_mean": "" if weighted_close_mean is None else str(round(weighted_close_mean, 4)),
            "weighted_close_median": "" if weighted_close_median is None else str(round(weighted_close_median, 4)),
            "weighted_upside_mean": "" if weighted_upside_mean is None else str(round(weighted_upside_mean, 4)),
            "weighted_downside_mean": "" if weighted_downside_mean is None else str(round(weighted_downside_mean, 4)),
            "model_version": MODEL_VERSION,
            "status": "pending",
            "actual_close": "",
            "actual_close_change_pct": "",
            "error_abs": "",
            "error_pct": "",
            "hit_direction": "",
            "drift_tag": "",
            "evaluated_at_utc": "",
        }

        rows_to_write.append(row)

    if not rows_to_write:
        print("No new prediction rows to append.")
        return

    with open(PREDICTIONS_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "asset",
            "created_at_utc",
            "created_at_local",
            "entry_ts_utc",
            "entry_close",
            "horizon_h",
            "target_ts_utc",
            "predicted_close_change_pct",
            "predicted_price",
            "prediction_method",
            "confidence",
            "analogue_quality",
            "neighbors_used",
            "best_distance",
            "weighted_close_mean",
            "weighted_close_median",
            "weighted_upside_mean",
            "weighted_downside_mean",
            "model_version",
            "status",
            "actual_close",
            "actual_close_change_pct",
            "error_abs",
            "error_pct",
            "hit_direction",
            "drift_tag",
            "evaluated_at_utc",
        ])

        for row in rows_to_write:
            writer.writerow(row)

    print(f"Predictions appended to: {PREDICTIONS_PATH}")
    print(f"Rows added: {len(rows_to_write)}")
    for row in rows_to_write:
        print(
            f"h={row['horizon_h']}h | "
            f"pred_pct={row['predicted_close_change_pct']} | "
            f"pred_price={row['predicted_price']} | "
            f"confidence={row['confidence']} | "
            f"quality={row['analogue_quality']}"
        )


if __name__ == "__main__":
    main()