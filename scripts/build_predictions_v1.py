#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


# Prefer the new multi-asset forecast.
MULTI_SIMILARITY_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")

# Legacy fallback retained in case the multi file is not present.
LEGACY_SIMILARITY_PATH = os.path.join("data", "model", "similarity_forecast_v2.json")

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")

MODEL_VERSION = "similarity_forecast_v2_multi"
PREDICTION_METHOD = "blend_0p7_median_0p3_mean"

HORIZONS = [24, 48, 168, 336]
REQUIRED_ASSETS = ["ETH-USDT", "BTC-USDT"]

FIELDNAMES = [
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
    # Evaluation audit fields may already exist from evaluate_predictions_v1.py.
    "matched_target_ts_utc",
    "matched_target_ts_iso",
    "match_diff_seconds",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def target_ts_utc_from_entry(entry_ts_utc: str, horizon_h: int) -> str:
    ts = int(entry_ts_utc)
    target = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=horizon_h)
    return target.isoformat()


def analogue_quality(best_distance: Optional[float]) -> str:
    if best_distance is None:
        return "weak"
    if best_distance <= 1.75:
        return "strong"
    if best_distance <= 2.5:
        return "medium"
    return "weak"


def blended_prediction_pct(
    weighted_median: Optional[float],
    weighted_mean: Optional[float],
) -> Optional[float]:
    if weighted_median is None and weighted_mean is None:
        return None
    if weighted_median is None:
        return weighted_mean
    if weighted_mean is None:
        return weighted_median
    return 0.7 * weighted_median + 0.3 * weighted_mean


def predicted_price(entry_close: float, pred_pct: Optional[float]) -> Optional[float]:
    if pred_pct is None:
        return None
    return entry_close * (1.0 + pred_pct / 100.0)


def existing_prediction_keys(path: str) -> Set[Tuple[str, str, str]]:
    if not os.path.isfile(path):
        return set()

    keys = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            keys.add(
                (
                    r.get("asset") or "",
                    str(r.get("entry_ts_utc") or ""),
                    str(r.get("horizon_h") or ""),
                )
            )

    return keys


def existing_fieldnames(path: str) -> List[str]:
    if not os.path.isfile(path):
        return FIELDNAMES

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        found = list(reader.fieldnames or [])

    if not found:
        return FIELDNAMES

    # Preserve existing order, append any new fields.
    for field in FIELDNAMES:
        if field not in found:
            found.append(field)

    return found


def ensure_predictions_file(path: str) -> None:
    if os.path.isfile(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()


def normalize_row_for_fieldnames(row: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in fieldnames}


def load_forecasts() -> Dict[str, dict]:
    """
    Returns asset -> similarity forecast.

    Preferred source:
      data/model/similarity_forecast_v2_multi.json

    Fallback:
      data/model/similarity_forecast_v2.json
    """

    if os.path.isfile(MULTI_SIMILARITY_PATH):
        payload = load_json(MULTI_SIMILARITY_PATH)

        forecasts = payload.get("forecasts", {})
        if not isinstance(forecasts, dict) or not forecasts:
            raise ValueError(
                f"{MULTI_SIMILARITY_PATH} exists but contains no forecasts object."
            )

        assets_failed = payload.get("assets_failed", {})
        if assets_failed:
            raise ValueError(f"Similarity forecast has failed assets: {assets_failed}")

        return forecasts

    if os.path.isfile(LEGACY_SIMILARITY_PATH):
        legacy = load_json(LEGACY_SIMILARITY_PATH)
        asset = legacy.get("asset")

        if not asset:
            raise ValueError(
                f"{LEGACY_SIMILARITY_PATH} exists but does not contain an asset."
            )

        return {asset: legacy}

    raise FileNotFoundError(
        f"Missing both {MULTI_SIMILARITY_PATH} and {LEGACY_SIMILARITY_PATH}"
    )


def validate_forecast(asset: str, sim: dict) -> None:
    current_state = sim.get("current_state", {})
    entry_ts_utc = current_state.get("entry_ts_utc")
    entry_close = safe_float(current_state.get("entry_close"))
    primary_summary = sim.get("forecast_summary_primary_top20", {})
    scorecard = sim.get("directional_scorecard", {})

    if sim.get("asset") != asset:
        raise ValueError(f"Forecast key {asset} does not match payload asset {sim.get('asset')}")

    if not entry_ts_utc:
        raise ValueError(f"{asset} forecast missing current_state.entry_ts_utc")

    if entry_close is None:
        raise ValueError(f"{asset} forecast missing current_state.entry_close")

    if not isinstance(primary_summary, dict) or not primary_summary:
        raise ValueError(f"{asset} forecast missing forecast_summary_primary_top20")

    if not isinstance(scorecard, dict) or not scorecard:
        raise ValueError(f"{asset} forecast missing directional_scorecard")


def build_prediction_rows_for_asset(
    asset: str,
    sim: dict,
    existing_keys: Set[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    validate_forecast(asset, sim)

    created_at_utc = sim.get("as_of_utc") or sim.get("generated_at_utc") or utc_now()
    created_at_local = sim.get("as_of_local") or ""

    current_state = sim.get("current_state", {})
    entry_ts_utc = str(current_state.get("entry_ts_utc"))
    entry_close = safe_float(current_state.get("entry_close"))

    overall_confidence = sim.get("overall_confidence", "low")
    best_distance = safe_float(sim.get("similarity", {}).get("best_distance"))
    primary_neighbors_found = sim.get("similarity", {}).get("primary_neighbors_found", 0)

    scorecard = sim.get("directional_scorecard", {})
    primary_summary = sim.get("forecast_summary_primary_top20", {})

    if entry_close is None:
        raise ValueError(f"{asset} entry_close is missing or invalid.")

    rows_to_write = []

    for horizon in HORIZONS:
        horizon_key = f"{horizon}h"
        close_field = f"close_change_pct_{horizon}"
        up_field = f"max_up_pct_{horizon}"
        down_field = f"max_down_pct_{horizon}"

        dedupe_key = (asset, entry_ts_utc, str(horizon))
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
            "entry_ts_utc": entry_ts_utc,
            "entry_close": "" if entry_close is None else str(entry_close),
            "horizon_h": str(horizon),
            "target_ts_utc": target_ts_utc_from_entry(entry_ts_utc, horizon),
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
            "matched_target_ts_utc": "",
            "matched_target_ts_iso": "",
            "match_diff_seconds": "",
        }

        rows_to_write.append(row)

    return rows_to_write


def main() -> None:
    ensure_predictions_file(PREDICTIONS_PATH)

    existing_keys = existing_prediction_keys(PREDICTIONS_PATH)
    fieldnames = existing_fieldnames(PREDICTIONS_PATH)

    forecasts = load_forecasts()

    missing_required = [asset for asset in REQUIRED_ASSETS if asset not in forecasts]
    if missing_required:
        raise RuntimeError(
            f"Missing required forecast assets for prediction build: {missing_required}"
        )

    all_rows_to_write: List[Dict[str, Any]] = []

    for asset in REQUIRED_ASSETS:
        rows = build_prediction_rows_for_asset(
            asset=asset,
            sim=forecasts[asset],
            existing_keys=existing_keys,
        )

        all_rows_to_write.extend(rows)

        for row in rows:
            existing_keys.add((row["asset"], row["entry_ts_utc"], row["horizon_h"]))

    if not all_rows_to_write:
        print("No new prediction rows to append.")
        return

    with open(PREDICTIONS_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for row in all_rows_to_write:
            writer.writerow(normalize_row_for_fieldnames(row, fieldnames))

    print(f"Predictions appended to: {PREDICTIONS_PATH}")
    print(f"Rows added: {len(all_rows_to_write)}")

    by_asset: Dict[str, int] = {}
    for row in all_rows_to_write:
        by_asset[row["asset"]] = by_asset.get(row["asset"], 0) + 1

    print(f"Rows added by asset: {by_asset}")

    for row in all_rows_to_write:
        print(
            f"{row['asset']} | "
            f"h={row['horizon_h']}h | "
            f"pred_pct={row['predicted_close_change_pct']} | "
            f"pred_price={row['predicted_price']} | "
            f"confidence={row['confidence']} | "
            f"quality={row['analogue_quality']}"
        )


if __name__ == "__main__":
    main()
