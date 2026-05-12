#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
SIMILARITY_MULTI_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")
PERFORMANCE_SUMMARY_PATH = os.path.join("data", "model", "performance_summary_v1.json")
MODEL_READINESS_PATH = os.path.join("data", "diagnostics", "model_readiness.json")
PREDICTION_SUMMARY_PATH = os.path.join("data", "diagnostics", "prediction_summary.json")

OUT_PATH = os.path.join("data", "model", "actionability_v1.json")

ACTIVE_ASSETS = ["ETH-USDT", "BTC-USDT"]
HORIZONS = [24, 48, 168, 336]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def latest_prediction_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}

    for asset in ACTIVE_ASSETS:
        asset_rows = [r for r in rows if r.get("asset") == asset]

        if not asset_rows:
            out[asset] = []
            continue

        latest_entry_ts = max((r.get("entry_ts_utc", "") for r in asset_rows), default="")
        batch = [r for r in asset_rows if r.get("entry_ts_utc") == latest_entry_ts]
        batch.sort(key=lambda r: safe_int(r.get("horizon_h")) or 0)

        out[asset] = batch

    return out


def get_similarity_asset_block(similarity_multi: Optional[Dict[str, Any]], asset: str) -> Dict[str, Any]:
    if not similarity_multi:
        return {}

    forecasts = similarity_multi.get("forecasts", {})
    if not isinstance(forecasts, dict):
        return {}

    block = forecasts.get(asset)
    if not isinstance(block, dict):
        return {}

    return block


def get_performance_asset_horizon(
    performance_summary: Optional[Dict[str, Any]],
    asset: str,
    horizon_h: int,
) -> Dict[str, Any]:
    if not performance_summary:
        return {}

    by_asset_horizon = performance_summary.get("by_asset_horizon", {})
    key = f"{asset}|{horizon_h}"

    if isinstance(by_asset_horizon, dict) and key in by_asset_horizon:
        return by_asset_horizon.get(key, {}) or {}

    by_asset = performance_summary.get("by_asset", {})
    if isinstance(by_asset, dict):
        return by_asset.get(asset, {}) or {}

    return {}


def get_readiness_asset(
    model_readiness: Optional[Dict[str, Any]],
    asset: str,
) -> Dict[str, Any]:
    if not model_readiness:
        return {}

    assets = model_readiness.get("assets", {})
    if not isinstance(assets, dict):
        return {}

    return assets.get(asset, {}) or {}


def confidence_points(confidence: str) -> float:
    c = (confidence or "").strip().lower()

    if c == "high":
        return 2.0
    if c == "medium":
        return 1.0
    return 0.0


def analogue_quality_points(quality: str) -> float:
    q = (quality or "").strip().lower()

    if q == "strong":
        return 2.0
    if q == "medium":
        return 1.0
    return 0.0


def best_distance_points(best_distance: Optional[float]) -> float:
    if best_distance is None:
        return 0.0

    if best_distance <= 1.50:
        return 2.0
    if best_distance <= 2.25:
        return 1.0
    if best_distance <= 3.00:
        return 0.5
    return 0.0


def neighbour_points(neighbors_used: Optional[int]) -> float:
    if neighbors_used is None:
        return 0.0

    if neighbors_used >= 20:
        return 1.0
    if neighbors_used >= 10:
        return 0.5
    return 0.0


def magnitude_points(predicted_pct: Optional[float], horizon_h: int) -> float:
    if predicted_pct is None:
        return 0.0

    mag = abs(predicted_pct)

    # Short horizons should not be called actionable on tiny predicted movement.
    if horizon_h in (24, 48):
        if mag >= 1.0:
            return 1.0
        if mag >= 0.5:
            return 0.5
        return 0.0

    # Longer horizons can tolerate broader moves.
    if mag >= 2.0:
        return 1.0
    if mag >= 1.0:
        return 0.5

    return 0.0


def performance_points(
    evaluated_predictions: Optional[int],
    direction_accuracy_pct: Optional[float],
) -> float:
    if evaluated_predictions is None or evaluated_predictions < 20:
        return 0.0

    if direction_accuracy_pct is None:
        return 0.0

    if direction_accuracy_pct >= 55:
        return 1.0
    if direction_accuracy_pct >= 50:
        return 0.5
    return 0.0


def cap_actionability(
    raw_actionability: str,
    hard_blocks: List[str],
    caution_flags: List[str],
) -> str:
    if hard_blocks:
        return "NOT_ACTIONABLE"

    if raw_actionability == "ACTIONABLE" and caution_flags:
        return "CAUTION"

    return raw_actionability


def classify_score(score: float) -> str:
    if score >= 7.0:
        return "ACTIONABLE"
    if score >= 4.0:
        return "CAUTION"
    return "NOT_ACTIONABLE"


def evaluate_row(
    row: Dict[str, str],
    similarity_block: Dict[str, Any],
    performance_block: Dict[str, Any],
    readiness_block: Dict[str, Any],
) -> Dict[str, Any]:
    asset = row.get("asset") or "UNKNOWN"
    horizon_h = safe_int(row.get("horizon_h")) or 0

    predicted_pct = safe_float(row.get("predicted_close_change_pct"))
    predicted_price = safe_float(row.get("predicted_price"))
    confidence = row.get("confidence") or "low"
    analogue_quality = row.get("analogue_quality") or "weak"
    neighbors_used = safe_int(row.get("neighbors_used"))
    best_distance = safe_float(row.get("best_distance"))

    evaluated_predictions = safe_int(performance_block.get("evaluated_predictions"))
    direction_accuracy_pct = safe_float(performance_block.get("direction_accuracy_pct"))

    similarity = similarity_block.get("similarity", {}) if similarity_block else {}
    similarity_neighbors_total = safe_int(similarity.get("neighbors_found_total"))
    similarity_primary_neighbors = safe_int(similarity.get("primary_neighbors_found"))
    similarity_best_distance = safe_float(similarity.get("best_distance"))

    # Prefer row-level values, but fall back to similarity block values.
    if neighbors_used is None:
        neighbors_used = similarity_primary_neighbors

    if best_distance is None:
        best_distance = similarity_best_distance

    score_components = {
        "confidence": confidence_points(confidence),
        "analogue_quality": analogue_quality_points(analogue_quality),
        "best_distance": best_distance_points(best_distance),
        "neighbors_used": neighbour_points(neighbors_used),
        "prediction_magnitude": magnitude_points(predicted_pct, horizon_h),
        "performance_history": performance_points(evaluated_predictions, direction_accuracy_pct),
    }

    score = round(sum(score_components.values()), 2)

    hard_blocks: List[str] = []
    caution_flags: List[str] = []

    if predicted_pct is None or predicted_price is None:
        hard_blocks.append("missing_prediction_value")

    if confidence.strip().lower() == "low" and analogue_quality.strip().lower() == "weak":
        hard_blocks.append("low_confidence_and_weak_analogue_quality")

    if best_distance is not None and best_distance > 3.0:
        hard_blocks.append("best_distance_too_high")

    if neighbors_used is not None and neighbors_used < 10:
        hard_blocks.append("too_few_primary_neighbors")

    if similarity_neighbors_total is not None and similarity_neighbors_total < 30:
        hard_blocks.append("too_few_total_similarity_neighbors")

    if evaluated_predictions is None or evaluated_predictions < 20:
        caution_flags.append("limited_evaluated_prediction_history")

    if asset == "BTC-USDT" and (evaluated_predictions is None or evaluated_predictions < 20):
        caution_flags.append("btc_prediction_history_newly_active")

    if direction_accuracy_pct is not None and direction_accuracy_pct < 45 and evaluated_predictions and evaluated_predictions >= 20:
        caution_flags.append("weak_recent_directional_accuracy")

    if analogue_quality.strip().lower() == "weak":
        caution_flags.append("weak_analogue_quality")

    raw_actionability = classify_score(score)
    final_actionability = cap_actionability(raw_actionability, hard_blocks, caution_flags)

    if predicted_pct is None:
        direction = "unknown"
    elif predicted_pct > 0:
        direction = "up"
    elif predicted_pct < 0:
        direction = "down"
    else:
        direction = "flat"

    return {
        "asset": asset,
        "horizon_h": horizon_h,
        "status": row.get("status"),
        "entry_ts_utc": row.get("entry_ts_utc"),
        "target_ts_utc": row.get("target_ts_utc"),
        "predicted_close_change_pct": predicted_pct,
        "predicted_price": predicted_price,
        "direction": direction,
        "confidence": confidence,
        "analogue_quality": analogue_quality,
        "neighbors_used": neighbors_used,
        "best_distance": best_distance,
        "evaluated_predictions": evaluated_predictions,
        "direction_accuracy_pct": direction_accuracy_pct,
        "score": score,
        "score_components": score_components,
        "raw_actionability": raw_actionability,
        "actionability": final_actionability,
        "hard_blocks": hard_blocks,
        "caution_flags": caution_flags,
        "interpretation": build_interpretation(
            actionability=final_actionability,
            direction=direction,
            predicted_pct=predicted_pct,
            confidence=confidence,
            analogue_quality=analogue_quality,
            caution_flags=caution_flags,
            hard_blocks=hard_blocks,
        ),
        "readiness_context": {
            "model_rows": readiness_block.get("model_rows"),
            "prediction_rows": readiness_block.get("prediction_rows"),
            "active_prediction_asset": readiness_block.get("active_prediction_asset"),
            "horizon_readiness": (readiness_block.get("horizons") or {}).get(str(horizon_h)),
        },
    }


def build_interpretation(
    actionability: str,
    direction: str,
    predicted_pct: Optional[float],
    confidence: str,
    analogue_quality: str,
    caution_flags: List[str],
    hard_blocks: List[str],
) -> str:
    if hard_blocks:
        return (
            f"Not actionable: {', '.join(hard_blocks)}. "
            "Treat this as a model observation only."
        )

    pct_text = "unknown move" if predicted_pct is None else f"{predicted_pct:.2f}% {direction}"

    if actionability == "ACTIONABLE":
        return (
            f"Actionable model signal: predicted {pct_text}, with {confidence} confidence "
            f"and {analogue_quality} analogue quality."
        )

    if actionability == "CAUTION":
        if caution_flags:
            return (
                f"Caution signal: predicted {pct_text}, but capped by "
                f"{', '.join(caution_flags)}."
            )
        return f"Caution signal: predicted {pct_text}, but confirmation is not strong enough."

    return (
        f"Not actionable: predicted {pct_text}, but confidence/quality/history are insufficient."
    )


def build_asset_summary(asset_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not asset_rows:
        return {
            "available": False,
            "overall_posture": "NO_SIGNAL",
            "reason": "No latest predictions available.",
        }

    action_counts: Dict[str, int] = {}
    directions: Dict[str, int] = {}

    for r in asset_rows:
        action = r.get("actionability", "UNKNOWN")
        direction = r.get("direction", "unknown")

        action_counts[action] = action_counts.get(action, 0) + 1
        directions[direction] = directions.get(direction, 0) + 1

    if action_counts.get("ACTIONABLE", 0) >= 2:
        posture = "MODEL_SUPPORTIVE"
    elif action_counts.get("ACTIONABLE", 0) == 1 or action_counts.get("CAUTION", 0) >= 2:
        posture = "WATCHLIST"
    else:
        posture = "NO_ACTION"

    return {
        "available": True,
        "overall_posture": posture,
        "action_counts": action_counts,
        "direction_counts": directions,
        "strongest_horizon": strongest_horizon(asset_rows),
        "notes": asset_summary_notes(asset_rows, posture),
    }


def strongest_horizon(asset_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not asset_rows:
        return None

    ranked = sorted(
        asset_rows,
        key=lambda r: (
            1 if r.get("actionability") == "ACTIONABLE" else 0,
            r.get("score") or 0,
            abs(r.get("predicted_close_change_pct") or 0),
        ),
        reverse=True,
    )

    top = ranked[0]

    return {
        "horizon_h": top.get("horizon_h"),
        "actionability": top.get("actionability"),
        "score": top.get("score"),
        "predicted_close_change_pct": top.get("predicted_close_change_pct"),
        "confidence": top.get("confidence"),
        "analogue_quality": top.get("analogue_quality"),
    }


def asset_summary_notes(asset_rows: List[Dict[str, Any]], posture: str) -> List[str]:
    notes: List[str] = []

    if posture == "MODEL_SUPPORTIVE":
        notes.append("At least two horizons are actionable under the current calibration rules.")
    elif posture == "WATCHLIST":
        notes.append("There is some model support, but the signal should be treated cautiously.")
    else:
        notes.append("No sufficiently strong actionability signal under current calibration rules.")

    if any("btc_prediction_history_newly_active" in r.get("caution_flags", []) for r in asset_rows):
        notes.append("BTC prediction history is newly active, so actionability is capped until more evaluations mature.")

    if any(r.get("actionability") == "NOT_ACTIONABLE" for r in asset_rows):
        notes.append("One or more horizons are blocked or too weak to act on.")

    return notes


def main() -> None:
    prediction_rows = load_csv_rows(PREDICTIONS_PATH)
    similarity_multi = load_json(SIMILARITY_MULTI_PATH)
    performance_summary = load_json(PERFORMANCE_SUMMARY_PATH)
    model_readiness = load_json(MODEL_READINESS_PATH)
    prediction_summary = load_json(PREDICTION_SUMMARY_PATH)

    latest_by_asset = latest_prediction_rows(prediction_rows)

    by_asset: Dict[str, Any] = {}

    for asset in ACTIVE_ASSETS:
        similarity_block = get_similarity_asset_block(similarity_multi, asset)
        readiness_block = get_readiness_asset(model_readiness, asset)
        asset_latest_rows = latest_by_asset.get(asset, [])

        evaluated_rows = []

        for row in asset_latest_rows:
            horizon_h = safe_int(row.get("horizon_h")) or 0
            performance_block = get_performance_asset_horizon(performance_summary, asset, horizon_h)

            evaluated_rows.append(
                evaluate_row(
                    row=row,
                    similarity_block=similarity_block,
                    performance_block=performance_block,
                    readiness_block=readiness_block,
                )
            )

        by_asset[asset] = {
            "available": bool(evaluated_rows),
            "latest_entry_ts_utc": asset_latest_rows[0].get("entry_ts_utc") if asset_latest_rows else None,
            "summary": build_asset_summary(evaluated_rows),
            "horizons": evaluated_rows,
        }

    payload = {
        "schema": "actionability_v1",
        "generated_at_utc": utc_now(),
        "source_files": {
            "predictions_v1": PREDICTIONS_PATH if os.path.isfile(PREDICTIONS_PATH) else None,
            "similarity_forecast_v2_multi": SIMILARITY_MULTI_PATH if os.path.isfile(SIMILARITY_MULTI_PATH) else None,
            "performance_summary_v1": PERFORMANCE_SUMMARY_PATH if os.path.isfile(PERFORMANCE_SUMMARY_PATH) else None,
            "model_readiness": MODEL_READINESS_PATH if os.path.isfile(MODEL_READINESS_PATH) else None,
            "prediction_summary": PREDICTION_SUMMARY_PATH if os.path.isfile(PREDICTION_SUMMARY_PATH) else None,
        },
        "active_assets": ACTIVE_ASSETS,
        "horizons": HORIZONS,
        "method": {
            "labels": ["ACTIONABLE", "CAUTION", "NOT_ACTIONABLE"],
            "summary": (
                "Actionability is a calibration layer. It does not change raw predictions. "
                "It classifies whether the latest prediction is usable based on confidence, "
                "analogue quality, distance, neighbour count, prediction magnitude, and evaluated performance history."
            ),
            "btc_note": (
                "BTC is newly active as a prediction asset. Until enough BTC predictions mature, "
                "BTC signals are capped by a caution flag."
            ),
        },
        "by_asset": by_asset,
    }

    write_json(OUT_PATH, payload)

    print(f"Wrote actionability file: {OUT_PATH}")

    for asset, block in by_asset.items():
        summary = block.get("summary", {})
        print(
            f"{asset}: posture={summary.get('overall_posture')} "
            f"actions={summary.get('action_counts')}"
        )


if __name__ == "__main__":
    main()
