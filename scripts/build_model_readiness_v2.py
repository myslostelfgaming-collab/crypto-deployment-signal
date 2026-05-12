#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


MODEL_DATASET_PATH = os.path.join("data", "model", "model_dataset_v1.csv")
PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
PERFORMANCE_SUMMARY_PATH = os.path.join("data", "model", "performance_summary_v1.json")
SIMILARITY_MULTI_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")
ACTIONABILITY_PATH = os.path.join("data", "model", "actionability_v1.json")
PREDICTION_SUMMARY_PATH = os.path.join("data", "diagnostics", "prediction_summary.json")

OUT_PATH = os.path.join("data", "diagnostics", "model_readiness_v2.json")

ACTIVE_ASSETS = ["ETH-USDT", "BTC-USDT"]
CONTEXT_ASSETS = ["ETH-BTC"]
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


def count_model_rows_by_asset_and_horizon(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for row in rows:
        asset = row.get("asset") or "UNKNOWN"

        out.setdefault(
            asset,
            {
                "model_rows": 0,
                "label_rows_by_horizon": {str(h): 0 for h in HORIZONS},
            },
        )

        out[asset]["model_rows"] += 1

        for h in HORIZONS:
            field = f"close_change_pct_{h}"
            if row.get(field) not in ("", None):
                out[asset]["label_rows_by_horizon"][str(h)] += 1

    return out


def prediction_counts(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for row in rows:
        asset = row.get("asset") or "UNKNOWN"
        horizon = str(row.get("horizon_h") or "UNKNOWN")
        status = (row.get("status") or "UNKNOWN").lower()

        out.setdefault(
            asset,
            {
                "total": 0,
                "pending": 0,
                "evaluated": 0,
                "by_horizon": {
                    str(h): {
                        "total": 0,
                        "pending": 0,
                        "evaluated": 0,
                    }
                    for h in HORIZONS
                },
                "latest_entry_ts_utc": None,
            },
        )

        out[asset]["total"] += 1

        if status == "evaluated":
            out[asset]["evaluated"] += 1
        else:
            out[asset]["pending"] += 1

        if horizon in out[asset]["by_horizon"]:
            out[asset]["by_horizon"][horizon]["total"] += 1
            if status == "evaluated":
                out[asset]["by_horizon"][horizon]["evaluated"] += 1
            else:
                out[asset]["by_horizon"][horizon]["pending"] += 1

        entry_ts = row.get("entry_ts_utc")
        if entry_ts:
            out[asset]["latest_entry_ts_utc"] = entry_ts

    return out


def latest_prediction_batch(rows: List[Dict[str, str]], asset: str) -> List[Dict[str, str]]:
    asset_rows = [r for r in rows if r.get("asset") == asset]

    if not asset_rows:
        return []

    latest_entry_ts = max((r.get("entry_ts_utc", "") for r in asset_rows), default="")
    batch = [r for r in asset_rows if r.get("entry_ts_utc") == latest_entry_ts]
    batch.sort(key=lambda r: safe_int(r.get("horizon_h")) or 0)

    return batch


def compact_latest_prediction(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "horizon_h": safe_int(row.get("horizon_h")),
        "entry_ts_utc": row.get("entry_ts_utc"),
        "target_ts_utc": row.get("target_ts_utc"),
        "predicted_close_change_pct": safe_float(row.get("predicted_close_change_pct")),
        "predicted_price": safe_float(row.get("predicted_price")),
        "confidence": row.get("confidence"),
        "analogue_quality": row.get("analogue_quality"),
        "neighbors_used": safe_int(row.get("neighbors_used")),
        "best_distance": safe_float(row.get("best_distance")),
        "status": row.get("status"),
    }


def get_performance_block(performance: Optional[Dict[str, Any]], asset: str, horizon_h: int) -> Dict[str, Any]:
    if not performance:
        return {}

    by_asset_horizon = performance.get("by_asset_horizon", {})
    key = f"{asset}|{horizon_h}"

    if isinstance(by_asset_horizon, dict) and key in by_asset_horizon:
        return by_asset_horizon.get(key, {}) or {}

    return {}


def get_similarity_block(similarity: Optional[Dict[str, Any]], asset: str) -> Dict[str, Any]:
    if not similarity:
        return {}

    forecasts = similarity.get("forecasts", {})
    if not isinstance(forecasts, dict):
        return {}

    return forecasts.get(asset, {}) or {}


def get_actionability_horizon(actionability: Optional[Dict[str, Any]], asset: str, horizon_h: int) -> Dict[str, Any]:
    if not actionability:
        return {}

    asset_block = (actionability.get("by_asset") or {}).get(asset, {})
    horizons = asset_block.get("horizons", [])

    for h in horizons:
        if safe_int(h.get("horizon_h")) == horizon_h:
            return h

    return {}


def row_score(
    label_rows: int,
    prediction_rows: int,
    evaluated_rows: int,
    accuracy_pct: Optional[float],
    best_distance: Optional[float],
    neighbors_used: Optional[int],
    actionability_label: Optional[str],
) -> Dict[str, Any]:
    components: Dict[str, float] = {}

    # Historical labelled data
    if label_rows >= 300:
        components["label_depth"] = 2.0
    elif label_rows >= 100:
        components["label_depth"] = 1.5
    elif label_rows >= 30:
        components["label_depth"] = 1.0
    elif label_rows > 0:
        components["label_depth"] = 0.5
    else:
        components["label_depth"] = 0.0

    # Prediction history
    if prediction_rows >= 100:
        components["prediction_history"] = 2.0
    elif prediction_rows >= 40:
        components["prediction_history"] = 1.5
    elif prediction_rows >= 10:
        components["prediction_history"] = 1.0
    elif prediction_rows > 0:
        components["prediction_history"] = 0.5
    else:
        components["prediction_history"] = 0.0

    # Evaluation maturity
    if evaluated_rows >= 50:
        components["evaluation_maturity"] = 2.0
    elif evaluated_rows >= 20:
        components["evaluation_maturity"] = 1.5
    elif evaluated_rows >= 5:
        components["evaluation_maturity"] = 1.0
    elif evaluated_rows > 0:
        components["evaluation_maturity"] = 0.5
    else:
        components["evaluation_maturity"] = 0.0

    # Observed performance
    if accuracy_pct is None:
        components["performance"] = 0.0
    elif accuracy_pct >= 55:
        components["performance"] = 2.0
    elif accuracy_pct >= 50:
        components["performance"] = 1.0
    elif accuracy_pct >= 45:
        components["performance"] = 0.5
    else:
        components["performance"] = 0.0

    # Similarity quality
    if best_distance is None:
        components["similarity_quality"] = 0.0
    elif best_distance <= 1.50:
        components["similarity_quality"] = 2.0
    elif best_distance <= 2.25:
        components["similarity_quality"] = 1.5
    elif best_distance <= 3.00:
        components["similarity_quality"] = 0.75
    else:
        components["similarity_quality"] = 0.0

    # Neighbour support
    if neighbors_used is None:
        components["neighbor_support"] = 0.0
    elif neighbors_used >= 20:
        components["neighbor_support"] = 1.0
    elif neighbors_used >= 10:
        components["neighbor_support"] = 0.5
    else:
        components["neighbor_support"] = 0.0

    # Actionability overlay
    if actionability_label == "ACTIONABLE":
        components["actionability"] = 1.0
    elif actionability_label == "CAUTION":
        components["actionability"] = 0.5
    else:
        components["actionability"] = 0.0

    total = round(sum(components.values()), 2)
    return {"score": total, "components": components}


def readiness_label(score: float, hard_blocks: List[str], caution_flags: List[str]) -> str:
    if hard_blocks:
        return "NOT_READY"

    if score >= 8.0 and not caution_flags:
        return "READY"

    if score >= 5.5:
        return "LIMITED"

    if score >= 3.0:
        return "EARLY"

    return "NOT_READY"


def build_horizon_readiness(
    asset: str,
    horizon_h: int,
    model_cov: Dict[str, Any],
    pred_cov: Dict[str, Any],
    latest_prediction: Optional[Dict[str, str]],
    similarity_block: Dict[str, Any],
    performance_block: Dict[str, Any],
    actionability_horizon: Dict[str, Any],
) -> Dict[str, Any]:
    label_rows = safe_int((model_cov.get("label_rows_by_horizon") or {}).get(str(horizon_h))) or 0

    pred_h = (pred_cov.get("by_horizon") or {}).get(str(horizon_h), {})
    prediction_rows = safe_int(pred_h.get("total")) or 0
    evaluated_rows = safe_int(pred_h.get("evaluated")) or 0
    pending_rows = safe_int(pred_h.get("pending")) or 0

    accuracy_pct = safe_float(performance_block.get("direction_accuracy_pct"))

    sim = similarity_block.get("similarity", {}) if similarity_block else {}
    best_distance = None
    neighbors_used = None

    if latest_prediction:
        best_distance = safe_float(latest_prediction.get("best_distance"))
        neighbors_used = safe_int(latest_prediction.get("neighbors_used"))

    if best_distance is None:
        best_distance = safe_float(sim.get("best_distance"))

    if neighbors_used is None:
        neighbors_used = safe_int(sim.get("primary_neighbors_found"))

    actionability_label = actionability_horizon.get("actionability")

    score_result = row_score(
        label_rows=label_rows,
        prediction_rows=prediction_rows,
        evaluated_rows=evaluated_rows,
        accuracy_pct=accuracy_pct,
        best_distance=best_distance,
        neighbors_used=neighbors_used,
        actionability_label=actionability_label,
    )

    hard_blocks: List[str] = []
    caution_flags: List[str] = []

    if label_rows == 0:
        hard_blocks.append("no_matured_labels")

    if prediction_rows == 0:
        hard_blocks.append("no_prediction_rows")

    if best_distance is not None and best_distance > 3.0:
        hard_blocks.append("similarity_distance_too_high")

    if neighbors_used is not None and neighbors_used < 10:
        hard_blocks.append("too_few_primary_neighbors")

    if evaluated_rows < 20:
        caution_flags.append("limited_evaluated_prediction_history")

    if asset == "BTC-USDT" and evaluated_rows < 20:
        caution_flags.append("btc_prediction_history_newly_active")

    if accuracy_pct is not None and accuracy_pct < 45 and evaluated_rows >= 20:
        caution_flags.append("weak_directional_accuracy")

    label = readiness_label(score_result["score"], hard_blocks, caution_flags)

    latest = compact_latest_prediction(latest_prediction) if latest_prediction else None

    return {
        "horizon_h": horizon_h,
        "readiness": label,
        "score": score_result["score"],
        "score_components": score_result["components"],
        "label_rows": label_rows,
        "prediction_rows": prediction_rows,
        "evaluated_prediction_rows": evaluated_rows,
        "pending_prediction_rows": pending_rows,
        "direction_accuracy_pct": accuracy_pct,
        "best_distance": best_distance,
        "neighbors_used": neighbors_used,
        "actionability": actionability_label,
        "hard_blocks": hard_blocks,
        "caution_flags": caution_flags,
        "latest_prediction": latest,
    }


def summarise_asset(horizons: Dict[str, Any], asset: str) -> Dict[str, Any]:
    labels = [h.get("readiness") for h in horizons.values()]
    scores = [safe_float(h.get("score")) for h in horizons.values()]
    scores = [s for s in scores if s is not None]

    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    if counts.get("READY", 0) >= 2:
        overall = "READY"
    elif counts.get("READY", 0) == 1 or counts.get("LIMITED", 0) >= 2:
        overall = "LIMITED"
    elif counts.get("LIMITED", 0) == 1 or counts.get("EARLY", 0) >= 2:
        overall = "EARLY"
    else:
        overall = "NOT_READY"

    notes: List[str] = []

    if asset == "BTC-USDT":
        if any("btc_prediction_history_newly_active" in h.get("caution_flags", []) for h in horizons.values()):
            notes.append("BTC prediction history is newly active; readiness is capped until more BTC predictions mature.")

    if any("weak_directional_accuracy" in h.get("caution_flags", []) for h in horizons.values()):
        notes.append("One or more horizons have weak recent directional accuracy.")

    if any(h.get("hard_blocks") for h in horizons.values()):
        notes.append("One or more horizons have hard readiness blockers.")

    if not notes:
        notes.append("No major readiness blockers detected under current rules.")

    return {
        "overall_readiness": overall,
        "readiness_counts": counts,
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "notes": notes,
    }


def main() -> None:
    model_rows = load_csv_rows(MODEL_DATASET_PATH)
    prediction_rows = load_csv_rows(PREDICTIONS_PATH)

    performance_summary = load_json(PERFORMANCE_SUMMARY_PATH)
    similarity_multi = load_json(SIMILARITY_MULTI_PATH)
    actionability = load_json(ACTIONABILITY_PATH)
    prediction_summary = load_json(PREDICTION_SUMMARY_PATH)

    model_cov = count_model_rows_by_asset_and_horizon(model_rows)
    pred_cov = prediction_counts(prediction_rows)

    by_asset: Dict[str, Any] = {}

    for asset in ACTIVE_ASSETS:
        asset_model_cov = model_cov.get(
            asset,
            {
                "model_rows": 0,
                "label_rows_by_horizon": {str(h): 0 for h in HORIZONS},
            },
        )

        asset_pred_cov = pred_cov.get(
            asset,
            {
                "total": 0,
                "pending": 0,
                "evaluated": 0,
                "by_horizon": {
                    str(h): {
                        "total": 0,
                        "pending": 0,
                        "evaluated": 0,
                    }
                    for h in HORIZONS
                },
                "latest_entry_ts_utc": None,
            },
        )

        similarity_block = get_similarity_block(similarity_multi, asset)
        latest_batch = latest_prediction_batch(prediction_rows, asset)
        latest_by_horizon = {
            safe_int(row.get("horizon_h")): row
            for row in latest_batch
            if safe_int(row.get("horizon_h")) is not None
        }

        horizons: Dict[str, Any] = {}

        for h in HORIZONS:
            perf = get_performance_block(performance_summary, asset, h)
            action_h = get_actionability_horizon(actionability, asset, h)

            horizons[str(h)] = build_horizon_readiness(
                asset=asset,
                horizon_h=h,
                model_cov=asset_model_cov,
                pred_cov=asset_pred_cov,
                latest_prediction=latest_by_horizon.get(h),
                similarity_block=similarity_block,
                performance_block=perf,
                actionability_horizon=action_h,
            )

        by_asset[asset] = {
            "asset": asset,
            "active_prediction_asset": asset_pred_cov.get("total", 0) > 0,
            "model_rows": asset_model_cov.get("model_rows", 0),
            "prediction_rows_total": asset_pred_cov.get("total", 0),
            "prediction_rows_pending": asset_pred_cov.get("pending", 0),
            "prediction_rows_evaluated": asset_pred_cov.get("evaluated", 0),
            "latest_entry_ts_utc": asset_pred_cov.get("latest_entry_ts_utc"),
            "summary": summarise_asset(horizons, asset),
            "horizons": horizons,
        }

    payload = {
        "schema": "model_readiness_v2",
        "generated_at_utc": utc_now(),
        "source_files": {
            "model_dataset_v1": MODEL_DATASET_PATH if os.path.isfile(MODEL_DATASET_PATH) else None,
            "predictions_v1": PREDICTIONS_PATH if os.path.isfile(PREDICTIONS_PATH) else None,
            "performance_summary_v1": PERFORMANCE_SUMMARY_PATH if os.path.isfile(PERFORMANCE_SUMMARY_PATH) else None,
            "similarity_forecast_v2_multi": SIMILARITY_MULTI_PATH if os.path.isfile(SIMILARITY_MULTI_PATH) else None,
            "actionability_v1": ACTIONABILITY_PATH if os.path.isfile(ACTIONABILITY_PATH) else None,
            "prediction_summary": PREDICTION_SUMMARY_PATH if os.path.isfile(PREDICTION_SUMMARY_PATH) else None,
        },
        "active_assets": ACTIVE_ASSETS,
        "context_assets": CONTEXT_ASSETS,
        "horizons": HORIZONS,
        "method": {
            "readiness_labels": ["READY", "LIMITED", "EARLY", "NOT_READY"],
            "summary": (
                "Readiness v2 combines historical label depth, prediction history, "
                "evaluation maturity, directional accuracy, similarity quality, neighbour support, "
                "and actionability labels. It is stricter than the legacy row-count readiness file."
            ),
            "btc_note": (
                "BTC-USDT is newly active as a prediction asset, so readiness remains cautious until "
                "more BTC prediction rows have matured and been evaluated."
            ),
        },
        "prediction_summary_brief": {
            "available": bool(prediction_summary),
            "total_rows": (prediction_summary or {}).get("total_rows"),
            "by_asset": (prediction_summary or {}).get("by_asset"),
            "by_status": (prediction_summary or {}).get("by_status"),
        },
        "by_asset": by_asset,
    }

    write_json(OUT_PATH, payload)

    print(f"Wrote: {OUT_PATH}")

    for asset, block in by_asset.items():
        summary = block.get("summary", {})
        print(
            f"{asset}: readiness={summary.get('overall_readiness')} "
            f"mean_score={summary.get('mean_score')} "
            f"counts={summary.get('readiness_counts')}"
        )


if __name__ == "__main__":
    main()
