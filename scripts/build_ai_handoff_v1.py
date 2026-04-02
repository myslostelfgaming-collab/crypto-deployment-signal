#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

HOURLY_REPORT_PATH = os.path.join("data", "hourly_report.json")
SIMILARITY_V2_PATH = os.path.join("data", "model", "similarity_forecast_v2.json")
PREDICTIONS_V1_PATH = os.path.join("data", "model", "predictions_v1.csv")
OUT_PATH = os.path.join("data", "ai_handoff_v1.json")


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


def compact_signal(hourly_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not hourly_report:
        return {"available": False}

    signal = hourly_report.get("signal", {})
    return {
        "available": True,
        "date": signal.get("date"),
        "timezone": signal.get("timezone"),
        "published_at_utc": signal.get("published_at_utc"),
        "published_at_local": signal.get("published_at_local"),
        "eth_usdt": signal.get("eth_usdt"),
        "btc_usdt": signal.get("btc_usdt"),
        "eth_btc": signal.get("eth_btc"),
        "integrity": signal.get("integrity"),
    }


def compact_baseline_forecast(hourly_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not hourly_report:
        return {"available": False}

    forecast = hourly_report.get("forecast")
    if not forecast:
        return {"available": False}

    targets = forecast.get("targets", [])
    keep_horizons = {24, 48, 168, 336}

    selected = []
    for t in targets:
        h = safe_int(t.get("horizon_h"))
        if h in keep_horizons:
            selected.append(t)

    return {
        "available": True,
        "schema": forecast.get("schema"),
        "as_of_utc": forecast.get("as_of_utc"),
        "source": forecast.get("source"),
        "selected_targets": selected,
        "notes": forecast.get("notes", []),
    }


def compact_similarity_forecast(sim_v2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not sim_v2:
        return {"available": False}

    return {
        "available": True,
        "schema": sim_v2.get("schema"),
        "as_of_utc": sim_v2.get("as_of_utc"),
        "as_of_local": sim_v2.get("as_of_local"),
        "asset": sim_v2.get("asset"),
        "settings": sim_v2.get("settings"),
        "current_state": sim_v2.get("current_state"),
        "similarity": sim_v2.get("similarity"),
        "directional_scorecard": sim_v2.get("directional_scorecard"),
        "overall_confidence": sim_v2.get("overall_confidence"),
        "forecast_summary_primary_top20": sim_v2.get("forecast_summary_primary_top20"),
        "notes": sim_v2.get("notes", []),
    }


def latest_prediction_batch(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}

    latest_created = max((r.get("created_at_utc", "") for r in rows), default="")
    batch = [r for r in rows if r.get("created_at_utc") == latest_created]
    batch.sort(key=lambda r: safe_int(r.get("horizon_h")) or 0)

    compact_rows = []
    for r in batch:
        compact_rows.append(
            {
                "asset": r.get("asset"),
                "created_at_utc": r.get("created_at_utc"),
                "entry_ts_utc": r.get("entry_ts_utc"),
                "entry_close": safe_float(r.get("entry_close")),
                "horizon_h": safe_int(r.get("horizon_h")),
                "target_ts_utc": r.get("target_ts_utc"),
                "predicted_close_change_pct": safe_float(r.get("predicted_close_change_pct")),
                "predicted_price": safe_float(r.get("predicted_price")),
                "prediction_method": r.get("prediction_method"),
                "confidence": r.get("confidence"),
                "analogue_quality": r.get("analogue_quality"),
                "neighbors_used": safe_int(r.get("neighbors_used")),
                "best_distance": safe_float(r.get("best_distance")),
                "status": r.get("status"),
                "actual_close": safe_float(r.get("actual_close")),
                "actual_close_change_pct": safe_float(r.get("actual_close_change_pct")),
                "error_abs": safe_float(r.get("error_abs")),
                "error_pct": safe_float(r.get("error_pct")),
                "hit_direction": r.get("hit_direction"),
                "drift_tag": r.get("drift_tag"),
                "evaluated_at_utc": r.get("evaluated_at_utc"),
            }
        )

    return {
        "available": True,
        "latest_created_at_utc": latest_created,
        "rows": compact_rows,
    }


def prediction_evaluation_summary(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}

    evaluated = [r for r in rows if (r.get("status") or "").lower() != "pending" and r.get("actual_close") not in ("", None)]
    pending = [r for r in rows if (r.get("status") or "").lower() == "pending" or r.get("actual_close") in ("", None)]

    summary: Dict[str, Any] = {
        "available": True,
        "n_total": len(rows),
        "n_evaluated": len(evaluated),
        "n_pending": len(pending),
        "by_horizon": {},
    }

    horizons = sorted({safe_int(r.get("horizon_h")) for r in rows if safe_int(r.get("horizon_h")) is not None})

    for h in horizons:
        h_rows = [r for r in evaluated if safe_int(r.get("horizon_h")) == h]
        if not h_rows:
            summary["by_horizon"][str(h)] = {
                "n_evaluated": 0
            }
            continue

        err_abs = [safe_float(r.get("error_abs")) for r in h_rows]
        err_pct = [safe_float(r.get("error_pct")) for r in h_rows]

        err_abs = [x for x in err_abs if x is not None]
        err_pct = [x for x in err_pct if x is not None]

        summary["by_horizon"][str(h)] = {
            "n_evaluated": len(h_rows),
            "mean_error_abs": round(sum(err_abs) / len(err_abs), 4) if err_abs else None,
            "mean_error_pct": round(sum(err_pct) / len(err_pct), 4) if err_pct else None,
        }

    return summary


def build_interpretation_instructions() -> Dict[str, Any]:
    return {
        "purpose": "This file is a single pasteable AI handoff for interpreting the latest crypto deployment signal in plain English.",
        "audience": "A non-technical user making cautious crypto deployment decisions.",
        "task": [
            "Read the file as a market interpretation artifact.",
            "Explain the latest ETH market state first.",
            "Then explain what the forecast suggests for 24h, 48h, 7d, and 14d.",
            "Then explain what was actually predicted and how confident the system is.",
            "If prediction evaluations are available, explain whether the model has recently been accurate or drifting."
        ],
        "required_output_format": [
            "1. Plain-English market summary",
            "2. 24h outlook",
            "3. 48h outlook",
            "4. 7d outlook",
            "5. 14d outlook",
            "6. Main upside case",
            "7. Main downside risk",
            "8. Confidence assessment",
            "9. Deployment posture for a cautious user",
            "10. Final verdict in 1-2 sentences"
        ],
        "rules": [
            "Use plain English.",
            "Do not assume the reader understands quant or trading jargon.",
            "Separate observed facts from model inference.",
            "If confidence is low, say so clearly.",
            "Do not pretend the model is certain.",
            "Treat 24h and 48h as tactical horizons, and 7d and 14d as broader directional horizons.",
            "If forecasts conflict across horizons, explicitly explain the conflict.",
            "If recent evaluated predictions are unavailable, say that the live tracking layer is not yet mature."
        ]
    }


def main():
    hourly_report = load_json(HOURLY_REPORT_PATH)
    sim_v2 = load_json(SIMILARITY_V2_PATH)
    pred_rows = load_csv_rows(PREDICTIONS_V1_PATH)

    payload = {
        "schema": "ai_handoff_v1",
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "hourly_report": HOURLY_REPORT_PATH if os.path.isfile(HOURLY_REPORT_PATH) else None,
                "similarity_forecast_v2": SIMILARITY_V2_PATH if os.path.isfile(SIMILARITY_V2_PATH) else None,
                "predictions_v1": PREDICTIONS_V1_PATH if os.path.isfile(PREDICTIONS_V1_PATH) else None,
            },
        },
        "market_state": compact_signal(hourly_report),
        "baseline_forecast_state": compact_baseline_forecast(hourly_report),
        "similarity_forecast_state": compact_similarity_forecast(sim_v2),
        "latest_prediction_state": latest_prediction_batch(pred_rows),
        "prediction_evaluation_state": prediction_evaluation_summary(pred_rows),
        "interpretation_instructions": build_interpretation_instructions(),
    }

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"AI handoff written to: {OUT_PATH}")


if __name__ == "__main__":
    main()