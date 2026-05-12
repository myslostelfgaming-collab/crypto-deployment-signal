#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


LOCAL_TZ = "Africa/Johannesburg"

SIGNAL_PATH = os.path.join("data", "hourly_signal.json")
BASELINE_SUMMARY_PATH = os.path.join("data", "forecast", "baseline_summary_v1.json")
FORECAST_FALLBACK_PATH = os.path.join("data", "hourly_forecast.json")

SIMILARITY_MULTI_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")
SIMILARITY_LEGACY_PATH = os.path.join("data", "model", "similarity_forecast_v2.json")
PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
PERFORMANCE_SUMMARY_PATH = os.path.join("data", "model", "performance_summary_v1.json")

REPO_STATUS_PATH = os.path.join("data", "diagnostics", "repo_status.json")
MODEL_READINESS_PATH = os.path.join("data", "diagnostics", "model_readiness.json")
PREDICTION_SUMMARY_PATH = os.path.join("data", "diagnostics", "prediction_summary.json")

OUT_PATH = os.path.join("data", "hourly_report.json")

HORIZONS = [24, 48, 168, 336]
ACTIVE_PREDICTION_ASSETS = ["ETH-USDT", "BTC-USDT"]
CONTEXT_ASSETS = ["ETH-BTC"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Optional[dict]:
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


def compact_prediction_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "asset": row.get("asset"),
        "created_at_utc": row.get("created_at_utc"),
        "created_at_local": row.get("created_at_local"),
        "entry_ts_utc": row.get("entry_ts_utc"),
        "entry_close": safe_float(row.get("entry_close")),
        "horizon_h": safe_int(row.get("horizon_h")),
        "target_ts_utc": row.get("target_ts_utc"),
        "predicted_close_change_pct": safe_float(row.get("predicted_close_change_pct")),
        "predicted_price": safe_float(row.get("predicted_price")),
        "prediction_method": row.get("prediction_method"),
        "confidence": row.get("confidence"),
        "analogue_quality": row.get("analogue_quality"),
        "neighbors_used": safe_int(row.get("neighbors_used")),
        "best_distance": safe_float(row.get("best_distance")),
        "weighted_close_mean": safe_float(row.get("weighted_close_mean")),
        "weighted_close_median": safe_float(row.get("weighted_close_median")),
        "weighted_upside_mean": safe_float(row.get("weighted_upside_mean")),
        "weighted_downside_mean": safe_float(row.get("weighted_downside_mean")),
        "model_version": row.get("model_version"),
        "status": row.get("status"),
        "actual_close": safe_float(row.get("actual_close")),
        "actual_close_change_pct": safe_float(row.get("actual_close_change_pct")),
        "error_abs": safe_float(row.get("error_abs")),
        "error_pct": safe_float(row.get("error_pct")),
        "hit_direction": row.get("hit_direction"),
        "drift_tag": row.get("drift_tag"),
        "evaluated_at_utc": row.get("evaluated_at_utc"),
    }


def flatten_baseline_summary(summary: dict, as_of_utc: str) -> dict:
    thresholds = summary.get("thresholds", [])
    horizons = summary.get("horizons", [])
    results = summary.get("results", {})
    n_labels = summary.get("n_rows_labels")
    labels_path = summary.get("source_labels")

    targets: List[Dict[str, Any]] = []

    for thr_key, by_h in results.items():
        thr = safe_float(thr_key)
        if thr is None or not isinstance(by_h, dict):
            continue

        for h_key, row in by_h.items():
            h = safe_int(h_key)
            if h is None or not isinstance(row, dict):
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

    targets.sort(key=lambda x: (x.get("target_pct", 0.0), x.get("horizon_h", 0)))

    return {
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
            "Baseline probabilities are empirical from historical label outcomes.",
            "Similarity predictions are reported separately under prediction_state and similarity_forecast_state.",
        ],
    }


def resolve_report_timestamps(signal: dict) -> Tuple[str, str, str]:
    tz = signal.get("timezone", LOCAL_TZ)
    published_at_utc = signal.get("published_at_utc")
    published_at_local = signal.get("published_at_local")

    if not published_at_utc or not published_at_local:
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        published_at_utc = published_at_utc or now_utc
        published_at_local = published_at_local or now_utc

    return tz, published_at_utc, published_at_local


def load_similarity_forecasts() -> Dict[str, Any]:
    multi = load_json(SIMILARITY_MULTI_PATH)

    if multi and isinstance(multi.get("forecasts"), dict):
        return {
            "available": True,
            "schema": multi.get("schema"),
            "generated_at_utc": multi.get("generated_at_utc"),
            "as_of_utc": multi.get("as_of_utc"),
            "as_of_local": multi.get("as_of_local"),
            "assets_requested": multi.get("assets_requested", []),
            "assets_generated": multi.get("assets_generated", []),
            "assets_failed": multi.get("assets_failed", {}),
            "legacy_asset": multi.get("legacy_asset"),
            "forecast_files": multi.get("forecast_files", {}),
            "by_asset": {
                asset: compact_similarity_forecast(forecast)
                for asset, forecast in multi.get("forecasts", {}).items()
            },
        }

    legacy = load_json(SIMILARITY_LEGACY_PATH)
    if legacy:
        asset = legacy.get("asset", "UNKNOWN")
        return {
            "available": True,
            "schema": "similarity_forecast_v2_legacy_only",
            "assets_generated": [asset],
            "assets_failed": {},
            "by_asset": {
                asset: compact_similarity_forecast(legacy)
            },
            "warning": "Only legacy similarity_forecast_v2.json was available.",
        }

    return {"available": False}


def compact_similarity_forecast(forecast: dict) -> Dict[str, Any]:
    return {
        "available": True,
        "schema": forecast.get("schema"),
        "generated_at_utc": forecast.get("generated_at_utc"),
        "as_of_utc": forecast.get("as_of_utc"),
        "as_of_local": forecast.get("as_of_local"),
        "asset": forecast.get("asset"),
        "candle_key": forecast.get("candle_key"),
        "settings": forecast.get("settings"),
        "current_state": forecast.get("current_state"),
        "model_dataset": forecast.get("model_dataset"),
        "similarity": forecast.get("similarity"),
        "directional_scorecard": forecast.get("directional_scorecard"),
        "overall_confidence": forecast.get("overall_confidence"),
        "notes": forecast.get("notes", []),
    }


def latest_predictions_by_asset(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": bool(rows),
        "assets": {},
    }

    if not rows:
        return out

    for asset in ACTIVE_PREDICTION_ASSETS:
        asset_rows = [r for r in rows if r.get("asset") == asset]

        if not asset_rows:
            out["assets"][asset] = {
                "available": False,
                "rows": [],
            }
            continue

        latest_entry_ts = max((r.get("entry_ts_utc", "") for r in asset_rows), default="")
        latest_rows = [r for r in asset_rows if r.get("entry_ts_utc") == latest_entry_ts]
        latest_rows.sort(key=lambda r: safe_int(r.get("horizon_h")) or 0)

        out["assets"][asset] = {
            "available": True,
            "latest_entry_ts_utc": latest_entry_ts,
            "latest_created_at_utc": max((r.get("created_at_utc", "") for r in latest_rows), default=""),
            "rows": [compact_prediction_row(r) for r in latest_rows],
        }

    return out


def prediction_evaluation_summary(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}

    out: Dict[str, Any] = {
        "available": True,
        "n_total": len(rows),
        "by_asset": {},
    }

    for asset in ACTIVE_PREDICTION_ASSETS:
        asset_rows = [r for r in rows if r.get("asset") == asset]
        evaluated = [
            r for r in asset_rows
            if (r.get("status") or "").lower() == "evaluated"
            and r.get("actual_close") not in ("", None)
        ]
        pending = [
            r for r in asset_rows
            if (r.get("status") or "").lower() != "evaluated"
            or r.get("actual_close") in ("", None)
        ]

        asset_summary: Dict[str, Any] = {
            "n_total": len(asset_rows),
            "n_evaluated": len(evaluated),
            "n_pending": len(pending),
            "by_horizon": {},
        }

        for h in HORIZONS:
            h_rows = [r for r in evaluated if safe_int(r.get("horizon_h")) == h]
            if not h_rows:
                asset_summary["by_horizon"][str(h)] = {"n_evaluated": 0}
                continue

            err_abs = [safe_float(r.get("error_abs")) for r in h_rows]
            err_pct = [safe_float(r.get("error_pct")) for r in h_rows]
            err_abs = [x for x in err_abs if x is not None]
            err_pct = [x for x in err_pct if x is not None]

            correct = sum(
                1 for r in h_rows
                if "correct" in (r.get("hit_direction") or "").lower()
                and "incorrect" not in (r.get("hit_direction") or "").lower()
            )

            asset_summary["by_horizon"][str(h)] = {
                "n_evaluated": len(h_rows),
                "direction_correct": correct,
                "direction_accuracy_pct": round((correct / len(h_rows)) * 100, 2) if h_rows else None,
                "mean_error_abs": round(sum(err_abs) / len(err_abs), 4) if err_abs else None,
                "mean_error_pct": round(sum(err_pct) / len(err_pct), 4) if err_pct else None,
            }

        out["by_asset"][asset] = asset_summary

    return out


def build_prediction_status_block(
    prediction_summary: Optional[dict],
    model_readiness: Optional[dict],
    repo_status: Optional[dict],
    performance_summary: Optional[dict],
) -> Dict[str, Any]:
    return {
        "prediction_summary": prediction_summary or {"available": False},
        "model_readiness": model_readiness or {"available": False},
        "repo_status_brief": {
            "available": bool(repo_status),
            "summary": (repo_status or {}).get("summary"),
            "missing_watched_files": (repo_status or {}).get("missing_watched_files"),
        },
        "performance_summary": performance_summary or {"available": False},
    }


def main() -> None:
    signal = load_json(SIGNAL_PATH)
    if not signal:
        print(f"Missing required file: {SIGNAL_PATH}")
        raise SystemExit(1)

    tz, published_at_utc, published_at_local = resolve_report_timestamps(signal)

    forecast = None
    baseline_summary = load_json(BASELINE_SUMMARY_PATH)
    if baseline_summary:
        try:
            forecast = flatten_baseline_summary(baseline_summary, as_of_utc=published_at_utc)
        except Exception as e:
            print(f"WARN: Failed to load/flatten baseline summary: {e}")

    if forecast is None:
        forecast = load_json(FORECAST_FALLBACK_PATH)

    prediction_rows = load_csv_rows(PREDICTIONS_PATH)

    similarity_state = load_similarity_forecasts()
    latest_prediction_state = latest_predictions_by_asset(prediction_rows)
    prediction_eval_state = prediction_evaluation_summary(prediction_rows)

    repo_status = load_json(REPO_STATUS_PATH)
    model_readiness = load_json(MODEL_READINESS_PATH)
    prediction_summary = load_json(PREDICTION_SUMMARY_PATH)
    performance_summary = load_json(PERFORMANCE_SUMMARY_PATH)

    report = {
        "schema": "hourly_report_v2",
        "timezone": tz,
        "published_at_utc": published_at_utc,
        "published_at_local": published_at_local,
        "active_prediction_assets": ACTIVE_PREDICTION_ASSETS,
        "context_assets": CONTEXT_ASSETS,

        # Preserve old fields for compatibility.
        "signal": signal,
        "forecast": forecast,

        # New Task 5 reporting blocks.
        "market_state": {
            "date": signal.get("date"),
            "timezone": signal.get("timezone"),
            "published_at_utc": signal.get("published_at_utc"),
            "published_at_local": signal.get("published_at_local"),
            "eth_usdt": signal.get("eth_usdt"),
            "btc_usdt": signal.get("btc_usdt"),
            "eth_btc": signal.get("eth_btc"),
            "integrity": signal.get("integrity"),
        },
        "baseline_forecast_state": forecast or {"available": False},
        "similarity_forecast_state": similarity_state,
        "latest_prediction_state": latest_prediction_state,
        "prediction_evaluation_state": prediction_eval_state,
        "prediction_diagnostics": build_prediction_status_block(
            prediction_summary=prediction_summary,
            model_readiness=model_readiness,
            repo_status=repo_status,
            performance_summary=performance_summary,
        ),
        "notes": [
            "ETH-USDT and BTC-USDT are active prediction assets.",
            "ETH-BTC is currently treated as context, not an active prediction asset.",
            "Forecast horizons are 24h, 48h, 168h, and 336h.",
            "Current live candle snapshots may be limited to the exchange/free-plan candle limit; longer-horizon forecasts use historical matured outcomes from similar states.",
        ],
    }

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {OUT_PATH}")
    print(f"Schema: {report['schema']}")
    print(f"Active prediction assets: {ACTIVE_PREDICTION_ASSETS}")

    for asset, state in latest_prediction_state.get("assets", {}).items():
        print(f"{asset}: latest predictions available={state.get('available')}, rows={len(state.get('rows', []))}")


if __name__ == "__main__":
    main()
