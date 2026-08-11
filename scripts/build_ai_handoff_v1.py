#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


HOURLY_REPORT_PATH = os.path.join("data", "hourly_report.json")
SIMILARITY_MULTI_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")
SIMILARITY_LEGACY_PATH = os.path.join("data", "model", "similarity_forecast_v2.json")
PREDICTIONS_V1_PATH = os.path.join("data", "model", "predictions_v1.csv")
PERFORMANCE_SUMMARY_PATH = os.path.join("data", "model", "performance_summary_v1.json")
ACTIONABILITY_PATH = os.path.join("data", "model", "actionability_v1.json")

REPO_STATUS_PATH = os.path.join("data", "diagnostics", "repo_status.json")
MODEL_READINESS_PATH = os.path.join("data", "diagnostics", "model_readiness.json")
MODEL_READINESS_V2_PATH = os.path.join("data", "diagnostics", "model_readiness_v2.json")
PREDICTION_SUMMARY_PATH = os.path.join("data", "diagnostics", "prediction_summary.json")
PREDICTION_AUDIT_PATH = os.path.join("data", "diagnostics", "prediction_audit_v2.json")
MODEL_TOURNAMENT_PATH = os.path.join("data", "diagnostics", "model_tournament_v1.json")
SHADOW_PERFORMANCE_PATH = os.path.join("data", "model", "shadow_challenger_performance_v1.json")
LUNO_PROXY_PATH = os.path.join("data", "luno", "luno_price_proxy_latest_v1.json")
LUNO_TOURNAMENT_PATH = os.path.join("data", "diagnostics", "luno_point_tournament_v1.json")
LUNO_SHADOW_LATEST_PATH = os.path.join("data", "luno", "luno_shadow_latest_v1.json")
LUNO_SHADOW_PERFORMANCE_PATH = os.path.join("data", "luno", "luno_shadow_performance_v1.json")

OUT_PATH = os.path.join("data", "ai_handoff_v1.json")

ACTIVE_PREDICTION_ASSETS = ["ETH-USDT", "BTC-USDT"]
CONTEXT_ASSETS = ["ETH-BTC"]
HORIZONS = [24, 48, 168, 336]


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

    if hourly_report.get("market_state"):
        market = hourly_report.get("market_state", {})
        market["available"] = True
        return market

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

    forecast = hourly_report.get("baseline_forecast_state") or hourly_report.get("forecast")
    if not forecast:
        return {"available": False}

    targets = forecast.get("targets", [])
    keep_horizons = set(HORIZONS)
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


def compact_similarity_forecast(forecast: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "available": True,
        "schema": forecast.get("schema"),
        "generated_at_utc": forecast.get("generated_at_utc"),
        "as_of_utc": forecast.get("as_of_utc"),
        "as_of_local": forecast.get("as_of_local"),
        "asset": forecast.get("asset"),
        "candle_key": forecast.get("candle_key"),
        "current_state": forecast.get("current_state"),
        "model_dataset": forecast.get("model_dataset"),
        "similarity": forecast.get("similarity"),
        "directional_scorecard": forecast.get("directional_scorecard"),
        "overall_confidence": forecast.get("overall_confidence"),
        "forecast_summary_primary_top20": forecast.get("forecast_summary_primary_top20"),
        "notes": forecast.get("notes", []),
    }


def compact_similarity_forecasts() -> Dict[str, Any]:
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



def compact_prediction_audit(audit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not audit:
        return {"available": False}

    def compact_metrics(block: Any) -> Dict[str, Any]:
        if not isinstance(block, dict):
            return {}
        return {
            "binary_direction_rows": block.get("binary_direction_rows"),
            "direction_accuracy_pct": block.get("direction_accuracy_pct"),
            "balanced_accuracy_pct": block.get("balanced_accuracy_pct"),
            "actual_up_rate_pct": block.get("actual_up_rate_pct"),
            "majority_class_baseline_pct": block.get("majority_class_baseline_pct"),
            "inverse_model_accuracy_pct": block.get("inverse_model_accuracy_pct"),
            "edge_vs_majority_baseline_pp": block.get("edge_vs_majority_baseline_pp"),
        }

    conviction = audit.get("conviction_thresholds_overall", {})
    selected_conviction = {
        key: conviction.get(key)
        for key in ["0.00", "0.50", "1.00", "2.00", "3.00"]
        if key in conviction
    }

    by_horizon = {
        str(k): compact_metrics(v)
        for k, v in (audit.get("by_horizon") or {}).items()
    }
    by_asset_horizon = {
        str(k): compact_metrics(v)
        for k, v in (audit.get("by_asset_horizon") or {}).items()
    }

    selected_baselines: Dict[str, Any] = {}
    for horizon, block in (audit.get("baseline_tournament_by_horizon") or {}).items():
        if not isinstance(block, dict):
            continue
        selected_baselines[str(horizon)] = {
            "model_v2": block.get("model_v2"),
            "ret_48h_reversion": block.get("ret_48h_reversion"),
            "ret_48h_trend": block.get("ret_48h_trend"),
        }

    return {
        "available": bool(audit.get("available", True)),
        "schema": audit.get("schema"),
        "generated_at_utc": audit.get("generated_at_utc"),
        "data_quality": audit.get("data_quality"),
        "overall": compact_metrics(audit.get("overall")),
        "overall_independent_sample": compact_metrics(audit.get("overall_independent_sample")),
        "by_horizon": by_horizon,
        "by_asset_horizon": by_asset_horizon,
        "selected_baseline_tournament_by_horizon": selected_baselines,
        "paired_short_horizon_benchmark_independent": audit.get("paired_short_horizon_benchmark_independent"),
        "selected_conviction_thresholds": selected_conviction,
        "warnings": audit.get("warnings", []),
    }



def compact_model_tournament(tournament: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not tournament:
        return {"available": False}

    tournaments = tournament.get("tournaments") or {}
    short_rules = {}
    for h in ("24", "48"):
        block = (tournaments.get("by_horizon_independent") or {}).get(h) or {}
        rules = block.get("rules") or {}
        short_rules[h] = {
            name: rules.get(name)
            for name in [
                "similarity_v2",
                "mean_reversion_48h",
                "momentum_48h",
                "sma24_trend",
                "sma48_trend",
            ]
            if name in rules
        }

    selector = tournament.get("walk_forward_selector") or {}
    return {
        "available": bool(tournament.get("available", True)),
        "schema": tournament.get("schema"),
        "status": tournament.get("status"),
        "generated_at_utc": tournament.get("generated_at_utc"),
        "data_quality": tournament.get("data_quality"),
        "regime_definition": tournament.get("regime_definition"),
        "short_horizon_independent_rules": short_rules,
        "walk_forward_selector_short": selector.get("short_horizons_24_48"),
        "walk_forward_selector_by_horizon": selector.get("by_horizon"),
        "promotion_rule": tournament.get("promotion_rule"),
        "warnings": tournament.get("warnings", []),
    }

def compact_luno_tournament(tournament: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not tournament:
        return {"available": False}
    return {
        "available": True,
        "schema": tournament.get("schema"),
        "status": tournament.get("status"),
        "generated_at_utc": tournament.get("generated_at_utc"),
        "objective": tournament.get("objective"),
        "settlement_truth_status": tournament.get("settlement_truth_status"),
        "schedule": tournament.get("schedule"),
        "data_quality": tournament.get("data_quality"),
        "weekly": tournament.get("weekly"),
        "monthly": tournament.get("monthly"),
        "promotion_rule": tournament.get("promotion_rule"),
    }


def compact_prediction_row(r: Dict[str, str]) -> Dict[str, Any]:
    return {
        "asset": r.get("asset"),
        "created_at_utc": r.get("created_at_utc"),
        "created_at_local": r.get("created_at_local"),
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
        "weighted_close_mean": safe_float(r.get("weighted_close_mean")),
        "weighted_close_median": safe_float(r.get("weighted_close_median")),
        "weighted_upside_mean": safe_float(r.get("weighted_upside_mean")),
        "weighted_downside_mean": safe_float(r.get("weighted_downside_mean")),
        "model_version": r.get("model_version"),
        "status": r.get("status"),
        "actual_close": safe_float(r.get("actual_close")),
        "actual_close_change_pct": safe_float(r.get("actual_close_change_pct")),
        "error_abs": safe_float(r.get("error_abs")),
        "error_pct": safe_float(r.get("error_pct")),
        "hit_direction": r.get("hit_direction"),
        "drift_tag": r.get("drift_tag"),
        "evaluated_at_utc": r.get("evaluated_at_utc"),
    }


def latest_prediction_state(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False}

    by_asset: Dict[str, Any] = {}

    for asset in ACTIVE_PREDICTION_ASSETS:
        asset_rows = [r for r in rows if r.get("asset") == asset]

        if not asset_rows:
            by_asset[asset] = {
                "available": False,
                "rows": [],
            }
            continue

        latest_entry_ts = max((r.get("entry_ts_utc", "") for r in asset_rows), default="")
        batch = [r for r in asset_rows if r.get("entry_ts_utc") == latest_entry_ts]
        batch.sort(key=lambda r: safe_int(r.get("horizon_h")) or 0)

        by_asset[asset] = {
            "available": True,
            "latest_entry_ts_utc": latest_entry_ts,
            "latest_created_at_utc": max((r.get("created_at_utc", "") for r in batch), default=""),
            "rows": [compact_prediction_row(r) for r in batch],
        }

    return {
        "available": True,
        "assets": by_asset,
    }


def prediction_evaluation_state(rows: List[Dict[str, str]]) -> Dict[str, Any]:
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


def diagnostics_state() -> Dict[str, Any]:
    repo_status = load_json(REPO_STATUS_PATH)
    model_readiness = load_json(MODEL_READINESS_PATH)
    model_readiness_v2 = load_json(MODEL_READINESS_V2_PATH)
    prediction_audit = load_json(PREDICTION_AUDIT_PATH)
    model_tournament = load_json(MODEL_TOURNAMENT_PATH)
    prediction_summary = load_json(PREDICTION_SUMMARY_PATH)
    performance_summary = load_json(PERFORMANCE_SUMMARY_PATH)
    actionability_state = load_json(ACTIONABILITY_PATH)

    return {
        "available": any([repo_status, model_readiness, model_readiness_v2, prediction_summary, performance_summary, actionability_state]),
        "repo_status_summary": (repo_status or {}).get("summary"),
        "missing_watched_files": (repo_status or {}).get("missing_watched_files"),
        "model_readiness": model_readiness,
        "model_readiness_v2": model_readiness_v2,
        "prediction_summary": prediction_summary,
        "performance_summary": performance_summary,
        "actionability_state": actionability_state,
        "model_tournament": compact_model_tournament(model_tournament),
    }


def build_interpretation_instructions() -> Dict[str, Any]:
    return {
        "purpose": "This file is a single pasteable AI handoff for interpreting the latest crypto deployment signal in plain English.",
        "audience": "A cautious crypto user comparing ETH and BTC deployment opportunities.",
        "task": [
            "Read the file as a market interpretation artifact.",
            "Explain the latest ETH market state first.",
            "Then explain the latest BTC market state.",
            "For each active asset, explain the 24h, 48h, 7d, and 14d outlook.",
            "Separate observed market facts from model inference.",
            "Explain confidence and analogue quality clearly.",
            "Use readiness_state to distinguish model maturity from raw prediction strength.",
            "Use the actionability_state block to distinguish raw predictions from actionable/caution/not-actionable signals.",
            "If prediction evaluations are available, explain whether the model has been accurate or drifting.",
            "Use prediction_audit_state to compare V2 against simple baselines and to flag anti-calibrated confidence or magnitude.",
            "Use model_tournament_state to compare transparent challengers and regime behaviour; it is diagnostic-only and must not be described as a live replacement model.",
            "Use shadow_challenger_state as the strongest post-discovery evidence because those challenger calls are recorded prospectively before outcomes mature.",
            "Treat luno_price_predict_state as a separate point-price objective from general direction forecasting.",
            "For Luno Price Predict, compare every candidate against persistence/current-price MAE and distinguish KuCoin market-proxy evaluation from exact Luno internal-rate results.",
            "Give a cautious deployment posture, not financial advice.",
        ],
        "required_output_format": [
            "1. Plain-English market summary",
            "2. ETH outlook: 24h / 48h / 7d / 14d",
            "3. BTC outlook: 24h / 48h / 7d / 14d",
            "4. Similarity quality and confidence",
            "5. Model readiness: READY / LIMITED / EARLY / NOT_READY",
            "6. Actionability: ACTIONABLE / CAUTION / NOT_ACTIONABLE",
            "7. Recent prediction performance",
            "8. Main upside case",
            "9. Main downside risk",
            "10. Deployment posture for a cautious user",
            "11. Final verdict in 1-2 sentences",
        ],
        "rules": [
            "Use plain English.",
            "Do not assume the reader understands quant or trading jargon.",
            "Do not pretend the model is certain.",
            "Treat 24h and 48h as tactical horizons, and 7d and 14d as broader directional horizons.",
            "If ETH and BTC conflict, explain the conflict.",
            "If confidence is low, say so clearly.",
            "Treat READY, LIMITED, EARLY, and NOT_READY as model maturity labels, not direct buy/sell signals.",
            "Do not treat every prediction as actionable; follow the ACTIONABLE / CAUTION / NOT_ACTIONABLE labels.",
            "Do not promote an experimental tournament rule merely because it wins an in-sample or small-sample slice.",
            "BTC prediction history is newly active, so be cautious when interpreting BTC performance.",
        ],
    }


def main() -> None:
    hourly_report = load_json(HOURLY_REPORT_PATH)
    pred_rows = load_csv_rows(PREDICTIONS_V1_PATH)
    actionability_state = load_json(ACTIONABILITY_PATH)
    model_readiness_v2 = load_json(MODEL_READINESS_V2_PATH)
    prediction_audit = load_json(PREDICTION_AUDIT_PATH)
    model_tournament = load_json(MODEL_TOURNAMENT_PATH)
    shadow_performance = load_json(SHADOW_PERFORMANCE_PATH)
    luno_proxy = load_json(LUNO_PROXY_PATH)
    luno_tournament = load_json(LUNO_TOURNAMENT_PATH)
    luno_shadow_latest = load_json(LUNO_SHADOW_LATEST_PATH)
    luno_shadow_performance = load_json(LUNO_SHADOW_PERFORMANCE_PATH)

    payload = {
        "schema": "ai_handoff_v2",
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "hourly_report": HOURLY_REPORT_PATH if os.path.isfile(HOURLY_REPORT_PATH) else None,
                "similarity_forecast_v2_multi": SIMILARITY_MULTI_PATH if os.path.isfile(SIMILARITY_MULTI_PATH) else None,
                "similarity_forecast_v2_legacy": SIMILARITY_LEGACY_PATH if os.path.isfile(SIMILARITY_LEGACY_PATH) else None,
                "predictions_v1": PREDICTIONS_V1_PATH if os.path.isfile(PREDICTIONS_V1_PATH) else None,
                "performance_summary_v1": PERFORMANCE_SUMMARY_PATH if os.path.isfile(PERFORMANCE_SUMMARY_PATH) else None,
                "actionability_v1": ACTIONABILITY_PATH if os.path.isfile(ACTIONABILITY_PATH) else None,
                "model_readiness": MODEL_READINESS_PATH if os.path.isfile(MODEL_READINESS_PATH) else None,
                "model_readiness_v2": MODEL_READINESS_V2_PATH if os.path.isfile(MODEL_READINESS_V2_PATH) else None,
                "repo_status": REPO_STATUS_PATH if os.path.isfile(REPO_STATUS_PATH) else None,
                "prediction_summary": PREDICTION_SUMMARY_PATH if os.path.isfile(PREDICTION_SUMMARY_PATH) else None,
                "prediction_audit_v2": PREDICTION_AUDIT_PATH if os.path.isfile(PREDICTION_AUDIT_PATH) else None,
                "model_tournament_v1": MODEL_TOURNAMENT_PATH if os.path.isfile(MODEL_TOURNAMENT_PATH) else None,
                "shadow_challenger_performance_v1": SHADOW_PERFORMANCE_PATH if os.path.isfile(SHADOW_PERFORMANCE_PATH) else None,
                "luno_price_proxy_v1": LUNO_PROXY_PATH if os.path.isfile(LUNO_PROXY_PATH) else None,
                "luno_point_tournament_v1": LUNO_TOURNAMENT_PATH if os.path.isfile(LUNO_TOURNAMENT_PATH) else None,
                "luno_shadow_latest_v1": LUNO_SHADOW_LATEST_PATH if os.path.isfile(LUNO_SHADOW_LATEST_PATH) else None,
                "luno_shadow_performance_v1": LUNO_SHADOW_PERFORMANCE_PATH if os.path.isfile(LUNO_SHADOW_PERFORMANCE_PATH) else None,
            },
        },
        "active_prediction_assets": ACTIVE_PREDICTION_ASSETS,
        "context_assets": CONTEXT_ASSETS,
        "market_state": compact_signal(hourly_report),
        "baseline_forecast_state": compact_baseline_forecast(hourly_report),
        "similarity_forecast_state": compact_similarity_forecasts(),
        "latest_prediction_state": latest_prediction_state(pred_rows),
        "actionability_state": actionability_state or {"available": False},
        "readiness_state": model_readiness_v2 or {"available": False},
        "prediction_evaluation_state": prediction_evaluation_state(pred_rows),
        "prediction_audit_state": compact_prediction_audit(prediction_audit),
        "model_tournament_state": compact_model_tournament(model_tournament),
        "shadow_challenger_state": shadow_performance or {"available": False},
        "luno_price_predict_state": {
            "available": any([luno_proxy, luno_tournament, luno_shadow_latest, luno_shadow_performance]),
            "price_proxy": luno_proxy or {"available": False},
            "point_price_tournament": compact_luno_tournament(luno_tournament),
            "prospective_shadow": luno_shadow_latest or {"available": False},
            "prospective_performance": luno_shadow_performance or {"available": False},
        },
        "diagnostics_state": diagnostics_state(),
        "interpretation_instructions": build_interpretation_instructions(),
    }

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"AI handoff written to: {OUT_PATH}")
    print(f"Schema: {payload['schema']}")

    if actionability_state:
        print(f"Actionability schema: {actionability_state.get('schema')}")

    if model_readiness_v2:
        print(f"Readiness schema: {model_readiness_v2.get('schema')}")

    for asset, state in payload["latest_prediction_state"].get("assets", {}).items():
        print(f"{asset}: latest predictions available={state.get('available')}, rows={len(state.get('rows', []))}")


if __name__ == "__main__":
    main()
