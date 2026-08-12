#!/usr/bin/env python3

import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

DATASET_PATH = os.path.join("data", "model", "model_dataset_v1.csv")
PROFILE_PATH = os.path.join("data", "pionex", "pionex_grid_profile_v1.json")
OUT_PATH = os.path.join("data", "diagnostics", "grid_tournament_v1.json")

ASSET = "ETH-USDT"
HORIZON_H = 24
MIN_TRAIN_ROWS = 100

BASE_FEATURES = {
    "atr14_raw": "atr14_pct",
    "recent_range_24_raw": "range_24h_pct",
    "recent_range_48_raw": "range_48h_pct",
}

SCALED_CANDIDATES = {
    "atr14_scaled_walkforward": "atr14_pct",
    "recent_range_24_scaled_walkforward": "range_24h_pct",
    "recent_range_48_scaled_walkforward": "range_48h_pct",
}


def to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def to_int(value) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def quantile(values: List[float], q: float) -> Optional[float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.mean(xvals)
    my = statistics.mean(yvals)
    sx = sum((x - mx) ** 2 for x in xvals)
    sy = sum((y - my) ** 2 for y in yvals)
    if sx <= 0 or sy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(sx * sy)


def describe(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6),
        "median": round(statistics.median(vals), 6),
        "p20": round(quantile(vals, 0.20), 6),
        "p50": round(quantile(vals, 0.50), 6),
        "p80": round(quantile(vals, 0.80), 6),
        "p90": round(quantile(vals, 0.90), 6),
        "p95": round(quantile(vals, 0.95), 6),
    }


def load_profile() -> dict:
    if not os.path.isfile(PROFILE_PATH):
        return {"available": False}
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {"available": True, "payload": payload}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def load_rows() -> List[dict]:
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Missing model dataset: {DATASET_PATH}")

    required = {
        "asset",
        "entry_ts_utc",
        "entry_close",
        "atr14_pct",
        "range_24h_pct",
        "range_48h_pct",
        "ret_24h_pct",
        "ret_48h_pct",
        "close_vs_sma_24_pct",
        "close_vs_sma_48_pct",
        "max_up_pct_24",
        "max_down_pct_24",
        "close_change_pct_24",
        "range_pct_24",
    }

    rows: List[dict] = []
    with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"Dataset missing required Phase 4 fields: {missing}")

        for raw in reader:
            if raw.get("asset") != ASSET:
                continue

            ts = to_int(raw.get("entry_ts_utc"))
            entry_close = to_float(raw.get("entry_close"))
            target_range = to_float(raw.get("range_pct_24"))
            close_change = to_float(raw.get("close_change_pct_24"))
            max_up = to_float(raw.get("max_up_pct_24"))
            max_down = to_float(raw.get("max_down_pct_24"))

            if None in (ts, entry_close, target_range, close_change, max_up, max_down):
                continue
            if entry_close <= 0 or target_range <= 0:
                continue

            row = {
                "ts": ts,
                "entry_close": entry_close,
                "target_range": target_range,
                "close_change": close_change,
                "max_up": max_up,
                "max_down": max_down,
            }

            valid = True
            for field in [
                "atr14_pct",
                "range_24h_pct",
                "range_48h_pct",
                "ret_24h_pct",
                "ret_48h_pct",
                "close_vs_sma_24_pct",
                "close_vs_sma_48_pct",
            ]:
                val = to_float(raw.get(field))
                if val is None:
                    valid = False
                    break
                row[field] = val

            if not valid:
                continue

            row["two_sided_excursion_pct"] = max(
                0.0, min(max_up, abs(min(0.0, max_down)))
            )
            row["trend_efficiency"] = min(
                1.0, abs(close_change) / target_range
            ) if target_range > 0 else None
            row["oscillation_residual_pct"] = max(
                0.0, target_range - abs(close_change)
            )

            rows.append(row)

    rows.sort(key=lambda r: r["ts"])

    # Deduplicate identical market-state timestamps.
    deduped: Dict[int, dict] = {}
    for row in rows:
        deduped[row["ts"]] = row

    return [deduped[k] for k in sorted(deduped)]


def matured_train(rows: List[dict], test_ts: int) -> List[dict]:
    cutoff = test_ts - HORIZON_H * 3600
    return [r for r in rows if r["ts"] <= cutoff]


def median_ratio(train: List[dict], feature: str) -> Optional[float]:
    ratios = []
    for r in train:
        x = r.get(feature)
        y = r.get("target_range")
        if x is None or y is None or x <= 0 or y <= 0:
            continue
        ratios.append(y / x)
    if len(ratios) < MIN_TRAIN_ROWS:
        return None
    return statistics.median(ratios)


def build_predictions(rows: List[dict]) -> List[dict]:
    predictions: List[dict] = []

    for idx, test in enumerate(rows):
        train = matured_train(rows[:idx], test["ts"])
        if len(train) < MIN_TRAIN_ROWS:
            continue

        item = {
            "ts": test["ts"],
            "actual": test["target_range"],
            "entry_close": test["entry_close"],
        }

        for name, feature in BASE_FEATURES.items():
            x = test.get(feature)
            item[name] = x if x is not None and x > 0 else None

        scaled_values = []
        for name, feature in SCALED_CANDIDATES.items():
            x = test.get(feature)
            ratio = median_ratio(train, feature)
            pred = None
            if x is not None and x > 0 and ratio is not None:
                pred = x * ratio
                scaled_values.append(pred)
            item[name] = pred

        item["median_scaled_blend_walkforward"] = (
            statistics.median(scaled_values) if scaled_values else None
        )
        predictions.append(item)

    return predictions


def independent_24h_sample(predictions: List[dict]) -> List[dict]:
    out = []
    last_ts = None
    min_gap = HORIZON_H * 3600
    for row in predictions:
        if last_ts is None or row["ts"] - last_ts >= min_gap:
            out.append(row)
            last_ts = row["ts"]
    return out


def evaluate_candidate(rows: List[dict], candidate: str) -> dict:
    pairs: List[Tuple[float, float]] = []
    for r in rows:
        pred = r.get(candidate)
        actual = r.get("actual")
        if pred is None or actual is None:
            continue
        if pred <= 0 or actual <= 0:
            continue
        pairs.append((pred, actual))

    if not pairs:
        return {"n": 0, "available": False}

    errors = [p - a for p, a in pairs]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e * e for e in errors]
    rel_errors = [abs(p - a) / a for p, a in pairs if a > 0]
    preds = [p for p, _ in pairs]
    actuals = [a for _, a in pairs]

    return {
        "available": True,
        "n": len(pairs),
        "mae_pct_points": round(statistics.mean(abs_errors), 6),
        "median_abs_error_pct_points": round(statistics.median(abs_errors), 6),
        "rmse_pct_points": round(math.sqrt(statistics.mean(sq_errors)), 6),
        "mean_bias_pct_points": round(statistics.mean(errors), 6),
        "pearson_corr": (
            round(pearson(preds, actuals), 6)
            if pearson(preds, actuals) is not None
            else None
        ),
        "within_25pct_relative_error_pct": round(
            100.0 * sum(1 for e in rel_errors if e <= 0.25) / len(rel_errors), 4
        ),
        "within_50pct_relative_error_pct": round(
            100.0 * sum(1 for e in rel_errors if e <= 0.50) / len(rel_errors), 4
        ),
        "predicted_range_summary": describe(preds),
        "actual_range_summary": describe(actuals),
    }


def feature_correlations(rows: List[dict]) -> dict:
    features = [
        "atr14_pct",
        "range_24h_pct",
        "range_48h_pct",
        "ret_24h_pct",
        "ret_48h_pct",
        "close_vs_sma_24_pct",
        "close_vs_sma_48_pct",
    ]
    targets = [
        "target_range",
        "two_sided_excursion_pct",
        "trend_efficiency",
        "oscillation_residual_pct",
    ]

    out = {}
    for feature in features:
        block = {}
        for target in targets:
            xs = []
            ys = []
            for r in rows:
                x = r.get(feature)
                y = r.get(target)
                if x is None or y is None:
                    continue
                # Magnitude of directional state is usually more relevant to volatility.
                if feature in {
                    "ret_24h_pct",
                    "ret_48h_pct",
                    "close_vs_sma_24_pct",
                    "close_vs_sma_48_pct",
                }:
                    x = abs(x)
                xs.append(x)
                ys.append(y)
            corr = pearson(xs, ys)
            block[target] = round(corr, 6) if corr is not None else None
        out[feature] = block
    return out


def current_grid_derived(profile_wrapper: dict) -> dict:
    if not profile_wrapper.get("available"):
        return {"available": False}
    p = profile_wrapper.get("payload") or {}
    state = p.get("current_reference_state") or {}
    lower = to_float(state.get("lower_limit_usdt"))
    upper = to_float(state.get("upper_limit_usdt"))
    current = to_float(state.get("current_price_usdt"))
    grids = to_int(state.get("grids"))

    if None in (lower, upper, current, grids) or current <= 0 or upper <= lower:
        return {"available": False, "reason": "Incomplete current_reference_state"}

    return {
        "available": True,
        "width_usdt": round(upper - lower, 6),
        "width_pct_of_current": round((upper - lower) / current * 100.0, 6),
        "distance_to_lower_pct": round((current - lower) / current * 100.0, 6),
        "distance_to_upper_pct": round((upper - current) / current * 100.0, 6),
        "grids": grids,
        "profit_per_grid_pct_fee_deducted": state.get(
            "profit_per_grid_pct_fee_deducted"
        ),
        "policy": p.get("adjustment_policy"),
    }


def main() -> None:
    rows = load_rows()
    if len(rows) < MIN_TRAIN_ROWS + 10:
        raise SystemExit(
            f"Insufficient ETH rows for grid tournament: {len(rows)}"
        )

    predictions = build_predictions(rows)
    independent = independent_24h_sample(predictions)

    candidate_names = list(BASE_FEATURES) + list(SCALED_CANDIDATES) + [
        "median_scaled_blend_walkforward"
    ]

    full_metrics = {
        name: evaluate_candidate(predictions, name) for name in candidate_names
    }
    independent_metrics = {
        name: evaluate_candidate(independent, name) for name in candidate_names
    }

    ranking = []
    for name, metrics in independent_metrics.items():
        if metrics.get("available"):
            ranking.append({
                "candidate": name,
                "mae_pct_points": metrics.get("mae_pct_points"),
                "pearson_corr": metrics.get("pearson_corr"),
                "n": metrics.get("n"),
            })
    ranking.sort(key=lambda x: (x["mae_pct_points"], -1 * (x["pearson_corr"] or -999)))

    outcomes = {
        "future_24h_range_pct": describe([r["target_range"] for r in rows]),
        "two_sided_excursion_pct": describe(
            [r["two_sided_excursion_pct"] for r in rows]
        ),
        "trend_efficiency": describe([r["trend_efficiency"] for r in rows]),
        "oscillation_residual_pct": describe(
            [r["oscillation_residual_pct"] for r in rows]
        ),
    }

    profile = load_profile()

    payload = {
        "schema": "grid_tournament_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY",
        "scope": {
            "platform": "Pionex",
            "bot_type": "Spot Grid",
            "asset": ASSET,
            "horizon_h": HORIZON_H,
            "objective": (
                "Forecast ETH 24h range/volatility and grid-friendly two-way movement. "
                "Profit probability is deferred to the Pionex path simulator in Phase 4B."
            ),
        },
        "methodology": {
            "target": "range_pct_24",
            "walkforward_rule": (
                "For every test row, calibration uses only rows whose 24h outcome "
                "had matured by the test timestamp."
            ),
            "min_mature_training_rows": MIN_TRAIN_ROWS,
            "independence_check": (
                "Greedy 24h-spaced test sample to reduce hourly pseudo-replication."
            ),
            "promotion": "No live model or trading decision changes in Phase 4A.",
        },
        "rows": {
            "eth_rows_loaded": len(rows),
            "walkforward_test_rows": len(predictions),
            "independent_24h_rows": len(independent),
        },
        "actual_outcome_distributions": outcomes,
        "feature_correlations": feature_correlations(rows),
        "candidates": {
            "full_hourly": full_metrics,
            "independent_24h": independent_metrics,
            "independent_ranking_by_mae": ranking,
        },
        "historical_champion_independent": ranking[0] if ranking else None,
        "current_pionex_reference": current_grid_derived(profile),
        "next_phase": {
            "phase": "4B",
            "name": "Pionex path simulator",
            "required_capabilities": [
                "simulate actual grid crossings on ETH hourly/intrahour paths",
                "apply Pionex fee-aware net profit per completed round",
                "model range escape and inventory state",
                "estimate P(grid return >= x) for user-selected 24h thresholds",
            ],
            "profit_probability_status": "NOT_YET_MODELLED",
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)

    print(f"Grid tournament written: {OUT_PATH}")
    print(
        "Rows:",
        payload["rows"],
    )
    print("Independent ranking:")
    for item in ranking:
        print(
            f"  {item['candidate']}: "
            f"MAE={item['mae_pct_points']}pp "
            f"corr={item['pearson_corr']} n={item['n']}"
        )
    print("Historical champion:", payload["historical_champion_independent"])


if __name__ == "__main__":
    main()
