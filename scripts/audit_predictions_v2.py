#!/usr/bin/env python3
"""
Prediction Audit V2
===================

Non-destructive diagnostic for data/model/predictions_v1.csv.

Goals:
- Recompute direction accuracy independently of the legacy evaluator.
- Compare the model with simple directional baselines.
- Detect systematic anti-signal (inverse predictions outperforming originals).
- Report class balance (how often the market actually went up/down).
- Slice performance by asset, horizon, confidence and analogue quality.
- Test conviction thresholds: does accuracy improve when the predicted move is larger?
- Produce a less-overlapping "independent" sample for each asset/horizon.
- Quantify signed prediction bias and absolute price/percentage error.

Uses Python standard library only.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
FEATURES_PATH = os.path.join("data", "features", "features_v1.csv")
OUTPUT_PATH = os.path.join("data", "diagnostics", "prediction_audit_v2.json")

# |predicted return| thresholds used to test "conviction" versus accuracy.
CONVICTION_THRESHOLDS_PCT = [0.0, 0.10, 0.25, 0.50, 1.00, 2.00, 3.00]

BASELINE_FEATURES = {
    "ret_6h": "ret_6h_pct",
    "ret_12h": "ret_12h_pct",
    "ret_24h": "ret_24h_pct",
    "ret_48h": "ret_48h_pct",
    "sma_24": "close_vs_sma_24_pct",
    "sma_48": "close_vs_sma_48_pct",
}

SCHEMA = "prediction_audit_v2"


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def sign(x: Optional[float], eps: float = 1e-12) -> Optional[int]:
    if x is None:
        return None
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def pct(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return round(100.0 * numer / denom, 4)


def round_or_none(x: Optional[float], digits: int = 6) -> Optional[float]:
    if x is None:
        return None
    return round(x, digits)


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in values if x is not None and math.isfinite(x)]
    return mean(xs) if xs else None


def med(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in values if x is not None and math.isfinite(x)]
    return median(xs) if xs else None


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            status = (raw.get("status") or "").strip().lower()
            pred_ret = safe_float(raw.get("predicted_close_change_pct"))
            actual_ret = safe_float(raw.get("actual_close_change_pct"))

            # Only evaluated rows with both prediction and realised return
            # can participate in direction testing.
            evaluated = (
                status == "evaluated"
                and pred_ret is not None
                and actual_ret is not None
            )

            rows.append({
                **raw,
                "_evaluated": evaluated,
                "_asset": (raw.get("asset") or "UNKNOWN").strip(),
                "_horizon": safe_int(raw.get("horizon_h")),
                "_entry_ts": safe_int(raw.get("entry_ts_utc")),
                "_pred_ret": pred_ret,
                "_actual_ret": actual_ret,
                "_pred_dir": sign(pred_ret),
                "_actual_dir": sign(actual_ret),
                "_entry_close": safe_float(raw.get("entry_close")),
                "_pred_price": safe_float(raw.get("predicted_price")),
                "_actual_close": safe_float(raw.get("actual_close")),
                "_error_abs": safe_float(raw.get("error_abs")),
                "_error_pct": safe_float(raw.get("error_pct")),
                "_confidence": (raw.get("confidence") or "unknown").strip().lower(),
                "_analogue_quality": (raw.get("analogue_quality") or "unknown").strip().lower(),
                "_model_version": (raw.get("model_version") or "unknown").strip(),
            })
    return rows


def confusion_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in rows if r["_evaluated"]]

    actual_up = sum(r["_actual_dir"] == 1 for r in evaluated)
    actual_down = sum(r["_actual_dir"] == -1 for r in evaluated)
    actual_flat = sum(r["_actual_dir"] == 0 for r in evaluated)

    pred_up = sum(r["_pred_dir"] == 1 for r in evaluated)
    pred_down = sum(r["_pred_dir"] == -1 for r in evaluated)
    pred_flat = sum(r["_pred_dir"] == 0 for r in evaluated)

    # Binary subset: excludes exact-zero realised or predicted moves.
    binary = [
        r for r in evaluated
        if r["_actual_dir"] in (-1, 1) and r["_pred_dir"] in (-1, 1)
    ]

    tp_up = sum(r["_pred_dir"] == 1 and r["_actual_dir"] == 1 for r in binary)
    fp_up = sum(r["_pred_dir"] == 1 and r["_actual_dir"] == -1 for r in binary)
    tn_down = sum(r["_pred_dir"] == -1 and r["_actual_dir"] == -1 for r in binary)
    fn_up = sum(r["_pred_dir"] == -1 and r["_actual_dir"] == 1 for r in binary)

    correct = tp_up + tn_down
    n_binary = len(binary)

    recall_up = tp_up / (tp_up + fn_up) if (tp_up + fn_up) else None
    recall_down = tn_down / (tn_down + fp_up) if (tn_down + fp_up) else None
    precision_up = tp_up / (tp_up + fp_up) if (tp_up + fp_up) else None
    precision_down = tn_down / (tn_down + fn_up) if (tn_down + fn_up) else None

    balanced_accuracy = (
        (recall_up + recall_down) / 2.0
        if recall_up is not None and recall_down is not None
        else None
    )

    actual_up_binary = tp_up + fn_up
    actual_down_binary = tn_down + fp_up
    majority_correct = max(actual_up_binary, actual_down_binary)

    accuracy = correct / n_binary if n_binary else None
    majority_accuracy = majority_correct / n_binary if n_binary else None
    inverse_accuracy = (n_binary - correct) / n_binary if n_binary else None

    result = {
        "rows_total": len(rows),
        "rows_evaluated": len(evaluated),
        "binary_direction_rows": n_binary,
        "actual_direction_counts": {
            "up": actual_up,
            "down": actual_down,
            "flat": actual_flat,
        },
        "predicted_direction_counts": {
            "up": pred_up,
            "down": pred_down,
            "flat": pred_flat,
        },
        "confusion_matrix": {
            "pred_up_actual_up": tp_up,
            "pred_up_actual_down": fp_up,
            "pred_down_actual_down": tn_down,
            "pred_down_actual_up": fn_up,
        },
        "direction_accuracy_pct": pct(correct, n_binary),
        "balanced_accuracy_pct": round_or_none(
            100.0 * balanced_accuracy if balanced_accuracy is not None else None, 4
        ),
        "precision_up_pct": round_or_none(
            100.0 * precision_up if precision_up is not None else None, 4
        ),
        "precision_down_pct": round_or_none(
            100.0 * precision_down if precision_down is not None else None, 4
        ),
        "recall_up_pct": round_or_none(
            100.0 * recall_up if recall_up is not None else None, 4
        ),
        "recall_down_pct": round_or_none(
            100.0 * recall_down if recall_down is not None else None, 4
        ),
        "actual_up_rate_pct": pct(actual_up_binary, n_binary),
        "actual_down_rate_pct": pct(actual_down_binary, n_binary),
        "majority_class_baseline_pct": round_or_none(
            100.0 * majority_accuracy if majority_accuracy is not None else None, 4
        ),
        "inverse_model_accuracy_pct": round_or_none(
            100.0 * inverse_accuracy if inverse_accuracy is not None else None, 4
        ),
        "edge_vs_majority_baseline_pp": round_or_none(
            100.0 * (accuracy - majority_accuracy)
            if accuracy is not None and majority_accuracy is not None
            else None,
            4,
        ),
    }

    return result


def error_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in rows if r["_evaluated"]]

    signed_return_errors = [
        r["_pred_ret"] - r["_actual_ret"]
        for r in evaluated
        if r["_pred_ret"] is not None and r["_actual_ret"] is not None
    ]

    absolute_return_errors = [abs(x) for x in signed_return_errors]

    price_abs_errors: List[float] = []
    for r in evaluated:
        if r["_pred_price"] is not None and r["_actual_close"] is not None:
            price_abs_errors.append(abs(r["_pred_price"] - r["_actual_close"]))
        elif r["_error_abs"] is not None:
            price_abs_errors.append(abs(r["_error_abs"]))

    legacy_error_pct = [
        abs(r["_error_pct"]) for r in evaluated if r["_error_pct"] is not None
    ]

    return {
        "mean_predicted_return_pct": round_or_none(avg(r["_pred_ret"] for r in evaluated)),
        "median_predicted_return_pct": round_or_none(med(r["_pred_ret"] for r in evaluated)),
        "mean_actual_return_pct": round_or_none(avg(r["_actual_ret"] for r in evaluated)),
        "median_actual_return_pct": round_or_none(med(r["_actual_ret"] for r in evaluated)),
        "mean_signed_return_error_pp": round_or_none(avg(signed_return_errors)),
        "mean_absolute_return_error_pp": round_or_none(avg(absolute_return_errors)),
        "median_absolute_return_error_pp": round_or_none(med(absolute_return_errors)),
        "mean_absolute_price_error": round_or_none(avg(price_abs_errors)),
        "median_absolute_price_error": round_or_none(med(price_abs_errors)),
        "mean_legacy_abs_error_pct": round_or_none(avg(legacy_error_pct)),
    }


def full_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = confusion_metrics(rows)
    result["error"] = error_metrics(rows)
    return result


def group_by(rows: List[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(key_fn(r))].append(r)
    return dict(groups)


def grouped_metrics(rows: List[Dict[str, Any]], key_fn) -> Dict[str, Any]:
    return {
        key: full_metrics(group)
        for key, group in sorted(group_by(rows, key_fn).items())
    }


def conviction_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in rows if r["_evaluated"]]
    total_binary = sum(
        r["_actual_dir"] in (-1, 1) and r["_pred_dir"] in (-1, 1)
        for r in evaluated
    )

    output: Dict[str, Any] = {}
    for threshold in CONVICTION_THRESHOLDS_PCT:
        subset = [
            r for r in evaluated
            if r["_pred_ret"] is not None and abs(r["_pred_ret"]) >= threshold
        ]
        m = confusion_metrics(subset)
        n = m["binary_direction_rows"]
        output[f"{threshold:.2f}"] = {
            "threshold_abs_predicted_return_pct": threshold,
            "coverage_pct_of_binary_evaluated": pct(n, total_binary),
            **m,
        }
    return output



def load_feature_index(path: str) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    index: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    if not os.path.isfile(path):
        return index
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            asset = (r.get("asset") or "").strip()
            entry_ts = (r.get("entry_ts_utc") or "").strip()
            if asset and entry_ts:
                index[(asset, entry_ts)].append(r)
    return index


def attach_features(rows: List[Dict[str, Any]], feature_index: Dict[Tuple[str, str], List[Dict[str, str]]]) -> int:
    matched = 0
    for r in rows:
        key = (r.get("_asset", ""), str(r.get("entry_ts_utc") or r.get("_entry_ts") or ""))
        candidates = feature_index.get(key, [])
        if not candidates:
            r["_features"] = {}
            continue
        entry_close = r.get("_entry_close")
        def distance_to_entry(fr: Dict[str, str]) -> float:
            f_close = safe_float(fr.get("entry_close"))
            if entry_close is None or f_close is None:
                return 0.0
            return abs(f_close - entry_close)
        best = min(candidates, key=distance_to_entry)
        r["_features"] = best
        matched += 1
    return matched


def rule_metrics(rows: List[Dict[str, Any]], feature_name: str, invert: bool = False) -> Dict[str, Any]:
    synthetic: List[Dict[str, Any]] = []
    for r in rows:
        if not r.get("_evaluated"):
            continue
        fr = r.get("_features") or {}
        value = safe_float(fr.get(feature_name))
        pred_dir = sign(value)
        if pred_dir not in (-1, 1):
            continue
        if invert:
            pred_dir = -pred_dir
        r2 = dict(r)
        r2["_pred_dir"] = pred_dir
        synthetic.append(r2)
    return confusion_metrics(synthetic)


def baseline_tournament(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "model_v2": confusion_metrics(rows),
    }
    for label, feature in BASELINE_FEATURES.items():
        out[f"{label}_trend"] = rule_metrics(rows, feature, invert=False)
        out[f"{label}_reversion"] = rule_metrics(rows, feature, invert=True)
    return out


def compact_rule_result(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n": metrics.get("binary_direction_rows", 0),
        "direction_accuracy_pct": metrics.get("direction_accuracy_pct"),
        "balanced_accuracy_pct": metrics.get("balanced_accuracy_pct"),
        "majority_class_baseline_pct": metrics.get("majority_class_baseline_pct"),
    }


def monthly_short_horizon_benchmark(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r.get("_evaluated") or r.get("_horizon") not in (24, 48):
            continue
        ts = r.get("_entry_ts")
        if ts is None:
            continue
        month = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
        groups[(r.get("_asset", "UNKNOWN"), int(r["_horizon"]), month)].append(r)

    out: Dict[str, Any] = {}
    for (asset, horizon, month), group in sorted(groups.items()):
        model = confusion_metrics(group)
        rev48 = rule_metrics(group, "ret_48h_pct", invert=True)
        key = f"{asset}|{horizon}|{month}"
        out[key] = {
            "asset": asset,
            "horizon_h": horizon,
            "month": month,
            "model_v2": compact_rule_result(model),
            "ret_48h_reversion": compact_rule_result(rev48),
        }
    return out


def wilson_interval(correct: int, total: int, z: float = 1.96) -> Optional[List[float]]:
    if total <= 0:
        return None
    p = correct / total
    den = 1.0 + (z * z / total)
    centre = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return [round(100.0 * (centre - half), 4), round(100.0 * (centre + half), 4)]


def exact_two_sided_binomial_half(k: int, n: int) -> Optional[float]:
    if n <= 0:
        return None
    m = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_short_horizon_test(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [r for r in rows if r.get("_horizon") in (24, 48) and r.get("_evaluated")]
    model_correct = 0
    rev_correct = 0
    model_only = 0
    reversion_only = 0
    n = 0

    for r in usable:
        actual = r.get("_actual_dir")
        model_pred = r.get("_pred_dir")
        feature_value = safe_float((r.get("_features") or {}).get("ret_48h_pct"))
        feature_dir = sign(feature_value)
        if actual not in (-1, 1) or model_pred not in (-1, 1) or feature_dir not in (-1, 1):
            continue
        reversion_pred = -feature_dir
        mc = model_pred == actual
        rc = reversion_pred == actual
        model_correct += int(mc)
        rev_correct += int(rc)
        if mc and not rc:
            model_only += 1
        elif rc and not mc:
            reversion_only += 1
        n += 1

    discordant = model_only + reversion_only
    p_value = exact_two_sided_binomial_half(min(model_only, reversion_only), discordant) if discordant else None

    return {
        "horizons": [24, 48],
        "n": n,
        "model_v2": {
            "correct": model_correct,
            "accuracy_pct": pct(model_correct, n),
            "wilson_95_pct": wilson_interval(model_correct, n),
        },
        "ret_48h_reversion": {
            "correct": rev_correct,
            "accuracy_pct": pct(rev_correct, n),
            "wilson_95_pct": wilson_interval(rev_correct, n),
        },
        "paired_comparison": {
            "model_only_correct": model_only,
            "reversion_only_correct": reversion_only,
            "discordant_pairs": discordant,
            "mcnemar_exact_two_sided_p": round(p_value, 8) if p_value is not None else None,
        },
    }

def independent_sample(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Greedy non-overlapping sample.

    For each asset+horizon, keep the earliest evaluated prediction, then only
    keep another prediction when its entry timestamp is >= one full forecast
    horizon after the previously kept entry.

    This does not make observations magically independent, but it removes the
    most obvious hourly overlap from 24/48/168/336-hour forecasts.
    """
    output: List[Dict[str, Any]] = []

    groups = group_by(
        [r for r in rows if r["_evaluated"]],
        lambda r: (r["_asset"], r["_horizon"]),
    )

    for _, group in groups.items():
        group = sorted(
            [r for r in group if r["_entry_ts"] is not None and r["_horizon"] is not None],
            key=lambda r: r["_entry_ts"],
        )
        if not group:
            continue

        last_kept: Optional[int] = None
        horizon_seconds = int(group[0]["_horizon"]) * 3600

        for r in group:
            ts = int(r["_entry_ts"])
            if last_kept is None or ts - last_kept >= horizon_seconds:
                output.append(r)
                last_kept = ts

    return output


def warnings_for_metrics(label: str, metrics: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    acc = metrics.get("direction_accuracy_pct")
    majority = metrics.get("majority_class_baseline_pct")
    inverse = metrics.get("inverse_model_accuracy_pct")
    bal = metrics.get("balanced_accuracy_pct")
    n = metrics.get("binary_direction_rows", 0)

    if n and n < 30:
        warnings.append(f"{label}: fewer than 30 binary evaluated observations.")

    if acc is not None and majority is not None and acc < majority:
        warnings.append(
            f"{label}: model direction accuracy ({acc:.2f}%) is below "
            f"the majority-class baseline ({majority:.2f}%)."
        )

    if inverse is not None and inverse >= 55.0:
        warnings.append(
            f"{label}: inverted model direction reaches {inverse:.2f}%; "
            "investigate systematic anti-signal / target alignment before using inversion."
        )

    if bal is not None and bal < 50.0:
        warnings.append(
            f"{label}: balanced accuracy is below 50%, suggesting weakness is not "
            "explained only by an up/down class imbalance."
        )

    return warnings


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()

    if not os.path.isfile(PREDICTIONS_PATH):
        payload = {
            "schema": SCHEMA,
            "generated_at_utc": generated_at,
            "available": False,
            "reason": f"Missing predictions file: {PREDICTIONS_PATH}",
        }
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(payload, indent=2))
        return

    rows = load_rows(PREDICTIONS_PATH)
    feature_index = load_feature_index(FEATURES_PATH)
    feature_matches = attach_features(rows, feature_index)
    evaluated = [r for r in rows if r["_evaluated"]]
    independent = independent_sample(rows)

    by_asset = grouped_metrics(evaluated, lambda r: r["_asset"])
    by_horizon = grouped_metrics(evaluated, lambda r: r["_horizon"])
    by_asset_horizon = grouped_metrics(
        evaluated, lambda r: f'{r["_asset"]}|{r["_horizon"]}'
    )
    by_confidence = grouped_metrics(evaluated, lambda r: r["_confidence"])
    by_analogue_quality = grouped_metrics(evaluated, lambda r: r["_analogue_quality"])
    by_model_version = grouped_metrics(evaluated, lambda r: r["_model_version"])

    warnings: List[str] = []
    overall = full_metrics(evaluated)
    warnings.extend(warnings_for_metrics("overall", overall))

    for horizon, metrics in by_horizon.items():
        warnings.extend(warnings_for_metrics(f"horizon {horizon}h", metrics))

    short_test = paired_short_horizon_test(independent)
    if short_test.get("n", 0):
        model_acc = (short_test.get("model_v2") or {}).get("accuracy_pct")
        reversion_acc = (short_test.get("ret_48h_reversion") or {}).get("accuracy_pct")
        pval = (short_test.get("paired_comparison") or {}).get("mcnemar_exact_two_sided_p")
        if model_acc is not None and reversion_acc is not None and reversion_acc > model_acc:
            warnings.append(
                f"Independent 24h/48h benchmark: simple 48h mean-reversion ({reversion_acc:.2f}%) "
                f"beats model V2 ({model_acc:.2f}%); paired p={pval}."
            )

    conv = conviction_metrics(evaluated)
    base_acc = (conv.get("0.00") or {}).get("direction_accuracy_pct")
    high_mag_acc = (conv.get("2.00") or {}).get("direction_accuracy_pct")
    if base_acc is not None and high_mag_acc is not None and high_mag_acc < base_acc:
        warnings.append(
            f"Conviction is anti-calibrated: >=2% predicted moves score {high_mag_acc:.2f}% "
            f"versus {base_acc:.2f}% overall."
        )

    overlap_ratio = None
    if evaluated:
        overlap_ratio = 1.0 - (len(independent) / len(evaluated))

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": generated_at,
        "source_file": PREDICTIONS_PATH,
        "features_file": FEATURES_PATH if os.path.isfile(FEATURES_PATH) else None,
        "available": True,
        "method_notes": {
            "direction_definition": (
                "Predicted direction = sign(predicted_close_change_pct); "
                "actual direction = sign(actual_close_change_pct)."
            ),
            "majority_baseline": (
                "Accuracy obtained by always predicting whichever realised direction "
                "(UP or DOWN) is more common in the evaluated subset."
            ),
            "inverse_model": (
                "Diagnostic only: percentage that would be correct if every binary "
                "model direction were reversed. Do not deploy inversion solely from "
                "this statistic."
            ),
            "balanced_accuracy": (
                "Average of UP recall and DOWN recall; useful when UP/DOWN frequencies "
                "are unequal."
            ),
            "independent_sample": (
                "Greedy sample retaining predictions at least one full forecast horizon "
                "apart within each asset/horizon. This reduces obvious overlap but does "
                "not prove statistical independence."
            ),
            "conviction_thresholds": (
                "Tests direction accuracy after abstaining from predictions whose "
                "absolute predicted move is below each threshold."
            ),
        },
        "data_quality": {
            "total_rows": len(rows),
            "evaluated_usable_rows": len(evaluated),
            "feature_rows_matched_to_predictions": feature_matches,
            "independent_sample_rows": len(independent),
            "estimated_overlap_fraction": round_or_none(overlap_ratio),
            "estimated_overlap_pct": round_or_none(
                100.0 * overlap_ratio if overlap_ratio is not None else None, 2
            ),
        },
        "overall": overall,
        "overall_independent_sample": full_metrics(independent),
        "by_asset": by_asset,
        "by_horizon": by_horizon,
        "by_asset_horizon": by_asset_horizon,
        "by_confidence": by_confidence,
        "by_analogue_quality": by_analogue_quality,
        "by_model_version": by_model_version,
        "baseline_tournament_overall": {k: compact_rule_result(v) for k, v in baseline_tournament(evaluated).items()},
        "baseline_tournament_by_horizon": {
            horizon: {k: compact_rule_result(v) for k, v in baseline_tournament(group).items()}
            for horizon, group in sorted(group_by(evaluated, lambda r: r["_horizon"]).items())
        },
        "baseline_tournament_by_asset_horizon": {
            key: {k: compact_rule_result(v) for k, v in baseline_tournament(group).items()}
            for key, group in sorted(group_by(evaluated, lambda r: f'{r["_asset"]}|{r["_horizon"]}').items())
        },
        "baseline_tournament_independent_sample": {k: compact_rule_result(v) for k, v in baseline_tournament(independent).items()},
        "monthly_short_horizon_benchmark": monthly_short_horizon_benchmark(independent),
        "paired_short_horizon_benchmark_independent": paired_short_horizon_test(independent),
        "conviction_thresholds_overall": conviction_metrics(evaluated),
        "conviction_thresholds_by_horizon": {
            horizon: conviction_metrics(group)
            for horizon, group in sorted(
                group_by(evaluated, lambda r: r["_horizon"]).items()
            )
        },
        "warnings": warnings,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Wrote {OUTPUT_PATH}: "
        f"{len(evaluated)} usable evaluated rows; "
        f"{len(independent)} less-overlapping rows."
    )

    # Short human-readable console summary for GitHub Actions.
    print("\n=== Prediction Audit V2 ===")
    print(
        "Overall direction accuracy:",
        payload["overall"].get("direction_accuracy_pct"),
    )
    print(
        "Majority baseline:",
        payload["overall"].get("majority_class_baseline_pct"),
    )
    print(
        "Inverse-model diagnostic:",
        payload["overall"].get("inverse_model_accuracy_pct"),
    )
    print(
        "Balanced accuracy:",
        payload["overall"].get("balanced_accuracy_pct"),
    )
    print(
        "Independent-sample direction accuracy:",
        payload["overall_independent_sample"].get("direction_accuracy_pct"),
    )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print("-", warning)


if __name__ == "__main__":
    main()
