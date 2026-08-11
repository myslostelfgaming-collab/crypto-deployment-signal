#!/usr/bin/env python3
"""
Model Tournament V1
===================

Diagnostic-only, walk-forward-safe directional model tournament.

This script does NOT replace similarity_forecast_v2 and does NOT alter live
predictions. It asks a simpler question first: which transparent directional
rules have historically worked, on which horizon and in which market state?

Inputs
------
- data/model/predictions_v1.csv
- data/features/features_v1.csv

Output
------
- data/diagnostics/model_tournament_v1.json

Key safeguards
--------------
1. Every challenger uses only features that were available at prediction time.
2. The adaptive selector only learns from outcomes whose target timestamp had
   already matured before the new prediction timestamp.
3. Selector training uses a less-overlapping history (entries spaced by at
   least one forecast horizon) to reduce false confidence from hourly overlap.
4. The existing V2 prediction is kept as a contestant, not silently replaced.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
FEATURES_PATH = os.path.join("data", "features", "features_v1.csv")
OUT_PATH = os.path.join("data", "diagnostics", "model_tournament_v1.json")

SCHEMA = "model_tournament_v1"
SHORT_HORIZONS = {24, 48}

RULES = [
    "similarity_v2",
    "momentum_24h",
    "momentum_48h",
    "mean_reversion_24h",
    "mean_reversion_48h",
    "sma24_trend",
    "sma48_trend",
    "sma24_reversion",
    "sma48_reversion",
]

# Adaptive selector safeguards. These are fixed in advance; the selector does
# not optimise these thresholds using future outcomes.
MIN_REGIME_TRAIN = 12
MIN_FALLBACK_TRAIN = 20
PRIOR_STRENGTH = 10.0  # shrink small samples toward 50%


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def sign(value: Optional[float], eps: float = 1e-12) -> int:
    if value is None:
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def pct(n: float, d: float) -> Optional[float]:
    if not d:
        return None
    return round(100.0 * n / d, 4)


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def parse_iso_epoch(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def load_features(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Keep the latest feature row for each asset+entry timestamp."""
    latest: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return latest

    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            asset = (row.get("asset") or "UNKNOWN").strip()
            ts = safe_int(row.get("entry_ts_utc"))
            if ts is None:
                continue
            key = (asset, ts)
            published = row.get("published_at_utc") or ""
            current = latest.get(key)
            if current is None or published >= (current.get("published_at_utc") or ""):
                latest[key] = dict(row)
    return latest


def classify_regime(row: Dict[str, Any]) -> str:
    """
    Transparent, contemporaneous market-state classifier.

    'trend_up'/'trend_down' requires:
    - 48h return and distance from 48h SMA to agree in direction, and
    - the 48h return magnitude to be at least 2x current ATR14%.

    Everything else is called 'range'. No future data or fitted threshold is
    used here.
    """
    ret48 = safe_float(row.get("ret_48h_pct"))
    sma48 = safe_float(row.get("close_vs_sma_48_pct"))
    atr = safe_float(row.get("atr14_pct"))

    if ret48 is None or sma48 is None or atr is None or atr <= 0:
        return "unknown"

    ret_sign = sign(ret48)
    sma_sign = sign(sma48)
    is_stretched = abs(ret48) >= 2.0 * atr

    if ret_sign != 0 and ret_sign == sma_sign and is_stretched:
        return "trend_up" if ret_sign > 0 else "trend_down"
    return "range"


def rule_predictions(row: Dict[str, Any]) -> Dict[str, int]:
    ret24 = safe_float(row.get("ret_24h_pct"))
    ret48 = safe_float(row.get("ret_48h_pct"))
    sma24 = safe_float(row.get("close_vs_sma_24_pct"))
    sma48 = safe_float(row.get("close_vs_sma_48_pct"))
    v2_ret = safe_float(row.get("predicted_close_change_pct"))

    return {
        "similarity_v2": sign(v2_ret),
        "momentum_24h": sign(ret24),
        "momentum_48h": sign(ret48),
        "mean_reversion_24h": -sign(ret24),
        "mean_reversion_48h": -sign(ret48),
        "sma24_trend": sign(sma24),
        "sma48_trend": sign(sma48),
        "sma24_reversion": -sign(sma24),
        "sma48_reversion": -sign(sma48),
    }


def load_rows() -> Tuple[List[Dict[str, Any]], int]:
    features = load_features(FEATURES_PATH)
    rows: List[Dict[str, Any]] = []
    matched = 0

    with open(PREDICTIONS_PATH, "r", encoding="utf-8", newline="") as f:
        for pred in csv.DictReader(f):
            if (pred.get("status") or "").strip().lower() != "evaluated":
                continue

            actual_ret = safe_float(pred.get("actual_close_change_pct"))
            pred_ret = safe_float(pred.get("predicted_close_change_pct"))
            asset = (pred.get("asset") or "UNKNOWN").strip()
            entry_ts = safe_int(pred.get("entry_ts_utc"))
            horizon = safe_int(pred.get("horizon_h"))

            if actual_ret is None or pred_ret is None or entry_ts is None or horizon is None:
                continue

            feat = features.get((asset, entry_ts))
            if feat is None:
                continue
            matched += 1

            target_epoch = parse_iso_epoch(pred.get("matched_target_ts_iso") or "")
            if target_epoch is None:
                target_epoch = parse_iso_epoch(pred.get("target_ts_utc") or "")
            if target_epoch is None:
                target_epoch = entry_ts + horizon * 3600

            merged: Dict[str, Any] = dict(feat)
            merged.update(pred)
            merged["asset"] = asset
            merged["entry_ts"] = entry_ts
            merged["horizon_h_int"] = horizon
            merged["target_epoch"] = target_epoch
            merged["actual_direction"] = sign(actual_ret)
            merged["regime"] = classify_regime(merged)
            merged["rule_predictions"] = rule_predictions(merged)
            rows.append(merged)

    rows.sort(key=lambda r: (r["entry_ts"], r["asset"], r["horizon_h_int"]))
    return rows, matched


def rule_metrics(rows: Iterable[Dict[str, Any]], rule: str) -> Dict[str, Any]:
    total = correct = up_calls = down_calls = 0
    actual_up = actual_down = 0

    for r in rows:
        actual = r.get("actual_direction", 0)
        pred = (r.get("rule_predictions") or {}).get(rule, 0)
        if actual not in (-1, 1) or pred not in (-1, 1):
            continue
        total += 1
        correct += int(actual == pred)
        up_calls += int(pred == 1)
        down_calls += int(pred == -1)
        actual_up += int(actual == 1)
        actual_down += int(actual == -1)

    return {
        "n": total,
        "correct": correct,
        "accuracy_pct": pct(correct, total),
        "predicted_up": up_calls,
        "predicted_down": down_calls,
        "actual_up": actual_up,
        "actual_down": actual_down,
    }


def tournament(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {rule: rule_metrics(rows, rule) for rule in RULES}
    ranked = sorted(
        [
            {"rule": rule, **m}
            for rule, m in metrics.items()
            if m.get("accuracy_pct") is not None
        ],
        key=lambda x: (-x["accuracy_pct"], -x["n"], x["rule"]),
    )
    return {"rules": metrics, "ranking": ranked}


def independent_sample(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedy horizon-spaced sample within each asset+horizon."""
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["asset"], r["horizon_h_int"])].append(r)

    out: List[Dict[str, Any]] = []
    for (_, horizon), group in groups.items():
        group.sort(key=lambda r: r["entry_ts"])
        spacing = horizon * 3600
        last: Optional[int] = None
        for r in group:
            if last is None or r["entry_ts"] - last >= spacing:
                out.append(r)
                last = r["entry_ts"]
    return sorted(out, key=lambda r: (r["entry_ts"], r["asset"], r["horizon_h_int"]))


def shrinkage_score(correct: int, n: int) -> float:
    return (correct + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)


def choose_best_rule(train_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for priority, rule in enumerate(RULES):
        m = rule_metrics(train_rows, rule)
        n = int(m["n"])
        if n <= 0:
            continue
        score = shrinkage_score(int(m["correct"]), n)
        candidate = {
            "rule": rule,
            "n": n,
            "accuracy_pct": m["accuracy_pct"],
            "shrinkage_score": score,
            "priority": priority,
        }
        if best is None:
            best = candidate
            continue
        if score > best["shrinkage_score"] + 1e-12:
            best = candidate
        elif abs(score - best["shrinkage_score"]) <= 1e-12:
            if n > best["n"] or (n == best["n"] and priority < best["priority"]):
                best = candidate
    return best


def walk_forward_selector(all_rows: List[Dict[str, Any]], independent: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    At each test timestamp, select the historically best rule using only
    independent rows whose realised target had already matured.
    """
    # Organise independent training rows for fast filtering.
    train_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in independent:
        train_groups[(r["asset"], r["horizon_h_int"])].append(r)
    for group in train_groups.values():
        group.sort(key=lambda r: r["target_epoch"])

    results: List[Dict[str, Any]] = []

    # Evaluate selector on the cleaner independent sample itself. A test row can
    # never train on itself because its target_epoch is after its entry_ts.
    for test in independent:
        key = (test["asset"], test["horizon_h_int"])
        history = [r for r in train_groups.get(key, []) if r["target_epoch"] <= test["entry_ts"]]
        same_regime = [r for r in history if r.get("regime") == test.get("regime")]

        scope = None
        training = None
        if len(same_regime) >= MIN_REGIME_TRAIN:
            scope = "same_regime"
            training = same_regime
        elif len(history) >= MIN_FALLBACK_TRAIN:
            scope = "asset_horizon_fallback"
            training = history
        else:
            continue

        chosen = choose_best_rule(training)
        if not chosen:
            continue

        chosen_rule = chosen["rule"]
        pred_dir = test["rule_predictions"].get(chosen_rule, 0)
        actual = test["actual_direction"]
        if pred_dir not in (-1, 1) or actual not in (-1, 1):
            continue

        results.append({
            "asset": test["asset"],
            "horizon_h": test["horizon_h_int"],
            "entry_ts": test["entry_ts"],
            "regime": test["regime"],
            "selection_scope": scope,
            "selected_rule": chosen_rule,
            "selected_rule_train_n": chosen["n"],
            "selected_rule_train_accuracy_pct": chosen["accuracy_pct"],
            "selected_rule_shrinkage_score": round(chosen["shrinkage_score"], 6),
            "predicted_direction": pred_dir,
            "actual_direction": actual,
            "correct": pred_dir == actual,
            "v2_direction": test["rule_predictions"].get("similarity_v2", 0),
            "mr48_direction": test["rule_predictions"].get("mean_reversion_48h", 0),
        })

    def selector_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        correct = sum(bool(x["correct"]) for x in items)
        v2_valid = [x for x in items if x["v2_direction"] in (-1, 1)]
        v2_correct = sum(x["v2_direction"] == x["actual_direction"] for x in v2_valid)
        mr_valid = [x for x in items if x["mr48_direction"] in (-1, 1)]
        mr_correct = sum(x["mr48_direction"] == x["actual_direction"] for x in mr_valid)

        rule_usage: Dict[str, int] = defaultdict(int)
        scope_usage: Dict[str, int] = defaultdict(int)
        for x in items:
            rule_usage[x["selected_rule"]] += 1
            scope_usage[x["selection_scope"]] += 1

        return {
            "n": n,
            "selector_accuracy_pct": pct(correct, n),
            "similarity_v2_accuracy_same_rows_pct": pct(v2_correct, len(v2_valid)),
            "mean_reversion_48h_accuracy_same_rows_pct": pct(mr_correct, len(mr_valid)),
            "edge_vs_v2_pp": round_or_none(
                (100.0 * correct / n) - (100.0 * v2_correct / len(v2_valid))
                if n and v2_valid else None,
                4,
            ),
            "edge_vs_mr48_pp": round_or_none(
                (100.0 * correct / n) - (100.0 * mr_correct / len(mr_valid))
                if n and mr_valid else None,
                4,
            ),
            "selected_rule_usage": dict(sorted(rule_usage.items())),
            "selection_scope_usage": dict(sorted(scope_usage.items())),
        }

    by_horizon: Dict[str, Any] = {}
    for h in sorted({x["horizon_h"] for x in results}):
        by_horizon[str(h)] = selector_summary([x for x in results if x["horizon_h"] == h])

    by_asset_horizon: Dict[str, Any] = {}
    for asset, h in sorted({(x["asset"], x["horizon_h"]) for x in results}):
        by_asset_horizon[f"{asset}|{h}"] = selector_summary(
            [x for x in results if x["asset"] == asset and x["horizon_h"] == h]
        )

    short = [x for x in results if x["horizon_h"] in SHORT_HORIZONS]

    return {
        "method": {
            "evaluation_sample": "horizon-spaced independent sample",
            "maturity_rule": "training row target_epoch <= test row entry_epoch",
            "regime_training_min_n": MIN_REGIME_TRAIN,
            "fallback_training_min_n": MIN_FALLBACK_TRAIN,
            "selection_score": (
                "shrunken accuracy = (correct + 0.5*prior_strength) / "
                "(n + prior_strength)"
            ),
            "prior_strength": PRIOR_STRENGTH,
        },
        "overall": selector_summary(results),
        "short_horizons_24_48": selector_summary(short),
        "by_horizon": by_horizon,
        "by_asset_horizon": by_asset_horizon,
        "rows": results,
    }


def grouped_tournaments(rows: List[Dict[str, Any]], independent: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["overall_full"] = tournament(rows)
    out["overall_independent"] = tournament(independent)

    out["by_horizon_full"] = {}
    out["by_horizon_independent"] = {}
    horizons = sorted({r["horizon_h_int"] for r in rows})
    for h in horizons:
        out["by_horizon_full"][str(h)] = tournament([r for r in rows if r["horizon_h_int"] == h])
        out["by_horizon_independent"][str(h)] = tournament(
            [r for r in independent if r["horizon_h_int"] == h]
        )

    out["by_asset_horizon_independent"] = {}
    keys = sorted({(r["asset"], r["horizon_h_int"]) for r in independent})
    for asset, h in keys:
        out["by_asset_horizon_independent"][f"{asset}|{h}"] = tournament(
            [r for r in independent if r["asset"] == asset and r["horizon_h_int"] == h]
        )

    out["by_horizon_regime_full"] = {}
    for h in horizons:
        for regime in ("range", "trend_up", "trend_down", "unknown"):
            subset = [
                r for r in rows
                if r["horizon_h_int"] == h and r.get("regime") == regime
            ]
            if subset:
                out["by_horizon_regime_full"][f"{h}|{regime}"] = tournament(subset)

    return out


def warnings(payload: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    wf = payload.get("walk_forward_selector", {})
    short = wf.get("short_horizons_24_48", {})
    n = short.get("n") or 0
    acc = short.get("selector_accuracy_pct")
    v2 = short.get("similarity_v2_accuracy_same_rows_pct")
    mr = short.get("mean_reversion_48h_accuracy_same_rows_pct")

    if n < 100:
        notes.append("Walk-forward short-horizon selector has fewer than 100 eligible independent tests.")
    if acc is not None and v2 is not None and acc <= v2:
        notes.append("Adaptive selector does not yet beat similarity V2 on eligible short-horizon rows.")
    if acc is not None and mr is not None and acc <= mr:
        notes.append("Adaptive selector does not yet beat the fixed 48h mean-reversion benchmark.")
    if acc is not None and acc < 55:
        notes.append("Adaptive selector remains below the provisional 55% directional promotion floor.")
    return notes


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()

    if not os.path.isfile(PREDICTIONS_PATH) or not os.path.isfile(FEATURES_PATH):
        payload = {
            "schema": SCHEMA,
            "generated_at_utc": generated,
            "available": False,
            "reason": "predictions_v1.csv or features_v1.csv missing",
        }
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(payload, indent=2))
        return

    rows, matched = load_rows()
    independent = independent_sample(rows)
    grouped = grouped_tournaments(rows, independent)
    selector = walk_forward_selector(rows, independent)

    short_independent = [r for r in independent if r["horizon_h_int"] in SHORT_HORIZONS]

    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": generated,
        "available": True,
        "status": "EXPERIMENTAL_DIAGNOSTIC_ONLY",
        "source_files": {
            "predictions": PREDICTIONS_PATH,
            "features": FEATURES_PATH,
        },
        "data_quality": {
            "evaluated_feature_matched_rows": matched,
            "independent_rows": len(independent),
            "short_horizon_independent_rows": len(short_independent),
        },
        "regime_definition": {
            "trend": (
                "sign(ret_48h_pct) == sign(close_vs_sma_48_pct) and "
                "abs(ret_48h_pct) >= 2 * atr14_pct"
            ),
            "trend_direction": "trend_up or trend_down from sign(ret_48h_pct)",
            "range": "all other feature-complete states",
            "fitted_from_future_data": False,
        },
        "candidate_rules": {
            "similarity_v2": "sign(existing V2 predicted close return)",
            "momentum_24h": "sign(previous 24h return)",
            "momentum_48h": "sign(previous 48h return)",
            "mean_reversion_24h": "opposite sign of previous 24h return",
            "mean_reversion_48h": "opposite sign of previous 48h return",
            "sma24_trend": "sign(current close distance from SMA24)",
            "sma48_trend": "sign(current close distance from SMA48)",
            "sma24_reversion": "opposite sign of close distance from SMA24",
            "sma48_reversion": "opposite sign of close distance from SMA48",
        },
        "tournaments": grouped,
        "walk_forward_selector": selector,
        "promotion_rule": {
            "purpose": "guardrail only; no automatic live promotion",
            "provisional_short_horizon_floor_pct": 55.0,
            "must_beat": ["similarity_v2", "mean_reversion_48h"],
            "minimum_eligible_independent_tests": 100,
            "automatic_live_switch": False,
        },
    }
    payload["warnings"] = warnings(payload)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print("Evaluated feature-matched rows:", matched)
    print("Independent rows:", len(independent))
    print("Short independent tournament ranking:")
    short_t = tournament(short_independent)
    for item in short_t["ranking"][:6]:
        print(f"  {item['rule']}: {item['accuracy_pct']}% (n={item['n']})")
    print("Walk-forward selector short horizons:", selector["short_horizons_24_48"])


if __name__ == "__main__":
    main()
