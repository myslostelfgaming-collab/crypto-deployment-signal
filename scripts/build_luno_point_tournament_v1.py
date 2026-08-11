#!/usr/bin/env python3
"""
Luno Point-Price Tournament V1
==============================

Historical, target-aligned diagnostic for Luno Price Predict.

Unlike the general crypto model tournament, this script does NOT score whether
BTC went up or down. It scores absolute USD price error at the competition's
actual clock times:

Weekly:
  entry cutoff   Wednesday 12:00 UTC
  settlement     Friday    12:00 UTC

Monthly:
  entry cutoff   day 14    12:00 UTC
  settlement     day 28    12:00 UTC

Because the exact Luno internal settlement feed is not available in this repo,
historical scoring uses the repo's KuCoin BTC-USDT snapshots as the settlement
proxy. This is intentionally labelled a proxy and must not be confused with the
actual Luno competition close.

Outputs
-------
- data/diagnostics/luno_point_tournament_v1.json
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

HISTORY_INDEX = os.path.join("data", "history", "index.csv")
PREDICTIONS = os.path.join("data", "model", "predictions_v1.csv")
FEATURES = os.path.join("data", "features", "features_v1.csv")
OUT_PATH = os.path.join("data", "diagnostics", "luno_point_tournament_v1.json")
SCHEMA = "luno_point_tournament_v1"

ENTRY_LOOKBACK_HOURS = 3.0
SETTLEMENT_TOLERANCE_MIN = 90.0
FEATURE_LOOKBACK_HOURS = 4.0
PREDICTION_LOOKBACK_HOURS = 4.0

# This coefficient is diagnostic only. It came from the broad 48h historical
# research sample and is NOT promoted into the live Luno recommendation.
MR48_SHRINK = 0.13


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


def parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def pct_change(a: float, b: float) -> float:
    return 100.0 * (b / a - 1.0)


def load_btc_price_points() -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if not os.path.isfile(HISTORY_INDEX):
        return points

    with open(HISTORY_INDEX, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pub = parse_iso(row.get("published_at_utc") or "")
            rel = row.get("history_file") or ""
            if pub is None or not rel or not os.path.isfile(rel):
                continue
            try:
                with open(rel, "r", encoding="utf-8") as hf:
                    payload = json.load(hf)
                btc = payload.get("btc_usdt") or {}
                close = safe_float(btc.get("close"))
                if close is None:
                    continue
                points.append({
                    "time": pub,
                    "price": close,
                    "history_file": rel,
                })
            except Exception:
                continue

    points.sort(key=lambda x: x["time"])
    return points


def load_prediction_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(PREDICTIONS):
        return rows
    with open(PREDICTIONS, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("asset") or "") != "BTC-USDT":
                continue
            created = parse_iso(row.get("created_at_utc") or "")
            horizon = safe_int(row.get("horizon_h"))
            pred_ret = safe_float(row.get("predicted_close_change_pct"))
            if created is None or horizon is None or pred_ret is None:
                continue
            rows.append({
                "time": created,
                "horizon": horizon,
                "pred_ret": pred_ret,
                "model_version": row.get("model_version"),
            })
    rows.sort(key=lambda x: x["time"])
    return rows


def load_feature_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(FEATURES):
        return rows
    with open(FEATURES, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("asset") or "") != "BTC-USDT":
                continue
            pub = parse_iso(row.get("published_at_utc") or "")
            ret48 = safe_float(row.get("ret_48h_pct"))
            if pub is None or ret48 is None:
                continue
            rows.append({"time": pub, "ret48": ret48})
    rows.sort(key=lambda x: x["time"])
    return rows


def latest_before(
    rows: List[Dict[str, Any]],
    cutoff: datetime,
    lookback_hours: float,
    predicate=None,
) -> Optional[Dict[str, Any]]:
    start = cutoff - timedelta(hours=lookback_hours)
    found = None
    for row in rows:
        t = row["time"]
        if t < start:
            continue
        if t > cutoff:
            break
        if predicate is not None and not predicate(row):
            continue
        found = row
    return found


def nearest_to(
    rows: List[Dict[str, Any]],
    target: datetime,
    tolerance_minutes: float,
) -> Optional[Dict[str, Any]]:
    tolerance = timedelta(minutes=tolerance_minutes)
    best = None
    best_diff = None
    for row in rows:
        diff = abs(row["time"] - target)
        if diff > tolerance:
            continue
        if best_diff is None or diff < best_diff:
            best = row
            best_diff = diff
    if best is None:
        return None
    out = dict(best)
    out["diff_minutes"] = best_diff.total_seconds() / 60.0
    return out


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def competition_specs(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not points:
        return []
    start = points[0]["time"].date()
    end = points[-1]["time"].date()
    specs: List[Dict[str, Any]] = []

    for d in daterange(start, end):
        # Weekly: Wednesday cutoff, Friday settlement.
        if d.weekday() == 2:
            cutoff = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
            settlement = cutoff + timedelta(days=2)
            if settlement.date() <= end:
                specs.append({
                    "challenge_type": "weekly",
                    "cutoff": cutoff,
                    "settlement": settlement,
                    "reference_horizon_h": 48,
                })

        # Monthly: day 14 cutoff, day 28 settlement.
        if d.day == 14:
            cutoff = datetime(d.year, d.month, 14, 12, tzinfo=timezone.utc)
            settlement = datetime(d.year, d.month, 28, 12, tzinfo=timezone.utc)
            if settlement.date() <= end:
                specs.append({
                    "challenge_type": "monthly",
                    "cutoff": cutoff,
                    "settlement": settlement,
                    "reference_horizon_h": 336,
                })

    specs.sort(key=lambda x: x["cutoff"])
    return specs


def build_challenge_rows(
    points: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for spec in competition_specs(points):
        cutoff = spec["cutoff"]
        settlement = spec["settlement"]
        ref_h = spec["reference_horizon_h"]

        entry = latest_before(points, cutoff, ENTRY_LOOKBACK_HOURS)
        actual = nearest_to(points, settlement, SETTLEMENT_TOLERANCE_MIN)
        if entry is None or actual is None:
            continue

        pred = latest_before(
            predictions,
            cutoff,
            PREDICTION_LOOKBACK_HOURS,
            predicate=lambda r, h=ref_h: r["horizon"] == h,
        )
        feat = latest_before(features, cutoff, FEATURE_LOOKBACK_HOURS)

        entry_price = float(entry["price"])
        actual_price = float(actual["price"])
        actual_ret = pct_change(entry_price, actual_price)
        actual_hours = (settlement - entry["time"]).total_seconds() / 3600.0

        forecasts: Dict[str, Optional[float]] = {
            "persistence": entry_price,
            "similarity_v2": None,
            "mean_reversion_48h_full": None,
            "mean_reversion_48h_shrunk_0p13": None,
        }

        if pred is not None:
            forecasts["similarity_v2"] = entry_price * (1.0 + pred["pred_ret"] / 100.0)
        if feat is not None:
            forecasts["mean_reversion_48h_full"] = entry_price * (1.0 - feat["ret48"] / 100.0)
            forecasts["mean_reversion_48h_shrunk_0p13"] = entry_price * (
                1.0 - MR48_SHRINK * feat["ret48"] / 100.0
            )

        errors = {
            name: abs(value - actual_price) if value is not None else None
            for name, value in forecasts.items()
        }

        out.append({
            "challenge_type": spec["challenge_type"],
            "cutoff_utc": cutoff.isoformat(),
            "settlement_utc": settlement.isoformat(),
            "reference_horizon_h": ref_h,
            "entry_snapshot_utc": entry["time"].isoformat(),
            "entry_snapshot_minutes_before_cutoff": round(
                (cutoff - entry["time"]).total_seconds() / 60.0, 2
            ),
            "entry_price": round(entry_price, 4),
            "settlement_proxy_snapshot_utc": actual["time"].isoformat(),
            "settlement_proxy_diff_minutes": round(actual["diff_minutes"], 2),
            "settlement_proxy_price": round(actual_price, 4),
            "actual_return_pct": round(actual_ret, 6),
            "actual_hours_from_entry_snapshot": round(actual_hours, 4),
            "horizon_alignment_gap_hours": round(actual_hours - ref_h, 4),
            "similarity_prediction_created_utc": pred["time"].isoformat() if pred else None,
            "feature_snapshot_utc": feat["time"].isoformat() if feat else None,
            "ret48_at_entry_pct": round_or_none(feat["ret48"] if feat else None, 6),
            "forecasts": {k: round_or_none(v, 4) for k, v in forecasts.items()},
            "absolute_errors": {k: round_or_none(v, 4) for k, v in errors.items()},
        })

    return out


def metric_for_model(rows: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    vals: List[Tuple[float, float]] = []
    wins = 0
    for row in rows:
        err = (row.get("absolute_errors") or {}).get(model)
        persist = (row.get("absolute_errors") or {}).get("persistence")
        if err is None:
            continue
        vals.append((float(err), float(persist) if persist is not None else float("nan")))
        if persist is not None and err < persist:
            wins += 1

    errs = [v[0] for v in vals]
    paired_persist = [v[1] for v in vals if math.isfinite(v[1])]
    return {
        "n": len(errs),
        "mae_usd": round_or_none(mean(errs) if errs else None, 4),
        "median_ae_usd": round_or_none(median(errs) if errs else None, 4),
        "wins_vs_persistence": wins,
        "win_rate_vs_persistence_pct": round_or_none(100.0 * wins / len(errs) if errs else None, 2),
        "paired_persistence_mae_usd": round_or_none(mean(paired_persist) if paired_persist else None, 4),
        "mae_improvement_vs_persistence_pct": round_or_none(
            100.0 * (mean(paired_persist) - mean(errs)) / mean(paired_persist)
            if errs and paired_persist and mean(paired_persist) != 0
            else None,
            4,
        ),
    }


def summarize(rows: List[Dict[str, Any]], challenge_type: str) -> Dict[str, Any]:
    subset = [r for r in rows if r["challenge_type"] == challenge_type]
    models = [
        "persistence",
        "similarity_v2",
        "mean_reversion_48h_full",
        "mean_reversion_48h_shrunk_0p13",
    ]
    metrics = {m: metric_for_model(subset, m) for m in models}

    gaps = [abs(float(r["horizon_alignment_gap_hours"])) for r in subset]
    settle_diffs = [float(r["settlement_proxy_diff_minutes"]) for r in subset]

    # Conservative historical champion rule: a challenger must have at least 10
    # paired observations and reduce MAE by >=5% versus persistence. This does
    # not promote it live; it merely calls it a historical challenger.
    champion = "persistence"
    qualifying = []
    for model in models[1:]:
        m = metrics[model]
        if (m.get("n") or 0) >= 10 and (m.get("mae_improvement_vs_persistence_pct") or -999) >= 5.0:
            qualifying.append((m["mae_usd"], model))
    if qualifying:
        qualifying.sort()
        champion = qualifying[0][1]

    return {
        "n_challenges_with_proxy_settlement": len(subset),
        "historical_proxy_champion": champion,
        "promotion_status": (
            "PERSISTENCE_CHAMPION" if champion == "persistence" else "HISTORICAL_CHALLENGER_ONLY"
        ),
        "models": metrics,
        "alignment": {
            "median_absolute_gap_vs_generic_horizon_hours": round_or_none(median(gaps) if gaps else None, 4),
            "median_settlement_proxy_timestamp_error_minutes": round_or_none(median(settle_diffs) if settle_diffs else None, 2),
        },
    }


def main() -> None:
    points = load_btc_price_points()
    predictions = load_prediction_rows()
    features = load_feature_rows()
    rows = build_challenge_rows(points, predictions, features)

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY",
        "objective": "minimise absolute BTC USD price error at Luno Price Predict settlement time",
        "settlement_truth_status": "KUCOIN_BTC_USDT_PROXY_NOT_LUNO_INTERNAL_RATE",
        "schedule": {
            "weekly": {
                "submission_cutoff_utc": "Wednesday 12:00",
                "settlement_utc": "Friday 12:00",
                "generic_reference_horizon_h": 48,
            },
            "monthly": {
                "submission_cutoff_utc": "day 14 12:00",
                "settlement_utc": "day 28 12:00",
                "generic_reference_horizon_h": 336,
            },
        },
        "method": {
            "entry_proxy": f"latest KuCoin BTC-USDT repo snapshot within {ENTRY_LOOKBACK_HOURS}h before submission cutoff",
            "settlement_proxy": f"nearest KuCoin BTC-USDT repo snapshot within {SETTLEMENT_TOLERANCE_MIN} minutes of settlement",
            "warning": "Historical results measure target-aligned market-price forecasting, not exact Luno internal-rate accuracy.",
            "mean_reversion_shrink_note": (
                "0.13 coefficient is research-only and was discovered on prior broad 48h history; it must not be treated as an independent validation result."
            ),
        },
        "data_quality": {
            "btc_price_points": len(points),
            "btc_prediction_rows": len(predictions),
            "btc_feature_rows": len(features),
            "challenge_rows": len(rows),
        },
        "weekly": summarize(rows, "weekly"),
        "monthly": summarize(rows, "monthly"),
        "challenge_rows": rows,
        "promotion_rule": {
            "historical_label_only": True,
            "minimum_paired_observations": 10,
            "minimum_mae_improvement_vs_persistence_pct": 5.0,
            "live_promotion_requires_prospective_shadow_evidence": True,
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Wrote", OUT_PATH)
    for name in ("weekly", "monthly"):
        block = payload[name]
        print(
            name,
            "n=", block["n_challenges_with_proxy_settlement"],
            "champion=", block["historical_proxy_champion"],
            "persistence_mae=", block["models"]["persistence"].get("mae_usd"),
            "similarity_mae=", block["models"]["similarity_v2"].get("mae_usd"),
        )


if __name__ == "__main__":
    main()
