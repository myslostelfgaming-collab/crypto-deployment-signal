#!/usr/bin/env python3
"""
Prospective Shadow Challengers V1
=================================

Starts a forward-only shadow scoreboard for transparent challenger rules.
It intentionally does NOT backfill old predictions: rows are created only for
the latest prediction batch seen when this script runs, then evaluated later
when predictions_v1 reports the realised outcome.

No shadow rule changes the live V2 forecast or actionability state.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PREDICTIONS_PATH = Path("data/model/predictions_v1.csv")
FEATURES_PATH = Path("data/features/features_v1.csv")
SHADOW_PATH = Path("data/model/shadow_predictions_v1.csv")
PERF_PATH = Path("data/model/shadow_challenger_performance_v1.json")

RULE_COLUMNS = [
    "similarity_v2_direction",
    "mean_reversion_48h_direction",
    "mean_reversion_24h_direction",
    "sma24_trend_direction",
    "sma48_trend_direction",
]

FIELDS = [
    "created_at_utc",
    "asset",
    "entry_ts_utc",
    "entry_close",
    "horizon_h",
    "target_ts_utc",
    "regime",
    "ret_24h_pct",
    "ret_48h_pct",
    "close_vs_sma_24_pct",
    "close_vs_sma_48_pct",
    "atr14_pct",
    *RULE_COLUMNS,
    "status",
    "actual_close_change_pct",
    "actual_direction",
    "evaluated_at_utc",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def direction(x: Optional[float]) -> int:
    if x is None:
        return 0
    if x > 1e-12:
        return 1
    if x < -1e-12:
        return -1
    return 0


def classify_regime(feat: Dict[str, str]) -> str:
    ret48 = safe_float(feat.get("ret_48h_pct"))
    sma48 = safe_float(feat.get("close_vs_sma_48_pct"))
    atr = safe_float(feat.get("atr14_pct"))
    if ret48 is None or sma48 is None or atr is None or atr <= 0:
        return "unknown"
    if direction(ret48) == direction(sma48) != 0 and abs(ret48) >= 2.0 * atr:
        return "trend_up" if ret48 > 0 else "trend_down"
    return "range"


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def feature_index() -> Dict[Tuple[str, int], Dict[str, str]]:
    idx: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in load_csv(FEATURES_PATH):
        asset = row.get("asset") or "UNKNOWN"
        ts = safe_int(row.get("entry_ts_utc"))
        if ts is None:
            continue
        key = (asset, ts)
        current = idx.get(key)
        if current is None or (row.get("published_at_utc") or "") >= (current.get("published_at_utc") or ""):
            idx[key] = row
    return idx


def write_shadow(rows: List[Dict[str, Any]]) -> None:
    SHADOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SHADOW_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})


def build_performance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in rows if (r.get("status") or "").lower() == "evaluated"]

    def summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n_evaluated": len(items), "rules": {}}
        for rule in RULE_COLUMNS:
            valid = []
            for r in items:
                pred = safe_int(r.get(rule))
                actual = safe_int(r.get("actual_direction"))
                if pred in (-1, 1) and actual in (-1, 1):
                    valid.append((pred, actual))
            n = len(valid)
            correct = sum(p == a for p, a in valid)
            out["rules"][rule] = {
                "n": n,
                "correct": correct,
                "accuracy_pct": round(100 * correct / n, 4) if n else None,
            }
        return out

    by_horizon = {}
    for h in (24, 48, 168, 336):
        by_horizon[str(h)] = summary([r for r in evaluated if safe_int(r.get("horizon_h")) == h])

    by_asset_horizon = {}
    keys = sorted({(r.get("asset"), safe_int(r.get("horizon_h"))) for r in evaluated})
    for asset, h in keys:
        if asset and h:
            by_asset_horizon[f"{asset}|{h}"] = summary(
                [r for r in evaluated if r.get("asset") == asset and safe_int(r.get("horizon_h")) == h]
            )

    created_times = [r.get("created_at_utc") for r in rows if r.get("created_at_utc")]
    return {
        "schema": "shadow_challenger_performance_v1",
        "generated_at_utc": now(),
        "status": "PROSPECTIVE_SHADOW_ONLY",
        "shadow_start_utc": min(created_times) if created_times else None,
        "rows_total": len(rows),
        "rows_pending": sum((r.get("status") or "").lower() != "evaluated" for r in rows),
        "rows_evaluated": len(evaluated),
        "overall": summary(evaluated),
        "by_horizon": by_horizon,
        "by_asset_horizon": by_asset_horizon,
        "promotion_warning": (
            "Shadow results are prospective evidence. Do not promote a challenger until "
            "there is adequate matured sample size across more than one market regime."
        ),
    }


def main() -> None:
    predictions = load_csv(PREDICTIONS_PATH)
    if not predictions:
        raise SystemExit(f"Missing/empty {PREDICTIONS_PATH}")

    features = feature_index()
    shadow: List[Dict[str, Any]] = [dict(r) for r in load_csv(SHADOW_PATH)]
    by_key = {
        (r.get("asset"), safe_int(r.get("entry_ts_utc")), safe_int(r.get("horizon_h"))): r
        for r in shadow
    }

    # First, mature any already-recorded prospective rows using the canonical
    # evaluator output from predictions_v1.
    pred_index = {}
    for p in predictions:
        key = (p.get("asset"), safe_int(p.get("entry_ts_utc")), safe_int(p.get("horizon_h")))
        pred_index[key] = p

    matured_now = 0
    for key, row in by_key.items():
        p = pred_index.get(key)
        if not p or (p.get("status") or "").lower() != "evaluated":
            continue
        if (row.get("status") or "").lower() == "evaluated":
            continue
        actual_ret = safe_float(p.get("actual_close_change_pct"))
        if actual_ret is None:
            continue
        row["status"] = "evaluated"
        row["actual_close_change_pct"] = actual_ret
        row["actual_direction"] = direction(actual_ret)
        row["evaluated_at_utc"] = p.get("evaluated_at_utc") or now()
        matured_now += 1

    # Add ONLY each asset's latest batch. This is what makes the record
    # genuinely prospective rather than a disguised historical backfill.
    latest_ts: Dict[str, int] = {}
    for p in predictions:
        asset = p.get("asset") or "UNKNOWN"
        ts = safe_int(p.get("entry_ts_utc"))
        if ts is not None and ts > latest_ts.get(asset, -1):
            latest_ts[asset] = ts

    added = 0
    created = now()
    for p in predictions:
        asset = p.get("asset") or "UNKNOWN"
        ts = safe_int(p.get("entry_ts_utc"))
        h = safe_int(p.get("horizon_h"))
        if ts is None or h is None or ts != latest_ts.get(asset):
            continue
        key = (asset, ts, h)
        if key in by_key:
            continue
        feat = features.get((asset, ts))
        if not feat:
            continue

        ret24 = safe_float(feat.get("ret_24h_pct"))
        ret48 = safe_float(feat.get("ret_48h_pct"))
        sma24 = safe_float(feat.get("close_vs_sma_24_pct"))
        sma48 = safe_float(feat.get("close_vs_sma_48_pct"))
        v2_ret = safe_float(p.get("predicted_close_change_pct"))

        row: Dict[str, Any] = {
            "created_at_utc": created,
            "asset": asset,
            "entry_ts_utc": ts,
            "entry_close": p.get("entry_close") or feat.get("entry_close"),
            "horizon_h": h,
            "target_ts_utc": p.get("target_ts_utc"),
            "regime": classify_regime(feat),
            "ret_24h_pct": ret24,
            "ret_48h_pct": ret48,
            "close_vs_sma_24_pct": sma24,
            "close_vs_sma_48_pct": sma48,
            "atr14_pct": safe_float(feat.get("atr14_pct")),
            "similarity_v2_direction": direction(v2_ret),
            "mean_reversion_48h_direction": -direction(ret48),
            "mean_reversion_24h_direction": -direction(ret24),
            "sma24_trend_direction": direction(sma24),
            "sma48_trend_direction": direction(sma48),
            "status": "pending",
            "actual_close_change_pct": "",
            "actual_direction": "",
            "evaluated_at_utc": "",
        }
        shadow.append(row)
        by_key[key] = row
        added += 1

    shadow.sort(key=lambda r: (safe_int(r.get("entry_ts_utc")) or 0, r.get("asset") or "", safe_int(r.get("horizon_h")) or 0))
    write_shadow(shadow)
    perf = build_performance(shadow)
    PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERF_PATH.write_text(json.dumps(perf, indent=2), encoding="utf-8")

    print(f"Shadow challenger log: {len(shadow)} rows; added={added}; matured_now={matured_now}")
    print(f"Wrote {SHADOW_PATH}")
    print(f"Wrote {PERF_PATH}")


if __name__ == "__main__":
    main()
