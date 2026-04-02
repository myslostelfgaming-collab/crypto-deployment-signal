#!/usr/bin/env python

import csv
import json
import os
from glob import glob
from datetime import datetime, timezone

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
HISTORY_ROOT = os.path.join("data", "history")


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_history_files():
    pattern = os.path.join(HISTORY_ROOT, "*", "*.json")
    return sorted(glob(pattern))


def build_eth_close_map():
    """
    Build:
      ts_utc -> ETH close
    from all history snapshots.
    """
    close_map = {}

    for path in iter_history_files():
        try:
            snap = load_json(path)
        except Exception:
            continue

        candles = snap.get("candles", {}).get("eth_usdt_1h", [])
        for c in candles:
            if not isinstance(c, list) or len(c) < 5:
                continue

            try:
                ts = int(c[0])
                close = float(c[4])
            except Exception:
                continue

            if ts not in close_map:
                close_map[ts] = close

    return close_map


def classify_hit_direction(pred_pct, actual_pct):
    if pred_pct is None or actual_pct is None:
        return ""

    if pred_pct > 0 and actual_pct > 0:
        return "correct_up"
    if pred_pct < 0 and actual_pct < 0:
        return "correct_down"
    if pred_pct > 0 and actual_pct <= 0:
        return "wrong_up"
    if pred_pct < 0 and actual_pct >= 0:
        return "wrong_down"
    return "flat"


def classify_drift_tag(pred_price, actual_close, error_pct):
    if pred_price is None or actual_close is None or error_pct is None:
        return ""

    if abs(error_pct) <= 0.5:
        return "on_target"
    if abs(error_pct) <= 1.5:
        return "near_miss"

    if actual_close > pred_price:
        return "undershot"
    return "overshot"


def main():
    if not os.path.isfile(PREDICTIONS_PATH):
        print(f"Missing predictions file: {PREDICTIONS_PATH}")
        return

    close_map = build_eth_close_map()
    if not close_map:
        print("No ETH close map could be built from history.")
        return

    with open(PREDICTIONS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        print("Predictions CSV has no header.")
        return

    updated = 0
    still_pending = 0

    now_utc = datetime.now(timezone.utc).isoformat()

    for row in rows:
        status = row.get("status", "")
        if status == "evaluated":
            continue

        asset = row.get("asset")
        if asset != "ETH-USDT":
            still_pending += 1
            continue

        target_ts_utc = row.get("target_ts_utc")
        entry_close = safe_float(row.get("entry_close"))
        pred_price = safe_float(row.get("predicted_price"))
        pred_pct = safe_float(row.get("predicted_close_change_pct"))

        if not target_ts_utc or entry_close is None:
            still_pending += 1
            continue

        try:
            target_dt = datetime.fromisoformat(target_ts_utc)
            target_ts = int(target_dt.timestamp())
        except Exception:
            still_pending += 1
            continue

        actual_close = close_map.get(target_ts)
        if actual_close is None:
            still_pending += 1
            continue

        actual_pct = ((actual_close / entry_close) - 1.0) * 100.0
        error_abs = None
        error_pct = None

        if pred_price is not None:
            error_abs = actual_close - pred_price
            error_pct = ((actual_close / pred_price) - 1.0) * 100.0 if pred_price != 0 else None

        row["actual_close"] = f"{actual_close:.4f}"
        row["actual_close_change_pct"] = f"{actual_pct:.4f}"
        row["error_abs"] = "" if error_abs is None else f"{error_abs:.4f}"
        row["error_pct"] = "" if error_pct is None else f"{error_pct:.4f}"
        row["hit_direction"] = classify_hit_direction(pred_pct, actual_pct)
        row["drift_tag"] = classify_drift_tag(pred_price, actual_close, error_pct)
        row["evaluated_at_utc"] = now_utc
        row["status"] = "evaluated"

        updated += 1

    with open(PREDICTIONS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predictions evaluated: {updated}")
    print(f"Predictions still pending: {still_pending}")
    print(f"File updated: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()