#!/usr/bin/env python

import csv
import json
import os
from glob import glob
from datetime import datetime, timezone

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
HISTORY_ROOT = os.path.join("data", "history")

# Accept nearest candle within ±45 minutes
TOLERANCE_SECONDS = 45 * 60


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


def extract_eth_candles(snapshot: dict):
    candles = snapshot.get("candles", {}).get("eth_usdt_1h")
    if candles is None:
        candles = snapshot.get("signal", {}).get("candles", {}).get("eth_usdt_1h", [])
    if not isinstance(candles, list):
        return []
    return candles


def build_eth_close_map():
    """
    Build:
      ts_utc -> ETH close
    from all history snapshots.

    Expected compact format:
      [ts_utc, open, high, low, close, volume]
    """
    close_map = {}

    for path in iter_history_files():
        try:
            snap = load_json(path)
        except Exception:
            continue

        candles = extract_eth_candles(snap)
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


def find_nearest_close(target_ts: int, close_map: dict, tolerance_seconds: int = TOLERANCE_SECONDS):
    """
    Find the nearest available candle close to target_ts,
    but only if within tolerance_seconds.
    Returns:
      matched_ts, matched_close, diff_seconds
    or:
      None, None, None
    """
    if not close_map:
        return None, None, None

    best_ts = None
    best_diff = None

    for ts in close_map.keys():
        diff = abs(ts - target_ts)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_ts = ts

    if best_ts is None or best_diff is None:
        return None, None, None

    if best_diff > tolerance_seconds:
        return None, None, None

    return best_ts, close_map[best_ts], best_diff


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
    already_evaluated = 0
    pending_not_matured = 0
    pending_no_match = 0
    still_pending_other_asset = 0

    now_dt = datetime.now(timezone.utc)
    now_ts = int(now_dt.timestamp())
    now_utc = now_dt.isoformat()

    for row in rows:
        status = row.get("status", "")
        if status == "evaluated":
            already_evaluated += 1
            continue

        asset = row.get("asset")
        if asset != "ETH-USDT":
            still_pending_other_asset += 1
            continue

        target_ts_utc = row.get("target_ts_utc")
        entry_close = safe_float(row.get("entry_close"))
        pred_price = safe_float(row.get("predicted_price"))
        pred_pct = safe_float(row.get("predicted_close_change_pct"))

        if not target_ts_utc or entry_close is None:
            pending_no_match += 1
            continue

        try:
            target_dt = datetime.fromisoformat(target_ts_utc)
            target_ts = int(target_dt.timestamp())
        except Exception:
            pending_no_match += 1
            continue

        # Do not evaluate before the target time has actually arrived
        if now_ts < target_ts:
            pending_not_matured += 1
            continue

        matched_ts, actual_close, diff_seconds = find_nearest_close(target_ts, close_map, TOLERANCE_SECONDS)
        if actual_close is None:
            pending_no_match += 1
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

        # Optional debug fields if they already exist in the CSV header
        if "matched_target_ts_utc" in fieldnames:
            row["matched_target_ts_utc"] = datetime.fromtimestamp(matched_ts, tz=timezone.utc).isoformat()
        if "matched_target_diff_seconds" in fieldnames:
            row["matched_target_diff_seconds"] = str(diff_seconds)

        updated += 1

    with open(PREDICTIONS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predictions evaluated this run: {updated}")
    print(f"Already evaluated: {already_evaluated}")
    print(f"Pending (target time not yet reached): {pending_not_matured}")
    print(f"Pending (no candle within tolerance): {pending_no_match}")
    print(f"Pending (other asset): {still_pending_other_asset}")
    print(f"Tolerance used: ±{TOLERANCE_SECONDS} seconds")
    print(f"ETH close map size: {len(close_map)}")
    print(f"File updated: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()