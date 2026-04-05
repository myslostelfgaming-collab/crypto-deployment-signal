#!/usr/bin/env python

import csv
import json
import os
from glob import glob
from datetime import datetime, timezone

PREDICTIONS_PATH = os.path.join("data", "model", "predictions_v1.csv")
HISTORY_ROOT = os.path.join("data", "history")

ASSET_CANDLE_KEYS = {
    "ETH-USDT": "eth_usdt_1h",
    "BTC-USDT": "btc_usdt_1h",
}

# 1h candles -> allow nearest match within 60 minutes
MATCH_TOLERANCE_SECONDS = 3600


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_history_files():
    pattern = os.path.join(HISTORY_ROOT, "*", "*.json")
    return sorted(glob(pattern))


def extract_candles(snapshot, candle_key):
    candles = snapshot.get("candles", {}).get(candle_key)
    if candles is None:
        candles = snapshot.get("signal", {}).get("candles", {}).get(candle_key, [])

    if not isinstance(candles, list):
        return []

    out = []
    for c in candles:
        if not isinstance(c, list) or len(c) < 5:
            continue
        try:
            ts = int(c[0])
            # compact format in your history files is [ts, open, high, low, close, volume]
            close = float(c[4])
            out.append((ts, close))
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    return out


def build_close_map(asset):
    candle_key = ASSET_CANDLE_KEYS.get(asset)
    if not candle_key:
        return {}

    close_map = {}
    for path in iter_history_files():
        try:
            snap = load_json(path)
        except Exception:
            continue

        for ts, close in extract_candles(snap, candle_key):
            if ts not in close_map:
                close_map[ts] = close

    return dict(sorted(close_map.items()))


def nearest_timestamp(ts_list, target_ts, tolerance_seconds):
    if not ts_list:
        return None, None

    best_ts = None
    best_diff = None

    for ts in ts_list:
        diff = abs(ts - target_ts)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_ts = ts

    if best_ts is None or best_diff is None:
        return None, None

    if best_diff > tolerance_seconds:
        return None, best_diff

    return best_ts, best_diff


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


def ensure_field(fieldnames, field):
    if field not in fieldnames:
        fieldnames.append(field)
    return fieldnames


def main():
    if not os.path.isfile(PREDICTIONS_PATH):
        print(f"Missing predictions file: {PREDICTIONS_PATH}")
        return

    with open(PREDICTIONS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        print("Predictions CSV has no header.")
        return

    # add audit fields if missing
    fieldnames = ensure_field(fieldnames, "matched_target_ts_utc")
    fieldnames = ensure_field(fieldnames, "matched_target_ts_iso")
    fieldnames = ensure_field(fieldnames, "match_diff_seconds")

    close_maps = {
        asset: build_close_map(asset)
        for asset in ASSET_CANDLE_KEYS.keys()
    }
    ts_lists = {
        asset: sorted(close_maps[asset].keys())
        for asset in close_maps
    }

    now_ts = int(datetime.now(timezone.utc).timestamp())
    now_iso = datetime.now(timezone.utc).isoformat()

    updated = 0
    already_done = 0
    pending_not_mature = 0
    pending_no_map = 0
    pending_no_match = 0

    debug_examples = []

    for row in rows:
        status = (row.get("status") or "").strip()
        if status == "evaluated":
            already_done += 1
            continue

        asset = (row.get("asset") or "").strip()
        if asset not in close_maps:
            pending_no_map += 1
            continue

        entry_close = safe_float(row.get("entry_close"))
        pred_price = safe_float(row.get("predicted_price"))
        pred_pct = safe_float(row.get("predicted_close_change_pct"))
        target_ts_utc = (row.get("target_ts_utc") or "").strip()

        if entry_close is None or target_ts_utc == "":
            pending_no_match += 1
            continue

        try:
            target_dt = datetime.fromisoformat(target_ts_utc)
            target_ts = int(target_dt.timestamp())
        except Exception:
            pending_no_match += 1
            continue

        if target_ts > now_ts:
            pending_not_mature += 1
            continue

        matched_ts, diff_seconds = nearest_timestamp(
            ts_lists[asset],
            target_ts,
            MATCH_TOLERANCE_SECONDS
        )

        if matched_ts is None:
            pending_no_match += 1
            if len(debug_examples) < 10:
                debug_examples.append({
                    "asset": asset,
                    "target_ts_utc": target_ts_utc,
                    "target_ts": target_ts,
                    "reason": "no_match_within_tolerance",
                    "closest_diff_seconds": diff_seconds,
                })
            continue

        actual_close = close_maps[asset].get(matched_ts)
        if actual_close is None:
            pending_no_match += 1
            continue

        actual_pct = ((actual_close / entry_close) - 1.0) * 100.0
        error_abs = None
        error_pct = None

        if pred_price is not None and pred_price != 0:
            error_abs = actual_close - pred_price
            error_pct = ((actual_close / pred_price) - 1.0) * 100.0

        row["actual_close"] = f"{actual_close:.4f}"
        row["actual_close_change_pct"] = f"{actual_pct:.4f}"
        row["error_abs"] = "" if error_abs is None else f"{error_abs:.4f}"
        row["error_pct"] = "" if error_pct is None else f"{error_pct:.4f}"
        row["hit_direction"] = classify_hit_direction(pred_pct, actual_pct)
        row["drift_tag"] = classify_drift_tag(pred_price, actual_close, error_pct)
        row["evaluated_at_utc"] = now_iso
        row["status"] = "evaluated"
        row["matched_target_ts_utc"] = str(matched_ts)
        row["matched_target_ts_iso"] = datetime.fromtimestamp(
            matched_ts, tz=timezone.utc
        ).isoformat()
        row["match_diff_seconds"] = str(diff_seconds or 0)

        updated += 1

    with open(PREDICTIONS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=== EVALUATE PREDICTIONS V1 ===")
    print(f"Predictions file: {PREDICTIONS_PATH}")
    for asset in sorted(close_maps.keys()):
        print(f"{asset} close-map candles: {len(close_maps[asset])}")

    print(f"Already evaluated: {already_done}")
    print(f"Newly evaluated: {updated}")
    print(f"Pending not yet mature: {pending_not_mature}")
    print(f"Pending unsupported asset/no map: {pending_no_map}")
    print(f"Pending mature but no match: {pending_no_match}")

    if debug_examples:
        print("\n--- Sample unmatched matured rows ---")
        for d in debug_examples:
            print(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    main()