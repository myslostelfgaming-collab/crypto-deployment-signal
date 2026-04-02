#!/usr/bin/env python

import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

MODEL_DATASET_PATH = os.path.join("data", "model", "model_dataset_v1.csv")
HOURLY_SIGNAL_PATH = os.path.join("data", "hourly_signal.json")
OUT_PATH = os.path.join("data", "model", "similarity_forecast_v1.json")

ASSET = "ETH-USDT"
CANDLE_KEY = "eth_usdt_1h"

# Core feature columns used for similarity
FEATURE_COLS = [
    "ret_6h_pct",
    "ret_12h_pct",
    "ret_24h_pct",
    "ret_48h_pct",
    "range_24h_pct",
    "range_48h_pct",
    "atr14_pct",
    "dist_from_24h_high_pct",
    "dist_from_24h_low_pct",
    "dist_from_48h_high_pct",
    "dist_from_48h_low_pct",
    "close_vs_sma_24_pct",
    "close_vs_sma_48_pct",
]

# Binary feature columns
BINARY_COLS = [
    "is_up_24h",
    "is_up_48h",
]

TOP_K = 50
MIN_REQUIRED_NUMERIC_FEATURES = 8

# Output targets of interest for your use-case
TARGET_SUMMARY = [
    ("max_up_pct_24", "24h"),
    ("max_up_pct_48", "48h"),
    ("max_up_pct_168", "168h"),
    ("max_up_pct_336", "336h"),
    ("close_change_pct_24", "24h"),
    ("close_change_pct_48", "48h"),
    ("close_change_pct_168", "168h"),
    ("close_change_pct_336", "336h"),
    ("max_down_pct_24", "24h"),
    ("max_down_pct_48", "48h"),
    ("max_down_pct_168", "168h"),
    ("max_down_pct_336", "336h"),
]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    m = n // 2
    if n % 2 == 1:
        return xs[m]
    return (xs[m - 1] + xs[m]) / 2.0


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def stddev(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mu = mean(values)
    if mu is None:
        return None
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def read_model_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def snapshot_signal(signal_root: dict) -> dict:
    if "signal" in signal_root and isinstance(signal_root["signal"], dict):
        return signal_root["signal"]
    return signal_root


def extract_asset_candles(signal_obj: dict, candle_key: str) -> List[List[float]]:
    candles = signal_obj.get("candles", {}).get(candle_key, [])
    out = []

    for c in candles:
        if not isinstance(c, list) or len(c) < 6:
            continue
        try:
            out.append([
                int(c[0]),
                float(c[1]),  # open
                float(c[2]),  # close
                float(c[3]),  # high
                float(c[4]),  # low
                float(c[5]),  # volume
            ])
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    return out


def pct_change(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def window_last(candles: List[List[float]], n: int) -> Optional[List[List[float]]]:
    if len(candles) < n:
        return None
    return candles[-n:]


def calc_return_feature(candles: List[List[float]], lookback_h: int) -> Optional[float]:
    w = window_last(candles, lookback_h + 1)
    if not w:
        return None
    start_close = w[0][2]
    end_close = w[-1][2]
    return pct_change(end_close, start_close)


def calc_range_pct(candles: List[List[float]], lookback_h: int) -> Optional[float]:
    w = window_last(candles, lookback_h)
    if not w:
        return None
    highs = [c[3] for c in w]
    lows = [c[4] for c in w]
    entry_close = w[-1][2]
    if entry_close == 0:
        return None
    return ((max(highs) - min(lows)) / entry_close) * 100.0


def calc_dist_from_high_low(candles: List[List[float]], lookback_h: int) -> Tuple[Optional[float], Optional[float]]:
    w = window_last(candles, lookback_h)
    if not w:
        return None, None

    highs = [c[3] for c in w]
    lows = [c[4] for c in w]
    close = w[-1][2]

    high = max(highs)
    low = min(lows)

    if close == 0:
        return None, None

    dist_from_high = ((close - high) / close) * 100.0
    dist_from_low = ((close - low) / close) * 100.0
    return dist_from_high, dist_from_low


def calc_sma(candles: List[List[float]], lookback_h: int) -> Optional[float]:
    w = window_last(candles, lookback_h)
    if not w:
        return None
    closes = [c[2] for c in w]
    return sum(closes) / len(closes)


def calc_close_vs_sma(candles: List[List[float]], lookback_h: int) -> Optional[float]:
    sma = calc_sma(candles, lookback_h)
    if sma is None or sma == 0:
        return None
    close = candles[-1][2]
    return pct_change(close, sma)


def calc_wilder_atr_pct(candles: List[List[float]], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None

    trs = []
    for i in range(1, len(candles)):
        high = candles[i][3]
        low = candles[i][4]
        prev_close = candles[i - 1][2]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if len(trs) < period:
        return None

    atr = sum(trs[:period]) / period
    for j in range(period, len(trs)):
        atr = ((atr * (period - 1)) + trs[j]) / period

    close = candles[-1][2]
    if close == 0:
        return None

    return (atr / close) * 100.0


def build_current_feature_vector(signal_obj: dict, asset: str, candle_key: str) -> dict:
    candles = extract_asset_candles(signal_obj, candle_key)
    if not candles:
        raise ValueError(f"No candles found for {asset} ({candle_key}) in hourly_signal.json")

    entry_ts = candles[-1][0]
    entry_close = candles[-1][2]

    ret_6h = calc_return_feature(candles, 6)
    ret_12h = calc_return_feature(candles, 12)
    ret_24h = calc_return_feature(candles, 24)
    ret_48h = calc_return_feature(candles, 48)

    range_24h = calc_range_pct(candles, 24)
    range_48h = calc_range_pct(candles, 48)
    atr14_pct = calc_wilder_atr_pct(candles, 14)

    dist_24h_high, dist_24h_low = calc_dist_from_high_low(candles, 24)
    dist_48h_high, dist_48h_low = calc_dist_from_high_low(candles, 48)

    close_vs_sma_24 = calc_close_vs_sma(candles, 24)
    close_vs_sma_48 = calc_close_vs_sma(candles, 48)

    return {
        "asset": asset,
        "entry_ts_utc": str(entry_ts),
        "entry_close": entry_close,
        "n_candles_snapshot": len(candles),
        "ret_6h_pct": ret_6h,
        "ret_12h_pct": ret_12h,
        "ret_24h_pct": ret_24h,
        "ret_48h_pct": ret_48h,
        "range_24h_pct": range_24h,
        "range_48h_pct": range_48h,
        "atr14_pct": atr14_pct,
        "dist_from_24h_high_pct": dist_24h_high,
        "dist_from_24h_low_pct": dist_24h_low,
        "dist_from_48h_high_pct": dist_48h_high,
        "dist_from_48h_low_pct": dist_48h_low,
        "close_vs_sma_24_pct": close_vs_sma_24,
        "close_vs_sma_48_pct": close_vs_sma_48,
        "is_up_24h": "1" if (ret_24h is not None and ret_24h > 0) else "0",
        "is_up_48h": "1" if (ret_48h is not None and ret_48h > 0) else "0",
    }


def compute_feature_stats(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Standardization stats per numeric feature, restricted to the chosen asset.
    """
    stats = {}
    for col in FEATURE_COLS:
        vals = [safe_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        mu = mean(vals)
        sd = stddev(vals)
        if mu is None or sd is None:
            continue
        if sd == 0:
            sd = 1.0
        stats[col] = {"mean": mu, "std": sd}
    return stats


def similarity_distance(current: dict, row: dict, feat_stats: Dict[str, Dict[str, float]]) -> Tuple[Optional[float], int]:
    """
    Lower = more similar.
    Returns (distance, n_numeric_used)
    """
    sq = 0.0
    used_numeric = 0

    for col in FEATURE_COLS:
        cur = current.get(col)
        past = safe_float(row.get(col))
        if cur is None or past is None:
            continue
        if col not in feat_stats:
            continue
        mu = feat_stats[col]["mean"]
        sd = feat_stats[col]["std"]
        z_cur = (cur - mu) / sd
        z_past = (past - mu) / sd
        sq += (z_cur - z_past) ** 2
        used_numeric += 1

    if used_numeric < MIN_REQUIRED_NUMERIC_FEATURES:
        return None, used_numeric

    # Penalize regime mismatches a bit
    bin_penalty = 0.0
    for col in BINARY_COLS:
        cur = str(current.get(col, ""))
        past = str(row.get(col, ""))
        if cur == "" or past == "":
            continue
        if cur != past:
            bin_penalty += 0.5

    dist = math.sqrt(sq) + bin_penalty
    return dist, used_numeric


def summarize_target(rows: List[dict], field: str) -> dict:
    vals = [safe_float(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4) if vals else None,
        "median": round(median(vals), 4) if vals else None,
        "p25": round(percentile(vals, 0.25), 4) if vals else None,
        "p75": round(percentile(vals, 0.75), 4) if vals else None,
        "min": round(min(vals), 4) if vals else None,
        "max": round(max(vals), 4) if vals else None,
    }


def top_neighbors_payload(neighbors: List[dict], top_n: int = 10) -> List[dict]:
    out = []
    for r in neighbors[:top_n]:
        out.append({
            "asset": r.get("asset"),
            "entry_ts_utc": r.get("entry_ts_utc"),
            "published_at_utc": r.get("published_at_utc"),
            "similarity_distance": round(float(r["_distance"]), 6),
            "ret_24h_pct": safe_float(r.get("ret_24h_pct")),
            "ret_48h_pct": safe_float(r.get("ret_48h_pct")),
            "range_24h_pct": safe_float(r.get("range_24h_pct")),
            "atr14_pct": safe_float(r.get("atr14_pct")),
            "max_up_pct_24": safe_float(r.get("max_up_pct_24")),
            "max_up_pct_48": safe_float(r.get("max_up_pct_48")),
            "max_up_pct_168": safe_float(r.get("max_up_pct_168")),
            "max_up_pct_336": safe_float(r.get("max_up_pct_336")),
            "close_change_pct_24": safe_float(r.get("close_change_pct_24")),
            "close_change_pct_48": safe_float(r.get("close_change_pct_48")),
            "close_change_pct_168": safe_float(r.get("close_change_pct_168")),
            "close_change_pct_336": safe_float(r.get("close_change_pct_336")),
            "max_down_pct_24": safe_float(r.get("max_down_pct_24")),
            "max_down_pct_48": safe_float(r.get("max_down_pct_48")),
        })
    return out


def main():
    if not os.path.isfile(MODEL_DATASET_PATH):
        print(f"Missing model dataset: {MODEL_DATASET_PATH}")
        return

    if not os.path.isfile(HOURLY_SIGNAL_PATH):
        print(f"Missing hourly signal: {HOURLY_SIGNAL_PATH}")
        return

    signal_root = load_json(HOURLY_SIGNAL_PATH)
    signal_obj = snapshot_signal(signal_root)

    current = build_current_feature_vector(signal_obj, ASSET, CANDLE_KEY)
    model_rows = read_model_rows(MODEL_DATASET_PATH)

    # Restrict to selected asset
    asset_rows = [r for r in model_rows if r.get("asset") == ASSET]
    feat_stats = compute_feature_stats(asset_rows)

    scored = []
    skipped_same_timestamp = 0
    skipped_insufficient = 0

    for row in asset_rows:
        # Avoid matching current row against itself if current timestamp already exists historically
        if str(row.get("entry_ts_utc")) == str(current["entry_ts_utc"]):
            skipped_same_timestamp += 1
            continue

        dist, used = similarity_distance(current, row, feat_stats)
        if dist is None:
            skipped_insufficient += 1
            continue

        row2 = dict(row)
        row2["_distance"] = dist
        row2["_n_used"] = used
        scored.append(row2)

    scored.sort(key=lambda r: r["_distance"])
    neighbors = scored[:TOP_K]

    target_summary = {}
    for field, label in TARGET_SUMMARY:
        target_summary[field] = summarize_target(neighbors, field)

    payload = {
        "schema": "similarity_forecast_v1",
        "as_of_utc": signal_obj.get("published_at_utc"),
        "as_of_local": signal_obj.get("published_at_local"),
        "asset": ASSET,
        "current_state": {
            "entry_ts_utc": current["entry_ts_utc"],
            "entry_close": current["entry_close"],
            "n_candles_snapshot": current["n_candles_snapshot"],
            "features": {k: current.get(k) for k in FEATURE_COLS + BINARY_COLS},
        },
        "model_dataset": {
            "path": MODEL_DATASET_PATH,
            "n_rows_total": len(model_rows),
            "n_rows_asset": len(asset_rows),
        },
        "similarity": {
            "top_k": TOP_K,
            "neighbors_found": len(neighbors),
            "skipped_same_timestamp": skipped_same_timestamp,
            "skipped_insufficient_features": skipped_insufficient,
        },
        "forecast_summary": target_summary,
        "nearest_neighbors": top_neighbors_payload(neighbors, top_n=10),
        "notes": [
            "Similarity uses standardized Euclidean distance across numeric market-state features.",
            "Binary regime mismatches add a small penalty.",
            "Current version is ETH-USDT only; BTC can be added once BTC history matures.",
        ],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Similarity forecast written to: {OUT_PATH}")
    print(f"Model rows total: {len(model_rows)}")
    print(f"Asset rows considered: {len(asset_rows)}")
    print(f"Neighbors found: {len(neighbors)}")
    if neighbors:
        print(f"Best distance: {round(neighbors[0]['_distance'], 6)}")
        print(f"Worst kept distance: {round(neighbors[-1]['_distance'], 6)}")
    else:
        print("No neighbors found.")


if __name__ == "__main__":
    main()