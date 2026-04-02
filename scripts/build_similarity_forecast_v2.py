#!/usr/bin/env python

import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

MODEL_DATASET_PATH = os.path.join("data", "model", "model_dataset_v1.csv")
HOURLY_SIGNAL_PATH = os.path.join("data", "hourly_signal.json")
OUT_PATH = os.path.join("data", "model", "similarity_forecast_v2.json")

ASSET = "ETH-USDT"
CANDLE_KEY = "eth_usdt_1h"

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

BINARY_COLS = [
    "is_up_24h",
    "is_up_48h",
]

PRIMARY_TOP_K = 20
CONTEXT_TOP_K = 50
MIN_REQUIRED_NUMERIC_FEATURES = 8

REGIME_MISMATCH_PENALTY = 1.0
WEIGHT_EPS = 1e-6
WEIGHT_POWER = 2.0  # weight = 1 / (distance^power + eps)

HORIZONS = [24, 48, 168, 336]


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


def median(values: List[float]) -> Optional[float]:
    return percentile(values, 0.5)


def weighted_mean(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    wsum = sum(weights)
    if wsum == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / wsum


def weighted_std(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    mu = weighted_mean(values, weights)
    if mu is None:
        return None
    wsum = sum(weights)
    if wsum == 0:
        return None
    var = sum(w * ((v - mu) ** 2) for v, w in zip(values, weights)) / wsum
    return math.sqrt(var)


def weighted_quantile(values: List[float], weights: List[float], q: float) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None

    threshold = total_w * q
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= threshold:
            return value
    return pairs[-1][0]


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
        raise ValueError(f"No candles found for {asset} ({candle_key})")

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
    sq = 0.0
    used_numeric = 0

    for col in FEATURE_COLS:
        cur = current.get(col)
        past = safe_float(row.get(col))
        if cur is None or past is None or col not in feat_stats:
            continue

        mu = feat_stats[col]["mean"]
        sd = feat_stats[col]["std"]
        z_cur = (cur - mu) / sd
        z_past = (past - mu) / sd
        sq += (z_cur - z_past) ** 2
        used_numeric += 1

    if used_numeric < MIN_REQUIRED_NUMERIC_FEATURES:
        return None, used_numeric

    bin_penalty = 0.0
    for col in BINARY_COLS:
        cur = str(current.get(col, ""))
        past = str(row.get(col, ""))
        if cur != "" and past != "" and cur != past:
            bin_penalty += REGIME_MISMATCH_PENALTY

    dist = math.sqrt(sq) + bin_penalty
    return dist, used_numeric


def distance_to_weight(distance: float) -> float:
    return 1.0 / ((distance ** WEIGHT_POWER) + WEIGHT_EPS)


def summarize_weighted(rows: List[dict], field: str) -> dict:
    vals = []
    weights = []

    for r in rows:
        v = safe_float(r.get(field))
        d = safe_float(r.get("_distance"))
        if v is None or d is None:
            continue
        vals.append(v)
        weights.append(distance_to_weight(d))

    raw_vals = [v for v in vals]

    return {
        "n": len(vals),
        "mean": round(mean(raw_vals), 4) if raw_vals else None,
        "median": round(median(raw_vals), 4) if raw_vals else None,
        "p25": round(percentile(raw_vals, 0.25), 4) if raw_vals else None,
        "p75": round(percentile(raw_vals, 0.75), 4) if raw_vals else None,
        "weighted_mean": round(weighted_mean(vals, weights), 4) if vals else None,
        "weighted_median": round(weighted_quantile(vals, weights, 0.5), 4) if vals else None,
        "weighted_p25": round(weighted_quantile(vals, weights, 0.25), 4) if vals else None,
        "weighted_p75": round(weighted_quantile(vals, weights, 0.75), 4) if vals else None,
        "weighted_std": round(weighted_std(vals, weights), 4) if vals else None,
        "min": round(min(raw_vals), 4) if raw_vals else None,
        "max": round(max(raw_vals), 4) if raw_vals else None,
    }


def build_horizon_score(rows: List[dict], horizon: int) -> dict:
    close_field = f"close_change_pct_{horizon}"
    up_field = f"max_up_pct_{horizon}"
    down_field = f"max_down_pct_{horizon}"

    close_vals = []
    up_vals = []
    down_vals = []
    weights = []

    for r in rows:
        close_v = safe_float(r.get(close_field))
        up_v = safe_float(r.get(up_field))
        down_v = safe_float(r.get(down_field))
        d = safe_float(r.get("_distance"))

        if close_v is None or up_v is None or down_v is None or d is None:
            continue

        w = distance_to_weight(d)
        close_vals.append(close_v)
        up_vals.append(up_v)
        down_vals.append(down_v)
        weights.append(w)

    if not close_vals:
        return {
            "horizon_h": horizon,
            "signal": "unknown",
            "confidence": "low",
            "weighted_close_mean": None,
            "weighted_close_median": None,
            "weighted_max_up_mean": None,
            "weighted_max_down_mean": None,
            "score": None,
        }

    w_close_mean = weighted_mean(close_vals, weights)
    w_close_median = weighted_quantile(close_vals, weights, 0.5)
    w_up_mean = weighted_mean(up_vals, weights)
    w_down_mean = weighted_mean(down_vals, weights)

    # Composite score
    # positive close helps
    # upside potential helps a bit
    # downside hurts
    score = 0.0
    if w_close_mean is not None:
        score += 0.6 * w_close_mean
    if w_up_mean is not None:
        score += 0.25 * w_up_mean
    if w_down_mean is not None:
        score += 0.15 * w_down_mean  # down is negative, so this subtracts

    # Confidence from consistency
    close_std = weighted_std(close_vals, weights)
    agreement = 0.0
    if w_close_mean is not None and w_close_median is not None:
        agreement = abs(w_close_mean - w_close_median)

    if close_std is None:
        confidence = "low"
    elif close_std < 1.5 and agreement < 1.0:
        confidence = "high"
    elif close_std < 3.0 and agreement < 2.0:
        confidence = "medium"
    else:
        confidence = "low"

    if score >= 1.0:
        signal = "bullish"
    elif score <= -1.0:
        signal = "bearish"
    else:
        signal = "mixed"

    return {
        "horizon_h": horizon,
        "signal": signal,
        "confidence": confidence,
        "weighted_close_mean": round(w_close_mean, 4) if w_close_mean is not None else None,
        "weighted_close_median": round(w_close_median, 4) if w_close_median is not None else None,
        "weighted_max_up_mean": round(w_up_mean, 4) if w_up_mean is not None else None,
        "weighted_max_down_mean": round(w_down_mean, 4) if w_down_mean is not None else None,
        "score": round(score, 4),
    }


def top_neighbors_payload(neighbors: List[dict], top_n: int = 10) -> List[dict]:
    out = []
    for r in neighbors[:top_n]:
        out.append({
            "asset": r.get("asset"),
            "entry_ts_utc": r.get("entry_ts_utc"),
            "published_at_utc": r.get("published_at_utc"),
            "similarity_distance": round(float(r["_distance"]), 6),
            "weight": round(distance_to_weight(float(r["_distance"])), 6),
            "ret_24h_pct": safe_float(r.get("ret_24h_pct")),
            "ret_48h_pct": safe_float(r.get("ret_48h_pct")),
            "range_24h_pct": safe_float(r.get("range_24h_pct")),
            "atr14_pct": safe_float(r.get("atr14_pct")),
            "close_change_pct_24": safe_float(r.get("close_change_pct_24")),
            "close_change_pct_48": safe_float(r.get("close_change_pct_48")),
            "close_change_pct_168": safe_float(r.get("close_change_pct_168")),
            "close_change_pct_336": safe_float(r.get("close_change_pct_336")),
            "max_up_pct_24": safe_float(r.get("max_up_pct_24")),
            "max_up_pct_48": safe_float(r.get("max_up_pct_48")),
            "max_up_pct_168": safe_float(r.get("max_up_pct_168")),
            "max_up_pct_336": safe_float(r.get("max_up_pct_336")),
            "max_down_pct_24": safe_float(r.get("max_down_pct_24")),
            "max_down_pct_48": safe_float(r.get("max_down_pct_48")),
            "max_down_pct_168": safe_float(r.get("max_down_pct_168")),
            "max_down_pct_336": safe_float(r.get("max_down_pct_336")),
        })
    return out


def build_target_summary(rows: List[dict]) -> dict:
    out = {}
    for horizon in HORIZONS:
        for prefix in ["max_up_pct", "max_down_pct", "close_change_pct"]:
            field = f"{prefix}_{horizon}"
            out[field] = summarize_weighted(rows, field)
    return out


def overall_confidence(primary_rows: List[dict], scorecard: Dict[str, dict]) -> str:
    if not primary_rows:
        return "low"

    distances = [safe_float(r.get("_distance")) for r in primary_rows]
    distances = [d for d in distances if d is not None]

    if not distances:
        return "low"

    mean_distance = mean(distances)
    strong_horizons = sum(1 for v in scorecard.values() if v.get("confidence") == "high")
    non_mixed = sum(1 for v in scorecard.values() if v.get("signal") != "mixed")

    if mean_distance is not None and mean_distance < 1.6 and strong_horizons >= 2 and non_mixed >= 2:
        return "high"
    if mean_distance is not None and mean_distance < 2.2 and non_mixed >= 1:
        return "medium"
    return "low"


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
    asset_rows = [r for r in model_rows if r.get("asset") == ASSET]
    feat_stats = compute_feature_stats(asset_rows)

    scored = []
    skipped_same_timestamp = 0
    skipped_insufficient = 0

    for row in asset_rows:
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

    primary_neighbors = scored[:PRIMARY_TOP_K]
    context_neighbors = scored[:CONTEXT_TOP_K]

    primary_summary = build_target_summary(primary_neighbors)
    context_summary = build_target_summary(context_neighbors)

    scorecard = {
        f"{h}h": build_horizon_score(primary_neighbors, h)
        for h in HORIZONS
    }

    overall = overall_confidence(primary_neighbors, scorecard)

    payload = {
        "schema": "similarity_forecast_v2",
        "as_of_utc": signal_obj.get("published_at_utc"),
        "as_of_local": signal_obj.get("published_at_local"),
        "asset": ASSET,
        "settings": {
            "primary_top_k": PRIMARY_TOP_K,
            "context_top_k": CONTEXT_TOP_K,
            "min_required_numeric_features": MIN_REQUIRED_NUMERIC_FEATURES,
            "regime_mismatch_penalty": REGIME_MISMATCH_PENALTY,
            "weight_power": WEIGHT_POWER,
        },
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
            "neighbors_found_total": len(scored),
            "primary_neighbors_found": len(primary_neighbors),
            "context_neighbors_found": len(context_neighbors),
            "skipped_same_timestamp": skipped_same_timestamp,
            "skipped_insufficient_features": skipped_insufficient,
            "best_distance": round(primary_neighbors[0]["_distance"], 6) if primary_neighbors else None,
            "worst_primary_distance": round(primary_neighbors[-1]["_distance"], 6) if primary_neighbors else None,
            "worst_context_distance": round(context_neighbors[-1]["_distance"], 6) if context_neighbors else None,
        },
        "directional_scorecard": scorecard,
        "overall_confidence": overall,
        "forecast_summary_primary_top20": primary_summary,
        "forecast_summary_context_top50": context_summary,
        "nearest_neighbors_primary": top_neighbors_payload(primary_neighbors, top_n=10),
        "nearest_neighbors_context": top_neighbors_payload(context_neighbors, top_n=10),
        "notes": [
            "v2 uses distance-weighted summaries.",
            "Closest matches count more via inverse-distance weighting.",
            "Directional scorecard is based on weighted close change, upside, and downside across each horizon.",
        ],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Similarity forecast written to: {OUT_PATH}")
    print(f"Model rows total: {len(model_rows)}")
    print(f"Asset rows considered: {len(asset_rows)}")
    print(f"Neighbors found total: {len(scored)}")
    print(f"Primary neighbors: {len(primary_neighbors)}")
    print(f"Context neighbors: {len(context_neighbors)}")
    if primary_neighbors:
        print(f"Best distance: {round(primary_neighbors[0]['_distance'], 6)}")
        print(f"Worst primary distance: {round(primary_neighbors[-1]['_distance'], 6)}")
    if context_neighbors:
        print(f"Worst context distance: {round(context_neighbors[-1]['_distance'], 6)}")
    print(f"Overall confidence: {overall}")


if __name__ == "__main__":
    main()