#!/usr/bin/env python

import csv
import json
import os
from glob import glob
from typing import Dict, List, Tuple, Optional

HISTORY_ROOT = os.path.join("data", "history")
OUT_DIR = os.path.join("data", "features")
OUT_CSV = os.path.join(OUT_DIR, "features_v1.csv")

ASSET_CANDLE_KEYS = {
    "ETH-USDT": "eth_usdt_1h",
    "BTC-USDT": "btc_usdt_1h",
}

Candle = List[float]  # [ts_utc, open, close, high, low, volume]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_history_files() -> List[str]:
    pattern = os.path.join(HISTORY_ROOT, "*", "*.json")
    return sorted(glob(pattern))


def read_existing_keys() -> set:
    if not os.path.isfile(OUT_CSV):
        return set()

    keys = set()
    with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.add((r.get("asset"), r.get("published_at_utc"), r.get("history_file")))
    return keys


def ensure_out_header(fieldnames: List[str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.isfile(OUT_CSV):
        return

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)


def build_header() -> List[str]:
    return [
        "asset",
        "published_at_utc",
        "published_at_local",
        "date_local",
        "history_file",
        "entry_ts_utc",
        "entry_close",
        "n_candles_snapshot",
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
        "is_up_24h",
        "is_up_48h",
    ]


def pct_change(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def safe_round(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return ""
    return str(round(x, digits))


def snapshot_meta(snapshot: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    pub_utc = snapshot.get("published_at_utc")
    pub_local = snapshot.get("published_at_local")
    date_local = snapshot.get("date")

    if pub_utc is None:
        signal = snapshot.get("signal", {})
        pub_utc = signal.get("published_at_utc")
        pub_local = signal.get("published_at_local")
        date_local = signal.get("date")

    return pub_utc, pub_local, date_local


def extract_asset_candles(snapshot: dict, candle_key: str) -> List[Candle]:
    candles = snapshot.get("signal", {}).get("candles", {}).get(candle_key)
    if candles is None:
        candles = snapshot.get("candles", {}).get(candle_key, [])

    if not isinstance(candles, list):
        return []

    out: List[Candle] = []
    for c in candles:
        if not isinstance(c, list) or len(c) < 6:
            continue
        try:
            out.append([
                int(c[0]),
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
            ])
        except Exception:
            continue

    out.sort(key=lambda x: x[0])
    return out


def get_entry_from_snapshot(candles: List[Candle]) -> Tuple[Optional[int], Optional[float]]:
    if not candles:
        return None, None
    last = candles[-1]
    try:
        return int(last[0]), float(last[2])
    except Exception:
        return None, None


def window_last(candles: List[Candle], n: int) -> Optional[List[Candle]]:
    if len(candles) < n:
        return None
    return candles[-n:]


def calc_return_feature(candles: List[Candle], lookback_h: int) -> Optional[float]:
    w = window_last(candles, lookback_h + 1)
    if not w:
        return None
    start_close = w[0][2]
    end_close = w[-1][2]
    return pct_change(end_close, start_close)


def calc_range_pct(candles: List[Candle], lookback_h: int) -> Optional[float]:
    w = window_last(candles, lookback_h)
    if not w:
        return None
    highs = [c[3] for c in w]
    lows = [c[4] for c in w]
    entry_close = w[-1][2]
    if entry_close == 0:
        return None
    return ((max(highs) - min(lows)) / entry_close) * 100.0


def calc_dist_from_high_low(candles: List[Candle], lookback_h: int) -> Tuple[Optional[float], Optional[float]]:
    w = window_last(candles, lookback_h)
    if not w:
        return None, None

    highs = [c[3] for c in w]
    lows = [c[4] for c in w]
    close = w[-1][2]

    high = max(highs)
    low = min(lows)

    if high == 0 or close == 0:
        return None, None

    dist_from_high = ((close - high) / close) * 100.0
    dist_from_low = ((close - low) / close) * 100.0
    return dist_from_high, dist_from_low


def calc_sma(candles: List[Candle], lookback_h: int) -> Optional[float]:
    w = window_last(candles, lookback_h)
    if not w:
        return None
    closes = [c[2] for c in w]
    return sum(closes) / len(closes)


def calc_close_vs_sma(candles: List[Candle], lookback_h: int) -> Optional[float]:
    sma = calc_sma(candles, lookback_h)
    if sma is None or sma == 0:
        return None
    close = candles[-1][2]
    return pct_change(close, sma)


def calc_wilder_atr_pct(candles: List[Candle], period: int = 14) -> Optional[float]:
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


def bool_flag_str(v: Optional[float]) -> str:
    if v is None:
        return ""
    return "1" if v > 0 else "0"


def main():
    history_paths = iter_history_files()
    if not history_paths:
        print("No history files found under data/history/. Nothing to featurize.")
        return

    header = build_header()
    ensure_out_header(header)
    existing = read_existing_keys()

    added = 0
    skipped_dupe = 0
    skipped_missing_asset = 0

    with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        for path in history_paths:
            rel = os.path.relpath(path, start=".")
            snap = load_json(path)
            pub_utc, pub_local, date_local = snapshot_meta(snap)

            for asset, candle_key in ASSET_CANDLE_KEYS.items():
                key = (asset, pub_utc, rel)
                if key in existing:
                    skipped_dupe += 1
                    continue

                candles = extract_asset_candles(snap, candle_key)
                if not candles:
                    skipped_missing_asset += 1
                    continue

                entry_ts, entry_close = get_entry_from_snapshot(candles)
                if entry_ts is None or entry_close is None:
                    skipped_missing_asset += 1
                    continue

                ret_6h = calc_return_feature(candles, 6)
                ret_12h = calc_return_feature(candles, 12)
                ret_24h = calc_return_feature(candles, 24)
                ret_48h = calc_return_feature(candles, 48)

                range_24h