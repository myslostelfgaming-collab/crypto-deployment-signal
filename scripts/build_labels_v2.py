#!/usr/bin/env python

import csv
import json
import os
from glob import glob
from typing import Dict, List, Tuple, Optional

HISTORY_ROOT = os.path.join("data", "history")
OUT_DIR = os.path.join("data", "labels")
OUT_CSV = os.path.join(OUT_DIR, "labels_v2.csv")

# v2 forecast horizons
HORIZONS = [24, 48, 168, 336]

# Thresholds can stay familiar for now
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]

# History candle keys to support
ASSET_CANDLE_KEYS = {
    "ETH-USDT": "eth_usdt_1h",
    "BTC-USDT": "btc_usdt_1h",
}

# Correct compact candle format:
# [ts_utc, open, high, low, close, volume]
Candle = List[float]


def pct_change(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def fmt_thr(x: float) -> str:
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return str(x).replace(".", "p")


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
    fields = [
        "asset",
        "published_at_utc",
        "published_at_local",
        "date_local",
        "history_file",
        "entry_ts_utc",
        "entry_close",
    ]

    for h in HORIZONS:
        fields += [
            f"max_up_pct_{h}",
            f"max_down_pct_{h}",
            f"close_change_pct_{h}",
            f"range_pct_{h}",
        ]

    for thr in THRESHOLDS:
        k = fmt_thr(thr)
        fields += [
            f"t_hit_up_{k}",
            f"t_hit_down_{k}",
            f"mdd_before_hit_up_{k}",
        ]

    return fields


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
            # Correct format: [ts_utc, open, high, low, close, volume]
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


def build_master_candle_map(history_paths: List[str], candle_key: str) -> Dict[int, Candle]:
    """
    Build a stitched ts -> candle map for one asset.
    Keeps first seen candle for each timestamp.
    """
    master: Dict[int, Candle] = {}

    for path in history_paths:
        snap = load_json(path)
        candles = extract_asset_candles(snap, candle_key)

        for c in candles:
            ts = int(c[0])
            if ts not in master:
                master[ts] = c

    return master


def get_entry_from_snapshot(snapshot: dict, candle_key: str) -> Tuple[Optional[int], Optional[float]]:
    candles = extract_asset_candles(snapshot, candle_key)
    if not candles:
        return None, None

    last = candles[-1]
    try:
        entry_ts = int(last[0])
        entry_close = float(last[4])  # close is index 4
    except Exception:
        return None, None

    return entry_ts, entry_close


def forward_window(master: Dict[int, Candle], entry_ts: int, hours: int) -> Optional[List[Candle]]:
    """
    Requires exact hourly continuity from entry_ts + 1h ... +hours*h.
    """
    out: List[Candle] = []
    for k in range(1, hours + 1):
        ts = entry_ts + 3600 * k
        c = master.get(ts)
        if c is None:
            return None
        out.append(c)
    return out


def compute_continuous_labels(entry_close: float, fwd: List[Candle]) -> dict:
    highs = [c[2] for c in fwd]
    lows = [c[3] for c in fwd]
    closes = [c[4] for c in fwd]

    max_high = max(highs)
    min_low = min(lows)
    end_close = closes[-1]

    return {
        "max_up_pct": round(pct_change(max_high, entry_close), 4),
        "max_down_pct": round(pct_change(min_low, entry_close), 4),
        "close_change_pct": round(pct_change(end_close, entry_close), 4),
        "range_pct": round(((max_high - min_low) / entry_close * 100.0) if entry_close else 0.0, 4),
    }


def compute_time_to_hit_and_mdd(entry_close: float, fwd_max: List[Candle]) -> dict:
    """
    Time-to-hit and max drawdown before upside hit, computed across the
    longest available forward window in v2 (336h).
    """
    running_min_low = []
    cur_min = float("inf")
    for c in fwd_max:
        low = c[3]
        cur_min = min(cur_min, low)
        running_min_low.append(cur_min)

    running_max_high = []
    cur_max = -float("inf")
    for c in fwd_max:
        high = c[2]
        cur_max = max(cur_max, high)
        running_max_high.append(cur_max)

    out = {}

    for thr in THRESHOLDS:
        thr_key = fmt_thr(thr)

        hit_up_t = ""
        target_up = entry_close * (1.0 + thr / 100.0)
        for i, rmh in enumerate(running_max_high):
            if rmh >= target_up:
                hit_up_t = str(i + 1)
                break
        out[f"t_hit_up_{thr_key}"] = hit_up_t

        hit_dn_t = ""
        target_dn = entry_close * (1.0 - thr / 100.0)
        for i, rml in enumerate(running_min_low):
            if rml <= target_dn:
                hit_dn_t = str(i + 1)
                break
        out[f"t_hit_down_{thr_key}"] = hit_dn_t

        mdd_val = ""
        if hit_up_t != "":
            t = int(hit_up_t)
            min_before = min(running_min_low[:t])
            mdd_pct = round(pct_change(min_before, entry_close), 4)
            mdd_pct = min(0.0, mdd_pct)
            mdd_val = str(mdd_pct)
        out[f"mdd_before_hit_up_{thr_key}"] = mdd_val

    return out


def snapshot_meta(snapshot: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Supports both:
    - old style: top-level published_at_utc, published_at_local, date
    - report style: signal.published_at_utc, signal.published_at_local, signal.date
    """
    pub_utc = snapshot.get("published_at_utc")
    pub_local = snapshot.get("published_at_local")
    date_local = snapshot.get("date")

    if pub_utc is None:
        signal = snapshot.get("signal", {})
        pub_utc = signal.get("published_at_utc")
        pub_local = signal.get("published_at_local")
        date_local = signal.get("date")

    return pub_utc, pub_local, date_local


def main():
    history_paths = iter_history_files()
    if not history_paths:
        print("No history files found under data/history/. Nothing to label yet.")
        return

    header = build_header()
    ensure_out_header(header)
    existing = read_existing_keys()

    # Build stitched candle maps once per asset
    masters: Dict[str, Dict[int, Candle]] = {}
    for asset, candle_key in ASSET_CANDLE_KEYS.items():
        masters[asset] = build_master_candle_map(history_paths, candle_key)
        print(f"{asset}: stitched candles = {len(masters[asset])}")

    added = 0
    skipped_dupe = 0
    skipped_missing_asset = 0
    skipped_missing_forward = 0

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

                entry_ts, entry_close = get_entry_from_snapshot(snap, candle_key)
                if entry_ts is None or entry_close is None:
                    skipped_missing_asset += 1
                    continue

                master = masters[asset]

                # Need the max horizon available to compute v2 labels
                max_h = max(HORIZONS)
                fwd_max = forward_window(master, entry_ts, max_h)
                if fwd_max is None:
                    skipped_missing_forward += 1
                    continue

                row = {
                    "asset": asset,
                    "published_at_utc": pub_utc,
                    "published_at_local": pub_local,
                    "date_local": date_local,
                    "history_file": rel,
                    "entry_ts_utc": str(entry_ts),
                    "entry_close": str(entry_close),
                }

                for h in HORIZONS:
                    fwd = fwd_max[:h]
                    cont = compute_continuous_labels(entry_close, fwd)
                    row[f"max_up_pct_{h}"] = str(cont["max_up_pct"])
                    row[f"max_down_pct_{h}"] = str(cont["max_down_pct"])
                    row[f"close_change_pct_{h}"] = str(cont["close_change_pct"])
                    row[f"range_pct_{h}"] = str(cont["range_pct"])

                row.update(compute_time_to_hit_and_mdd(entry_close, fwd_max))

                writer.writerow(row)
                added += 1

    print(f"Labels written to: {OUT_CSV}")
    print(f"Added: {added}")
    print(f"Skipped duplicates: {skipped_dupe}")
    print(f"Skipped missing asset candles in snapshot: {skipped_missing_asset}")
    print(f"Skipped missing forward window: {skipped_missing_forward}")


if __name__ == "__main__":
    main()