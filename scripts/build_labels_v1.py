#!/usr/bin/env python

import csv
import json
import os
from glob import glob
from typing import Dict, List, Tuple, Optional

HISTORY_ROOT = os.path.join("data", "history")
OUT_DIR = os.path.join("data", "labels")
OUT_CSV = os.path.join(OUT_DIR, "labels_v1.csv")

# Confirmed horizons (hours)
HORIZONS = [12, 24, 36, 48, 60, 72, 84, 96]

# Confirmed thresholds (%)
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]

# Candle compact format in history snapshots:
# [ts_utc, open, high, low, close, volume]
Candle = List[float]


def pct_change(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def fmt_thr(x: float) -> str:
    # 0.5 -> "0p5", 1.0 -> "1", 5.0 -> "5"
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return str(x).replace(".", "p")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_history_files() -> List[str]:
    # data/history/YYYY-MM-DD/*.json
    pattern = os.path.join(HISTORY_ROOT, "*", "*.json")
    return sorted(glob(pattern))


def read_existing_keys() -> set:
    if not os.path.isfile(OUT_CSV):
        return set()
    keys = set()
    with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.add((r.get("published_at_utc"), r.get("history_file")))
    return keys


def ensure_out_header(fieldnames: List[str]):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.isfile(OUT_CSV):
        return
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)


def build_master_candle_map(history_paths: List[str]) -> Dict[int, Candle]:
    """
    Build master candle map from all history snapshots:
      ts_utc -> [ts_utc, o, h, l, c, v]
    If duplicates exist for same ts_utc, we keep the first seen (stable).
    """
    master: Dict[int, Candle] = {}

    for p in history_paths:
        snap = load_json(p)
        candles = snap.get("candles", {}).get("eth_usdt_1h", [])
        for c in candles:
            if not c or len(c) < 6:
                continue
            ts = int(c[0])
            if ts not in master:
                master[ts] = c

    return master


def get_entry_ts_and_close_from_snapshot(snap: dict) -> Tuple[Optional[int], Optional[float]]:
    """
    Robust entry alignment:
      - Use the snapshot's own last candle timestamp as entry_ts_utc (candles[-1][0])
      - Use the snapshot's own last candle close as entry_close (candles[-1][4])

    This guarantees alignment with the master candle map (also built from c[0]).
    """
    candles = snap.get("candles", {}).get("eth_usdt_1h", [])

    if not candles or not isinstance(candles, list):
        return None, None

    last = candles[-1]
    if not isinstance(last, list) or len(last) < 5:
        return None, None

    try:
        entry_ts = int(last[0])
        entry_close = float(last[4])
    except Exception:
        return None, None

    return entry_ts, entry_close


def forward_window(master: Dict[int, Candle], entry_ts: int, hours: int) -> Optional[List[Candle]]:
    """
    Returns candles for (entry_ts + 1h) ... (entry_ts + hours*h)
    Requires exact hourly continuity.
    """
    out = []
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


def compute_time_to_hit_and_mdd(entry_close: float, fwd96: List[Candle]) -> dict:
    """
    Compute time-to-hit (earliest hour) for thresholds up to 96h,
    and max drawdown before the first hit of upside target.

    NOTE: Drawdown is clamped to <= 0.0. If price never goes below entry
    before the upside hit, drawdown is reported as 0.0 (not positive).
    """
    running_min_low = []
    cur_min = float("inf")
    for c in fwd96:
        low = c[3]
        cur_min = min(cur_min, low)
        running_min_low.append(cur_min)

    running_max_high = []
    cur_max = -float("inf")
    for c in fwd96:
        high = c[2]
        cur_max = max(cur_max, high)
        running_max_high.append(cur_max)

    out = {}

    for thr in THRESHOLDS:
        thr_key = fmt_thr(thr)

        # Upside hit time
        hit_up_t = ""
        target_up = entry_close * (1.0 + thr / 100.0)
        for i, rmh in enumerate(running_max_high):
            if rmh >= target_up:
                hit_up_t = str(i + 1)  # hours are 1-indexed
                break
        out[f"t_hit_up_{thr_key}"] = hit_up_t

        # Downside hit time
        hit_dn_t = ""
        target_dn = entry_close * (1.0 - thr / 100.0)
        for i, rml in enumerate(running_min_low):
            if rml <= target_dn:
                hit_dn_t = str(i + 1)
                break
        out[f"t_hit_down_{thr_key}"] = hit_dn_t

        # MDD before upside hit (within 96h window) — clamped to <= 0
        mdd_val = ""
        if hit_up_t != "":
            t = int(hit_up_t)
            min_before = min(running_min_low[:t])
            mdd_pct = round(pct_change(min_before, entry_close), 4)
            mdd_pct = min(0.0, mdd_pct)  # clamp: drawdown cannot be positive
            mdd_val = str(mdd_pct)
        out[f"mdd_before_hit_up_{thr_key}"] = mdd_val

    return out


def build_header() -> List[str]:
    fields = [
        "published_at_utc",
        "published_at_local",
        "date_local",
        "history_file",
        "entry_ts_utc",
        "entry_close",
    ]

    # Continuous labels per horizon
    for h in HORIZONS:
        fields += [
            f"max_up_pct_{h}",
            f"max_down_pct_{h}",
            f"close_change_pct_{h}",
            f"range_pct_{h}",
        ]

    # Time-to-hit + mdd (computed once up to 96)
    for thr in THRESHOLDS:
        k = fmt_thr(thr)
        fields.append(f"t_hit_up_{k}")
        fields.append(f"t_hit_down_{k}")
        fields.append(f"mdd_before_hit_up_{k}")

    return fields


def main():
    history_paths = iter_history_files()
    if not history_paths:
        print("No history files found under data/history/. Nothing to label yet.")
        return

    master = build_master_candle_map(history_paths)
    if not master:
        print("No candles found in history snapshots. Cannot build labels.")
        return

    existing = read_existing_keys()
    header = build_header()
    ensure_out_header(header)

    added = 0
    skipped_dupe = 0
    skipped_missing = 0

    with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        for path in history_paths:
            rel = os.path.relpath(path, start=".")
            snap = load_json(path)

            pub_utc = snap.get("published_at_utc")
            pub_local = snap.get("published_at_local")
            date_local = snap.get("date")

            key = (pub_utc, rel)
            if key in existing:
                skipped_dupe += 1
                continue

            entry_ts, entry_close = get_entry_ts_and_close_from_snapshot(snap)
            if entry_ts is None or entry_close is None:
                skipped_missing += 1
                continue

            row = {
                "published_at_utc": pub_utc,
                "published_at_local": pub_local,
                "date_local": date_local,
                "history_file": rel,
                "entry_ts_utc": str(entry_ts),
                "entry_close": str(entry_close),
            }

            # Need 96h forward window for time-to-hit/mdd; if not available, skip entirely for now.
            fwd96 = forward_window(master, entry_ts, 96)
            if fwd96 is None:
                skipped_missing += 1
                continue

            # Continuous labels per horizon
            for h in HORIZONS:
                fwd = fwd96[:h]  # first h hours
                cont = compute_continuous_labels(entry_close, fwd)
                row[f"max_up_pct_{h}"] = str(cont["max_up_pct"])
                row[f"max_down_pct_{h}"] = str(cont["max_down_pct"])
                row[f"close_change_pct_{h}"] = str(cont["close_change_pct"])
                row[f"range_pct_{h}"] = str(cont["range_pct"])

            # Time-to-hit + MDD (once, up to 96h)
            ttm = compute_time_to_hit_and_mdd(entry_close, fwd96)
            row.update(ttm)

            writer.writerow(row)
            added += 1

    print(f"Labels written to: {OUT_CSV}")
    print(
        f"Added: {added}, skipped duplicates: {skipped_dupe}, skipped (missing forward window/fields): {skipped_missing}"
    )


if __name__ == "__main__":
    main()