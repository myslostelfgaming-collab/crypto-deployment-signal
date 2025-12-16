#!/usr/bin/env python

import csv
import json
import os
from datetime import datetime

HISTORY_ROOT = os.path.join("data", "history")
INDEX_PATH = os.path.join(HISTORY_ROOT, "index.csv")

HEADER = [
    "published_at_utc",
    "published_at_local",
    "date_local",
    "eth_usdt_close",
    "eth_usdt_gap_pct",
    "atr_1h",
    "atr_trend",
    "intraday_momentum",
    "early_breakout",
    "last_candle_open_time_local",
    "last_candle_open_time_utc",
    "history_file",
]


def safe_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_existing_rows():
    if not os.path.isfile(INDEX_PATH):
        return set()

    seen = set()
    with open(INDEX_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("published_at_utc"), row.get("history_file"))
            seen.add(key)
    return seen


def ensure_index_header():
    if os.path.isfile(INDEX_PATH):
        return
    os.makedirs(HISTORY_ROOT, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def iter_history_files():
    if not os.path.isdir(HISTORY_ROOT):
        return

    for day in sorted(os.listdir(HISTORY_ROOT)):
        day_path = os.path.join(HISTORY_ROOT, day)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(os.listdir(day_path)):
            if not fname.endswith(".json"):
                continue
            yield os.path.join(day_path, fname)


def main():
    ensure_index_header()
    seen = read_existing_rows()

    new_rows = 0

    with open(INDEX_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)

        for path in iter_history_files():
            rel_path = os.path.relpath(path, start=".")
            with open(path, "r", encoding="utf-8") as jf:
                snap = json.load(jf)

            pub_utc = snap.get("published_at_utc")
            key = (pub_utc, rel_path)
            if key in seen:
                continue

            row = {
                "published_at_utc": pub_utc,
                "published_at_local": snap.get("published_at_local"),
                "date_local": snap.get("date"),
                "eth_usdt_close": safe_get(snap, ["eth_usdt", "close"]),
                "eth_usdt_gap_pct": safe_get(snap, ["eth_usdt", "gap_pct"]),
                "atr_1h": safe_get(snap, ["atr_1h", "value"]),
                "atr_trend": safe_get(snap, ["atr_1h", "trend"]),
                "intraday_momentum": snap.get("intraday_momentum"),
                "early_breakout": safe_get(snap, ["early_breakout", "occurred"]),
                "last_candle_open_time_local": safe_get(snap, ["integrity", "eth_usdt", "last_candle_open_time_local"]),
                "last_candle_open_time_utc": safe_get(snap, ["integrity", "eth_usdt", "last_candle_open_time_utc"]),
                "history_file": rel_path,
            }

            writer.writerow(row)
            new_rows += 1

    print(f"Index updated: {INDEX_PATH}. Added {new_rows} new rows.")


if __name__ == "__main__":
    main()