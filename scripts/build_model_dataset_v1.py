#!/usr/bin/env python

import csv
import os
from typing import Dict, List, Tuple

FEATURES_PATH = os.path.join("data", "features", "features_v1.csv")
LABELS_PATH = os.path.join("data", "labels", "labels_v2.csv")
OUT_DIR = os.path.join("data", "model")
OUT_PATH = os.path.join(OUT_DIR, "model_dataset_v1.csv")

KEY_FIELDS = ["asset", "entry_ts_utc"]


def read_csv(path: str) -> Tuple[List[dict], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def make_key(row: dict) -> Tuple[str, str]:
    return (row.get("asset", ""), row.get("entry_ts_utc", ""))


def choose_best_row(rows: List[dict], source_name: str) -> dict:
    """
    Deduplicate multiple rows for the same true market state.
    Preference:
    1. Higher n_candles_snapshot (for features rows)
    2. Later published_at_utc
    3. Last row as fallback
    """
    if len(rows) == 1:
        return rows[0]

    def score(r: dict):
        n_candles = 0
        try:
            n_candles = int(float(r.get("n_candles_snapshot", "") or 0))
        except Exception:
            n_candles = 0

        published = r.get("published_at_utc", "") or ""
        return (n_candles, published)

    best = sorted(rows, key=score)[-1]
    return best


def dedupe_rows(rows: List[dict], source_name: str) -> Tuple[Dict[Tuple[str, str], dict], int]:
    buckets: Dict[Tuple[str, str], List[dict]] = {}

    for row in rows:
        key = make_key(row)
        if not key[0] or not key[1]:
            continue
        buckets.setdefault(key, []).append(row)

    deduped: Dict[Tuple[str, str], dict] = {}
    duplicates_removed = 0

    for key, group in buckets.items():
        deduped[key] = choose_best_row(group, source_name)
        duplicates_removed += max(0, len(group) - 1)

    return deduped, duplicates_removed


def main():
    if not os.path.isfile(FEATURES_PATH):
        print(f"Missing features file: {FEATURES_PATH}")
        return

    if not os.path.isfile(LABELS_PATH):
        print(f"Missing labels file: {LABELS_PATH}")
        return

    feature_rows, feature_fields = read_csv(FEATURES_PATH)
    label_rows, label_fields = read_csv(LABELS_PATH)

    feature_map, feature_dupes_removed = dedupe_rows(feature_rows, "features")
    label_map, label_dupes_removed = dedupe_rows(label_rows, "labels")

    feature_keys = set(feature_map.keys())
    label_keys = set(label_map.keys())
    join_keys = sorted(feature_keys & label_keys)

    # Keep metadata from features, exclude repeated label metadata
    label_exclude = {
        "asset",
        "published_at_utc",
        "published_at_local",
        "date_local",
        "history_file",
        "entry_ts_utc",
        "entry_close",
    }

    label_value_fields = [f for f in label_fields if f not in label_exclude]

    out_fields = feature_fields + label_value_fields

    os.makedirs(OUT_DIR, exist_ok=True)

    written = 0
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for key in join_keys:
            frow = feature_map[key]
            lrow = label_map[key]

            out_row = dict(frow)
            for field in label_value_fields:
                out_row[field] = lrow.get(field, "")

            writer.writerow(out_row)
            written += 1

    print(f"Model dataset written to: {OUT_PATH}")
    print(f"Feature rows loaded: {len(feature_rows)}")
    print(f"Label rows loaded: {len(label_rows)}")
    print(f"Feature duplicates removed: {feature_dupes_removed}")
    print(f"Label duplicates removed: {label_dupes_removed}")
    print(f"Joined rows written: {written}")
    print(f"Feature unique keys: {len(feature_map)}")
    print(f"Label unique keys: {len(label_map)}")
    print(f"Intersection keys: {len(join_keys)}")


if __name__ == "__main__":
    main()