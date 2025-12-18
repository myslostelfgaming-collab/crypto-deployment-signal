#!/usr/bin/env python

import os
import json
from datetime import datetime, timezone, timedelta

HISTORY_DIR = os.path.join("data", "history")


def iso(dt):
    return dt.isoformat(timespec="seconds") if dt else None


def main():
    print("\nLABEL DEBUG REPORT")
    print("==================\n")

    if not os.path.isdir(HISTORY_DIR):
        print("ERROR: data/history directory not found.")
        return

    files = []
    for root, _, fnames in os.walk(HISTORY_DIR):
        for f in fnames:
            if f.endswith(".json") and f != "index.json":
                files.append(os.path.join(root, f))

    files.sort()
    print(f"History files found: {len(files)}")

    if not files:
        print("ERROR: No history snapshots available.")
        return

    snapshots = []
    candle_ts = set()
    candle_count_total = 0

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            continue

        pub_utc = snap.get("published_at_utc")
        pub_dt = None
        if pub_utc:
            try:
                pub_dt = datetime.fromisoformat(pub_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                pub_dt = None

        candles = (
            snap.get("candles", {})
                .get("eth_usdt_1h", [])
        )

        # candles are compact lists: [ts_utc, o, h, l, c, v]
        for c in candles:
            if not isinstance(c, list) or len(c) < 1:
                continue
            try:
                ts = int(c[0])
            except Exception:
                continue
            candle_ts.add(ts)
            candle_count_total += 1

        snapshots.append({
            "file": path,
            "published_at": pub_dt,
            "candles": len(candles),
        })

    snapshots = [s for s in snapshots if s["published_at"]]
    snapshots.sort(key=lambda x: x["published_at"])

    if not snapshots:
        print("ERROR: No valid published_at_utc timestamps found.")
        return

    oldest_snap = snapshots[0]["published_at"]
    newest_snap = snapshots[-1]["published_at"]

    print("\nSnapshot time span:")
    print(f"  Oldest snapshot : {iso(oldest_snap)}")
    print(f"  Newest snapshot : {iso(newest_snap)}")
    print(f"  Snapshot hours span: {round((newest_snap - oldest_snap).total_seconds() / 3600.0, 2)}")

    print("\nCandle aggregation (ETH/USDT 1h):")
    print(f"  Total candles scanned (incl duplicates across files): {candle_count_total}")
    print(f"  Unique candle timestamps: {len(candle_ts)}")

    if not candle_ts:
        print("ERROR: No candles found inside history snapshots.")
        return

    ts_sorted = sorted(candle_ts)
    oldest_candle = datetime.fromtimestamp(ts_sorted[0], tz=timezone.utc)
    newest_candle = datetime.fromtimestamp(ts_sorted[-1], tz=timezone.utc)

    print(f"  Oldest candle : {iso(oldest_candle)}")
    print(f"  Newest candle : {iso(newest_candle)}")
    print(f"  Candle hours span: {round((newest_candle - oldest_candle).total_seconds() / 3600.0, 2)}")

    # Check forward availability for oldest snapshot entry time using the candle timeline.
    # We assume snapshot "decision time" aligns closely with latest candle open time available at that snapshot.
    # For a robust check, use the oldest candle timestamp as an anchor.
    anchor_ts = ts_sorted[0]  # earliest candle open
    anchor_dt = datetime.fromtimestamp(anchor_ts, tz=timezone.utc)

    print("\nForward window availability check (from oldest candle open):")
    missing_any = False
    for h in [12, 24, 36, 48, 60, 72, 84, 96]:
        target_ts = anchor_ts + 3600 * h
        exists = target_ts in candle_ts
        status = "OK" if exists else "MISSING"
        if not exists:
            missing_any = True
        print(f"  +{h:>3}h : {status}")

    print("\nContinuity check (first 120 hours from oldest candle):")
    gaps = 0
    first_missing = None
    for k in range(1, 121):
        t = anchor_ts + 3600 * k
        if t not in candle_ts:
            gaps += 1
            if first_missing is None:
                first_missing = k
    if gaps == 0:
        print("  OK: no gaps detected in first 120 hours.")
    else:
        print(f"  WARN: {gaps} missing hours in first 120 hours (first missing at +{first_missing}h).")

    print("\nINTERPRETATION:")
    print("- If you have many missing hours (gaps), label stitching will fail.")
    print("- If 96h forward is missing, strict 96h labeling will produce 0 rows.")
    print("- If continuity is good but labels still empty, we need to inspect entry_ts alignment.\n")


if __name__ == "__main__":
    main()