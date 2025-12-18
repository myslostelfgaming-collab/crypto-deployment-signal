#!/usr/bin/env python

import os
import json
from datetime import datetime, timezone

HISTORY_DIR = os.path.join("data", "history")


def iso(dt):
    return dt.isoformat(timespec="seconds") if dt else None


def parse_ts(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


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
    all_candle_ts = set()

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            continue

        pub = snap.get("published_at_utc")
        pub_dt = parse_ts(pub)

        candles = (
            snap
            .get("candles", {})
            .get("eth_usdt_1h", [])
        )

        for c in candles:
            ts = parse_ts(c.get("open_time_utc"))
            if ts:
                all_candle_ts.add(ts)

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

    print("\nSnapshot time span:")
    print(f"  Oldest snapshot : {iso(snapshots[0]['published_at'])}")
    print(f"  Newest snapshot : {iso(snapshots[-1]['published_at'])}")

    print("\nCandle aggregation:")
    print(f"  Unique ETH/USDT 1h candles found: {len(all_candle_ts)}")

    if not all_candle_ts:
        print("ERROR: No candles found inside history snapshots.")
        return

    candle_list = sorted(all_candle_ts)
    print(f"  Oldest candle : {iso(candle_list[0])}")
    print(f"  Newest candle : {iso(candle_list[-1])}")

    # Check forward availability for oldest snapshot
    oldest_snap = snapshots[0]
    t0 = oldest_snap["published_at"]

    print("\nForward window availability check (oldest snapshot):")
    for h in [12, 24, 36, 48, 60, 72, 84, 96]:
        target = t0 + timedelta(hours=h)
        exists = target in all_candle_ts
        status = "OK" if exists else "MISSING"
        print(f"  +{h:>3}h : {status}")

    print("\nINTERPRETATION:")
    print("- If candles are missing at +12h or +24h, labels cannot be built yet.")
    print("- If candles exist but labels are empty, timestamp alignment is broken.")
    print("- If total candles < ~120, 96h labels are impossible.\n")


if __name__ == "__main__":
    main()