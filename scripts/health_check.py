#!/usr/bin/env python

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

LOCAL_TZ = timezone(timedelta(hours=2))  # Africa/Johannesburg (SAST)


@dataclass
class Findings:
    fails: list
    warns: list
    infos: list


def parse_iso(dt_str: str) -> datetime:
    # Handles "2025-12-14T03:32:14+00:00" or with Z if ever used
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


def load_index(index_path: str):
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"index.csv not found at {index_path}")

    rows = []
    with open(index_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_snapshot(history_file: str):
    if not os.path.isfile(history_file):
        return None
    with open(history_file, "r", encoding="utf-8") as f:
        return json.load(f)


def hour_bucket(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def main():
    ap = argparse.ArgumentParser(description="Health check for Deployment Signal history stream.")
    ap.add_argument("--days", type=int, default=7, help="How many past days to check (default 7).")
    ap.add_argument("--expected-per-day", type=int, default=24, help="Expected snapshots per day (default 24).")
    ap.add_argument("--max-staleness-min", type=float, default=90.0, help="Max acceptable freshness minutes (default 90).")
    ap.add_argument("--max-gap-hours-warn", type=float, default=2.0, help="Gap > this triggers WARN (default 2h).")
    ap.add_argument("--max-gap-hours-fail", type=float, default=6.0, help="Gap > this triggers FAIL (default 6h).")
    ap.add_argument("--index", default="data/history/index.csv", help="Path to index.csv")
    args = ap.parse_args()

    findings = Findings(fails=[], warns=[], infos=[])

    # --- Load index
    rows = load_index(args.index)
    if not rows:
        findings.fails.append("index.csv is empty (no history rows).")
        print_report(findings)
        raise SystemExit(2)

    # --- Filter to last N days by published_at_utc
    rows_parsed = []
    for r in rows:
        try:
            pub_utc = parse_iso(r["published_at_utc"]).astimezone(timezone.utc)
        except Exception:
            findings.warns.append(f"Bad published_at_utc in index row: {r.get('published_at_utc')}")
            continue
        rows_parsed.append((pub_utc, r))

    rows_parsed.sort(key=lambda x: x[0])

    newest = rows_parsed[-1][0]
    cutoff = newest - timedelta(days=args.days)
    recent = [(t, r) for (t, r) in rows_parsed if t >= cutoff]

    if not recent:
        findings.fails.append(f"No rows within last {args.days} days (cutoff {cutoff.isoformat()}).")
        print_report(findings)
        raise SystemExit(2)

    findings.infos.append(f"Checking {len(recent)} snapshots from {recent[0][0].isoformat()} to {recent[-1][0].isoformat()}.")

    # --- 1) Duplicates (published_at_utc)
    seen_pub = set()
    dup_pub = 0
    for t, _ in recent:
        if t in seen_pub:
            dup_pub += 1
        seen_pub.add(t)
    if dup_pub > 0:
        findings.warns.append(f"Duplicate published_at_utc timestamps found: {dup_pub} duplicates.")

    # --- 2) Coverage per day (local date)
    per_day_hours = defaultdict(set)
    for t, _ in recent:
        local = t.astimezone(LOCAL_TZ)
        per_day_hours[local.date()].add(hour_bucket(local))

    for d, hours in sorted(per_day_hours.items()):
        count = len(hours)
        if count < max(1, args.expected_per_day - 6):
            findings.warns.append(f"{d}: only {count}/{args.expected_per_day} hourly snapshots (low coverage).")
        else:
            findings.infos.append(f"{d}: {count}/{args.expected_per_day} snapshots.")

    # --- 3) Gaps between runs (based on published_at_utc)
    gaps_warn = 0
    gaps_fail = 0
    prev_t = None
    for t, _ in recent:
        if prev_t is not None:
            gap_h = (t - prev_t).total_seconds() / 3600.0
            if gap_h > args.max_gap_hours_fail:
                gaps_fail += 1
            elif gap_h > args.max_gap_hours_warn:
                gaps_warn += 1
        prev_t = t

    if gaps_fail:
        findings.fails.append(f"Large gaps between runs: {gaps_fail} gaps > {args.max_gap_hours_fail}h.")
    if gaps_warn:
        findings.warns.append(f"Gaps between runs: {gaps_warn} gaps > {args.max_gap_hours_warn}h (but <= {args.max_gap_hours_fail}h).")

    # --- 4) Deep checks via history JSON files
    missing_files = 0
    stale_count = 0
    bad_counts = 0
    candle_gaps_warn = 0
    candle_gaps_fail = 0

    prev_last_candle_utc = None

    for t, r in recent:
        rel = r.get("history_file")
        if not rel:
            findings.warns.append(f"Missing history_file in index row at {t.isoformat()}.")
            continue

        snap_path = rel if os.path.isabs(rel) else os.path.join(".", rel)
        snap = load_snapshot(snap_path)
        if snap is None:
            missing_files += 1
            continue

        # Returned count check (ETH-USDT)
        rc = (
            snap.get("integrity", {})
                .get("eth_usdt", {})
                .get("returned_count")
        )
        req = (
            snap.get("integrity", {})
                .get("eth_usdt", {})
                .get("requested_limit")
        )
        if isinstance(rc, int) and isinstance(req, int) and rc < req:
            bad_counts += 1

        # Freshness check if present
        freshness = (
            snap.get("integrity", {})
                .get("eth_usdt", {})
                .get("data_freshness_minutes")
        )
        if isinstance(freshness, (int, float)) and freshness > args.max_staleness_min:
            stale_count += 1

        # Candle continuity (last candle open utc should usually step by 1h between snapshots)
        last_candle_utc_str = (
            snap.get("integrity", {})
                .get("eth_usdt", {})
                .get("last_candle_open_time_utc")
        )
        if last_candle_utc_str:
            try:
                last_candle_utc = parse_iso(last_candle_utc_str).astimezone(timezone.utc)
                if prev_last_candle_utc is not None:
                    step_h = (last_candle_utc - prev_last_candle_utc).total_seconds() / 3600.0
                    if step_h > args.max_gap_hours_fail:
                        candle_gaps_fail += 1
                    elif step_h > args.max_gap_hours_warn:
                        candle_gaps_warn += 1
                prev_last_candle_utc = last_candle_utc
            except Exception:
                findings.warns.append(f"Bad last_candle_open_time_utc in {snap_path}: {last_candle_utc_str}")

    if missing_files:
        findings.warns.append(f"Missing history JSON files referenced by index: {missing_files}.")

    if bad_counts:
        findings.warns.append(f"ETH-USDT returned_count < requested_limit in {bad_counts} snapshots (KuCoin/API short return).")

    if stale_count:
        findings.warns.append(f"Stale snapshots: {stale_count} with freshness > {args.max_staleness_min} minutes.")

    if candle_gaps_fail:
        findings.fails.append(f"Candle continuity FAIL: {candle_gaps_fail} gaps > {args.max_gap_hours_fail}h in last_candle_open_time_utc.")
    if candle_gaps_warn:
        findings.warns.append(f"Candle continuity WARN: {candle_gaps_warn} gaps > {args.max_gap_hours_warn}h (but <= {args.max_gap_hours_fail}h).")

    print_report(findings)

    # Exit codes suitable for CI
    if findings.fails:
        raise SystemExit(2)
    if findings.warns:
        raise SystemExit(1)
    raise SystemExit(0)


def print_report(findings: Findings):
    status = "PASS"
    if findings.fails:
        status = "FAIL"
    elif findings.warns:
        status = "WARN"

    print("\n=== Deployment Signal History Health Check ===")
    print(f"STATUS: {status}\n")

    if findings.infos:
        print("Info:")
        for x in findings.infos[:50]:
            print(f"  - {x}")
        if len(findings.infos) > 50:
            print(f"  - (and {len(findings.infos) - 50} more info lines)")

    if findings.warns:
        print("\nWarnings:")
        for x in findings.warns:
            print(f"  - {x}")

    if findings.fails:
        print("\nFailures:")
        for x in findings.fails:
            print(f"  - {x}")

    print("\n=============================================\n")


if __name__ == "__main__":
    main()