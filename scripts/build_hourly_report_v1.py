#!/usr/bin/env python
import json
import os
from datetime import datetime, timezone

LOCAL_TZ = "Africa/Johannesburg"

SIGNAL_PATH = os.path.join("data", "hourly_signal.json")
FORECAST_PATH = os.path.join("data", "hourly_forecast.json")
OUT_PATH = os.path.join("data", "hourly_report.json")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if not os.path.isfile(SIGNAL_PATH):
        print(f"Missing required file: {SIGNAL_PATH}")
        raise SystemExit(1)

    signal = load_json(SIGNAL_PATH)

    forecast = None
    if os.path.isfile(FORECAST_PATH):
        forecast = load_json(FORECAST_PATH)

    # Prefer signal timestamps (authoritative)
    published_at_utc = signal.get("published_at_utc")
    published_at_local = signal.get("published_at_local")
    tz = signal.get("timezone", LOCAL_TZ)

    # Fallback if missing
    if not published_at_utc or not published_at_local:
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        now_local = now_utc
        published_at_utc = published_at_utc or now_utc
        published_at_local = published_at_local or now_local

    report = {
        "schema": "hourly_report_v1",
        "timezone": tz,
        "published_at_utc": published_at_utc,
        "published_at_local": published_at_local,
        "signal": signal,
        "forecast": forecast,  # may be null until we add the forecast script
    }

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()