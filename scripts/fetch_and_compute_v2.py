#!/usr/bin/env python

import requests
import json
import os
from datetime import datetime, timezone
import pytz

KUCOIN_BASE = "https://api.kucoin.com/api/v1/market/candles"

SYMBOLS = {
    "eth_usdt": "ETH-USDT",
    "btc_usdt": "BTC-USDT",
    "eth_btc": "ETH-BTC",
}

INTERVAL = "1hour"
LIMIT = 400  # IMPORTANT: wider history


def fetch_candles(symbol):
    params = {
        "symbol": symbol,
        "type": INTERVAL,
    }
    response = requests.get(KUCOIN_BASE, params=params)
    data = response.json()

    if "data" not in data:
        raise Exception(f"Failed to fetch {symbol}: {data}")

    candles = data["data"][:LIMIT]

    # Reverse (KuCoin gives newest first)
    candles = list(reversed(candles))

    # Convert to numeric format
    parsed = []
    for c in candles:
        parsed.append([
            int(c[0]),      # timestamp
            float(c[1]),    # open
            float(c[2]),    # close
            float(c[3]),    # high
            float(c[4]),    # low
            float(c[5]),    # volume
        ])

    return parsed


def compute_24h_stats(candles):
    last_24 = candles[-24:]
    opens = [c[1] for c in last_24]
    highs = [c[3] for c in last_24]
    lows = [c[4] for c in last_24]
    closes = [c[2] for c in last_24]

    return {
        "high": max(highs),
        "low": min(lows),
        "open": opens[0],
        "close": closes[-1],
        "gap_pct": round((max(highs) - min(lows)) / opens[0] * 100, 2),
    }


def build_integrity(candles, symbol):
    tz_local = pytz.timezone("Africa/Johannesburg")

    first = candles[0][0]
    last = candles[-1][0]

    return {
        "symbol": symbol,
        "interval": INTERVAL,
        "requested_limit": LIMIT,
        "returned_count": len(candles),
        "first_candle_open_time_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
        "last_candle_open_time_utc": datetime.fromtimestamp(last, timezone.utc).isoformat(),
        "first_candle_open_time_local": datetime.fromtimestamp(first, tz_local).isoformat(),
        "last_candle_open_time_local": datetime.fromtimestamp(last, tz_local).isoformat(),
    }


def run():
    print("RUNNING FETCH_AND_COMPUTE_V2")

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(pytz.timezone("Africa/Johannesburg"))

    result = {
        "date": now_local.strftime("%Y-%m-%d"),
        "timezone": "Africa/Johannesburg",
        "published_at_local": now_local.isoformat(),
        "published_at_utc": now_utc.isoformat(),
        "integrity": {},
        "candles": {},
    }

    for key, symbol in SYMBOLS.items():
        candles = fetch_candles(symbol)

        result["candles"][f"{key}_1h"] = candles
        result["integrity"][key] = build_integrity(candles, symbol)

        stats = compute_24h_stats(candles)
        result[key] = stats

    os.makedirs("data", exist_ok=True)

    with open("data/hourly_signal.json", "w") as f:
        json.dump(result, f, indent=2)

    print("DONE — hourly_signal.json written")


if __name__ == "__main__":
    run()