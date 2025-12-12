#!/usr/bin/env python

import json
import os
from datetime import datetime, timezone, timedelta
from statistics import mean

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore


# KuCoin public spot candles endpoint (no API key required)
KUCOIN_URL = "https://api.kucoin.com/api/v1/market/candles"
LOCAL_TZ = ZoneInfo("Africa/Johannesburg")


def fetch_klines(symbol: str, interval: str = "1hour", limit: int = 48):
    """
    Fetch OHLCV candles from KuCoin and return the last `limit` candles,
    sorted by time ascending.
    """
    params = {"symbol": symbol, "type": interval}
    resp = requests.get(KUCOIN_URL, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    data = raw.get("data", [])
    candles = []

    for entry in data:
        if len(entry) < 7:
            continue

        ts = int(entry[0])
        open_dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        open_dt_local = open_dt_utc.astimezone(LOCAL_TZ)

        candles.append(
            {
                "open_time": open_dt_local,
                "open": float(entry[1]),
                "close": float(entry[2]),
                "high": float(entry[3]),
                "low": float(entry[4]),
                "volume": float(entry[6]),
            }
        )

    candles.sort(key=lambda c: c["open_time"])
    return candles[-limit:]


def compute_24h_stats(candles):
    if len(candles) < 24:
        raise ValueError("Not enough candles for 24h stats")

    last_24 = candles[-24:]
    high = max(c["high"] for c in last_24)
    low = min(c["low"] for c in last_24)

    return {
        "high": round(high, 6),
        "low": round(low, 6),
        "open": round(last_24[0]["open"], 6),
        "close": round(last_24[-1]["close"], 6),
        "gap_pct": round((high - low) / low * 100, 2) if low else 0.0,
    }


def compute_yesterday_range(candles):
    """
    Rolling prior 24h window (candles -48 to -24).
    """
    if len(candles) < 48:
        return {"high": None, "low": None}

    prev_24 = candles[-48:-24]
    return {
        "high": round(max(c["high"] for c in prev_24), 6),
        "low": round(min(c["low"] for c in prev_24), 6),
    }


def compute_today_first_4h_range(candles, hours_window=4):
    today = datetime.now(LOCAL_TZ).date()
    early = [
        c for c in candles
        if c["open_time"].date() == today and c["open_time"].hour < hours_window
    ]

    if not early:
        return {"high": None, "low": None}

    return {
        "high": round(max(c["high"] for c in early), 6),
        "low": round(min(c["low"] for c in early), 6),
    }


def compute_atr_and_trend(candles, period=14, lookback=10):
    if len(candles) < period + 1:
        return None, "flat"

    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    atrs = [mean(trs[i - period + 1:i + 1]) for i in range(period - 1, len(trs))]
    latest_atr = atrs[-1]

    recent = atrs[-lookback:]
    change_pct = ((latest_atr - recent[0]) / recent[0] * 100) if recent[0] else 0

    if change_pct > 5:
        trend = "rising"
    elif change_pct < -5:
        trend = "falling"
    else:
        trend = "flat"

    return round(latest_atr, 4), trend


def compute_early_breakout(candles, hours_window=4):
    today = datetime.now(LOCAL_TZ).date()
    yesterday = today - timedelta(days=1)

    y = [c for c in candles if c["open_time"].date() == yesterday]
    t = [c for c in candles if c["open_time"].date() == today]

    if not y or not t:
        return False, "Not enough data."

    prev_high = max(c["high"] for c in y)
    prev_low = min(c["low"] for c in y)

    early = [c for c in t if c["open_time"].hour < hours_window]

    broke_above = any(c["high"] > prev_high for c in early)
    broke_below = any(c["low"] < prev_low for c in early)

    if broke_above and broke_below:
        return True, "Broke above and below yesterday's range early."
    if broke_above:
        return True, "Broke above yesterday's high early."
    if broke_below:
        return True, "Broke below yesterday's low early."

    return False, "No early breakout."


def compute_intraday_momentum(candles, atr):
    today = datetime.now(LOCAL_TZ).date()
    t = [c for c in candles if c["open_time"].date() == today]

    if t:
        open_p = t[0]["open"]
        close_p = t[-1]["close"]
    else:
        open_p = candles[-24]["open"]
        close_p = candles[-1]["close"]

    diff = close_p - open_p

    if not atr or abs(diff) < 0.3 * atr:
        return "sideways"
    return "up" if diff > 0 else "down"


def compute_precomputed_signal(stats, atr, atr_trend, early_breakout, momentum):
    gap = stats["gap_pct"]
    high = stats["high"]

    if not atr or not high:
        return {
            "suggested_target": "2%",
            "confidence": "low",
            "notes": "ATR unavailable; fallback mode.",
        }

    atr_pct = atr / high * 100
    high_vol = atr_pct > 1.5 or gap > 7

    if high_vol and early_breakout:
        return {
            "suggested_target": "skip",
            "confidence": "medium",
            "notes": "High volatility with early breakout.",
        }

    if early_breakout and momentum == "up":
        return {
            "suggested_target": "3%",
            "confidence": "high",
            "notes": "Upside early breakout with momentum.",
        }

    if early_breakout and momentum == "down":
        return {
            "suggested_target": "1%",
            "confidence": "medium",
            "notes": "Downside breakout; conservative target.",
        }

    if not early_breakout and atr_trend == "falling":
        return {
            "suggested_target": "3%",
            "confidence": "medium",
            "notes": "ATR contracting; range opportunity.",
        }

    return {
        "suggested_target": "2%",
        "confidence": "medium",
        "notes": "Mixed conditions.",
    }


def main():
    eth_usdt = fetch_klines("ETH-USDT")
    eth_btc = fetch_klines("ETH-BTC")

    eth_usdt_stats = compute_24h_stats(eth_usdt)
    eth_btc_stats = compute_24h_stats(eth_btc)

    yesterday = compute_yesterday_range(eth_usdt)
    first_4h = compute_today_first_4h_range(eth_usdt)

    atr, atr_trend = compute_atr_and_trend(eth_usdt)
    early_flag, early_desc = compute_early_breakout(eth_usdt)
    momentum = compute_intraday_momentum(eth_usdt, atr)

    signal = compute_precomputed_signal(
        eth_usdt_stats, atr, atr_trend, early_flag, momentum
    )

    now_local = datetime.now(LOCAL_TZ)

    payload = {
        "date": now_local.date().isoformat(),
        "timezone": "Africa/Johannesburg",
        "published_at_local": now_local.isoformat(timespec="seconds"),
        "published_at_utc": now_local.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "eth_usdt": eth_usdt_stats,
        "eth_btc": eth_btc_stats,
        "yesterday": yesterday,
        "today_first_4h": first_4h,
        "atr_1h": {"value": atr, "periods": 14, "trend": atr_trend},
        "intraday_momentum": momentum,
        "early_breakout": {"occurred": early_flag, "description": early_desc},
        "precomputed_signal": signal,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/today_signal.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()