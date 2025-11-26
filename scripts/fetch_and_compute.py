#!/usr/bin/env python

import json
import os
from datetime import datetime, timezone, timedelta
from statistics import mean

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    # Fallback if you ever run this manually on an older Python
    from backports.zoneinfo import ZoneInfo  # type: ignore

# KuCoin public spot candles endpoint (no API key required)
KUCOIN_URL = "https://api.kucoin.com/api/v1/market/candles"
LOCAL_TZ = ZoneInfo("Africa/Johannesburg")


def fetch_klines(symbol: str, interval: str = "1hour", limit: int = 48):
    """
    Fetch OHLCV candles from KuCoin for the given symbol.

    KuCoin response format (each entry):
      [
        "1545904980",  # Start time (unix seconds)
        "0.058",       # Open
        "0.049",       # Close
        "0.058",       # High
        "0.049",       # Low
        "0.018",       # Transaction amount
        "0.000945"     # Transaction volume
      ]

    We:
      - Convert times to Africa/Johannesburg
      - Convert numbers to floats
      - Sort by time ascending
      - Return only the last `limit` candles
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

        open_price = float(entry[1])
        close_price = float(entry[2])
        high = float(entry[3])
        low = float(entry[4])
        volume = float(entry[6])

        candles.append(
            {
                "open_time": open_dt_local,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
            }
        )

    candles.sort(key=lambda c: c["open_time"])
    if len(candles) > limit:
        candles = candles[-limit:]

    if not candles:
        raise ValueError(f"No candles returned for symbol {symbol}")

    return candles


def compute_24h_stats(candles):
    """
    Compute 24h high, low, open, close, and gap% from the last 24 candles.
    We treat:
      - open = first candle open in the last 24
      - close = last candle close in the last 24
      - high/low = extrema over the last 24 candles
    gap% = (high - low) / low * 100
    """
    if len(candles) < 24:
        raise ValueError("Not enough candles to compute 24h stats")

    last_24 = candles[-24:]
    high = max(c["high"] for c in last_24)
    low = min(c["low"] for c in last_24)
    open_24 = last_24[0]["open"]
    close_24 = last_24[-1]["close"]
    gap_pct = (high - low) / low * 100 if low != 0 else 0.0

    return {
        "high": round(high, 6),
        "low": round(low, 6),
        "open": round(open_24, 6),
        "close": round(close_24, 6),
        "gap_pct": round(gap_pct, 2),
    }


def compute_yesterday_range(candles):
    """
    Compute yesterday's high and low based on local Africa/Johannesburg dates.
    Returns dict: {"high": float|None, "low": float|None}
    """
    today = datetime.now(LOCAL_TZ).date()
    yesterday = today - timedelta(days=1)

    y_candles = [c for c in candles if c["open_time"].date() == yesterday]
    if not y_candles:
        return {"high": None, "low": None}

    y_high = max(c["high"] for c in y_candles)
    y_low = min(c["low"] for c in y_candles)

    return {"high": round(y_high, 6), "low": round(y_low, 6)}


def compute_today_first_4h_range(candles, hours_window: int = 4):
    """
    Compute today's high/low over the first `hours_window` hours of the day,
    using local Africa/Johannesburg dates and hour-of-day.

    Returns dict: {"high": float|None, "low": float|None}
    """
    today = datetime.now(LOCAL_TZ).date()
    t_candles = [
        c
        for c in candles
        if c["open_time"].date() == today and c["open_time"].hour < hours_window
    ]
    if not t_candles:
        return {"high": None, "low": None}

    t_high = max(c["high"] for c in t_candles)
    t_low = min(c["low"] for c in t_candles)

    return {"high": round(t_high, 6), "low": round(t_low, 6)}


def compute_atr_and_trend(candles, period: int = 14, lookback: int = 10):
    """
    Compute ATR(period) with a simple moving average of True Range (TR),
    and infer a basic trend: 'rising', 'falling', or 'flat'.

    Returns (atr_value: float|None, trend: str)
    """
    if len(candles) < period + 1:
        return None, "flat"

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    trs = []
    for i in range(1, len(candles)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)

    atr_values = []
    for i in range(period - 1, len(trs)):
        window = trs[i - period + 1 : i + 1]
        atr_values.append(mean(window))

    if not atr_values:
        return None, "flat"

    latest_atr = atr_values[-1]

    recent = atr_values[-lookback:] if len(atr_values) >= lookback else atr_values
    first = recent[0]
    change_pct = (latest_atr - first) / first * 100 if first else 0.0

    if change_pct > 5:
        trend = "rising"
    elif change_pct < -5:
        trend = "falling"
    else:
        trend = "flat"

    return round(latest_atr, 4), trend


def compute_early_breakout(candles, hours_window: int = 4):
    """
    Early breakout flag: did today's price break above/below
    yesterday's high/low in the first `hours_window` hours (local time)?

    Returns (occurred: bool, description: str)
    """
    today = datetime.now(LOCAL_TZ).date()
    yesterday = today - timedelta(days=1)

    y_candles = [c for c in candles if c["open_time"].date() == yesterday]
    t_candles = [c for c in candles if c["open_time"].date() == today]

    if not y_candles or not t_candles:
        return False, "Not enough data to evaluate early breakout."

    prev_high = max(c["high"] for c in y_candles)
    prev_low = min(c["low"] for c in y_candles)

    first_hours = [c for c in t_candles if c["open_time"].hour < hours_window]
    if not first_hours:
        return False, "No candles in the early session."

    broke_above = any(c["high"] > prev_high for c in first_hours)
    broke_below = any(c["low"] < prev_low for c in first_hours)

    if broke_above and broke_below:
        return (
            True,
            "Price broke both above yesterday's high and below yesterday's low in the first few hours.",
        )
    elif broke_above:
        return True, "Price broke above yesterday's high in the first few hours."
    elif broke_below:
        return True, "Price broke below yesterday's low in the first few hours."
    else:
        return False, "No breakout of yesterday's range in the first few hours."


def compute_intraday_momentum(candles, atr_value):
    """
    Compute basic intraday momentum for ETH/USDT:

      - Use today's local date.
      - day_open = first candle open of today (if available),
        otherwise use the first of the last-24h window.
      - day_close = last candle close of today (if available),
        otherwise use the last close of the last-24h window.

    Rules:
      close > open  -> "up"
      close < open  -> "down"
      |close - open| small vs ATR -> "sideways"
    """
    if not candles:
        return "sideways"

    today = datetime.now(LOCAL_TZ).date()
    today_candles = [c for c in candles if c["open_time"].date() == today]

    if today_candles:
        day_open = today_candles[0]["open"]
        day_close = today_candles[-1]["close"]
    else:
        # Fallback to last 24 candles as an approximation of "intraday"
        last_24 = candles[-24:] if len(candles) >= 24 else candles
        day_open = last_24[0]["open"]
        day_close = last_24[-1]["close"]

    diff = day_close - day_open
    diff_abs = abs(diff)

    # If ATR is missing or 0, use simple direction
    if not atr_value or atr_value == 0:
        if diff > 0:
            return "up"
        elif diff < 0:
            return "down"
        else:
            return "sideways"

    # If move is small relative to ATR, consider sideways
    if diff_abs < 0.3 * atr_value:
        return "sideways"
    else:
        if diff > 0:
            return "up"
        elif diff < 0:
            return "down"
        else:
            return "sideways"


def compute_precomputed_signal(
    eth_usdt_stats,
    atr_value,
    atr_trend,
    early_breakout_flag,
    intraday_momentum,
):
    """
    Generate a simple, interpretable deployment suggestion:

    suggested_target: "skip" | "1%" | "2%" | "3%"
    confidence: "low" | "medium" | "high"
    notes: free-text explanation
    """
    gap = eth_usdt_stats.get("gap_pct", 0.0)
    high = eth_usdt_stats.get("high", 0.0)

    # Default fallbacks
    target = "2%"
    confidence = "medium"
    notes = "Standard deployment conditions."

    if atr_value is None or high == 0:
        target = "2%"
        confidence = "low"
        notes = "ATR unavailable or price invalid; using fallback target."
        return {
            "suggested_target": target,
            "confidence": confidence,
            "notes": notes,
        }

    atr_pct_of_price = atr_value / high * 100 if high else 0.0
    high_vol = atr_pct_of_price > 1.5 or gap > 7.0

    if high_vol and early_breakout_flag:
        target = "skip"
        confidence = "medium"
        notes = (
            "High volatility with early breakout; consider skipping new deployments."
        )
    elif early_breakout_flag and intraday_momentum == "up":
        target = "3%"
        confidence = "high"
        notes = "Upside early breakout with positive momentum; more aggressive deployment target."
    elif early_breakout_flag and intraday_momentum == "down":
        target = "1%"
        confidence = "medium"
        notes = "Downside early breakout with negative momentum; use a conservative target."
    elif not early_breakout_flag and atr_trend == "falling":
        target = "3%"
        confidence = "medium"
        notes = "ATR trend is falling with no breakout; range contraction may favour moderate deployments."
    else:
        target = "2%"
        confidence = "medium"
        notes = "Mixed conditions; standard deployment target."

    return {
        "suggested_target": target,
        "confidence": confidence,
        "notes": notes,
    }


def main():
    # Fetch candles from KuCoin
    eth_usdt_candles = fetch_klines("ETH-USDT")
    eth_btc_candles = fetch_klines("ETH-BTC")

    # 24h stats (including open/close)
    eth_usdt_stats = compute_24h_stats(eth_usdt_candles)
    eth_btc_stats = compute_24h_stats(eth_btc_candles)

    # Yesterday's range (ETH/USDT)
    yesterday_range = compute_yesterday_range(eth_usdt_candles)

    # Today's first 4h range (ETH/USDT)
    today_first_4h = compute_today_first_4h_range(eth_usdt_candles, hours_window=4)

    # ATR + trend on ETH/USDT
    atr_value, atr_trend = compute_atr_and_trend(eth_usdt_candles, period=14)

    # Early breakout detection on ETH/USDT
    early_breakout_flag, early_breakout_desc = compute_early_breakout(
        eth_usdt_candles, hours_window=4
    )

    # Intraday momentum on ETH/USDT
    intraday_momentum = compute_intraday_momentum(eth_usdt_candles, atr_value)

    # Precomputed signal (uses gap, ATR, breakout, momentum)
    pre_signal = compute_precomputed_signal(
        eth_usdt_stats,
        atr_value,
        atr_trend,
        early_breakout_flag,
        intraday_momentum,
    )

    today_str = datetime.now(LOCAL_TZ).date().isoformat()

    payload = {
        "date": today_str,
        "timezone": "Africa/Johannesburg",
        "eth_usdt": eth_usdt_stats,
        "eth_btc": eth_btc_stats,
        "yesterday": yesterday_range,
        "today_first_4h": today_first_4h,
        "atr_1h": {
            "value": atr_value,
            "periods": 14,
            "trend": atr_trend,
        },
        "intraday_momentum": intraday_momentum,
        "early_breakout": {
            "occurred": early_breakout_flag,
            "description": early_breakout_desc,
        },
        "precomputed_signal": pre_signal,
    }

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "today_signal.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
