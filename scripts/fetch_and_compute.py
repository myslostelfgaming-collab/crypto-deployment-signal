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

    # KuCoin wraps data under "data"
    data = raw.get("data", [])
    candles = []
    for entry in data:
        # Safety: skip malformed entries
        if len(entry) < 7:
            continue

        # entry[0] is unix seconds as string
        ts = int(entry[0])
        open_dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        open_dt_local = open_dt_utc.astimezone(LOCAL_TZ)

        open_price = float(entry[1])
        close_price = float(entry[2])
        high = float(entry[3])
        low = float(entry[4])
        # entry[5] amount, entry[6] volume
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

    # Sort by time ascending, then trim to last `limit`
    candles.sort(key=lambda c: c["open_time"])
    if len(candles) > limit:
        candles = candles[-limit:]

    if not candles:
        raise ValueError(f"No candles returned for symbol {symbol}")

    return candles


def compute_24h_stats(candles):
    """
    Compute 24h high, low, and gap% from the last 24 candles.
    gap% = (high - low) / low * 100
    """
    if len(candles) < 24:
        raise ValueError("Not enough candles to compute 24h stats")

    last_24 = candles[-24:]
    high = max(c["high"] for c in last_24)
    low = min(c["low"] for c in last_24)
    gap_pct = (high - low) / low * 100 if low != 0 else 0.0

    return {
        "high": round(high, 6),
        "low": round(low, 6),
        "gap_pct": round(gap_pct, 2),
    }


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

    # True Range series starts from candle 1 (requires previous close)
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

    # Simple moving ATR over 'period' TR values
    atr_values = []
    for i in range(period - 1, len(trs)):
        window = trs[i - period + 1 : i + 1]
        atr_values.append(mean(window))

    if not atr_values:
        return None, "flat"

    latest_atr = atr_values[-1]

    # Trend: compare first vs last ATR in recent window
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


def compute_early_breakout(candles, hours_window: int = 4) -> bool:
    """
    Early breakout flag: did today's price break above/below
    yesterday's high/low in the first `hours_window` hours (local time)?

    Uses local dates in Africa/Johannesburg.
    """
    today = datetime.now(LOCAL_TZ).date()
    yesterday = today - timedelta(days=1)

    y_candles = [c for c in candles if c["open_time"].date() == yesterday]
    t_candles = [c for c in candles if c["open_time"].date() == today]

    if not y_candles or not t_candles:
        # Not enough data to determine
        return False

    prev_high = max(c["high"] for c in y_candles)
    prev_low = min(c["low"] for c in y_candles)

    first_hours = [c for c in t_candles if c["open_time"].hour < hours_window]
    if not first_hours:
        return False

    for c in first_hours:
        if c["high"] > prev_high or c["low"] < prev_low:
            return True

    return False


def compute_precomputed_signal(eth_usdt_stats, atr_value):
    """
    Simple placeholder rule for suggested_target.
    You can upgrade this logic later.

    Example rule:
      - If 24h gap is large (> 6%) or ATR is relatively big, be more conservative (2%).
      - Otherwise, default to 3%.
    """
    gap = eth_usdt_stats.get("gap_pct", 0.0)
    high = eth_usdt_stats.get("high", 0.0)

    if atr_value is None or high == 0:
        target = "3%"
    else:
        # ATR as % of price
        atr_pct_of_price = atr_value / high * 100 if high else 0.0
        if gap > 6 or atr_pct_of_price > 1.0:
            target = "2%"
        else:
            target = "3%"

    return {"suggested_target": target}


def main():
    # Fetch candles from KuCoin
    eth_usdt_candles = fetch_klines("ETH-USDT")
    eth_btc_candles = fetch_klines("ETH-BTC")

    # 24h stats
    eth_usdt_stats = compute_24h_stats(eth_usdt_candles)
    eth_btc_stats = compute_24h_stats(eth_btc_candles)

    # ATR + trend on ETH/USDT
    atr_value, atr_trend = compute_atr_and_trend(eth_usdt_candles)

    # Early breakout detection on ETH/USDT
    early_breakout_flag = compute_early_breakout(eth_usdt_candles)

    # Precomputed signal (placeholder logic)
    pre_signal = compute_precomputed_signal(eth_usdt_stats, atr_value)

    today_str = datetime.now(LOCAL_TZ).date().isoformat()

    payload = {
        "date": today_str,
        "timezone": "Africa/Johannesburg",
        "eth_usdt": eth_usdt_stats,
        "eth_btc": eth_btc_stats,
        "atr_1h": {
            "value": atr_value,
            "trend": atr_trend,
        },
        "early_breakout": {
            "occurred": early_breakout_flag,
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
