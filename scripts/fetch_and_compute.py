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

BINANCE_URL = "https://api.binance.com/api/v3/klines"
LOCAL_TZ = ZoneInfo("Africa/Johannesburg")


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 48):
    """
    Fetch OHLCV candles from Binance for the given symbol.

    Returns a list of dicts with:
      - open_time (datetime, local)
      - open, high, low, close, volume (floats)
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_URL, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    candles = []
    for entry in raw:
        # Binance kline format:
        # 0 open time (ms), 1 open, 2 high, 3 low, 4 close, 5 volume, ...
        open_time_ms = entry[0]
        open_price = float(entry[1])
        high = float(entry[2])
        low = float(entry[3])
        close = float(entry[4])
        volume = float(entry[5])

        open_dt_utc = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        open_dt_local = open_dt_utc.astimezone(LOCAL_TZ)

        candles.append(
            {
                "open_time": open_dt_local,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

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
    # Fetch candles
    eth_usdt_candles = fetch_klines("ETHUSDT")
    eth_btc_candles = fetch_klines("ETHBTC")

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
