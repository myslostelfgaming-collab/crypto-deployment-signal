#!/usr/bin/env python

import json
import os
from datetime import datetime, timezone
from statistics import mean

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore


KUCOIN_URL = "https://api.kucoin.com/api/v1/market/candles"
LOCAL_TZ = ZoneInfo("Africa/Johannesburg")


def fetch_klines(symbol: str, interval: str = "1hour", limit: int = 400):
    """
    Fetch OHLCV candles from KuCoin and return the last `limit` candles,
    sorted by time ascending. Uses LOCAL_TZ timestamps for open_time.
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
                "open_time_utc": open_dt_utc,
                "open": float(entry[1]),
                "close": float(entry[2]),
                "high": float(entry[3]),
                "low": float(entry[4]),
                "volume": float(entry[6]),
            }
        )

    candles.sort(key=lambda c: c["open_time"])
    return candles[-limit:]


def serialize_candles(candles):
    """
    Convert candle dicts to compact history format:
    [ts_utc, open, high, low, close, volume]
    """
    out = []
    for c in candles:
        out.append(
            [
                int(c["open_time_utc"].timestamp()),
                round(c["open"], 8),
                round(c["high"], 8),
                round(c["low"], 8),
                round(c["close"], 8),
                round(c["volume"], 12),
            ]
        )
    return out


def _window_slice(candles, start_idx_inclusive, end_idx_exclusive):
    w = candles[start_idx_inclusive:end_idx_exclusive]
    if not w:
        return None
    return w


def _range_high_low(window):
    return max(c["high"] for c in window), min(c["low"] for c in window)


def compute_24h_stats(candles):
    if len(candles) < 24:
        raise ValueError("Not enough candles for 24h stats")

    last_24 = candles[-24:]
    high, low = _range_high_low(last_24)

    return {
        "high": round(high, 6),
        "low": round(low, 6),
        "open": round(last_24[0]["open"], 6),
        "close": round(last_24[-1]["close"], 6),
        "gap_pct": round((high - low) / low * 100, 2) if low else 0.0,
    }


def compute_prior_24h_range(candles):
    """
    Rolling prior 24h window = candles[-48:-24].
    """
    if len(candles) < 48:
        return {"high": None, "low": None}

    prev_24 = candles[-48:-24]
    high, low = _range_high_low(prev_24)

    return {"high": round(high, 6), "low": round(low, 6)}


def compute_first_4h_of_current_24h(candles):
    """
    Rolling 'first 4h' of the CURRENT 24h window:
      current_24h = candles[-24:]
      first_4h = current_24h[:4] -> candles[-24:-20]
    """
    if len(candles) < 24:
        return {"high": None, "low": None}

    current_24 = candles[-24:]
    first_4 = current_24[:4]

    if len(first_4) < 4:
        return {"high": None, "low": None}

    high, low = _range_high_low(first_4)
    return {"high": round(high, 6), "low": round(low, 6)}


def compute_wilder_atr_and_trend(candles, period=14, lookback=10):
    """
    Wilder ATR (RMA):
      - TR computed from candle i and prev close (i-1)
      - ATR_0 = SMA(TR[0:period])
      - ATR_t = (ATR_{t-1}*(period-1) + TR_t) / period
    """
    if len(candles) < period + 1:
        return None, "flat"

    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if len(trs) < period:
        return None, "flat"

    atrs = []
    first_atr = mean(trs[:period])
    atrs.append(first_atr)

    for j in range(period, len(trs)):
        prev_atr = atrs[-1]
        atr = (prev_atr * (period - 1) + trs[j]) / period
        atrs.append(atr)

    latest_atr = atrs[-1]

    if len(atrs) < lookback:
        return round(latest_atr, 4), "flat"

    base = atrs[-lookback]
    change_pct = ((latest_atr - base) / base * 100) if base else 0.0

    if change_pct > 5:
        trend = "rising"
    elif change_pct < -5:
        trend = "falling"
    else:
        trend = "flat"

    return round(latest_atr, 4), trend


def compute_early_breakout_rolling(candles, hours_window=4):
    """
    Rolling breakout check:
      - prior_24h = candles[-48:-24]
      - current_first_4h = candles[-24:-20]
      - breakout if first 4h breaks prior range
    """
    if len(candles) < 48:
        return False, "Not enough candles for rolling breakout."

    prior_24 = candles[-48:-24]
    current_first_4 = candles[-24:-20]

    prev_high, prev_low = _range_high_low(prior_24)

    broke_above = any(c["high"] > prev_high for c in current_first_4)
    broke_below = any(c["low"] < prev_low for c in current_first_4)

    if broke_above and broke_below:
        return True, "Broke above and below prior 24h range in first 4h (rolling)."
    if broke_above:
        return True, "Broke above prior 24h high in first 4h (rolling)."
    if broke_below:
        return True, "Broke below prior 24h low in first 4h (rolling)."

    return False, "No early breakout (rolling)."


def compute_intraday_momentum_rolling(candles, atr):
    """
    Momentum computed over rolling current 24h window:
      diff = close(last) - open(first)
      classified using ATR threshold
    """
    if len(candles) < 24:
        return "sideways"

    w = candles[-24:]
    open_p = w[0]["open"]
    close_p = w[-1]["close"]
    diff = close_p - open_p

    if not atr or abs(diff) < 0.3 * atr:
        return "sideways"
    return "up" if diff > 0 else "down"


def compute_precomputed_signal(stats, atr, atr_trend, early_breakout, momentum):
    # Advisory-only (does not override your v2 framework)
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

    if (not early_breakout) and atr_trend == "falling":
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


def build_integrity_meta(symbol, interval, requested_limit, candles, now_local):
    returned_count = len(candles)
    last_local = candles[-1]["open_time"] if candles else None
    last_utc = candles[-1]["open_time_utc"] if candles else None
    first_local = candles[0]["open_time"] if candles else None
    first_utc = candles[0]["open_time_utc"] if candles else None

    now_utc = now_local.astimezone(timezone.utc)
    freshness_min = None
    if last_utc:
        freshness_min = round((now_utc - last_utc).total_seconds() / 60.0, 2)

    def iso(dt):
        return dt.isoformat(timespec="seconds") if dt else None

    w_last_24 = _window_slice