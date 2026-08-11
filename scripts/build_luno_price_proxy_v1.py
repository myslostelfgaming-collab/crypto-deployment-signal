#!/usr/bin/env python3
"""
Luno Price Proxy Collector V1
=============================

Collects public Luno exchange tickers and constructs a Luno-native BTC/USD
proxy without authentication.

Why a proxy?
------------
Luno Price Predict settles on Luno's internal market rate, which is not exposed
by the public exchange ticker API as a dedicated Price Predict settlement feed.
We therefore collect Luno-native exchange prices now so that we can measure the
basis between:
  - KuCoin BTC-USDT used by the existing forecasting engine, and
  - a Luno-local BTC/USD-ish proxy built from Luno exchange markets.

The collector is deliberately non-fatal. A temporary Luno API outage writes an
unavailable latest-state JSON but does not break the main hourly workflow.

Outputs
-------
- data/luno/luno_price_proxy_latest_v1.json
- data/luno/luno_price_proxy_history_v1.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import requests

LUNO_TICKERS_URL = "https://api.luno.com/api/1/tickers"
SIM_MULTI_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")
SIM_BTC_PATH = os.path.join("data", "model", "similarity_forecast_v2_BTC-USDT.json")
OUT_LATEST = os.path.join("data", "luno", "luno_price_proxy_latest_v1.json")
OUT_HISTORY = os.path.join("data", "luno", "luno_price_proxy_history_v1.csv")
SCHEMA = "luno_price_proxy_v1"

FIELDS = [
    "observed_at_utc",
    "status",
    "luno_api_timestamp_utc",
    "xbtzar_price",
    "usdtzar_price",
    "usdczar_price",
    "xbtusdt_price",
    "xbtusdc_price",
    "proxy_xbtzar_usdtzar",
    "proxy_xbtzar_usdczar",
    "proxy_direct_xbtusdt",
    "proxy_direct_xbtusdc",
    "luno_usd_proxy",
    "proxy_count",
    "market_btc_usdt",
    "basis_usd",
    "basis_pct",
    "error",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def representative_price(ticker: Dict[str, Any]) -> Optional[float]:
    """Use bid/ask midpoint when possible; otherwise fall back to last trade."""
    bid = safe_float(ticker.get("bid"))
    ask = safe_float(ticker.get("ask"))
    last = safe_float(ticker.get("last_trade"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    return None


def load_market_btc_usdt() -> Optional[float]:
    # Prefer the dedicated BTC similarity artifact.
    for path in (SIM_BTC_PATH, SIM_MULTI_PATH):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if path == SIM_MULTI_PATH:
                payload = (payload.get("forecasts") or {}).get("BTC-USDT") or {}
            value = safe_float((payload.get("current_state") or {}).get("entry_close"))
            if value is not None:
                return value
        except Exception:
            continue
    return None


def fetch_tickers() -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    resp = requests.get(LUNO_TICKERS_URL, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    tickers = payload.get("tickers") or []
    by_pair: Dict[str, Dict[str, Any]] = {}
    latest_ms: Optional[int] = None
    for ticker in tickers:
        pair = str(ticker.get("pair") or "").upper().strip()
        if not pair:
            continue
        by_pair[pair] = ticker
        try:
            ts = int(ticker.get("timestamp") or 0)
            if ts > 0:
                latest_ms = max(latest_ms or ts, ts)
        except Exception:
            pass

    latest_iso = None
    if latest_ms:
        latest_iso = datetime.fromtimestamp(latest_ms / 1000.0, tz=timezone.utc).isoformat()
    return by_pair, latest_iso


def pair_price(tickers: Dict[str, Dict[str, Any]], pair: str) -> Optional[float]:
    ticker = tickers.get(pair)
    return representative_price(ticker) if ticker else None


def cross_proxy(
    xbt_quote: Optional[float],
    stable_quote: Optional[float],
) -> Optional[float]:
    if xbt_quote is None or stable_quote is None or stable_quote <= 0:
        return None
    return xbt_quote / stable_quote


def ensure_history_file() -> None:
    os.makedirs(os.path.dirname(OUT_HISTORY), exist_ok=True)
    if os.path.isfile(OUT_HISTORY):
        return
    with open(OUT_HISTORY, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def append_history(row: Dict[str, Any]) -> None:
    ensure_history_file()
    with open(OUT_HISTORY, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def write_latest(payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(OUT_LATEST), exist_ok=True)
    with open(OUT_LATEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    observed = now_iso()
    market_btc = load_market_btc_usdt()

    try:
        tickers, luno_ts = fetch_tickers()

        xbtzar = pair_price(tickers, "XBTZAR")
        usdtzar = pair_price(tickers, "USDTZAR")
        usdczar = pair_price(tickers, "USDCZAR")
        xbtusdt = pair_price(tickers, "XBTUSDT")
        xbtusdc = pair_price(tickers, "XBTUSDC")

        proxy_cross_usdt = cross_proxy(xbtzar, usdtzar)
        proxy_cross_usdc = cross_proxy(xbtzar, usdczar)

        proxies = [
            p for p in [proxy_cross_usdt, proxy_cross_usdc, xbtusdt, xbtusdc]
            if p is not None and p > 0
        ]
        luno_proxy = median(proxies) if proxies else None

        basis_usd = None
        basis_pct = None
        if luno_proxy is not None and market_btc is not None and market_btc > 0:
            basis_usd = luno_proxy - market_btc
            basis_pct = 100.0 * basis_usd / market_btc

        status = "AVAILABLE" if luno_proxy is not None else "NO_USD_PROXY_MARKETS"
        row = {
            "observed_at_utc": observed,
            "status": status,
            "luno_api_timestamp_utc": luno_ts,
            "xbtzar_price": round_or_none(xbtzar, 4),
            "usdtzar_price": round_or_none(usdtzar, 6),
            "usdczar_price": round_or_none(usdczar, 6),
            "xbtusdt_price": round_or_none(xbtusdt, 4),
            "xbtusdc_price": round_or_none(xbtusdc, 4),
            "proxy_xbtzar_usdtzar": round_or_none(proxy_cross_usdt, 4),
            "proxy_xbtzar_usdczar": round_or_none(proxy_cross_usdc, 4),
            "proxy_direct_xbtusdt": round_or_none(xbtusdt, 4),
            "proxy_direct_xbtusdc": round_or_none(xbtusdc, 4),
            "luno_usd_proxy": round_or_none(luno_proxy, 4),
            "proxy_count": len(proxies),
            "market_btc_usdt": round_or_none(market_btc, 4),
            "basis_usd": round_or_none(basis_usd, 4),
            "basis_pct": round_or_none(basis_pct, 6),
            "error": "",
        }
        append_history(row)

        payload = {
            "schema": SCHEMA,
            "generated_at_utc": observed,
            "available": luno_proxy is not None,
            "status": status,
            "source": {
                "luno_endpoint": LUNO_TICKERS_URL,
                "luno_market_data_requires_auth": False,
                "settlement_warning": (
                    "This is a Luno exchange-derived proxy. Luno Price Predict uses "
                    "Luno's internal market rate, so this value must not be treated as "
                    "the exact competition settlement price."
                ),
            },
            "luno_api_timestamp_utc": luno_ts,
            "market_prices": {
                "XBTZAR": round_or_none(xbtzar, 4),
                "USDTZAR": round_or_none(usdtzar, 6),
                "USDCZAR": round_or_none(usdczar, 6),
                "XBTUSDT": round_or_none(xbtusdt, 4),
                "XBTUSDC": round_or_none(xbtusdc, 4),
            },
            "proxy_components": {
                "XBTZAR_div_USDTZAR": round_or_none(proxy_cross_usdt, 4),
                "XBTZAR_div_USDCZAR": round_or_none(proxy_cross_usdc, 4),
                "XBTUSDT_direct": round_or_none(xbtusdt, 4),
                "XBTUSDC_direct": round_or_none(xbtusdc, 4),
            },
            "luno_usd_proxy": round_or_none(luno_proxy, 4),
            "proxy_count": len(proxies),
            "market_btc_usdt": round_or_none(market_btc, 4),
            "basis": {
                "usd": round_or_none(basis_usd, 4),
                "pct": round_or_none(basis_pct, 6),
            },
        }
        write_latest(payload)
        print(
            "Luno price proxy:",
            payload.get("luno_usd_proxy"),
            "market BTC-USDT:",
            payload.get("market_btc_usdt"),
            "basis_pct:",
            (payload.get("basis") or {}).get("pct"),
        )

    except Exception as exc:
        row = {
            "observed_at_utc": observed,
            "status": "ERROR",
            "market_btc_usdt": round_or_none(market_btc, 4),
            "error": str(exc),
        }
        append_history(row)
        write_latest({
            "schema": SCHEMA,
            "generated_at_utc": observed,
            "available": False,
            "status": "ERROR",
            "error": str(exc),
            "market_btc_usdt": round_or_none(market_btc, 4),
            "note": "Collector failure is non-fatal to the main crypto workflow.",
        })
        print("Luno price proxy unavailable:", exc)


if __name__ == "__main__":
    main()
