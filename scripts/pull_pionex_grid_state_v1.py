#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytz
import requests

API_BASE = "https://api.pionex.com"
BOT_LIST_PATH = "/api/v1/bot/orders"
BOT_DETAIL_PATH = "/api/v1/bot/orders/spotGrid/order"
TICKERS_PATH = "/api/v1/market/tickers"
BASE, QUOTE, SYMBOL = "ETH", "USDT", "ETH_USDT"
LOCAL_TZ = pytz.timezone("Africa/Johannesburg")

LATEST_JSON = Path("data/pionex/api_bot_state_latest_v1.json")
HISTORY_CSV = Path("data/pionex/api_grid_states_v1.csv")
PARITY_JSON = Path("data/diagnostics/pionex_api_parity_v1.json")
MANUAL_CSV = Path("data/pionex/manual_grid_states_v1.csv")

SAFE_DATA_FIELDS = {
    "status", "top", "bottom", "row", "lossStopType", "lossStop", "lossStopDelay",
    "profitStopType", "profitStop", "profitStopDelay", "condition", "conditionDirection",
    "baseTotalInvestment", "quoteTotalInvestment", "gridType", "openPrice", "earnCoin",
    "trend", "createTime", "baseInvestment", "quoteInvestment", "perVolume", "baseAmount",
    "quoteAmount", "realizedProfit", "gridProfit", "totalCostInBase", "totalCostInQuote",
    "totalFeeInBase", "totalFeeInQuote", "averageCost", "profitWithdrawn", "reasonBy",
    "investCoin", "closeSellModel", "slippage", "pausePrice", "profitAutoReinvest",
    "breakEvenWithoutGridProfit", "breakEvenWithGridProfit", "openQuotePrice",
}
SAFE_OUTER_FIELDS = {"buOrderType", "base", "quote", "status", "createTime", "closeTime", "note"}

HISTORY_FIELDS = [
    "captured_at_local", "captured_at_utc", "source", "pair", "bot_id_hash", "bot_status",
    "current_price_usdt", "lower_limit_usdt", "upper_limit_usdt", "grids",
    "quantity_per_grid_eth", "eth_holdings", "usdt_holdings", "grid_profit_usdt",
    "realized_profit_usdt", "derived_trend_pnl_usdt", "base_total_investment",
    "quote_total_investment", "base_investment", "quote_investment", "start_price_usdt",
    "grid_type", "average_cost_usdt", "profit_withdrawn_usdt", "total_fee_eth",
    "total_fee_usdt", "take_profit_type", "take_profit", "stop_loss_type", "stop_loss",
    "trigger_price", "trigger_direction", "reinvest_profits_automatically",
    "break_even_without_grid_profit", "break_even_with_grid_profit", "runtime_hours",
    "unknown_detail_keys",
]


def fnum(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def canonical_query(params: dict[str, Any]) -> str:
    return "&".join(f"{k}={v}" for k, v in sorted((str(k), str(v)) for k, v in params.items() if v is not None))


def signed_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = os.environ.get("PIONEX_API_KEY")
    api_secret = os.environ.get("PIONEX_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing PIONEX_API_KEY or PIONEX_API_SECRET.")

    q = dict(params or {})
    q["timestamp"] = int(time.time() * 1000)
    query = canonical_query(q)
    payload = f"GET{path}?{query}"
    signature = hmac.new(api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    response = requests.get(
        API_BASE + path,
        params=sorted(q.items()),
        headers={"PIONEX-KEY": api_key, "PIONEX-SIGNATURE": signature},
        timeout=20,
    )
    try:
        body = response.json()
    except Exception as exc:
        raise RuntimeError(f"Pionex returned non-JSON: HTTP {response.status_code}") from exc
    if not response.ok or body.get("result") is not True:
        raise RuntimeError(
            f"Pionex API failed: HTTP {response.status_code}; "
            f"{body.get('code', 'UNKNOWN')}: {body.get('message', 'No message')}"
        )
    return body


def public_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    r = requests.get(API_BASE + path, params=params, timeout=20)
    r.raise_for_status()
    body = r.json()
    if body.get("result") is not True:
        raise RuntimeError(f"Pionex public API failed: {body.get('code')}: {body.get('message')}")
    return body


def choose_bot() -> tuple[dict[str, Any], int]:
    body = signed_get(BOT_LIST_PATH, {
        "status": "running", "base": BASE, "quote": QUOTE, "buOrderTypes": "spot_grid"
    })
    results = ((body.get("data") or {}).get("results") or [])
    matches = [
        x for x in results
        if str(x.get("buOrderType", "")).lower() == "spot_grid"
        and str(x.get("base", "")).upper() == BASE
        and str(x.get("quote", "")).upper() == QUOTE
    ]
    if not matches:
        raise RuntimeError("No running ETH/USDT spot-grid bot returned by Pionex Bot API.")
    matches.sort(key=lambda x: int(x.get("createTime") or 0), reverse=True)
    return matches[0], len(matches)


def current_price_and_ticker() -> tuple[float | None, dict[str, Any]]:
    body = public_get(TICKERS_PATH, {"symbol": SYMBOL})
    tickers = ((body.get("data") or {}).get("tickers") or [])
    if not tickers:
        return None, {}
    t = tickers[0]
    for key in ("close", "last", "lastPrice", "price"):
        p = fnum(t.get(key))
        if p is not None:
            return p, t
    return None, t


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def read_latest_manual() -> dict[str, str] | None:
    if not MANUAL_CSV.exists():
        return None
    with MANUAL_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def delta(a: Any, b: Any) -> float | None:
    x, y = fnum(a), fnum(b)
    return None if x is None or y is None else x - y


def main() -> None:
    selected, match_count = choose_bot()
    bot_id = str(selected.get("buOrderId") or "")
    if not bot_id:
        raise RuntimeError("Selected bot is missing buOrderId.")

    detail_body = signed_get(BOT_DETAIL_PATH, {"buOrderId": bot_id})
    detail = detail_body.get("data") or {}
    data = detail.get("buOrderData") or {}
    if not detail or not data:
        raise RuntimeError("Pionex spot-grid detail response is empty.")

    price, ticker = current_price_and_ticker()
    now = datetime.now(timezone.utc)
    local = now.astimezone(LOCAL_TZ)

    unknown_outer = sorted(k for k in detail if k not in SAFE_OUTER_FIELDS | {"buOrderData", "buOrderId", "userId", "keyId"})
    unknown_data = sorted(k for k in data if k not in SAFE_DATA_FIELDS | {"buOrderId"})

    safe_outer = {k: detail.get(k) for k in SAFE_OUTER_FIELDS if k in detail}
    safe_data = {k: data.get(k) for k in SAFE_DATA_FIELDS if k in data}

    bot_id_source = str(detail.get("buOrderId") or data.get("buOrderId") or bot_id)
    safe = {
        "schema": "pionex_api_bot_state_v1",
        "generated_at_utc": now.isoformat(),
        "source": {
            "provider": "Pionex official Open API",
            "mode": "READ_ONLY_BOT_READING",
            "endpoints": [BOT_LIST_PATH, BOT_DETAIL_PATH, TICKERS_PATH],
            "writes_to_pionex": False,
        },
        "selection": {
            "base": BASE, "quote": QUOTE, "running_spot_grid_matches": match_count,
            "bot_id_hash": short_hash(bot_id_source),
        },
        "order": safe_outer,
        "grid": safe_data,
        "market_ticker": {
            "symbol": SYMBOL,
            "current_price_usdt": price,
            "raw_safe": {k: ticker.get(k) for k in ("symbol", "time", "open", "close", "high", "low", "volume", "amount", "count") if k in ticker},
        },
        "schema_observation": {
            "unknown_outer_keys": unknown_outer,
            "unknown_grid_detail_keys": unknown_data,
            "note": "Only names of undocumented keys are persisted; their values are omitted.",
        },
    }

    realized = fnum(data.get("realizedProfit"))
    grid_profit = fnum(data.get("gridProfit"))
    trend = realized - grid_profit if realized is not None and grid_profit is not None else None
    create_ms = detail.get("createTime") or data.get("createTime")
    runtime_h = None
    if create_ms:
        try:
            runtime_h = (now.timestamp() - float(create_ms) / 1000.0) / 3600.0
        except Exception:
            pass

    row = {
        "captured_at_local": local.isoformat(), "captured_at_utc": now.isoformat(),
        "source": "PIONEX_BOT_API", "pair": f"{BASE}/{QUOTE}",
        "bot_id_hash": short_hash(bot_id_source), "bot_status": data.get("status") or detail.get("status"),
        "current_price_usdt": price, "lower_limit_usdt": fnum(data.get("bottom")),
        "upper_limit_usdt": fnum(data.get("top")), "grids": data.get("row"),
        "quantity_per_grid_eth": fnum(data.get("perVolume")), "eth_holdings": fnum(data.get("baseAmount")),
        "usdt_holdings": fnum(data.get("quoteAmount")), "grid_profit_usdt": grid_profit,
        "realized_profit_usdt": realized, "derived_trend_pnl_usdt": trend,
        "base_total_investment": fnum(data.get("baseTotalInvestment")),
        "quote_total_investment": fnum(data.get("quoteTotalInvestment")),
        "base_investment": fnum(data.get("baseInvestment")), "quote_investment": fnum(data.get("quoteInvestment")),
        "start_price_usdt": fnum(data.get("openPrice")), "grid_type": data.get("gridType"),
        "average_cost_usdt": fnum(data.get("averageCost")), "profit_withdrawn_usdt": fnum(data.get("profitWithdrawn")),
        "total_fee_eth": fnum(data.get("totalFeeInBase")), "total_fee_usdt": fnum(data.get("totalFeeInQuote")),
        "take_profit_type": data.get("profitStopType"), "take_profit": data.get("profitStop"),
        "stop_loss_type": data.get("lossStopType"), "stop_loss": data.get("lossStop"),
        "trigger_price": data.get("condition"), "trigger_direction": data.get("conditionDirection"),
        "reinvest_profits_automatically": data.get("profitAutoReinvest"),
        "break_even_without_grid_profit": fnum(data.get("breakEvenWithoutGridProfit")),
        "break_even_with_grid_profit": fnum(data.get("breakEvenWithGridProfit")),
        "runtime_hours": round(runtime_h, 4) if runtime_h is not None else None,
        "unknown_detail_keys": ";".join(unknown_data),
    }

    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    exists = HISTORY_CSV.exists() and HISTORY_CSV.stat().st_size > 0
    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})

    manual = read_latest_manual()
    parity = {
        "schema": "pionex_api_parity_v1",
        "generated_at_utc": now.isoformat(),
        "promotion_status": "VALIDATION_STAGE_DO_NOT_REPLACE_MANUAL_CSV",
        "api_snapshot": {k: row.get(k) for k in (
            "captured_at_local", "current_price_usdt", "lower_limit_usdt", "upper_limit_usdt", "grids",
            "quantity_per_grid_eth", "eth_holdings", "usdt_holdings", "grid_profit_usdt",
            "realized_profit_usdt", "derived_trend_pnl_usdt", "start_price_usdt",
            "reinvest_profits_automatically")},
        "field_availability": {
            "range": row["lower_limit_usdt"] is not None and row["upper_limit_usdt"] is not None,
            "grids": row["grids"] is not None,
            "quantity_per_grid": row["quantity_per_grid_eth"] is not None,
            "holdings": row["eth_holdings"] is not None and row["usdt_holdings"] is not None,
            "grid_profit": row["grid_profit_usdt"] is not None,
            "total_or_realized_profit": row["realized_profit_usdt"] is not None,
            "start_price": row["start_price_usdt"] is not None,
            "rounds_24h_documented": False,
            "rounds_total_documented": False,
            "avg_transactions_per_day_documented": False,
        },
        "undocumented_grid_keys_seen": unknown_data,
        "latest_manual_available": manual is not None,
        "comparison_note": "Dynamic values may drift because manual and API captures occur at different times; geometry/config fields are the primary parity check.",
    }
    if manual:
        parity["latest_manual"] = {k: manual.get(k) for k in (
            "captured_at_local", "current_price_usdt", "lower_limit_usdt", "upper_limit_usdt", "grids",
            "quantity_per_grid_eth", "eth_holdings", "usdt_holdings", "grid_profit_usdt", "start_price_usdt")}
        parity["geometry_match_vs_latest_manual"] = {
            "lower_match": delta(row["lower_limit_usdt"], manual.get("lower_limit_usdt")) == 0,
            "upper_match": delta(row["upper_limit_usdt"], manual.get("upper_limit_usdt")) == 0,
            "grids_match": delta(row["grids"], manual.get("grids")) == 0,
        }
        parity["dynamic_deltas_vs_latest_manual"] = {
            "price_usdt": delta(row["current_price_usdt"], manual.get("current_price_usdt")),
            "quantity_per_grid_eth": delta(row["quantity_per_grid_eth"], manual.get("quantity_per_grid_eth")),
            "eth_holdings": delta(row["eth_holdings"], manual.get("eth_holdings")),
            "usdt_holdings": delta(row["usdt_holdings"], manual.get("usdt_holdings")),
            "grid_profit_usdt": delta(row["grid_profit_usdt"], manual.get("grid_profit_usdt")),
            "start_price_usdt": delta(row["start_price_usdt"], manual.get("start_price_usdt")),
        }
    PARITY_JSON.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Pionex read-only API snapshot captured successfully.")
    print(f"Pair: {BASE}/{QUOTE}; running matches: {match_count}")
    print(f"Range: {row['lower_limit_usdt']} - {row['upper_limit_usdt']} / {row['grids']} grids")
    print(f"Current price: {row['current_price_usdt']}; grid profit: {row['grid_profit_usdt']}")
    print(f"Undocumented grid keys: {row['unknown_detail_keys'] or '(none)'}")
    print("No Pionex write endpoint was called.")


if __name__ == "__main__":
    main()
