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

# Officially documented fields plus a small set of read-only operational fields
# discovered in the live response. The latter remain SEMANTIC_CANDIDATES until
# checked against a contemporaneous UI screenshot.
SAFE_DATA_FIELDS = {
    "status", "top", "bottom", "row", "lossStopType", "lossStop", "lossStopDelay",
    "profitStopType", "profitStop", "profitStopDelay", "condition", "conditionDirection",
    "baseTotalInvestment", "quoteTotalInvestment", "gridType", "openPrice", "earnCoin",
    "trend", "createTime", "baseInvestment", "quoteInvestment", "perVolume", "baseAmount",
    "quoteAmount", "realizedProfit", "gridProfit", "totalCostInBase", "totalCostInQuote",
    "totalFeeInBase", "totalFeeInQuote", "averageCost", "profitWithdrawn", "reasonBy",
    "investCoin", "closeSellModel", "slippage", "pausePrice", "profitAutoReinvest",
    "breakEvenWithoutGridProfit", "breakEvenWithGridProfit", "openQuotePrice",
    # Undocumented operational candidates discovered 2026-08-20:
    "exchangeOrderPairedCount", "trx24h", "placedExchangeOrderCount",
    "closedExchangeOrderCount", "gridAverageOpenPrice",
}
SAFE_OUTER_FIELDS = {"buOrderType", "base", "quote", "status", "createTime", "closeTime", "note"}

HISTORY_FIELDS = [
    "captured_at_local", "captured_at_utc", "source", "pair", "bot_id_hash", "bot_status",
    "current_price_usdt", "lower_limit_usdt", "upper_limit_usdt", "grids",
    "quantity_per_grid_eth", "eth_holdings", "usdt_holdings", "current_value_usdt",
    "investment_usdt", "total_pnl_usdt_mark_to_market", "total_pnl_pct_mark_to_market",
    "grid_profit_usdt", "grid_profit_pct", "trend_pnl_usdt_mark_to_market",
    "trend_pnl_pct_mark_to_market", "raw_realized_profit_field",
    "paired_count_candidate", "trx24h_candidate", "placed_exchange_order_count",
    "closed_exchange_order_count", "base_total_investment", "quote_total_investment",
    "base_investment", "quote_investment", "start_price_usdt", "grid_type",
    "average_cost_usdt", "grid_average_open_price_usdt", "profit_withdrawn_usdt",
    "total_fee_eth", "total_fee_usdt", "take_profit_type", "take_profit",
    "stop_loss_type", "stop_loss", "trigger_price", "trigger_direction",
    "reinvest_profits_automatically", "break_even_without_grid_profit",
    "break_even_with_grid_profit", "runtime_hours", "unknown_detail_keys",
]


def fnum(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def canonical_query(params: dict[str, Any]) -> str:
    return "&".join(
        f"{k}={v}"
        for k, v in sorted((str(k), str(v)) for k, v in params.items() if v is not None)
    )


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
    body = signed_get(
        BOT_LIST_PATH,
        {"status": "running", "base": BASE, "quote": QUOTE, "buOrderTypes": "spot_grid"},
    )
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


def append_history(row: dict[str, Any]) -> None:
    """Append while safely migrating the single-source CSV if its schema evolves."""
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, str]] = []
    old_fields: list[str] = []

    if HISTORY_CSV.exists() and HISTORY_CSV.stat().st_size > 0:
        with HISTORY_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            old_rows = list(reader)

    if old_fields and old_fields != HISTORY_FIELDS:
        with HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for old in old_rows:
                migrated = {k: old.get(k, "") for k in HISTORY_FIELDS}
                # Preserve old raw fields in the closest semantically neutral destination.
                if "raw_realized_profit_field" in HISTORY_FIELDS and not migrated["raw_realized_profit_field"]:
                    migrated["raw_realized_profit_field"] = old.get("realized_profit_usdt", "")
                writer.writerow(migrated)
        old_fields = HISTORY_FIELDS

    exists = HISTORY_CSV.exists() and HISTORY_CSV.stat().st_size > 0
    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})


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

    unknown_outer = sorted(
        k for k in detail
        if k not in SAFE_OUTER_FIELDS | {"buOrderData", "buOrderId", "userId", "keyId"}
    )
    unknown_data = sorted(k for k in data if k not in SAFE_DATA_FIELDS | {"buOrderId"})
    safe_outer = {k: detail.get(k) for k in SAFE_OUTER_FIELDS if k in detail}
    safe_data = {k: data.get(k) for k in SAFE_DATA_FIELDS if k in data}

    bot_id_source = str(detail.get("buOrderId") or data.get("buOrderId") or bot_id)

    investment = fnum(data.get("quoteTotalInvestment"))
    base_amount = fnum(data.get("baseAmount"))
    quote_amount = fnum(data.get("quoteAmount"))
    grid_profit = fnum(data.get("gridProfit"))
    raw_realized = fnum(data.get("realizedProfit"))

    current_value = None
    total_pnl = None
    total_pnl_pct = None
    grid_profit_pct = None
    trend_pnl = None
    trend_pnl_pct = None

    if price is not None and base_amount is not None and quote_amount is not None:
        current_value = base_amount * price + quote_amount
    if current_value is not None and investment not in (None, 0):
        total_pnl = current_value - investment
        total_pnl_pct = total_pnl / investment * 100.0
    if grid_profit is not None and investment not in (None, 0):
        grid_profit_pct = grid_profit / investment * 100.0
    if total_pnl is not None and grid_profit is not None:
        trend_pnl = total_pnl - grid_profit
        if investment not in (None, 0):
            trend_pnl_pct = trend_pnl / investment * 100.0

    create_ms = detail.get("createTime") or data.get("createTime")
    runtime_h = None
    if create_ms:
        try:
            runtime_h = (now.timestamp() - float(create_ms) / 1000.0) / 3600.0
        except Exception:
            pass

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
            "base": BASE,
            "quote": QUOTE,
            "running_spot_grid_matches": match_count,
            "bot_id_hash": short_hash(bot_id_source),
        },
        "order": safe_outer,
        "grid": safe_data,
        "derived_ui_equivalents": {
            "investment_usdt": investment,
            "current_value_usdt": current_value,
            "total_pnl_usdt": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "grid_profit_usdt": grid_profit,
            "grid_profit_pct": grid_profit_pct,
            "trend_pnl_usdt": trend_pnl,
            "trend_pnl_pct": trend_pnl_pct,
            "method": (
                "mark-to-market: ETH balance * current ETH_USDT ticker + USDT balance - "
                "quoteTotalInvestment; trend = total P&L - gridProfit"
            ),
        },
        "counter_candidates": {
            "exchangeOrderPairedCount": data.get("exchangeOrderPairedCount"),
            "trx24h": data.get("trx24h"),
            "placedExchangeOrderCount": data.get("placedExchangeOrderCount"),
            "closedExchangeOrderCount": data.get("closedExchangeOrderCount"),
            "semantic_status": "CANDIDATE_REQUIRES_UI_PARITY_CHECK",
        },
        "market_ticker": {
            "symbol": SYMBOL,
            "current_price_usdt": price,
            "raw_safe": {
                k: ticker.get(k)
                for k in ("symbol", "time", "open", "close", "high", "low", "volume", "amount", "count")
                if k in ticker
            },
        },
        "schema_observation": {
            "unknown_outer_keys": unknown_outer,
            "unknown_grid_detail_keys": unknown_data,
            "note": "Unknown key names are persisted without values.",
        },
    }

    row = {
        "captured_at_local": local.isoformat(),
        "captured_at_utc": now.isoformat(),
        "source": "PIONEX_BOT_API",
        "pair": f"{BASE}/{QUOTE}",
        "bot_id_hash": short_hash(bot_id_source),
        "bot_status": data.get("status") or detail.get("status"),
        "current_price_usdt": price,
        "lower_limit_usdt": fnum(data.get("bottom")),
        "upper_limit_usdt": fnum(data.get("top")),
        "grids": data.get("row"),
        "quantity_per_grid_eth": fnum(data.get("perVolume")),
        "eth_holdings": base_amount,
        "usdt_holdings": quote_amount,
        "current_value_usdt": current_value,
        "investment_usdt": investment,
        "total_pnl_usdt_mark_to_market": total_pnl,
        "total_pnl_pct_mark_to_market": total_pnl_pct,
        "grid_profit_usdt": grid_profit,
        "grid_profit_pct": grid_profit_pct,
        "trend_pnl_usdt_mark_to_market": trend_pnl,
        "trend_pnl_pct_mark_to_market": trend_pnl_pct,
        "raw_realized_profit_field": raw_realized,
        "paired_count_candidate": data.get("exchangeOrderPairedCount"),
        "trx24h_candidate": data.get("trx24h"),
        "placed_exchange_order_count": data.get("placedExchangeOrderCount"),
        "closed_exchange_order_count": data.get("closedExchangeOrderCount"),
        "base_total_investment": fnum(data.get("baseTotalInvestment")),
        "quote_total_investment": investment,
        "base_investment": fnum(data.get("baseInvestment")),
        "quote_investment": fnum(data.get("quoteInvestment")),
        "start_price_usdt": fnum(data.get("openPrice")),
        "grid_type": data.get("gridType"),
        "average_cost_usdt": fnum(data.get("averageCost")),
        "grid_average_open_price_usdt": fnum(data.get("gridAverageOpenPrice")),
        "profit_withdrawn_usdt": fnum(data.get("profitWithdrawn")),
        "total_fee_eth": fnum(data.get("totalFeeInBase")),
        "total_fee_usdt": fnum(data.get("totalFeeInQuote")),
        "take_profit_type": data.get("profitStopType"),
        "take_profit": data.get("profitStop"),
        "stop_loss_type": data.get("lossStopType"),
        "stop_loss": data.get("lossStop"),
        "trigger_price": data.get("condition"),
        "trigger_direction": data.get("conditionDirection"),
        "reinvest_profits_automatically": data.get("profitAutoReinvest"),
        "break_even_without_grid_profit": fnum(data.get("breakEvenWithoutGridProfit")),
        "break_even_with_grid_profit": fnum(data.get("breakEvenWithGridProfit")),
        "runtime_hours": round(runtime_h, 4) if runtime_h is not None else None,
        "unknown_detail_keys": ";".join(unknown_data),
    }

    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_history(row)

    manual = read_latest_manual()
    parity = {
        "schema": "pionex_api_parity_v1",
        "generated_at_utc": now.isoformat(),
        "promotion_status": "VALIDATION_STAGE_COUNTER_SEMANTICS_PENDING",
        "api_snapshot": {
            k: row.get(k)
            for k in (
                "captured_at_local", "current_price_usdt", "lower_limit_usdt",
                "upper_limit_usdt", "grids", "quantity_per_grid_eth", "eth_holdings",
                "usdt_holdings", "current_value_usdt", "investment_usdt",
                "total_pnl_usdt_mark_to_market", "total_pnl_pct_mark_to_market",
                "grid_profit_usdt", "grid_profit_pct", "trend_pnl_usdt_mark_to_market",
                "trend_pnl_pct_mark_to_market", "start_price_usdt",
                "reinvest_profits_automatically", "paired_count_candidate", "trx24h_candidate",
                "placed_exchange_order_count", "closed_exchange_order_count",
            )
        },
        "semantic_corrections": {
            "realizedProfit": (
                "Raw API field retained for audit only. It returned 0 on the first live snapshot "
                "and is NOT treated as the Pionex UI Total P&L."
            ),
            "total_pnl": "Derived from mark-to-market bot balances minus quoteTotalInvestment.",
            "trend_pnl": "Derived as total_pnl - gridProfit.",
        },
        "counter_candidates": safe["counter_candidates"],
        "latest_manual_available": manual is not None,
        "comparison_note": (
            "Dynamic values drift between capture times. Geometry/config are direct parity checks; "
            "counter candidates require a contemporaneous UI screenshot before promotion."
        ),
        "undocumented_grid_keys_seen": unknown_data,
    }

    if manual:
        parity["latest_manual"] = {
            k: manual.get(k)
            for k in (
                "captured_at_local", "current_price_usdt", "lower_limit_usdt",
                "upper_limit_usdt", "grids", "quantity_per_grid_eth", "eth_holdings",
                "usdt_holdings", "grid_profit_usdt", "start_price_usdt",
                "current_profit_usdt", "current_profit_pct", "trend_pnl_usdt",
                "trend_pnl_pct", "rounds_24h", "rounds_total",
            )
        }
        parity["geometry_match_vs_latest_manual"] = {
            "lower_match": delta(row["lower_limit_usdt"], manual.get("lower_limit_usdt")) == 0,
            "upper_match": delta(row["upper_limit_usdt"], manual.get("upper_limit_usdt")) == 0,
            "grids_match": delta(row["grids"], manual.get("grids")) == 0,
            "quantity_per_grid_match": delta(
                row["quantity_per_grid_eth"], manual.get("quantity_per_grid_eth")
            ) == 0,
            "start_price_match": delta(row["start_price_usdt"], manual.get("start_price_usdt")) == 0,
        }

    PARITY_JSON.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Pionex read-only API snapshot captured successfully.")
    print(f"Range: {row['lower_limit_usdt']} - {row['upper_limit_usdt']} / {row['grids']} grids")
    print(
        "Mark-to-market UI equivalents:",
        f"total={row['total_pnl_usdt_mark_to_market']},",
        f"grid={row['grid_profit_usdt']},",
        f"trend={row['trend_pnl_usdt_mark_to_market']}",
    )
    print(
        "Counter candidates:",
        f"paired={row['paired_count_candidate']},",
        f"trx24h={row['trx24h_candidate']},",
        f"placed={row['placed_exchange_order_count']},",
        f"closed={row['closed_exchange_order_count']}",
    )
    print("No Pionex write endpoint was called.")


if __name__ == "__main__":
    main()
