#!/usr/bin/env python3
"""
Build the canonical Pionex model-state history from the read-only API ledger.

Architecture
------------
1) data/pionex/api_grid_states_v1.csv
   Raw API observations. This may be hourly (or more frequent).

2) data/pionex/model_grid_states_v1.csv
   Canonical calibration history. Seeded once from the historic manual screenshot
   CSV, then extended automatically:
     - append immediately when live grid geometry changes; or
     - append the first API observation >= 23h after the previous canonical row.

3) data/pionex/runtime_grid_states_v1.csv
   Canonical history plus exactly one fresh latest API observation when necessary.
   Phase 4D can consume this runtime file without polluting calibration with a
   long sequence of hourly adjacent rows.

The historic manual_grid_states_v1.csv remains an immutable audit source.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_API_PATH = Path("data/pionex/api_grid_states_v1.csv")
MANUAL_ARCHIVE_PATH = Path("data/pionex/manual_grid_states_v1.csv")
MODEL_PATH = Path("data/pionex/model_grid_states_v1.csv")
RUNTIME_PATH = Path("data/pionex/runtime_grid_states_v1.csv")
SOURCE_DIAG_PATH = Path("data/diagnostics/pionex_state_source_v1.json")

PAIR = "ETH/USDT"
DAILY_CANONICAL_MIN_H = 23.0
ROUND24_TARGET_H = 24.0
ROUND24_TOL_H = 1.5

MANUAL_FIELDS = [
    "captured_at_local",
    "captured_at_utc",
    "pair",
    "investment_usdt",
    "current_profit_usdt",
    "current_profit_pct",
    "grid_profit_usdt",
    "grid_profit_pct",
    "trend_pnl_usdt",
    "trend_pnl_pct",
    "grid_annualized_pct",
    "total_annualized_pct",
    "rounds_24h",
    "rounds_total",
    "avg_transactions_per_day",
    "lower_limit_usdt",
    "upper_limit_usdt",
    "grids",
    "current_price_usdt",
    "eth_holdings",
    "usdt_holdings",
    "quantity_per_grid_eth",
    "start_price_usdt",
    "profit_per_grid_min_pct_fee_deducted",
    "profit_per_grid_max_pct_fee_deducted",
    "take_profit",
    "stop_loss",
    "trigger_price",
    "trailing_up",
    "reinvest_profits_automatically",
    "runtime_display",
    "source_note",
]


def fnum(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def inum(v: Any) -> int | None:
    x = fnum(v)
    return None if x is None else int(round(x))


def parse_dt(v: str) -> datetime:
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manual_schema(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANUAL_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANUAL_FIELDS})


def normalize_bool(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v or "").strip().lower()
    return "true" if s in {"true", "1", "yes", "on"} else "false"


def blank_if_zero(v: Any) -> Any:
    x = fnum(v)
    if x is None or abs(x) < 1e-15:
        return ""
    return v


def runtime_display(hours: float | None) -> str:
    if hours is None or hours < 0:
        return ""
    total_minutes = int(round(hours * 60.0))
    days, rem = divmod(total_minutes, 24 * 60)
    hrs, mins = divmod(rem, 60)
    return f"{days}D {hrs}H {mins}m"


def profit_per_grid_bounds(lower: float | None, upper: float | None, grids: int | None) -> tuple[Any, Any]:
    """
    Pionex screenshot values are consistent with arithmetic spacing and roughly
    0.05% fee per side. We use a conservative display-compatible truncation to
    two decimals. These fields are descriptive; Phase 4D recomputes its own grid
    economics and does not rely on them.
    """
    if lower is None or upper is None or grids is None or grids <= 1 or upper <= lower:
        return "", ""
    spacing = (upper - lower) / (grids - 1)
    gross_at_low = spacing / lower * 100.0
    gross_at_high_buy = spacing / (upper - spacing) * 100.0
    round_trip_fee_pct = 0.10
    hi = max(0.0, gross_at_low - round_trip_fee_pct)
    lo = max(0.0, gross_at_high_buy - round_trip_fee_pct)
    trunc2 = lambda x: math.floor((x + 1e-12) * 100.0) / 100.0
    return trunc2(lo), trunc2(hi)


def same_geometry(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("lower_limit_usdt", "upper_limit_usdt", "grids", "quantity_per_grid_eth")
    tolerances = {
        "lower_limit_usdt": 1e-6,
        "upper_limit_usdt": 1e-6,
        "grids": 0.0,
        "quantity_per_grid_eth": 1e-10,
    }
    for key in keys:
        av, bv = fnum(a.get(key)), fnum(b.get(key))
        if av is None or bv is None:
            return False
        if abs(av - bv) > tolerances[key]:
            return False
    return True


def raw_paired_count(row: dict[str, Any]) -> int | None:
    # v1.1 field name retained for backward compatibility after UI parity
    # conclusively established exchangeOrderPairedCount == Pionex History rounds.
    return inum(
        row.get("rounds_total")
        if row.get("rounds_total") not in (None, "")
        else row.get("paired_count_candidate")
    )


def derive_rounds_24h(raw_rows: list[dict[str, str]], idx: int, current_count: int | None) -> int | None:
    if current_count is None:
        return None
    current_dt = parse_dt(raw_rows[idx]["captured_at_utc"])
    best = None
    best_abs = None
    for j in range(idx - 1, -1, -1):
        prior_count = raw_paired_count(raw_rows[j])
        if prior_count is None:
            continue
        prior_dt = parse_dt(raw_rows[j]["captured_at_utc"])
        age_h = (current_dt - prior_dt).total_seconds() / 3600.0
        if age_h > ROUND24_TARGET_H + ROUND24_TOL_H:
            break
        err = abs(age_h - ROUND24_TARGET_H)
        if err <= ROUND24_TOL_H and (best_abs is None or err < best_abs):
            best = current_count - prior_count
            best_abs = err
    return best if best is not None and best >= 0 else None


def api_row_to_model(
    raw: dict[str, str],
    raw_rows: list[dict[str, str]],
    raw_idx: int,
    reason: str,
    previous_model: dict[str, Any] | None,
) -> dict[str, Any]:
    lower = fnum(raw.get("lower_limit_usdt"))
    upper = fnum(raw.get("upper_limit_usdt"))
    grids = inum(raw.get("grids"))
    qty = fnum(raw.get("quantity_per_grid_eth"))
    paired = raw_paired_count(raw)

    investment = fnum(raw.get("investment_usdt"))
    if investment is None:
        investment = fnum(raw.get("quote_total_investment"))
    total_pnl = fnum(raw.get("total_pnl_usdt_mark_to_market"))
    total_pct = fnum(raw.get("total_pnl_pct_mark_to_market"))
    gp = fnum(raw.get("grid_profit_usdt"))
    gp_pct = fnum(raw.get("grid_profit_pct"))
    trend = fnum(raw.get("trend_pnl_usdt_mark_to_market"))
    trend_pct = fnum(raw.get("trend_pnl_pct_mark_to_market"))
    runtime_h = fnum(raw.get("runtime_hours"))

    rounds24 = derive_rounds_24h(raw_rows, raw_idx, paired)
    # Before a full 24h API history exists, a ~daily canonical delta is a useful
    # near-equivalent. Calibration itself uses rounds_total deltas, not this field.
    if rounds24 is None and previous_model is not None and paired is not None:
        prev_count = inum(previous_model.get("rounds_total"))
        try:
            elapsed_h = (
                parse_dt(raw["captured_at_utc"]) - parse_dt(previous_model["captured_at_utc"])
            ).total_seconds() / 3600.0
        except Exception:
            elapsed_h = None
        if prev_count is not None and elapsed_h is not None and 22.0 <= elapsed_h <= 26.0:
            d = paired - prev_count
            if d >= 0:
                rounds24 = d

    avg_tx = None
    if paired is not None and runtime_h and runtime_h > 0:
        avg_tx = paired / (runtime_h / 24.0)

    grid_ann = None
    if gp_pct is not None and runtime_h and runtime_h > 0:
        grid_ann = gp_pct * (365.0 * 24.0 / runtime_h)

    pmin, pmax = profit_per_grid_bounds(lower, upper, grids)

    row = {
        "captured_at_local": raw.get("captured_at_local", ""),
        "captured_at_utc": raw.get("captured_at_utc", ""),
        "pair": PAIR,
        "investment_usdt": investment if investment is not None else "",
        "current_profit_usdt": total_pnl if total_pnl is not None else "",
        "current_profit_pct": total_pct if total_pct is not None else "",
        "grid_profit_usdt": gp if gp is not None else "",
        "grid_profit_pct": gp_pct if gp_pct is not None else "",
        "trend_pnl_usdt": trend if trend is not None else "",
        "trend_pnl_pct": trend_pct if trend_pct is not None else "",
        "grid_annualized_pct": round(grid_ann, 6) if grid_ann is not None else "",
        # Pionex's displayed total annualized return does not follow the same
        # simple convention as grid annualized. Leave blank rather than invent it.
        "total_annualized_pct": "",
        "rounds_24h": rounds24 if rounds24 is not None else "",
        "rounds_total": paired if paired is not None else "",
        "avg_transactions_per_day": round(avg_tx, 6) if avg_tx is not None else "",
        "lower_limit_usdt": lower if lower is not None else "",
        "upper_limit_usdt": upper if upper is not None else "",
        "grids": grids if grids is not None else "",
        "current_price_usdt": fnum(raw.get("current_price_usdt")) or "",
        "eth_holdings": fnum(raw.get("eth_holdings")) if fnum(raw.get("eth_holdings")) is not None else "",
        "usdt_holdings": fnum(raw.get("usdt_holdings")) if fnum(raw.get("usdt_holdings")) is not None else "",
        "quantity_per_grid_eth": qty if qty is not None else "",
        "start_price_usdt": fnum(raw.get("start_price_usdt")) or "",
        "profit_per_grid_min_pct_fee_deducted": pmin,
        "profit_per_grid_max_pct_fee_deducted": pmax,
        "take_profit": blank_if_zero(raw.get("take_profit")),
        "stop_loss": blank_if_zero(raw.get("stop_loss")),
        "trigger_price": blank_if_zero(raw.get("trigger_price")),
        "trailing_up": "",
        "reinvest_profits_automatically": normalize_bool(raw.get("reinvest_profits_automatically")),
        "runtime_display": runtime_display(runtime_h),
        "source_note": (
            f"Automated read-only Pionex Bot API state; canonical_reason={reason}; "
            "rounds_total is confirmed exchangeOrderPairedCount; total/trend P&L are "
            "mark-to-market reconstructions from bot balances and ETH/USDT ticker."
        ),
    }
    return row


def ensure_model_seeded() -> tuple[list[dict[str, str]], bool]:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return read_csv(MODEL_PATH), False
    manual = read_csv(MANUAL_ARCHIVE_PATH)
    if not manual:
        raise SystemExit("Cannot seed model states: manual_grid_states_v1.csv is missing/empty.")
    write_manual_schema(MODEL_PATH, manual)
    return read_csv(MODEL_PATH), True


def main() -> None:
    raw_rows = read_csv(RAW_API_PATH)
    if not raw_rows:
        raise SystemExit("No API state history available.")

    # Sort and discard malformed timestamps.
    valid_raw = []
    for r in raw_rows:
        try:
            r["_dt"] = parse_dt(r["captured_at_utc"])
        except Exception:
            continue
        valid_raw.append(r)
    valid_raw.sort(key=lambda r: r["_dt"])
    # Re-materialize without private helper key for conversion functions.
    raw_rows = [{k: v for k, v in r.items() if k != "_dt"} for r in valid_raw]

    model_rows, seeded = ensure_model_seeded()
    model_rows.sort(key=lambda r: parse_dt(r["captured_at_utc"]))
    appended = []

    for idx, raw in enumerate(raw_rows):
        current_dt = parse_dt(raw["captured_at_utc"])
        latest_model = model_rows[-1]
        latest_dt = parse_dt(latest_model["captured_at_utc"])
        if current_dt <= latest_dt:
            continue

        provisional = api_row_to_model(raw, raw_rows, idx, "PROVISIONAL", latest_model)
        elapsed_h = (current_dt - latest_dt).total_seconds() / 3600.0
        geometry_changed = not same_geometry(latest_model, provisional)

        if geometry_changed:
            reason = "STRUCTURAL_BREAK_GEOMETRY_CHANGE"
        elif elapsed_h >= DAILY_CANONICAL_MIN_H:
            reason = "DAILY_CANONICAL_GE_23H"
        else:
            continue

        row = api_row_to_model(raw, raw_rows, idx, reason, latest_model)
        model_rows.append(row)
        appended.append({
            "captured_at_utc": row["captured_at_utc"],
            "reason": reason,
            "elapsed_h_from_previous_canonical": round(elapsed_h, 4),
            "geometry_changed": geometry_changed,
        })

    write_manual_schema(MODEL_PATH, model_rows)

    # Runtime view = canonical history plus one current API row only.
    latest_raw = raw_rows[-1]
    latest_model = model_rows[-1]
    runtime_rows = list(model_rows)
    if parse_dt(latest_raw["captured_at_utc"]) > parse_dt(latest_model["captured_at_utc"]):
        runtime_rows.append(
            api_row_to_model(
                latest_raw,
                raw_rows,
                len(raw_rows) - 1,
                "RUNTIME_LATEST_NOT_CANONICAL",
                latest_model,
            )
        )
    write_manual_schema(RUNTIME_PATH, runtime_rows)

    latest_runtime = runtime_rows[-1]
    source_diag = {
        "schema": "pionex_state_source_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "API_PRIMARY_UI_PARITY_CONFIRMED",
        "historic_manual_archive": str(MANUAL_ARCHIVE_PATH),
        "raw_api_history": str(RAW_API_PATH),
        "canonical_model_history": str(MODEL_PATH),
        "phase4d_runtime_history": str(RUNTIME_PATH),
        "manual_archive_mutated": False,
        "model_seeded_from_manual_this_run": seeded,
        "canonical_rows": len(model_rows),
        "runtime_rows": len(runtime_rows),
        "canonical_rows_appended_this_run": appended,
        "latest_runtime_state": {
            "captured_at_local": latest_runtime.get("captured_at_local"),
            "captured_at_utc": latest_runtime.get("captured_at_utc"),
            "lower_limit_usdt": fnum(latest_runtime.get("lower_limit_usdt")),
            "upper_limit_usdt": fnum(latest_runtime.get("upper_limit_usdt")),
            "grids": inum(latest_runtime.get("grids")),
            "quantity_per_grid_eth": fnum(latest_runtime.get("quantity_per_grid_eth")),
            "current_price_usdt": fnum(latest_runtime.get("current_price_usdt")),
            "rounds_total": inum(latest_runtime.get("rounds_total")),
            "rounds_24h": inum(latest_runtime.get("rounds_24h")),
            "source_note": latest_runtime.get("source_note"),
        },
        "counter_semantics": {
            "exchangeOrderPairedCount": "CONFIRMED_PIONEX_UI_HISTORY_COMPLETED_ROUNDS",
            "trx24h": "NOT_UI_24H_ROUNDS",
            "rounds_24h_policy": (
                "Prefer paired-count delta against nearest raw API observation 24h +/-1.5h; "
                "fallback to ~daily canonical paired-count delta for 22-26h intervals."
            ),
        },
        "canonical_policy": {
            "daily_min_elapsed_h": DAILY_CANONICAL_MIN_H,
            "append_immediately_on_geometry_change": True,
            "geometry_fields": [
                "lower_limit_usdt",
                "upper_limit_usdt",
                "grids",
                "quantity_per_grid_eth",
            ],
            "purpose": (
                "Keep prospective 14-30h calibration windows statistically usable while "
                "allowing raw API capture to run hourly."
            ),
        },
        "runtime_policy": (
            "Phase 4D receives canonical history plus at most one fresh latest API row. "
            "The hourly raw ledger is never passed wholesale into calibration."
        ),
    }
    SOURCE_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_DIAG_PATH.write_text(json.dumps(source_diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Pionex model-state build complete.")
    print("Seeded from historic manual:", seeded)
    print("Canonical rows:", len(model_rows))
    print("Appended this run:", appended or "none")
    print("Runtime rows:", len(runtime_rows))
    print("Latest runtime:", source_diag["latest_runtime_state"])


if __name__ == "__main__":
    main()
