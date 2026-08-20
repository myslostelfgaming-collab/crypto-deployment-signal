#!/usr/bin/env python3
"""
Phase 4D v3.1 — live-price bridge + edge/out-of-grid recovery adapter.

This wraps the existing Phase 4D v2 execution-resolution adapter.

Fixes two live-edge failure modes:

1) LIVE PRICE SOURCE
   The legacy Phase 4D optimizer used the latest hourly feature's entry_close as
   both the regime reference and the live geometry/sizing price. With automated
   Pionex API state, this can be 0-60 minutes stale relative to the bot state.
   v3.1 keeps the hourly feature timestamp/ATR/path-shape conditioning, but uses
   the fresh Pionex API state's current_price_usdt as the market price for:
     - current geometry feasibility,
     - candidate geometry placement,
     - quantity sizing,
     - historical-path remapping.

2) EDGE / OUT-OF-GRID LIVE BOT
   The legacy candidate quantity function requires >=2 buy intervals and >=2
   sell intervals. That is a sensible candidate-grid guard, but a valid live bot
   can naturally have only 0-1 intervals remaining on one side near/after a
   boundary. v3.1 evaluates such a live current geometry with its observed
   quantity/balances and enters a recenter-recovery selection mode instead of
   aborting.

Historical in-grid calibration logic is otherwise preserved. No Pionex write
endpoint is used.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import build_pionex_grid_geometry_optimizer_v1 as base
import build_pionex_grid_geometry_optimizer_v2 as integration
from pionex_out_of_grid_support_v1 import corrected_initial_states

GEO_PATH = Path("data/diagnostics/pionex_grid_geometry_optimizer_v1.json")

_ORIGINAL_EVALUATE = base.evaluate_geometry
_ORIGINAL_SELECT = base.select_geometry
_ORIGINAL_LOAD_FEATURES = base.sim.load_eth_features

RECOVERY_RISK_BAND_PP = 5.0
RECOVERY_TAIL_TOL_USDT = 0.50
MIN_ACTIVE_INTERVALS_EACH_SIDE = 2

_LIVE_PRICE_BRIDGE = {
    "installed": False,
    "state_price_usdt": None,
    "original_hourly_feature_close_usdt": None,
    "latest_feature_ts_utc": None,
}


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _same_live_geometry(state, lower, upper, grids):
    return (
        abs(_f(state.get("lower_limit_usdt")) - float(lower)) <= 1e-6
        and abs(_f(state.get("upper_limit_usdt")) - float(upper)) <= 1e-6
        and int(float(state.get("grids"))) == int(grids)
    )


def _current_grid_side_counts(state, market_price):
    lower = _f(state.get("lower_limit_usdt"))
    upper = _f(state.get("upper_limit_usdt"))
    grids = int(float(state.get("grids")))
    lines = base.sim.grid_lines(lower, upper, grids)
    states = corrected_initial_states(lines, market_price)
    return {
        "buy_intervals": sum(st == "buy" for st in states),
        "sell_intervals": sum(st == "sell" for st in states),
    }


def _recovery_trigger(state, market_price):
    lower = _f(state.get("lower_limit_usdt"))
    upper = _f(state.get("upper_limit_usdt"))

    if market_price > upper:
        return "ABOVE_UPPER"
    if market_price < lower:
        return "BELOW_LOWER"

    counts = _current_grid_side_counts(state, market_price)
    if counts["sell_intervals"] < MIN_ACTIVE_INTERVALS_EACH_SIDE:
        return "NEAR_UPPER_EDGE"
    if counts["buy_intervals"] < MIN_ACTIVE_INTERVALS_EACH_SIDE:
        return "NEAR_LOWER_EDGE"
    return None


def _install_live_price_bridge():
    """
    Preserve hourly regime/feature timing but replace only the latest feature's
    entry_close with the fresh Pionex API price consumed by this run.

    The optimizer's current_price is read from current_feature['entry_close'].
    This bridge therefore makes live geometry/sizing use the API price without
    rewriting the validated Phase 4D core.

    Historical feature rows remain untouched, so walk-forward calibration and
    matured analogue paths preserve their historical prices.
    """
    states = base.phase4c.read_manual_states()
    if not states:
        raise SystemExit("No Pionex state available for live-price bridge")

    live_state = states[-1]
    live_price = _f(live_state.get("current_price_usdt"), None)
    if live_price is None or live_price <= 0:
        raise SystemExit("Latest Pionex state lacks a valid current_price_usdt")

    original = _ORIGINAL_LOAD_FEATURES()
    if not original:
        raise SystemExit("No ETH feature rows for live-price bridge")

    original_close = _f(original[-1].get("entry_close"), None)
    latest_ts = original[-1].get("ts")

    def _load_features_with_live_price():
        rows = _ORIGINAL_LOAD_FEATURES()
        if not rows:
            return rows
        copied = [dict(r) for r in rows]
        copied[-1]["entry_close"] = live_price
        return copied

    base.sim.load_eth_features = _load_features_with_live_price

    _LIVE_PRICE_BRIDGE.update({
        "installed": True,
        "state_price_usdt": live_price,
        "original_hourly_feature_close_usdt": original_close,
        "latest_feature_ts_utc": latest_ts,
    })


def _candidate_rebalance_seed(
    state,
    current_price,
    lower,
    upper,
    grids,
    fee_rate,
    active_notional_proxy,
):
    """
    Estimate post-edit quantity and seed balances from preserved active-grid
    notional plus current mark-to-market equity.

    The pre-edit ETH/USDT split is not imposed on a hypothetical new geometry,
    because Pionex may rebalance assets when editing the live range. Pionex's
    edit screen remains authoritative for actual quantity/grid and minimums.
    """
    lines = base.sim.grid_lines(lower, upper, grids)
    states = base.sim.initial_states(lines, current_price)

    sell_idx = [idx for idx, st in enumerate(states) if st == "sell"]
    buy_idx = [idx for idx, st in enumerate(states) if st == "buy"]
    if len(sell_idx) < MIN_ACTIVE_INTERVALS_EACH_SIDE or len(buy_idx) < MIN_ACTIVE_INTERVALS_EACH_SIDE:
        return None

    trigger_sum = sum(
        lines[idx + 1] if states[idx] == "sell" else lines[idx]
        for idx in range(len(states))
    )
    if trigger_sum <= 0 or active_notional_proxy <= 0:
        return None

    qty_by_active_notional = active_notional_proxy / trigger_sum

    equity = (
        _f(state.get("eth_holdings")) * current_price
        + _f(state.get("usdt_holdings"))
    )
    buy_cost_per_qty = sum(
        lines[idx] * (1.0 + fee_rate) for idx in buy_idx
    )
    seed_equity_per_qty = len(sell_idx) * current_price + buy_cost_per_qty
    if equity <= 0 or seed_equity_per_qty <= 0:
        return None

    qty_by_equity = equity / seed_equity_per_qty
    qty = min(qty_by_active_notional, qty_by_equity)
    if qty <= 0:
        return None

    seed_eth = qty * len(sell_idx)
    seed_usdt = equity - seed_eth * current_price
    if seed_usdt < -1e-8:
        return None
    seed_usdt = max(0.0, seed_usdt)

    return {
        "qty": qty,
        "sell_intervals": len(sell_idx),
        "buy_intervals": len(buy_idx),
        "avg_order_notional_usdt": qty * current_price,
        "seed_eth": seed_eth,
        "seed_usdt": seed_usdt,
        "equity_usdt": equity,
        "qty_by_active_notional": qty_by_active_notional,
        "qty_by_equity": qty_by_equity,
    }


def _evaluate_recovery_geometry(
    lower,
    upper,
    grids,
    current_price,
    state,
    fee_rate,
    utilization,
    active_notional_proxy,
    paths,
    profit_scale,
    rounds_scale,
):
    trigger = _recovery_trigger(state, current_price)

    # Normal, comfortably in-grid state: preserve validated legacy behaviour.
    if trigger is None:
        return _ORIGINAL_EVALUATE(
            lower, upper, grids, current_price, state, fee_rate, utilization,
            active_notional_proxy, paths, profit_scale, rounds_scale
        )

    is_current = _same_live_geometry(state, lower, upper, grids)

    # Hypothetical recovery candidates must contain the current market price.
    # Only the observed current bot may legitimately sit at/outside an edge.
    if not is_current and not (lower < current_price < upper):
        return None
    if upper <= lower:
        return None

    min_net, max_net = base.net_grid_profit_bounds(lower, upper, grids, fee_rate)
    min_required = max(
        base.MIN_NET_PROFIT_GRID_PCT_FLOOR,
        fee_rate * 100.0 * 2.0 + base.FEE_BUFFER_PP,
    )
    if min_net < min_required:
        return None

    if is_current:
        qty = _f(state.get("quantity_per_grid_eth"))
        if qty <= 0:
            return None

        lines = base.sim.grid_lines(lower, upper, grids)
        states = base.sim.initial_states(lines, current_price)
        sell_n = sum(st == "sell" for st in states)
        buy_n = sum(st == "buy" for st in states)

        # Evaluate the live bot exactly as observed. Do not pretend it is
        # infeasible simply because one side of the grid is nearly exhausted.
        eth0 = _f(state.get("eth_holdings"))
        usdt0 = _f(state.get("usdt_holdings"))
        sizing_mode = "OBSERVED_LIVE_EDGE_OR_ESCAPED_BOT"
        sizing_meta = {
            "qty_by_active_notional": None,
            "qty_by_equity": None,
            "seed_equity_usdt": eth0 * current_price + usdt0,
        }
    else:
        qinfo = _candidate_rebalance_seed(
            state, current_price, lower, upper, grids, fee_rate,
            active_notional_proxy
        )
        if qinfo is None:
            return None

        qty = qinfo["qty"]
        sell_n = qinfo["sell_intervals"]
        buy_n = qinfo["buy_intervals"]
        eth0 = qinfo["seed_eth"]
        usdt0 = qinfo["seed_usdt"]
        sizing_mode = "POST_EDIT_REBALANCE_PROXY"
        sizing_meta = {
            "qty_by_active_notional": qinfo["qty_by_active_notional"],
            "qty_by_equity": qinfo["qty_by_equity"],
            "seed_equity_usdt": qinfo["equity_usdt"],
        }

    profits, rounds, pnls = [], [], []
    escapes, low_esc, up_esc = [], [], []

    initially_low = current_price < lower
    initially_high = current_price > upper

    for item in paths:
        a = base.simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "ohlc"
        )
        b = base.simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "olhc"
        )

        profits.append(
            (a["grid_profit_usdt"] + b["grid_profit_usdt"]) / 2.0
            * profit_scale
        )
        rounds.append(
            (a["rounds"] + b["rounds"]) / 2.0 * rounds_scale
        )
        pnls.append(
            (a["total_pnl_usdt"] + b["total_pnl_usdt"]) / 2.0
        )

        low = initially_low or a["lower_escape"] or b["lower_escape"]
        high = initially_high or a["upper_escape"] or b["upper_escape"]
        low_esc.append(low)
        up_esc.append(high)
        escapes.append(low or high)

    if not profits:
        return None

    n = len(profits)
    result = {
        "lower_usdt": round(lower, 4),
        "upper_usdt": round(upper, 4),
        "center_usdt": round((lower + upper) / 2.0, 4),
        "width_usdt": round(upper - lower, 4),
        "width_pct_of_market": round((upper - lower) / current_price * 100.0, 4),
        "grids": int(grids),
        "quantity_per_grid_eth_est": round(qty, 8),
        "avg_order_notional_usdt_est": round(qty * current_price, 4),
        "buy_intervals": int(buy_n),
        "sell_intervals": int(sell_n),
        "net_profit_per_grid_pct_min": round(min_net, 5),
        "net_profit_per_grid_pct_max": round(max_net, 5),
        "expected_grid_profit_usdt": round(statistics.fmean(profits), 6),
        "median_grid_profit_usdt": round(statistics.median(profits), 6),
        "expected_rounds": round(statistics.fmean(rounds), 6),
        "median_rounds": round(statistics.median(rounds), 6),
        "escape_probability_pct": round(sum(bool(x) for x in escapes) / n * 100.0, 4),
        "lower_escape_probability_pct": round(sum(bool(x) for x in low_esc) / n * 100.0, 4),
        "upper_escape_probability_pct": round(sum(bool(x) for x in up_esc) / n * 100.0, 4),
        "p_grid_profit_ge_0_25_pct": base.sim.probability_ge(profits, 0.25),
        "p_grid_profit_ge_0_50_pct": base.sim.probability_ge(profits, 0.50),
        "p_total_pnl_positive_pct": round(sum(x > 0 for x in pnls) / n * 100.0, 4),
        "expected_total_pnl_usdt": round(statistics.fmean(pnls), 6),
        "p20_total_pnl_usdt": round(base.percentile(pnls, 0.20), 6),
        "p10_total_pnl_usdt": round(base.percentile(pnls, 0.10), 6),
        "sample_n": n,
        "recovery_trigger": trigger if is_current else None,
        "out_of_grid_direction": (
            trigger if is_current and trigger in {"ABOVE_UPPER", "BELOW_LOWER"} else None
        ),
        "sizing_mode": sizing_mode,
        **sizing_meta,
    }
    return result


def _select_recovery_geometry(candidates, current):
    trigger = current.get("recovery_trigger")
    if not trigger:
        return _ORIGINAL_SELECT(candidates, current)

    feasible = [
        c for c in candidates
        if c is not current and not c.get("recovery_trigger")
    ]

    if not feasible:
        return {
            "current": current,
            "raw_selected": current,
            "selected": current,
            "action": "KEEP_CURRENT",
            "expected_gain_vs_current_usdt": 0.0,
            "rounds_gain_vs_current": 0.0,
            "escape_change_vs_current_pp": 0.0,
            "effective_escape_cap_pct": 100.0,
            "eligible_count": 0,
            "recovery_status": "NO_FEASIBLE_RECENTER_CANDIDATE",
        }

    min_escape = min(c["escape_probability_pct"] for c in feasible)
    risk_cap = min(100.0, min_escape + RECOVERY_RISK_BAND_PP)
    risk_band = [
        c for c in feasible
        if c["escape_probability_pct"] <= risk_cap + 1e-9
    ]

    best_tail = max(c["p20_total_pnl_usdt"] for c in risk_band)
    tail_band = [
        c for c in risk_band
        if c["p20_total_pnl_usdt"] >= best_tail - RECOVERY_TAIL_TOL_USDT
    ] or risk_band

    raw = max(
        tail_band,
        key=lambda c: (
            c["expected_grid_profit_usdt"],
            c["p_total_pnl_positive_pct"],
            c["p20_total_pnl_usdt"],
            -c["escape_probability_pct"],
        ),
    )

    chosen = raw

    prefix = {
        "ABOVE_UPPER": "RECENTER_AFTER_UPPER_ESCAPE",
        "BELOW_LOWER": "RECENTER_AFTER_LOWER_ESCAPE",
        "NEAR_UPPER_EDGE": "RECENTER_NEAR_UPPER_EDGE",
        "NEAR_LOWER_EDGE": "RECENTER_NEAR_LOWER_EDGE",
    }.get(trigger, "RECENTER_EDGE_GRID")

    parts = [prefix]

    centre_delta = chosen["center_usdt"] - current["center_usdt"]
    width_delta = chosen["width_usdt"] - current["width_usdt"]
    grid_delta = chosen["grids"] - current["grids"]

    if abs(centre_delta) >= 2.5:
        parts.append("SHIFT_UP" if centre_delta > 0 else "SHIFT_DOWN")
    if abs(width_delta) >= 2.5:
        parts.append("WIDEN" if width_delta > 0 else "NARROW")
    if grid_delta:
        parts.append("MORE_GRIDS" if grid_delta > 0 else "FEWER_GRIDS")

    return {
        "current": current,
        "raw_selected": raw,
        "selected": chosen,
        "action": "+".join(parts),
        "expected_gain_vs_current_usdt": round(
            chosen["expected_grid_profit_usdt"] - current["expected_grid_profit_usdt"], 6
        ),
        "rounds_gain_vs_current": round(
            chosen["expected_rounds"] - current["expected_rounds"], 6
        ),
        "escape_change_vs_current_pp": round(
            chosen["escape_probability_pct"] - current["escape_probability_pct"], 6
        ),
        "effective_escape_cap_pct": round(risk_cap, 6),
        "eligible_count": len(risk_band),
        "recovery_status": "RECENTER_CANDIDATE_SELECTED",
        "minimum_candidate_escape_probability_pct": round(min_escape, 6),
    }


def _annotate_output():
    if not GEO_PATH.is_file():
        return

    geo = json.loads(GEO_PATH.read_text(encoding="utf-8"))

    geo["live_market_price_bridge"] = {
        **_LIVE_PRICE_BRIDGE,
        "policy": (
            "Hourly features remain the regime/analogue descriptor. Fresh Pionex "
            "API current_price_usdt is authoritative for live grid geometry, "
            "quantity sizing and historical-path remapping."
        ),
    }

    source = geo.get("source_state") or {}
    source["geometry_market_price_source"] = "FRESH_PIONEX_API_STATE"
    source["hourly_regime_feature_close_usdt"] = _LIVE_PRICE_BRIDGE.get(
        "original_hourly_feature_close_usdt"
    )
    geo["source_state"] = source

    benches = geo.get("benchmarks") or {}
    current = benches.get("current") or {}
    selected = benches.get("selected") or {}
    trigger = current.get("recovery_trigger")

    geo["edge_or_out_of_grid_recovery"] = {
        "triggered": bool(trigger),
        "trigger": trigger,
        "status": (
            "RECOVERY_MODEL_ACTIVE" if trigger else "LIVE_GRID_HAS_BOTH_ACTIVE_SIDES"
        ),
        "current_buy_intervals": current.get("buy_intervals"),
        "current_sell_intervals": current.get("sell_intervals"),
        "minimum_active_intervals_each_side_for_normal_candidate_mode": MIN_ACTIVE_INTERVALS_EACH_SIDE,
        "selected_lower_usdt": selected.get("lower_usdt"),
        "selected_upper_usdt": selected.get("upper_usdt"),
        "selected_grids": selected.get("grids"),
        "selected_sizing_mode": selected.get("sizing_mode"),
    }

    if trigger:
        decision = geo.get("decision") or {}
        decision["warning"] = (
            "Live grid is at/through an edge. Recovery geometry is decision support "
            "only. Confirm Pionex's live edit-screen quantity/grid and minimum-order "
            "validation before applying any recenter."
        )
        geo["decision"] = decision

    GEO_PATH.write_text(json.dumps(geo, indent=2) + "\n", encoding="utf-8")


def main():
    # Correct mature-grid interval state at/through boundaries.
    base.sim.initial_states = corrected_initial_states

    # Fresh API state becomes the live market price; hourly feature still supplies
    # timestamp/ATR/path-shape regime context.
    _install_live_price_bridge()

    # Recovery-aware current benchmark and selection.
    base.evaluate_geometry = _evaluate_recovery_geometry
    base.select_geometry = _select_recovery_geometry

    integration.main()
    _annotate_output()

    print("Phase 4D v3.1 live-price / edge-recovery adapter complete.")
    print("Live-price bridge:", _LIVE_PRICE_BRIDGE)


if __name__ == "__main__":
    main()
