#!/usr/bin/env python3
"""
Phase 4D v3 — out-of-grid recovery adapter.

This wraps the existing Phase 4D v2 execution-resolution adapter and changes
behaviour only when the *live* Pionex market price is outside its configured
range.

Why this exists
---------------
Legacy Phase 4D intentionally required lower < market < upper when evaluating a
geometry. That is appropriate for candidate grids, but it made the entire
pipeline crash exactly when a live bot escaped its range — the moment when
decision support is most useful.

Out-of-grid recovery policy
---------------------------
1. Preserve all historical in-grid calibration behaviour unchanged.
2. Evaluate the escaped current bot with its observed quantity and balances.
3. Correct the mature grid state above the upper bound: all intervals wait to
   buy on re-entry (no synthetic remaining sell interval).
4. For a *new* candidate geometry, model the range edit as a rebalance of the
   existing active grid notional. This is a sizing proxy only; the Pionex edit
   screen remains authoritative.
5. When the current grid is already escaped, select among feasible in-grid
   candidates using a minimum-escape-risk band, then prefer higher expected grid
   profit / stronger tail metrics. Do not compare against the escaped geometry
   as though it were a normal active grid.

No Pionex write endpoint is used.
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
_ORIGINAL_INITIAL_STATES = base.sim.initial_states

OUT_OF_GRID_RISK_BAND_PP = 5.0
OUT_OF_GRID_TAIL_TOL_USDT = 0.50


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _live_escape_direction(state, market_price):
    lower = _f(state.get("lower_limit_usdt"))
    upper = _f(state.get("upper_limit_usdt"))
    if market_price > upper:
        return "ABOVE_UPPER"
    if market_price < lower:
        return "BELOW_LOWER"
    return None


def _same_live_geometry(state, lower, upper, grids):
    return (
        abs(_f(state.get("lower_limit_usdt")) - float(lower)) <= 1e-6
        and abs(_f(state.get("upper_limit_usdt")) - float(upper)) <= 1e-6
        and int(float(state.get("grids"))) == int(grids)
    )


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
    Estimate post-edit quantity and seed balances from:
      - preserved active grid notional, and
      - current bot mark-to-market equity.

    We intentionally do NOT constrain a new geometry by the pre-edit ETH/USDT
    split. Pionex range edits can require reallocating the bot's assets to seed
    the new buy/sell ladder; using the old split is what caused the legacy model
    to estimate implausibly tiny quantities near a range edge.

    This is still an offline proxy. The Pionex edit screen is authoritative.
    """
    lines = base.sim.grid_lines(lower, upper, grids)
    states = base.sim.initial_states(lines, current_price)

    sell_idx = [idx for idx, st in enumerate(states) if st == "sell"]
    buy_idx = [idx for idx, st in enumerate(states) if st == "buy"]
    if len(sell_idx) < 2 or len(buy_idx) < 2:
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


def _evaluate_special(
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
    escape_direction = _live_escape_direction(state, current_price)
    if escape_direction is None:
        return _ORIGINAL_EVALUATE(
            lower, upper, grids, current_price, state, fee_rate, utilization,
            active_notional_proxy, paths, profit_scale, rounds_scale
        )

    is_current = _same_live_geometry(state, lower, upper, grids)

    # Candidates still must contain the market. Only the observed live grid is
    # allowed to be outside the market because it has already escaped.
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
        eth0 = _f(state.get("eth_holdings"))
        usdt0 = _f(state.get("usdt_holdings"))
        sizing_mode = "OBSERVED_ESCAPED_LIVE_BOT"
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
        "out_of_grid_direction": escape_direction if is_current else None,
        "sizing_mode": sizing_mode,
        **sizing_meta,
    }
    return result


def _select_special(candidates, current):
    direction = current.get("out_of_grid_direction")
    if not direction:
        return _ORIGINAL_SELECT(candidates, current)

    feasible = [
        c for c in candidates
        if c is not current and not c.get("out_of_grid_direction")
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
            "out_of_grid_recovery_status": "NO_FEASIBLE_RECENTER_CANDIDATE",
        }

    min_escape = min(c["escape_probability_pct"] for c in feasible)
    risk_cap = min(100.0, min_escape + OUT_OF_GRID_RISK_BAND_PP)
    risk_band = [
        c for c in feasible
        if c["escape_probability_pct"] <= risk_cap + 1e-9
    ]

    # Avoid buying a tiny profit improvement at the cost of a materially worse
    # downside tail within the already-lowest escape-risk region.
    best_tail = max(c["p20_total_pnl_usdt"] for c in risk_band)
    tail_band = [
        c for c in risk_band
        if c["p20_total_pnl_usdt"] >= best_tail - OUT_OF_GRID_TAIL_TOL_USDT
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
    parts = [
        "RECENTER_AFTER_UPPER_ESCAPE"
        if direction == "ABOVE_UPPER"
        else "RECENTER_AFTER_LOWER_ESCAPE"
    ]

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
        "out_of_grid_recovery_status": "RECENTER_CANDIDATE_SELECTED",
        "minimum_candidate_escape_probability_pct": round(min_escape, 6),
    }


def _annotate_recovery_output():
    if not GEO_PATH.is_file():
        return

    geo = json.loads(GEO_PATH.read_text(encoding="utf-8"))
    source = geo.get("source_state") or {}
    benches = geo.get("benchmarks") or {}
    current = benches.get("current") or {}
    selected = benches.get("selected") or {}

    direction = current.get("out_of_grid_direction")
    if not direction:
        geo["out_of_grid_recovery"] = {
            "triggered": False,
            "status": "LIVE_MARKET_INSIDE_GRID",
        }
    else:
        geo["out_of_grid_recovery"] = {
            "triggered": True,
            "direction": direction,
            "status": "RECOVERY_MODEL_ACTIVE",
            "current_market_price_usdt": source.get("latest_market_price_usdt"),
            "escaped_lower_usdt": current.get("lower_usdt"),
            "escaped_upper_usdt": current.get("upper_usdt"),
            "selected_lower_usdt": selected.get("lower_usdt"),
            "selected_upper_usdt": selected.get("upper_usdt"),
            "selected_grids": selected.get("grids"),
            "candidate_sizing_mode": selected.get("sizing_mode"),
            "warning": (
                "Out-of-grid candidate sizing assumes the Pionex range edit can rebalance "
                "the bot's active grid capital into the new ladder. Confirm quantity/grid "
                "and minimum-order validation in the Pionex edit screen before applying."
            ),
        }

        decision = geo.get("decision") or {}
        decision["warning"] = (
            "LIVE GRID HAS ESCAPED. Recovery geometry is decision support only. "
            "Confirm Pionex's live edit-screen quantity/grid and minimum-order checks "
            "before applying the suggested recenter."
        )
        geo["decision"] = decision

    GEO_PATH.write_text(json.dumps(geo, indent=2) + "\n", encoding="utf-8")


def main():
    # Correct the edge-state model used by Phase 4D v2 during this process.
    base.sim.initial_states = corrected_initial_states
    base.evaluate_geometry = _evaluate_special
    base.select_geometry = _select_special

    integration.main()
    _annotate_recovery_output()

    print("Phase 4D v3 out-of-grid recovery adapter complete.")


if __name__ == "__main__":
    main()
