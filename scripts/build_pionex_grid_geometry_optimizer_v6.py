#!/usr/bin/env python3
"""
Phase 4.5 — joint paired-cycle geometry optimiser.

This layer runs the complete Phase 4D v4.4 stack first, then performs a
research-only joint search over:

    * band centre
    * band width
    * grid count

using the corrected paired buy->sell cycle accounting and the paired-specific
walk-forward calibration introduced in v4.4.

Safety
------
- No Pionex write endpoint is used.
- No candidate is automatically promoted.
- The live bot remains authoritative.
- Exact Pionex setup/edit validation is still required before any manual change.
- Switching/rebalance slippage and tax effects are not modelled.

The objective of Phase 4.5 is to answer the question v4.4 could not:
"Given that the current density is broadly sensible, should the *band itself*
move or resize, and what grid count belongs on that new band?"
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import build_pionex_grid_geometry_optimizer_v1 as base
import build_pionex_grid_geometry_optimizer_v4 as v4
import build_pionex_grid_geometry_optimizer_v5 as v5

OUT_PATH = Path("data/diagnostics/pionex_joint_paired_geometry_v1.json")
DENSITY_PATH = Path("data/diagnostics/pionex_grid_density_sweep_v1.json")
LEGACY_GEO_PATH = Path("data/diagnostics/pionex_grid_geometry_optimizer_v1.json")

VERSION = "4.5"

# Search mirrors the validated Phase 4D geometry envelope.
COARSE_GRID_COUNTS = tuple(range(10, 81, 5))
TOP_COARSE_TO_REFINE = 4
REFINE_CENTER_DELTAS_USDT = (-5.0, -2.5, 0.0, 2.5, 5.0)
REFINE_WIDTH_DELTAS_PCT = (-0.5, 0.0, 0.5)
REFINE_GRID_RADIUS = 3

# Robust rather than argmax-only selection.
NEAR_PEAK_EXPECTED_RATIO = 0.95
NEAR_BEST_MEDIAN_RATIO = 0.95
LOCAL_SUPPORT_CENTER_USDT = 20.0
LOCAL_SUPPORT_WIDTH_PCT = 2.05
LOCAL_SUPPORT_GRIDS = 5

# Materiality for research action.
MIN_EXPECTED_GAIN_USDT = 0.02
MIN_EXPECTED_GAIN_PCT = 5.0
MIN_ESCAPE_REDUCTION_PP = 2.5
MAX_MEDIAN_PROFIT_DROP_PCT = 5.0

PRACTICAL_PRICE_ROUND_USDT = 5.0


def _f(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _pct_gain(new: Any, old: Any) -> Optional[float]:
    a = _f(new)
    b = _f(old)
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return round((a / b - 1.0) * 100.0, 4)


def _prob_ge(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return round(sum(v >= threshold for v in values) / len(values) * 100.0, 4)


def _prob_zero(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(abs(v) < 1e-12 for v in values) / len(values) * 100.0, 4)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _geometry_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        round(float(row["center_usdt"]), 3),
        round(float(row["width_usdt"]), 3),
        int(row["grids"]),
    )


def _compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keys = (
        "lower_usdt", "upper_usdt", "center_usdt", "width_usdt",
        "width_pct_of_market", "center_offset_pct_of_market", "grids",
        "grid_spacing_usdt", "quantity_per_grid_eth_est",
        "avg_order_notional_usdt_est", "net_profit_per_grid_pct_min",
        "net_profit_per_grid_pct_max",
        "expected_paired_grid_profit_usdt_calibrated",
        "median_paired_grid_profit_usdt_calibrated",
        "expected_paired_rounds_calibrated",
        "median_paired_rounds_calibrated",
        "expected_paired_grid_profit_usdt_raw",
        "median_paired_grid_profit_usdt_raw",
        "expected_paired_rounds_raw", "median_paired_rounds_raw",
        "p_zero_paired_rounds_pct", "p_paired_rounds_ge_10_pct",
        "p_paired_rounds_ge_25_pct", "p_paired_rounds_ge_50_pct",
        "escape_probability_pct", "lower_escape_probability_pct",
        "upper_escape_probability_pct", "p_total_pnl_positive_pct",
        "expected_total_pnl_usdt", "p20_total_pnl_usdt",
        "p10_total_pnl_usdt", "standard_risk_eligible",
        "risk_failures", "local_near_peak_support_count",
        "geometry_change_distance", "sizing_model",
        "execution_feasibility", "metrics_recomputed_for_rounded_bounds",
        "status",
    )
    return {k: row.get(k) for k in keys if k in row}


def _paired_calibration(density: dict[str, Any]) -> dict[str, Any]:
    promo = density.get("promotion_readiness") or {}
    cal = promo.get("paired_calibration") or {}
    if not cal:
        cal = ((density.get("paired_cycle_correction") or {})
               .get("paired_specific_calibration") or {})
    return cal


def _repair_v44_current_anchor(density: dict[str, Any]) -> dict[str, Any]:
    """
    v4.4 originally looked for grid=22 before looking at the actual live grid
    count. Once the live bot moved to 40 grids, gain-vs-current percentages were
    therefore anchored to the wrong hypothetical density row.

    Repair only the current run. Phase 4.5 keeps its own history and never relies
    on the contaminated old v4.4 history for promotion.
    """
    plateau = ((density.get("promotion_readiness") or {})
               .get("plateau_selector") or {})
    rows = list(density.get("paired_cycle_current_band_integer_sweep") or [])
    live = density.get("current_benchmark") or {}
    live_grids = int(live.get("grids") or 0)

    exact = next((r for r in rows if int(r.get("grids") or -1) == live_grids), None)
    selected = plateau.get("robust_plateau_selection") or {}

    repaired = False
    if exact:
        plateau["current_band_live_density"] = v5._compact_candidate(exact)
        score = plateau.get("score_fields") or {}
        expected_field = score.get(
            "expected", "expected_paired_grid_profit_usdt_calibrated"
        )
        median_field = score.get(
            "median", "median_paired_grid_profit_usdt_calibrated"
        )
        if selected:
            plateau["selected_expected_profit_gain_vs_current_pct"] = _pct_gain(
                selected.get(expected_field), exact.get(expected_field)
            )
            plateau["selected_median_profit_gain_vs_current_pct"] = _pct_gain(
                selected.get(median_field), exact.get(median_field)
            )
        repaired = True

    method = density.get("method") or {}
    method["phase4_5_v44_current_anchor_hotfix"] = (
        "Current-band comparison is anchored to the actual live grid count, "
        "not the historical hard-coded 22-grid row."
    )
    density["method"] = method
    density["promotion_readiness"]["plateau_selector"] = plateau
    density["phase4_5_hotfix"] = {
        "v44_current_live_density_anchor_repaired": repaired,
        "actual_live_grids": live_grids,
    }
    return density


def _apply_calibrated_names(
    row: dict[str, Any],
    calibration: dict[str, Any],
    market_price: float,
    sizing_model: str,
) -> dict[str, Any]:
    out = dict(row)
    out["expected_paired_grid_profit_usdt_calibrated"] = out.pop(
        "expected_paired_grid_profit_usdt_scaled_provisional", None
    )
    out["median_paired_grid_profit_usdt_calibrated"] = out.pop(
        "median_paired_grid_profit_usdt_scaled_provisional", None
    )
    out["expected_paired_rounds_calibrated"] = out.pop(
        "expected_paired_rounds_scaled_provisional", None
    )
    out["median_paired_rounds_calibrated"] = out.pop(
        "median_paired_rounds_scaled_provisional", None
    )
    out["paired_calibration_status"] = calibration.get("status")
    out["calibration_status"] = "PAIRED_SPECIFIC_CALIBRATION_APPLIED"
    out["center_offset_pct_of_market"] = round(
        (float(out["center_usdt"]) / market_price - 1.0) * 100.0, 6
    )
    out["sizing_model"] = sizing_model
    return out


def _actual_live_paired_benchmark(
    ctx: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    state = ctx["state"]
    market = float(ctx["current_price"])
    lower = float(state["lower_limit_usdt"])
    upper = float(state["upper_limit_usdt"])
    grids = int(float(state["grids"]))
    qty = float(state["quantity_per_grid_eth"])
    eth0 = float(state.get("eth_holdings") or 0.0)
    usdt0 = float(state.get("usdt_holdings") or 0.0)
    fee_rate = float(ctx["fee_rate"])
    paths = ctx["paths"]

    profit_scale = float(calibration.get("profit_scale_applied") or 1.0)
    rounds_scale = float(calibration.get("rounds_scale_applied") or 1.0)

    raw_profits: list[float] = []
    raw_rounds: list[float] = []
    pnls: list[float] = []
    escapes: list[bool] = []
    lows: list[bool] = []
    ups: list[bool] = []

    for item in paths:
        a = v4._paired_simulate_portfolio_path(
            item["candles"], market, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "ohlc"
        )
        b = v4._paired_simulate_portfolio_path(
            item["candles"], market, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "olhc"
        )
        raw_profits.append(
            (float(a["paired_grid_profit_usdt"])
             + float(b["paired_grid_profit_usdt"])) / 2.0
        )
        raw_rounds.append(
            (float(a["paired_rounds"]) + float(b["paired_rounds"])) / 2.0
        )
        pnls.append(
            (float(a["total_pnl_usdt"]) + float(b["total_pnl_usdt"])) / 2.0
        )
        lo = bool(a["lower_escape"] or b["lower_escape"])
        up = bool(a["upper_escape"] or b["upper_escape"])
        lows.append(lo)
        ups.append(up)
        escapes.append(lo or up)

    if not raw_profits:
        raise SystemExit("Phase 4.5 could not simulate the live paired benchmark")

    lines = base.sim.grid_lines(lower, upper, grids)
    states = base.sim.initial_states(lines, market)
    min_net, max_net = base.net_grid_profit_bounds(
        lower, upper, grids, fee_rate
    )
    n = len(raw_profits)
    cal_profits = [x * profit_scale for x in raw_profits]
    cal_rounds = [x * rounds_scale for x in raw_rounds]

    return {
        "lower_usdt": round(lower, 4),
        "upper_usdt": round(upper, 4),
        "center_usdt": round((lower + upper) / 2.0, 4),
        "width_usdt": round(upper - lower, 4),
        "width_pct_of_market": round((upper - lower) / market * 100.0, 4),
        "center_offset_pct_of_market": round(
            (((lower + upper) / 2.0) / market - 1.0) * 100.0, 6
        ),
        "grids": grids,
        "grid_spacing_usdt": round((upper - lower) / max(1, grids - 1), 6),
        "quantity_per_grid_eth_est": round(qty, 8),
        "avg_order_notional_usdt_est": round(qty * market, 4),
        "buy_intervals": sum(st == "buy" for st in states),
        "sell_intervals": sum(st == "sell" for st in states),
        "net_profit_per_grid_pct_min": round(min_net, 5),
        "net_profit_per_grid_pct_max": round(max_net, 5),
        "expected_paired_grid_profit_usdt_raw": round(
            statistics.fmean(raw_profits), 6
        ),
        "median_paired_grid_profit_usdt_raw": round(
            statistics.median(raw_profits), 6
        ),
        "expected_paired_grid_profit_usdt_calibrated": round(
            statistics.fmean(cal_profits), 6
        ),
        "median_paired_grid_profit_usdt_calibrated": round(
            statistics.median(cal_profits), 6
        ),
        "expected_paired_rounds_raw": round(
            statistics.fmean(raw_rounds), 6
        ),
        "median_paired_rounds_raw": round(
            statistics.median(raw_rounds), 6
        ),
        "expected_paired_rounds_calibrated": round(
            statistics.fmean(cal_rounds), 6
        ),
        "median_paired_rounds_calibrated": round(
            statistics.median(cal_rounds), 6
        ),
        "p_zero_paired_rounds_pct": _prob_zero(raw_rounds),
        "p_paired_rounds_ge_10_pct": _prob_ge(raw_rounds, 10),
        "p_paired_rounds_ge_25_pct": _prob_ge(raw_rounds, 25),
        "p_paired_rounds_ge_50_pct": _prob_ge(raw_rounds, 50),
        "escape_probability_pct": round(sum(escapes) / n * 100.0, 4),
        "lower_escape_probability_pct": round(sum(lows) / n * 100.0, 4),
        "upper_escape_probability_pct": round(sum(ups) / n * 100.0, 4),
        "p_total_pnl_positive_pct": round(
            sum(x > 0 for x in pnls) / n * 100.0, 4
        ),
        "expected_total_pnl_usdt": round(statistics.fmean(pnls), 6),
        "p20_total_pnl_usdt": round(float(_percentile(pnls, 0.20)), 6),
        "p10_total_pnl_usdt": round(float(_percentile(pnls, 0.10)), 6),
        "sample_n": n,
        "standard_risk_eligible": True,
        "risk_failures": [],
        "sizing_model": (
            "STATUS_QUO_ACTUAL_LIVE_QUANTITY_AND_BALANCES; interval states are "
            "reconstructed from current price because per-grid order state is "
            "not available in the snapshot archive."
        ),
    }


def _risk_policy(current: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute_escape_cap_pct": base.ABS_ESCAPE_RISK_CAP_PCT,
        "effective_escape_cap_pct": min(
            base.ABS_ESCAPE_RISK_CAP_PCT,
            float(current["escape_probability_pct"]),
        ),
        "do_not_reduce_total_pnl_positive_probability_by_more_than_pp":
            base.MAX_POSITIVE_PNL_PROB_DROP_PP,
        "do_not_worsen_p20_total_pnl_by_more_than_usdt":
            base.MAX_P20_TOTAL_PNL_WORSEN_USDT,
    }


def _apply_risk_gate(
    row: dict[str, Any],
    current: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    failures: list[str] = []

    if (
        float(out["escape_probability_pct"])
        > float(policy["effective_escape_cap_pct"]) + 1e-9
    ):
        failures.append("ESCAPE_RISK_ABOVE_EFFECTIVE_CAP")

    if (
        float(out["p_total_pnl_positive_pct"])
        < float(current["p_total_pnl_positive_pct"])
        - float(
            policy[
                "do_not_reduce_total_pnl_positive_probability_by_more_than_pp"
            ]
        )
    ):
        failures.append("TOTAL_PNL_POSITIVE_PROBABILITY_DETERIORATION")

    if (
        float(out["p20_total_pnl_usdt"])
        < float(current["p20_total_pnl_usdt"])
        - float(policy["do_not_worsen_p20_total_pnl_by_more_than_usdt"])
    ):
        failures.append("P20_TOTAL_PNL_DETERIORATION")

    out["standard_risk_eligible"] = not failures
    out["risk_failures"] = failures
    return out


def _geometry_change_distance(
    row: dict[str, Any],
    current: dict[str, Any],
) -> float:
    center_component = abs(
        float(row["center_usdt"]) - float(current["center_usdt"])
    ) / max(1.0, base.COARSE_CENTER_STEP_USDT)
    width_component = abs(
        float(row["width_pct_of_market"])
        - float(current["width_pct_of_market"])
    ) / max(0.1, base.COARSE_WIDTH_STEP_PCT)
    grid_component = abs(
        int(row["grids"]) - int(current["grids"])
    ) / 5.0
    return round(center_component + width_component + grid_component, 6)


def _candidate(
    lower: float,
    upper: float,
    grids: int,
    ctx: dict[str, Any],
    calibration: dict[str, Any],
    current: dict[str, Any],
    policy: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any] | None:
    market = float(ctx["current_price"])
    profit_scale = float(calibration.get("profit_scale_applied") or 1.0)
    rounds_scale = float(calibration.get("rounds_scale_applied") or 1.0)

    raw = v4._paired_rebalanced_evaluate(
        lower,
        upper,
        int(grids),
        market,
        ctx["state"],
        float(ctx["fee_rate"]),
        float(ctx["active_notional_proxy"]),
        ctx["paths"],
        profit_scale,
        rounds_scale,
    )
    if raw is None:
        return None

    out = _apply_calibrated_names(
        raw,
        calibration,
        market,
        "POST_EDIT_REBALANCE_TOTAL_EQUITY_PLUS_ACTIVE_NOTIONAL_CAP",
    )
    out = _apply_risk_gate(out, current, policy)
    out["execution_feasibility"] = v5._execution_feasibility(
        out, constraints
    )
    out["geometry_change_distance"] = _geometry_change_distance(out, current)
    return out


def _candidate_from_center_width(
    center: float,
    width: float,
    grids: int,
    *args: Any,
) -> dict[str, Any] | None:
    return _candidate(
        center - width / 2.0,
        center + width / 2.0,
        grids,
        *args,
    )


def _search_joint(
    ctx: dict[str, Any],
    calibration: dict[str, Any],
    current: dict[str, Any],
    policy: dict[str, Any],
    constraints: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    market = float(ctx["current_price"])

    center_offsets: list[float] = []
    lim = market * base.CENTER_SEARCH_PCT / 100.0
    x = -lim
    while x <= lim + 1e-9:
        center_offsets.append(x)
        x += base.COARSE_CENTER_STEP_USDT
    center_offsets.extend([
        0.0,
        float(current["center_usdt"]) - market,
    ])

    width_pcts: list[float] = []
    w = base.MIN_WIDTH_PCT
    while w <= base.MAX_WIDTH_PCT + 1e-9:
        width_pcts.append(w)
        w += base.COARSE_WIDTH_STEP_PCT
    width_pcts.append(float(current["width_pct_of_market"]))

    grids_list = sorted(set(
        COARSE_GRID_COUNTS + (int(current["grids"]),)
    ))

    evaluated: dict[tuple[float, float, int], dict[str, Any]] = {}

    # Coarse paired search.
    for off in sorted(set(round(v, 6) for v in center_offsets)):
        center = market + off
        for width_pct in sorted(set(round(v, 6) for v in width_pcts)):
            width = market * width_pct / 100.0
            for grids in grids_list:
                row = _candidate_from_center_width(
                    center, width, grids,
                    ctx, calibration, current, policy, constraints,
                )
                if row:
                    evaluated[_geometry_key(row)] = row

    coarse_count = len(evaluated)

    feasible = [
        r for r in evaluated.values()
        if r.get("standard_risk_eligible") is True
        and (r.get("execution_feasibility") or {}).get("status")
            != "FAIL_KNOWN_CONSTRAINT"
    ]
    coarse_seeds = sorted(
        feasible,
        key=lambda r: (
            -float(r.get(
                "expected_paired_grid_profit_usdt_calibrated"
            ) or 0.0),
            -float(r.get(
                "median_paired_grid_profit_usdt_calibrated"
            ) or 0.0),
            float(r.get("escape_probability_pct") or 100.0),
        ),
    )[:TOP_COARSE_TO_REFINE]

    # Local paired refinement around strongest corrected coarse regions.
    for seed in coarse_seeds:
        seed_center = float(seed["center_usdt"])
        seed_width_pct = float(seed["width_pct_of_market"])
        seed_grids = int(seed["grids"])

        for dc in REFINE_CENTER_DELTAS_USDT:
            center = seed_center + dc
            for dw in REFINE_WIDTH_DELTAS_PCT:
                width_pct = seed_width_pct + dw
                if width_pct <= 0:
                    continue
                width = market * width_pct / 100.0
                for grids in range(
                    max(10, seed_grids - REFINE_GRID_RADIUS),
                    min(80, seed_grids + REFINE_GRID_RADIUS) + 1,
                ):
                    row = _candidate_from_center_width(
                        center, width, grids,
                        ctx, calibration, current, policy, constraints,
                    )
                    if row:
                        evaluated[_geometry_key(row)] = row

    meta = {
        "coarse_candidate_count": coarse_count,
        "refinement_seed_count": len(coarse_seeds),
        "total_candidate_count": len(evaluated),
        "coarse_center_step_usdt": base.COARSE_CENTER_STEP_USDT,
        "coarse_width_step_pct": base.COARSE_WIDTH_STEP_PCT,
        "coarse_grid_counts": list(grids_list),
        "refine_center_deltas_usdt": list(REFINE_CENTER_DELTAS_USDT),
        "refine_width_deltas_pct": list(REFINE_WIDTH_DELTAS_PCT),
        "refine_grid_radius": REFINE_GRID_RADIUS,
    }
    return list(evaluated.values()), meta


def _local_support(
    row: dict[str, Any],
    near_peak: list[dict[str, Any]],
) -> int:
    return sum(
        abs(float(other["center_usdt"]) - float(row["center_usdt"]))
            <= LOCAL_SUPPORT_CENTER_USDT + 1e-9
        and abs(
            float(other["width_pct_of_market"])
            - float(row["width_pct_of_market"])
        ) <= LOCAL_SUPPORT_WIDTH_PCT + 1e-9
        and abs(int(other["grids"]) - int(row["grids"]))
            <= LOCAL_SUPPORT_GRIDS
        for other in near_peak
    )


def _select_joint(
    candidates: list[dict[str, Any]],
    current: dict[str, Any],
) -> dict[str, Any]:
    eligible = [
        r for r in candidates
        if r.get("standard_risk_eligible") is True
        and (r.get("execution_feasibility") or {}).get("status")
            != "FAIL_KNOWN_CONSTRAINT"
    ]
    if not eligible:
        return {
            "status": "NO_ELIGIBLE_JOINT_PAIRED_CANDIDATES",
            "operational_override": False,
        }

    ef = "expected_paired_grid_profit_usdt_calibrated"
    mf = "median_paired_grid_profit_usdt_calibrated"

    peak = max(
        eligible,
        key=lambda r: (
            float(r.get(ef) or 0.0),
            float(r.get(mf) or 0.0),
        ),
    )
    median_champ = max(
        eligible,
        key=lambda r: (
            float(r.get(mf) or 0.0),
            float(r.get(ef) or 0.0),
        ),
    )

    expected_floor = float(peak[ef]) * NEAR_PEAK_EXPECTED_RATIO
    near_peak = [
        r for r in eligible
        if float(r.get(ef) or 0.0) >= expected_floor - 1e-12
    ]

    best_median_near_peak = max(
        float(r.get(mf) or 0.0) for r in near_peak
    )
    median_floor = best_median_near_peak * NEAR_BEST_MEDIAN_RATIO

    robust_pool = [
        r for r in near_peak
        if float(r.get(mf) or 0.0) >= median_floor - 1e-12
    ]

    for r in near_peak:
        r["local_near_peak_support_count"] = _local_support(r, near_peak)

    robust = max(
        robust_pool,
        key=lambda r: (
            int(r.get("local_near_peak_support_count") or 0),
            -float(r.get("escape_probability_pct") or 100.0),
            -float(r.get("geometry_change_distance") or 1e9),
            float(r.get(mf) or 0.0),
            float(r.get(ef) or 0.0),
            float(r.get("avg_order_notional_usdt_est") or 0.0),
        ),
    )

    expected_gain_usdt = (
        float(robust[ef])
        - float(current["expected_paired_grid_profit_usdt_calibrated"])
    )
    expected_gain_pct = _pct_gain(
        robust.get(ef),
        current.get("expected_paired_grid_profit_usdt_calibrated"),
    )
    median_gain_pct = _pct_gain(
        robust.get(mf),
        current.get("median_paired_grid_profit_usdt_calibrated"),
    )
    escape_reduction_pp = (
        float(current["escape_probability_pct"])
        - float(robust["escape_probability_pct"])
    )

    changed = (
        abs(float(robust["lower_usdt"]) - float(current["lower_usdt"])) > 0.01
        or abs(float(robust["upper_usdt"]) - float(current["upper_usdt"])) > 0.01
        or int(robust["grids"]) != int(current["grids"])
    )

    median_ok = (
        median_gain_pct is None
        or median_gain_pct >= -MAX_MEDIAN_PROFIT_DROP_PCT
    )
    profit_material = (
        expected_gain_usdt >= MIN_EXPECTED_GAIN_USDT
        and expected_gain_pct is not None
        and expected_gain_pct >= MIN_EXPECTED_GAIN_PCT
        and median_ok
    )
    risk_material = escape_reduction_pp >= MIN_ESCAPE_REDUCTION_PP

    material = bool(changed and (profit_material or risk_material))

    action = "KEEP_CURRENT"
    if material:
        parts: list[str] = []
        center_delta = float(robust["center_usdt"]) - float(current["center_usdt"])
        width_delta = float(robust["width_usdt"]) - float(current["width_usdt"])
        grid_delta = int(robust["grids"]) - int(current["grids"])
        if abs(center_delta) >= 2.5:
            parts.append("SHIFT_UP" if center_delta > 0 else "SHIFT_DOWN")
        if abs(width_delta) >= 2.5:
            parts.append("WIDEN" if width_delta > 0 else "NARROW")
        if grid_delta:
            parts.append("MORE_GRIDS" if grid_delta > 0 else "FEWER_GRIDS")
        action = "+".join(parts) or "CHANGE_GEOMETRY"

    return {
        "status": (
            "MATERIAL_RESEARCH_CANDIDATE"
            if material else "KEEP_CURRENT_RESEARCH"
        ),
        "selection_policy": {
            "expected_profit_near_peak_ratio": NEAR_PEAK_EXPECTED_RATIO,
            "median_profit_near_best_ratio": NEAR_BEST_MEDIAN_RATIO,
            "local_support_neighbourhood": {
                "center_usdt": LOCAL_SUPPORT_CENTER_USDT,
                "width_pct_points": LOCAL_SUPPORT_WIDTH_PCT,
                "grids": LOCAL_SUPPORT_GRIDS,
            },
            "robust_choice_rule": (
                "Keep only risk/execution-feasible candidates within 95% of "
                "the expected paired-profit peak, then within 95% of the best "
                "median profit in that set. Prefer the candidate with the most "
                "near-peak local neighbours; tie-break on lower escape risk, "
                "smaller geometry change, higher median/expected profit, and "
                "larger order notional."
            ),
        },
        "eligible_candidate_count": len(eligible),
        "near_peak_candidate_count": len(near_peak),
        "robust_pool_candidate_count": len(robust_pool),
        "expected_profit_champion": _compact(peak),
        "median_profit_champion": _compact(median_champ),
        "robust_joint_selection": _compact(robust),
        "research_action": action,
        "material_research_candidate": material,
        "expected_profit_gain_vs_current_usdt": round(expected_gain_usdt, 6),
        "expected_profit_gain_vs_current_pct": expected_gain_pct,
        "median_profit_gain_vs_current_pct": median_gain_pct,
        "escape_reduction_vs_current_pp": round(escape_reduction_pp, 6),
        "operational_override": False,
    }


def _evaluate_named_geometry(
    row: dict[str, Any] | None,
    ctx: dict[str, Any],
    calibration: dict[str, Any],
    current: dict[str, Any],
    policy: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any] | None:
    if not row:
        return None
    return _candidate(
        float(row["lower_usdt"]),
        float(row["upper_usdt"]),
        int(row["grids"]),
        ctx, calibration, current, policy, constraints,
    )


def _practicalize_and_recompute(
    selected: dict[str, Any] | None,
    ctx: dict[str, Any],
    calibration: dict[str, Any],
    current: dict[str, Any],
    policy: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any] | None:
    if not selected:
        return None

    lower = round(
        float(selected["lower_usdt"]) / PRACTICAL_PRICE_ROUND_USDT
    ) * PRACTICAL_PRICE_ROUND_USDT
    upper = round(
        float(selected["upper_usdt"]) / PRACTICAL_PRICE_ROUND_USDT
    ) * PRACTICAL_PRICE_ROUND_USDT

    if upper <= lower:
        return None

    practical = _candidate(
        lower, upper, int(selected["grids"]),
        ctx, calibration, current, policy, constraints,
    )
    if practical is None:
        return {
            "lower_usdt": lower,
            "upper_usdt": upper,
            "grids": int(selected["grids"]),
            "metrics_recomputed_for_rounded_bounds": False,
            "status": "ROUNDED_GEOMETRY_FAILED_MODEL_FEASIBILITY",
        }

    practical["metrics_recomputed_for_rounded_bounds"] = True
    practical["status"] = (
        "ROUNDED_GEOMETRY_RECOMPUTED_AND_ELIGIBLE"
        if practical.get("standard_risk_eligible") is True
        and (practical.get("execution_feasibility") or {}).get("status")
            != "FAIL_KNOWN_CONSTRAINT"
        else "ROUNDED_GEOMETRY_RECOMPUTED_BUT_NOT_ELIGIBLE"
    )
    return practical


def _best_by_grid(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grids = sorted({int(r["grids"]) for r in candidates})
    for g in grids:
        group = [
            r for r in candidates
            if int(r["grids"]) == g
            and r.get("standard_risk_eligible") is True
            and (r.get("execution_feasibility") or {}).get("status")
                != "FAIL_KNOWN_CONSTRAINT"
        ]
        if not group:
            continue
        best = max(
            group,
            key=lambda r: float(
                r.get("expected_paired_grid_profit_usdt_calibrated") or 0.0
            ),
        )
        out.append({
            "grids": g,
            "best_eligible": _compact(best),
        })
    return out


def _calibration_quality(calibration: dict[str, Any]) -> dict[str, Any]:
    holdout = calibration.get("holdout_validation") or {}
    pmape = _f(holdout.get("profit_mape_pct"))
    rmape = _f(holdout.get("rounds_mape_pct"))
    caution = bool(
        (pmape is not None and pmape > 100.0)
        or (rmape is not None and rmape > 100.0)
    )
    return {
        "status": (
            "CAUTION_HIGH_HOLDOUT_PERCENTAGE_ERROR"
            if caution else "NO_HIGH_ERROR_CAUTION"
        ),
        "profit_mape_pct": pmape,
        "rounds_mape_pct": rmape,
        "note": (
            "Percentage errors can become unstable when realised daily profit "
            "or rounds are small. Phase 4.5 therefore treats calibration quality "
            "as a caution flag, not an automatic promotion rule."
        ),
    }


def _write_joint_output() -> None:
    if not DENSITY_PATH.is_file():
        raise SystemExit("Phase 4.5 requires the v4.4 density diagnostic")
    if not LEGACY_GEO_PATH.is_file():
        raise SystemExit("Phase 4.5 requires the legacy joint geometry output")
    if not v4._LIVE_CONTEXT:
        raise SystemExit("Phase 4.5 live paired context was not retained")

    density = json.loads(DENSITY_PATH.read_text(encoding="utf-8"))
    density = _repair_v44_current_anchor(density)
    DENSITY_PATH.write_text(
        json.dumps(density, indent=2) + "\n", encoding="utf-8"
    )

    calibration = _paired_calibration(density)
    if not calibration:
        raise SystemExit("Paired-specific calibration missing for Phase 4.5")

    ctx = v4._LIVE_CONTEXT
    current = _actual_live_paired_benchmark(ctx, calibration)
    policy = _risk_policy(current)
    constraints = v5._load_constraints()

    candidates, search_meta = _search_joint(
        ctx, calibration, current, policy, constraints
    )
    selection = _select_joint(candidates, current)

    robust = selection.get("robust_joint_selection") or {}
    practical = _practicalize_and_recompute(
        robust if robust else None,
        ctx, calibration, current, policy, constraints,
    )

    # Reconfigure the current band from a fresh-start perspective.
    current_reconfigured = _candidate(
        float(current["lower_usdt"]),
        float(current["upper_usdt"]),
        int(current["grids"]),
        ctx, calibration, current, policy, constraints,
    )

    legacy_geo = json.loads(LEGACY_GEO_PATH.read_text(encoding="utf-8"))
    legacy_selected = (
        (legacy_geo.get("benchmarks") or {}).get("selected")
        or (legacy_geo.get("benchmarks") or {}).get("raw_selected")
    )
    legacy_paired = _evaluate_named_geometry(
        legacy_selected,
        ctx, calibration, current, policy, constraints,
    )

    comparison_to_legacy = {}
    if legacy_paired and robust:
        comparison_to_legacy = {
            "legacy_selected_geometry_under_paired_model":
                _compact(legacy_paired),
            "robust_joint_selection": _compact(robust),
            "robust_expected_profit_gain_vs_legacy_pct": _pct_gain(
                robust.get("expected_paired_grid_profit_usdt_calibrated"),
                legacy_paired.get(
                    "expected_paired_grid_profit_usdt_calibrated"
                ),
            ),
            "legacy_zero_cycle_probability_pct":
                legacy_paired.get("p_zero_paired_rounds_pct"),
            "legacy_standard_risk_eligible":
                legacy_paired.get("standard_risk_eligible"),
        }

    eligible = [
        r for r in candidates
        if r.get("standard_risk_eligible") is True
        and (r.get("execution_feasibility") or {}).get("status")
            != "FAIL_KNOWN_CONSTRAINT"
    ]
    near_peak = sorted(
        [
            r for r in eligible
            if robust
            and float(r.get(
                "expected_paired_grid_profit_usdt_calibrated"
            ) or 0.0)
            >= 0.95 * float(
                (selection.get("expected_profit_champion") or {})
                .get("expected_paired_grid_profit_usdt_calibrated") or 0.0
            )
        ],
        key=lambda r: -float(
            r.get("expected_paired_grid_profit_usdt_calibrated") or 0.0
        ),
    )[:100]

    source = density.get("source_state") or {}
    payload = {
        "schema": "pionex_joint_paired_geometry_v1",
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_RESEARCH_ONLY",
        "scope": {
            "platform": "Pionex",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "horizon_h": 24,
            "joint_variables": ["centre", "width", "grid_count"],
        },
        "source_state": source,
        "execution_resolution": density.get("execution_resolution"),
        "paired_calibration": {
            k: v for k, v in calibration.items() if k != "windows"
        },
        "calibration_quality": _calibration_quality(calibration),
        "current_status_quo_paired_benchmark": current,
        "current_geometry_fresh_rebalance_counterfactual":
            _compact(current_reconfigured),
        "risk_policy": policy,
        "search": search_meta,
        "selection": selection,
        "practical_candidate_exact_recompute":
            _compact(practical) if practical else None,
        "practical_candidate_status":
            practical.get("status") if practical else None,
        "comparison_to_legacy_optimizer": comparison_to_legacy,
        "best_eligible_by_grid_count": _best_by_grid(candidates),
        "near_peak_candidates_top100": [_compact(r) for r in near_peak],
        "candidate_counts": {
            "total": len(candidates),
            "risk_eligible": sum(
                r.get("standard_risk_eligible") is True for r in candidates
            ),
            "known_execution_failure": sum(
                (r.get("execution_feasibility") or {}).get("status")
                == "FAIL_KNOWN_CONSTRAINT"
                for r in candidates
            ),
        },
        "method": {
            "paired_cycle_semantics": (
                "Only sell events preceded by an actual replay buy in the same "
                "interval count as grid-profit cycles. Initial seeded ETH sales "
                "are inventory conversion, not completed grid cycles."
            ),
            "candidate_sizing": (
                "Hypothetical changed geometries use the same total live equity "
                "and preserved active-order-notional cap with a post-edit "
                "rebalance seed."
            ),
            "incumbent_benchmark": (
                "The status-quo benchmark uses the actual live quantity/grid and "
                "live ETH/USDT balances. Per-grid interval state is reconstructed "
                "from current price because exact open-order state is unavailable."
            ),
            "practical_rounding": (
                "Raw selected bounds are rounded to nearest $5 and the rounded "
                "geometry is then simulated again. Metrics are never borrowed "
                "from the unrounded candidate."
            ),
            "switching_cost_warning": (
                "Reconfiguration slippage, realised inventory effects outside "
                "the 24h replay, and tax consequences are not modelled."
            ),
            "operational_effect": (
                "Safety blocker / manual-review research only. Phase 4.5 does "
                "not automatically change the Pionex bot."
            ),
        },
        "next": {
            "collect": (
                "Accumulate multiple joint paired selections across changing "
                "market states and compare centre/width/grid stability."
            ),
            "promotion_gate": (
                "Require stable repeated joint evidence plus exact Pionex live "
                "setup validation before manual promotion review."
            ),
        },
        "operational_override": False,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Compact Phase 4.5 summary is also embedded into the density file so older
    # consumers still see the new stage even before the full runner annotates
    # pionex_full_decision_v1.json.
    density["phase4_5_joint_paired_geometry"] = {
        "schema": payload["schema"],
        "version": VERSION,
        "status": payload["status"],
        "source_state": payload["source_state"],
        "current_status_quo_paired_benchmark":
            _compact(current),
        "selection": selection,
        "practical_candidate_exact_recompute":
            _compact(practical) if practical else None,
        "comparison_to_legacy_optimizer": comparison_to_legacy,
        "operational_effect": "NONE_DIRECTLY",
    }
    DENSITY_PATH.write_text(
        json.dumps(density, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== PHASE 4.5 JOINT PAIRED GEOMETRY ===")
    print("Candidates:", len(candidates))
    print("Eligible:", payload["candidate_counts"]["risk_eligible"])
    print("Selection status:", selection.get("status"))
    print("Research action:", selection.get("research_action"))
    print(
        "Robust joint selection:",
        (selection.get("robust_joint_selection") or {}).get("lower_usdt"),
        "to",
        (selection.get("robust_joint_selection") or {}).get("upper_usdt"),
        "/",
        (selection.get("robust_joint_selection") or {}).get("grids"),
        "grids",
    )
    print("Operational override: False")
    print("Written:", OUT_PATH)


def main() -> None:
    # v5 runs the complete v4.4 stack and leaves v4._LIVE_CONTEXT populated in
    # this process. Phase 4.5 then builds on that exact same fresh context.
    v5.main()
    _write_joint_output()


if __name__ == "__main__":
    main()
