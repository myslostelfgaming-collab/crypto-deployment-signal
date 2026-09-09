#!/usr/bin/env python3
"""
Phase 4D v4.3 — paired-cycle grid-profit correction and density diagnostic.

This adapter wraps the validated v3.1 Phase 4D path and adds a diagnostic layer.
It does NOT alter the existing operational actionability decision.

Includes v4.3 fixes plus v4.3 paired-cycle correction
-------------
1. LIVE-ONLY CAPTURE
   Candidate evaluations performed inside historical calibration reconstruction
   are excluded from the density sweep. Only the current live search is captured.

2. CORRECT GRID SPACING
   Pionex "N grids" is represented by N price levels and N-1 arithmetic
   intervals, so spacing is width / (grids - 1), not width / grids.

3. INTEGER 5..80 CURRENT-BAND SWEEP
   The live lower/upper bounds are held fixed while every integer grid count
   from 5 through 80 is evaluated. Counts below the configured production
   minimum of 10 are diagnostic-only and reveal whether the objective is pushing
   against the lower search boundary.

4. SIZING COUNTERFACTUAL
   The same 5..80 current-band sweep is evaluated two ways:
     A) legacy/current-holdings sizing used by normal Phase 4D candidates;
     B) post-edit rebalanced sizing using total equity + preserved active-order
        notional, based on the v3 recovery sizing helper.
   This diagnoses whether the pre-edit ETH/USDT split unfairly starves denser
   hypothetical grids.

5. LOW-GRID PRESSURE AUDIT
   Reports whether the objective prefers the minimum tested grid count, whether
   it would prefer <10 grids, and how order size, spacing, cycles and profit/round
   interact across density.

Safety
------
Diagnostic / decision-support only.
No Pionex write endpoint is used.
Existing Phase 4D operational/actionability logic remains unchanged.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import build_pionex_grid_geometry_optimizer_v1 as base
import build_pionex_grid_geometry_optimizer_v3 as v3

OUT_PATH = Path("data/diagnostics/pionex_grid_density_sweep_v1.json")
ROUND_THRESHOLDS = (1, 2, 5, 10, 25, 50)
DIAGNOSTIC_MIN_GRIDS = 5
DIAGNOSTIC_MAX_GRIDS = 80
PRODUCTION_MIN_GRIDS = min(base.COARSE_GRID_COUNTS)

_ORIGINAL_INTEGRATION_MAIN = v3.integration.main

_CAPTURE_ENABLED = False
_CAPTURED_LIVE: dict[tuple[float, float, int], dict[str, Any]] = {}
_CAPTURED_DISTS: dict[tuple[float, float, int], dict[str, list[float]]] = {}
_LIVE_CONTEXT: dict[str, Any] = {}


def _key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        round(float(row["lower_usdt"]), 4),
        round(float(row["upper_usdt"]), 4),
        int(row["grids"]),
    )


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _prob_ge(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return round(sum(float(v) >= threshold for v in values) / len(values) * 100.0, 4)


def _prob_zero(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(abs(float(v)) < 1e-12 for v in values) / len(values) * 100.0, 4)


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


def _safe_ratio(a: Any, b: Any) -> float | None:
    aa = _f(a, float("nan"))
    bb = _f(b, float("nan"))
    if not math.isfinite(aa) or not math.isfinite(bb) or abs(bb) < 1e-12:
        return None
    return round(aa / bb, 6)


def _capture_distribution(
    evaluator: Callable[..., Any],
    lower: float,
    upper: float,
    grids: int,
    current_price: float,
    state: dict,
    fee_rate: float,
    utilization: float,
    active_notional_proxy: float,
    paths: list[dict],
    profit_scale: float,
    rounds_scale: float,
) -> tuple[dict[str, Any] | None, dict[str, list[float]]]:
    original_sim = base.simulate_portfolio_path
    sims: list[tuple[float, float]] = []

    def capture_sim(*args, **kwargs):
        result = original_sim(*args, **kwargs)
        sims.append(
            (
                float(result.get("rounds") or 0.0),
                float(result.get("grid_profit_usdt") or 0.0),
            )
        )
        return result

    base.simulate_portfolio_path = capture_sim
    try:
        result = evaluator(
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
        )
    finally:
        base.simulate_portfolio_path = original_sim

    dist: dict[str, list[float]] = {}
    if result is not None and sims and len(sims) % 2 == 0:
        raw_rounds: list[float] = []
        calibrated_rounds: list[float] = []
        calibrated_profits: list[float] = []
        for idx in range(0, len(sims), 2):
            a_rounds, a_profit = sims[idx]
            b_rounds, b_profit = sims[idx + 1]
            raw = (a_rounds + b_rounds) / 2.0
            raw_rounds.append(raw)
            calibrated_rounds.append(raw * float(rounds_scale))
            calibrated_profits.append(
                ((a_profit + b_profit) / 2.0) * float(profit_scale)
            )
        dist = {
            "raw_rounds": raw_rounds,
            "calibrated_rounds": calibrated_rounds,
            "calibrated_grid_profits": calibrated_profits,
        }
    return result, dist


def _instrumented_live_evaluator(evaluator):
    def wrapped(
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
        if not _CAPTURE_ENABLED:
            return evaluator(
                lower, upper, grids, current_price, state, fee_rate, utilization,
                active_notional_proxy, paths, profit_scale, rounds_scale
            )

        if not _LIVE_CONTEXT:
            _LIVE_CONTEXT.update({
                "current_price": float(current_price),
                "state": state,
                "fee_rate": float(fee_rate),
                "utilization": float(utilization),
                "active_notional_proxy": float(active_notional_proxy),
                "paths": paths,
                "profit_scale": float(profit_scale),
                "rounds_scale": float(rounds_scale),
            })

        result, dist = _capture_distribution(
            evaluator, lower, upper, grids, current_price, state, fee_rate,
            utilization, active_notional_proxy, paths, profit_scale, rounds_scale
        )
        if result is not None:
            k = _key(result)
            _CAPTURED_LIVE[k] = dict(result)
            if dist:
                _CAPTURED_DISTS[k] = dist
        return result

    return wrapped


def _capturing_integration_main() -> None:
    """
    Install capture only around the live Phase 4D search.

    base.main() calls reconstruct_calibration first, then the current/live
    evaluation and candidate search. The wrapper explicitly disables capture
    during calibration, preventing historical calibration geometries from
    contaminating the live density curve.
    """
    global _CAPTURE_ENABLED

    production_evaluator = base.evaluate_geometry
    production_reconstruct = base.reconstruct_calibration
    wrapped_evaluator = _instrumented_live_evaluator(production_evaluator)

    def reconstruct_without_capture(*args, **kwargs):
        global _CAPTURE_ENABLED
        previous = _CAPTURE_ENABLED
        _CAPTURE_ENABLED = False
        try:
            return production_reconstruct(*args, **kwargs)
        finally:
            _CAPTURE_ENABLED = True

    base.evaluate_geometry = wrapped_evaluator
    base.reconstruct_calibration = reconstruct_without_capture
    _CAPTURE_ENABLED = False
    try:
        _ORIGINAL_INTEGRATION_MAIN()
    finally:
        _CAPTURE_ENABLED = False
        base.evaluate_geometry = production_evaluator
        base.reconstruct_calibration = production_reconstruct


def _distribution_block(dist: dict[str, list[float]]) -> dict[str, Any]:
    raw = list(dist.get("raw_rounds") or [])
    cal = list(dist.get("calibrated_rounds") or [])
    profits = list(dist.get("calibrated_grid_profits") or [])
    out: dict[str, Any] = {}

    if raw:
        out["round_distribution"] = {
            "sample_n": len(raw),
            "p_zero_rounds_pct": _prob_zero(raw),
            "expected_raw_rounds": round(statistics.fmean(raw), 6),
            "median_raw_rounds": round(statistics.median(raw), 6),
            "p90_raw_rounds": round(float(_percentile(raw, 0.90)), 6),
            "p95_raw_rounds": round(float(_percentile(raw, 0.95)), 6),
            "max_raw_rounds": round(max(raw), 6),
            "expected_calibrated_rounds": round(statistics.fmean(cal), 6),
            "median_calibrated_rounds": round(statistics.median(cal), 6),
            "max_calibrated_rounds": round(max(cal), 6),
            "raw_probability_thresholds_pct": {
                f"p_rounds_ge_{n}": _prob_ge(raw, n) for n in ROUND_THRESHOLDS
            },
            "calibrated_probability_thresholds_pct": {
                f"p_rounds_ge_{n}": _prob_ge(cal, n) for n in ROUND_THRESHOLDS
            },
        }

    if profits:
        out["profit_distribution"] = {
            "p_zero_grid_profit_pct": _prob_zero(profits),
            "p_positive_grid_profit_pct": round(
                sum(p > 0 for p in profits) / len(profits) * 100.0, 4
            ),
            "expected_grid_profit_usdt_reconstructed": round(
                statistics.fmean(profits), 6
            ),
            "median_grid_profit_usdt_reconstructed": round(
                statistics.median(profits), 6
            ),
            "p90_grid_profit_usdt": round(float(_percentile(profits, 0.90)), 6),
            "max_grid_profit_usdt": round(max(profits), 6),
        }
    return out


def _enrich(
    row: dict[str, Any],
    dist: dict[str, list[float]] | None = None,
    sizing_model: str | None = None,
) -> dict[str, Any]:
    out = dict(row)
    grids = int(out["grids"])
    width = float(out["upper_usdt"]) - float(out["lower_usdt"])
    out["grid_spacing_usdt"] = round(width / max(1, grids - 1), 6)
    out["profit_per_expected_round_usdt"] = _safe_ratio(
        out.get("expected_grid_profit_usdt"), out.get("expected_rounds")
    )
    out["profit_per_median_round_usdt"] = _safe_ratio(
        out.get("median_grid_profit_usdt"), out.get("median_rounds")
    )
    out["gross_spacing_capture_proxy_usdt_per_eth"] = round(
        width / max(1, grids - 1), 6
    )
    if sizing_model:
        out["diagnostic_sizing_model"] = sizing_model
    if dist:
        out.update(_distribution_block(dist))
    return out


def _risk_eligible(candidate: dict[str, Any], geo: dict[str, Any]) -> bool:
    current = ((geo.get("benchmarks") or {}).get("current") or {})
    policy = geo.get("risk_policy") or {}
    if not current:
        return False
    try:
        return (
            float(candidate["escape_probability_pct"])
            <= float(policy["effective_escape_cap_pct"]) + 1e-9
            and float(candidate["p_total_pnl_positive_pct"])
            >= float(current["p_total_pnl_positive_pct"])
            - float(policy["do_not_reduce_total_pnl_positive_probability_by_more_than_pp"])
            and float(candidate["p20_total_pnl_usdt"])
            >= float(current["p20_total_pnl_usdt"])
            - float(policy["do_not_worsen_p20_total_pnl_by_more_than_usdt"])
        )
    except Exception:
        return False


def _best(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get(field) or 0.0))


def _rebalanced_evaluate(
    lower: float,
    upper: float,
    grids: int,
    current_price: float,
    state: dict,
    fee_rate: float,
    utilization: float,
    active_notional_proxy: float,
    paths: list[dict],
    profit_scale: float,
    rounds_scale: float,
) -> tuple[dict[str, Any] | None, dict[str, list[float]]]:
    if upper <= lower or not (lower < current_price < upper):
        return None, {}

    min_net, max_net = base.net_grid_profit_bounds(lower, upper, grids, fee_rate)
    min_required = max(
        base.MIN_NET_PROFIT_GRID_PCT_FLOOR,
        fee_rate * 100.0 * 2.0 + base.FEE_BUFFER_PP,
    )
    if min_net < min_required:
        return None, {}

    qinfo = v3._candidate_rebalance_seed(
        state, current_price, lower, upper, grids, fee_rate, active_notional_proxy
    )
    if qinfo is None:
        return None, {}

    qty = float(qinfo["qty"])
    eth0 = float(qinfo["seed_eth"])
    usdt0 = float(qinfo["seed_usdt"])

    raw_rounds: list[float] = []
    cal_rounds: list[float] = []
    profits: list[float] = []
    pnls: list[float] = []
    escapes: list[bool] = []
    lows: list[bool] = []
    ups: list[bool] = []

    for item in paths:
        a = base.simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "ohlc"
        )
        b = base.simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "olhc"
        )
        raw = (float(a["rounds"]) + float(b["rounds"])) / 2.0
        raw_rounds.append(raw)
        cal_rounds.append(raw * rounds_scale)
        profits.append(
            (float(a["grid_profit_usdt"]) + float(b["grid_profit_usdt"]))
            / 2.0 * profit_scale
        )
        pnls.append(
            (float(a["total_pnl_usdt"]) + float(b["total_pnl_usdt"])) / 2.0
        )
        lo = bool(a["lower_escape"] or b["lower_escape"])
        up = bool(a["upper_escape"] or b["upper_escape"])
        lows.append(lo)
        ups.append(up)
        escapes.append(lo or up)

    if not profits:
        return None, {}

    n = len(profits)
    row = {
        "lower_usdt": round(lower, 4),
        "upper_usdt": round(upper, 4),
        "center_usdt": round((lower + upper) / 2.0, 4),
        "width_usdt": round(upper - lower, 4),
        "width_pct_of_market": round((upper - lower) / current_price * 100.0, 4),
        "grids": int(grids),
        "quantity_per_grid_eth_est": round(qty, 8),
        "avg_order_notional_usdt_est": round(qty * current_price, 4),
        "buy_intervals": int(qinfo["buy_intervals"]),
        "sell_intervals": int(qinfo["sell_intervals"]),
        "net_profit_per_grid_pct_min": round(min_net, 5),
        "net_profit_per_grid_pct_max": round(max_net, 5),
        "expected_grid_profit_usdt": round(statistics.fmean(profits), 6),
        "median_grid_profit_usdt": round(statistics.median(profits), 6),
        "expected_rounds": round(statistics.fmean(cal_rounds), 6),
        "median_rounds": round(statistics.median(cal_rounds), 6),
        "escape_probability_pct": round(sum(escapes) / n * 100.0, 4),
        "lower_escape_probability_pct": round(sum(lows) / n * 100.0, 4),
        "upper_escape_probability_pct": round(sum(ups) / n * 100.0, 4),
        "p_grid_profit_ge_0_25_pct": base.sim.probability_ge(profits, 0.25),
        "p_grid_profit_ge_0_50_pct": base.sim.probability_ge(profits, 0.50),
        "p_total_pnl_positive_pct": round(sum(x > 0 for x in pnls) / n * 100.0, 4),
        "expected_total_pnl_usdt": round(statistics.fmean(pnls), 6),
        "p20_total_pnl_usdt": round(base.percentile(pnls, 0.20), 6),
        "p10_total_pnl_usdt": round(base.percentile(pnls, 0.10), 6),
        "sample_n": n,
        "seed_equity_usdt": round(float(qinfo["equity_usdt"]), 6),
        "qty_by_active_notional": round(float(qinfo["qty_by_active_notional"]), 10),
        "qty_by_equity": round(float(qinfo["qty_by_equity"]), 10),
    }
    dist = {
        "raw_rounds": raw_rounds,
        "calibrated_rounds": cal_rounds,
        "calibrated_grid_profits": profits,
    }
    return row, dist


def _integer_current_band_sweeps(
    geo: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _LIVE_CONTEXT:
        raise SystemExit("Live evaluation context was not captured")

    current = ((geo.get("benchmarks") or {}).get("current") or {})
    if not current:
        raise SystemExit("Current benchmark missing from geometry output")

    lower = float(current["lower_usdt"])
    upper = float(current["upper_usdt"])
    ctx = _LIVE_CONTEXT

    legacy_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []

    for grids in range(DIAGNOSTIC_MIN_GRIDS, DIAGNOSTIC_MAX_GRIDS + 1):
        legacy, legacy_dist = _capture_distribution(
            v3._evaluate_recovery_geometry,
            lower, upper, grids,
            ctx["current_price"], ctx["state"], ctx["fee_rate"],
            ctx["utilization"], ctx["active_notional_proxy"], ctx["paths"],
            ctx["profit_scale"], ctx["rounds_scale"],
        )
        if legacy is not None:
            r = _enrich(
                legacy, legacy_dist,
                "LEGACY_PRE_EDIT_CURRENT_HOLDINGS_PLUS_ACTIVE_NOTIONAL_CAP"
            )
            r["standard_risk_eligible"] = _risk_eligible(r, geo)
            legacy_rows.append(r)

        reb, reb_dist = _rebalanced_evaluate(
            lower, upper, grids,
            ctx["current_price"], ctx["state"], ctx["fee_rate"],
            ctx["utilization"], ctx["active_notional_proxy"], ctx["paths"],
            ctx["profit_scale"], ctx["rounds_scale"],
        )
        if reb is not None:
            r = _enrich(
                reb, reb_dist,
                "POST_EDIT_REBALANCE_TOTAL_EQUITY_PLUS_ACTIVE_NOTIONAL_CAP"
            )
            r["standard_risk_eligible"] = _risk_eligible(r, geo)
            rebalance_rows.append(r)

    return legacy_rows, rebalance_rows


def _joint_live_best_by_grid(geo: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = []
    for k, row in _CAPTURED_LIVE.items():
        r = _enrich(row, _CAPTURED_DISTS.get(k), "PRODUCTION_PHASE4D_SIZING")
        r["standard_risk_eligible"] = _risk_eligible(r, geo)
        enriched.append(r)

    out = []
    for grids in sorted({int(r["grids"]) for r in enriched}):
        group = [r for r in enriched if int(r["grids"]) == grids]
        eligible = [r for r in group if r.get("standard_risk_eligible") is True]
        out.append({
            "grids": grids,
            "feasible_candidate_count": len(group),
            "standard_risk_eligible_candidate_count": len(eligible),
            "best_feasible_by_expected_profit": _best(
                group, "expected_grid_profit_usdt"
            ),
            "best_standard_risk_eligible_by_expected_profit": _best(
                eligible, "expected_grid_profit_usdt"
            ),
        })
    return out


def _row_by_grid(rows: list[dict[str, Any]], grids: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row["grids"]) == int(grids):
            return row
    return None


def _compact_compare(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    rd = row.get("round_distribution") or {}
    probs = rd.get("raw_probability_thresholds_pct") or {}
    return {
        "grids": row.get("grids"),
        "grid_spacing_usdt": row.get("grid_spacing_usdt"),
        "quantity_per_grid_eth_est": row.get("quantity_per_grid_eth_est"),
        "avg_order_notional_usdt_est": row.get("avg_order_notional_usdt_est"),
        "expected_grid_profit_usdt": row.get("expected_grid_profit_usdt"),
        "median_grid_profit_usdt": row.get("median_grid_profit_usdt"),
        "expected_rounds": row.get("expected_rounds"),
        "median_rounds": row.get("median_rounds"),
        "profit_per_expected_round_usdt": row.get("profit_per_expected_round_usdt"),
        "p_zero_rounds_pct": rd.get("p_zero_rounds_pct"),
        "p_raw_rounds_ge_10_pct": probs.get("p_rounds_ge_10"),
        "p_raw_rounds_ge_25_pct": probs.get("p_rounds_ge_25"),
        "p_raw_rounds_ge_50_pct": probs.get("p_rounds_ge_50"),
        "standard_risk_eligible": row.get("standard_risk_eligible"),
    }


def _bias_audit(
    geo: dict[str, Any],
    legacy_rows: list[dict[str, Any]],
    rebalance_rows: list[dict[str, Any]],
    joint: list[dict[str, Any]],
) -> dict[str, Any]:
    legacy_eligible = [r for r in legacy_rows if r.get("standard_risk_eligible") is True]
    reb_eligible = [r for r in rebalance_rows if r.get("standard_risk_eligible") is True]

    legacy_pool = legacy_eligible or legacy_rows
    reb_pool = reb_eligible or rebalance_rows

    legacy_champ = _best(legacy_pool, "expected_grid_profit_usdt")
    reb_champ = _best(reb_pool, "expected_grid_profit_usdt")

    prod_selected = ((geo.get("benchmarks") or {}).get("selected") or {})
    prod_grid = int(prod_selected.get("grids") or 0)

    below_prod_min = [r for r in legacy_pool if int(r["grids"]) < PRODUCTION_MIN_GRIDS]
    below_champ = _best(below_prod_min, "expected_grid_profit_usdt")

    g10 = _row_by_grid(legacy_rows, 10)
    g22 = _row_by_grid(legacy_rows, 22)
    r10 = _row_by_grid(rebalance_rows, 10)
    r22 = _row_by_grid(rebalance_rows, 22)

    paired = []
    max_rebalance_uplift = None
    for l in legacy_rows:
        rr = _row_by_grid(rebalance_rows, int(l["grids"]))
        if rr is None:
            continue
        uplift = _safe_ratio(
            rr.get("expected_grid_profit_usdt"),
            l.get("expected_grid_profit_usdt"),
        )
        qty_ratio = _safe_ratio(
            rr.get("quantity_per_grid_eth_est"),
            l.get("quantity_per_grid_eth_est"),
        )
        item = {
            "grids": int(l["grids"]),
            "rebalance_vs_legacy_expected_profit_ratio": uplift,
            "rebalance_vs_legacy_qty_ratio": qty_ratio,
        }
        paired.append(item)
        if uplift is not None and (
            max_rebalance_uplift is None
            or uplift > max_rebalance_uplift["rebalance_vs_legacy_expected_profit_ratio"]
        ):
            max_rebalance_uplift = item

    codes = ["CALIBRATION_LEAK_EXCLUDED", "GRID_SPACING_N_MINUS_1_CORRECTED"]

    legacy_boundary = (
        legacy_champ is not None
        and int(legacy_champ["grids"]) <= PRODUCTION_MIN_GRIDS
    )
    if legacy_boundary:
        codes.append("LOW_GRID_BOUNDARY_PRESSURE_PRESENT")
    if (
        below_champ is not None
        and legacy_champ is not None
        and int(legacy_champ["grids"]) < PRODUCTION_MIN_GRIDS
    ):
        codes.append("OBJECTIVE_PREFERS_BELOW_PRODUCTION_MINIMUM")

    asset_split_material = (
        max_rebalance_uplift is not None
        and float(max_rebalance_uplift["rebalance_vs_legacy_expected_profit_ratio"] or 1.0)
        >= 1.10
    )
    if asset_split_material:
        codes.append("PRE_EDIT_ASSET_SPLIT_HAS_MATERIAL_DENSITY_EFFECT")

    if g10 and g22:
        qty_decay = _safe_ratio(g10.get("quantity_per_grid_eth_est"), g22.get("quantity_per_grid_eth_est"))
        spacing_decay = _safe_ratio(g10.get("grid_spacing_usdt"), g22.get("grid_spacing_usdt"))
        round_gain = _safe_ratio(g22.get("expected_rounds"), g10.get("expected_rounds"))
        ppr_decay = _safe_ratio(g10.get("profit_per_expected_round_usdt"), g22.get("profit_per_expected_round_usdt"))
    else:
        qty_decay = spacing_decay = round_gain = ppr_decay = None

    return {
        "diagnosis_codes": codes,
        "production_optimizer_selected_grid_count": prod_grid,
        "production_configured_min_grid_count": PRODUCTION_MIN_GRIDS,
        "production_selected_at_min_boundary": prod_grid == PRODUCTION_MIN_GRIDS,
        "legacy_current_band_champion": _compact_compare(legacy_champ),
        "legacy_best_below_production_minimum": _compact_compare(below_champ),
        "rebalanced_current_band_champion": _compact_compare(reb_champ),
        "legacy_10_vs_live_22_decomposition": {
            "grid_10": _compact_compare(g10),
            "grid_22": _compact_compare(g22),
            "qty_10_vs_22_ratio": qty_decay,
            "spacing_10_vs_22_ratio": spacing_decay,
            "rounds_22_vs_10_ratio": round_gain,
            "profit_per_round_10_vs_22_ratio": ppr_decay,
            "interpretation": (
                "The production sizing model preserves an active-order-notional "
                "budget. Higher density therefore tends to reduce quantity/grid "
                "while also reducing interval spacing. Cycle frequency must rise "
                "enough to overcome both effects."
            ),
        },
        "rebalanced_10_vs_22": {
            "grid_10": _compact_compare(r10),
            "grid_22": _compact_compare(r22),
        },
        "maximum_rebalance_vs_legacy_expected_profit_uplift": max_rebalance_uplift,
        "asset_split_effect_material_at_10pct_threshold": asset_split_material,
        "sizing_model_note": (
            "Normal Phase 4D hypothetical candidates use the current pre-edit "
            "ETH/USDT holdings plus an active-order-notional cap. The rebalanced "
            "counterfactual removes only the pre-edit asset-split constraint; it "
            "still preserves the same total equity and active-order-notional cap."
        ),
        "boundary_test_note": (
            "Grid counts 5-9 are diagnostic-only. They are not promoted into the "
            "production search. If they outperform grid 10, the existing objective "
            "is demonstrably pressing against the configured lower boundary."
        ),
        "operational_override": False,
    }


def _headline(
    legacy_rows: list[dict[str, Any]],
    rebalance_rows: list[dict[str, Any]],
    joint: list[dict[str, Any]],
) -> dict[str, Any]:
    legacy_eligible = [r for r in legacy_rows if r.get("standard_risk_eligible") is True] or legacy_rows
    reb_eligible = [r for r in rebalance_rows if r.get("standard_risk_eligible") is True] or rebalance_rows

    joint_rows = [
        x.get("best_standard_risk_eligible_by_expected_profit")
        or x.get("best_feasible_by_expected_profit")
        for x in joint
    ]
    joint_rows = [x for x in joint_rows if x]

    def p50champ(rows):
        best = None
        best_p = -1.0
        for r in rows:
            p = (
                ((r.get("round_distribution") or {}).get("raw_probability_thresholds_pct") or {})
                .get("p_rounds_ge_50")
            )
            p = float(p or 0.0)
            if p > best_p:
                best_p = p
                best = r
        return best_p, best

    p50, p50row = p50champ(reb_eligible)

    return {
        "joint_live_expected_profit_champion": _best(
            joint_rows, "expected_grid_profit_usdt"
        ),
        "legacy_current_band_expected_profit_champion": _best(
            legacy_eligible, "expected_grid_profit_usdt"
        ),
        "legacy_current_band_median_profit_champion": _best(
            legacy_eligible, "median_grid_profit_usdt"
        ),
        "rebalanced_current_band_expected_profit_champion": _best(
            reb_eligible, "expected_grid_profit_usdt"
        ),
        "rebalanced_current_band_median_profit_champion": _best(
            reb_eligible, "median_grid_profit_usdt"
        ),
        "rebalanced_current_band_activity_champion": _best(
            reb_eligible, "expected_rounds"
        ),
        "rebalanced_maximum_p_raw_rounds_ge_50_pct": round(p50, 4),
        "rebalanced_fifty_plus_rounds_champion": p50row,
        "operational_override": False,
    }



def _paired_simulate_portfolio_path(
    mapped_candles: list[list[float]],
    current_price: float,
    lower: float,
    upper: float,
    grids: int,
    qty: float,
    fee_rate: float,
    eth0: float,
    usdt0: float,
    mode: str,
) -> dict[str, Any]:
    """
    Fresh-start paired-cycle accounting.

    Critical distinction from the legacy simulator:
    - intervals initially above the market contain seeded ETH and are marked
      seed_sell;
    - the FIRST sell of seeded ETH is inventory conversion, NOT a completed
      buy->sell grid cycle;
    - only a sell that follows an actual replay buy in that same interval earns
      paired_grid_profit and increments paired_rounds.

    This removes the synthetic full-grid-spread credit that structurally rewards
    very wide / very low-density grids.
    """
    lines = base.sim.grid_lines(lower, upper, grids)
    initial = base.sim.initial_states(lines, current_price)
    states = ["buy" if st == "buy" else "seed_sell" for st in initial]

    balances = {"eth": float(eth0), "usdt": float(usdt0)}
    start_equity = balances["eth"] * current_price + balances["usdt"]

    paired_rounds = 0
    paired_grid_profit = 0.0
    seed_sells = 0
    seed_inventory_realized_pnl = 0.0
    lower_escape = False
    upper_escape = False
    prev = current_price

    for candle in mapped_candles:
        _, o, h, l, cl, _ = candle
        lower_escape = lower_escape or l < lower
        upper_escape = upper_escape or h > upper

        if mode == "ohlc":
            pts = [o, h, l, cl]
        elif mode == "olhc":
            pts = [o, l, h, cl]
        else:
            raise ValueError(mode)

        pts = [prev] + pts

        for a, b in zip(pts, pts[1:]):
            if b > a:
                for idx in range(len(lines) - 1):
                    trigger = lines[idx + 1]
                    if not (a < trigger <= b):
                        continue
                    st = states[idx]
                    if st not in {"seed_sell", "paired_sell"}:
                        continue
                    if balances["eth"] + 1e-12 < qty:
                        continue

                    balances["eth"] -= qty
                    balances["usdt"] += qty * trigger * (1.0 - fee_rate)

                    if st == "paired_sell":
                        paired_rounds += 1
                        paired_grid_profit += base.sim.interval_net_profit_usdt(
                            lines[idx], lines[idx + 1], qty, fee_rate
                        )
                    else:
                        # Seeded ETH existed at t0. Realize its mark-to-market
                        # trading contribution relative to the t0 market price,
                        # but DO NOT call it grid-cycle profit.
                        seed_sells += 1
                        seed_inventory_realized_pnl += (
                            qty * (trigger - current_price)
                            - fee_rate * qty * trigger
                        )

                    states[idx] = "buy"

            elif b < a:
                for idx in range(len(lines) - 2, -1, -1):
                    trigger = lines[idx]
                    if not (b <= trigger < a):
                        continue
                    if states[idx] != "buy":
                        continue

                    cost = qty * trigger * (1.0 + fee_rate)
                    if balances["usdt"] + 1e-9 < cost:
                        continue

                    balances["usdt"] -= cost
                    balances["eth"] += qty
                    states[idx] = "paired_sell"

        prev = cl

    end_equity = balances["usdt"] + balances["eth"] * prev
    return {
        "paired_rounds": paired_rounds,
        "paired_grid_profit_usdt": paired_grid_profit,
        "seed_sells": seed_sells,
        "seed_inventory_realized_pnl_usdt": seed_inventory_realized_pnl,
        "total_pnl_usdt": end_equity - start_equity,
        "end_equity_usdt": end_equity,
        "end_eth": balances["eth"],
        "end_usdt": balances["usdt"],
        "lower_escape": lower_escape,
        "upper_escape": upper_escape,
        "any_escape": lower_escape or upper_escape,
        "end_price": prev,
    }


def _paired_rebalanced_evaluate(
    lower: float,
    upper: float,
    grids: int,
    current_price: float,
    state: dict,
    fee_rate: float,
    active_notional_proxy: float,
    paths: list[dict],
    profit_scale: float,
    rounds_scale: float,
) -> dict[str, Any] | None:
    """
    Evaluate one hypothetical fresh/reconfigured grid using:
    - the same total equity as the live bot,
    - the same active-order-notional cap,
    - post-edit rebalance sizing,
    - paired buy->sell cycle profit only.

    Existing legacy calibration scales are applied provisionally only so the
    magnitudes remain comparable. Ranking is unchanged by a common positive
    scale. The paired model needs its own future calibration before promotion.
    """
    if upper <= lower or not (lower < current_price < upper):
        return None

    min_net, max_net = base.net_grid_profit_bounds(lower, upper, grids, fee_rate)
    min_required = max(
        base.MIN_NET_PROFIT_GRID_PCT_FLOOR,
        fee_rate * 100.0 * 2.0 + base.FEE_BUFFER_PP,
    )
    if min_net < min_required:
        return None

    qinfo = v3._candidate_rebalance_seed(
        state, current_price, lower, upper, grids, fee_rate, active_notional_proxy
    )
    if qinfo is None:
        return None

    qty = float(qinfo["qty"])
    eth0 = float(qinfo["seed_eth"])
    usdt0 = float(qinfo["seed_usdt"])

    raw_profits: list[float] = []
    scaled_profits: list[float] = []
    raw_rounds: list[float] = []
    scaled_rounds: list[float] = []
    seed_sells: list[float] = []
    seed_realized: list[float] = []
    pnls: list[float] = []
    escapes: list[bool] = []
    lows: list[bool] = []
    ups: list[bool] = []

    for item in paths:
        a = _paired_simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "ohlc"
        )
        b = _paired_simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "olhc"
        )

        raw_profit = (
            float(a["paired_grid_profit_usdt"])
            + float(b["paired_grid_profit_usdt"])
        ) / 2.0
        raw_round = (
            float(a["paired_rounds"]) + float(b["paired_rounds"])
        ) / 2.0

        raw_profits.append(raw_profit)
        scaled_profits.append(raw_profit * profit_scale)
        raw_rounds.append(raw_round)
        scaled_rounds.append(raw_round * rounds_scale)
        seed_sells.append(
            (float(a["seed_sells"]) + float(b["seed_sells"])) / 2.0
        )
        seed_realized.append(
            (
                float(a["seed_inventory_realized_pnl_usdt"])
                + float(b["seed_inventory_realized_pnl_usdt"])
            ) / 2.0
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
        return None

    n = len(raw_profits)
    spacing = (upper - lower) / max(1, grids - 1)

    return {
        "lower_usdt": round(lower, 4),
        "upper_usdt": round(upper, 4),
        "center_usdt": round((lower + upper) / 2.0, 4),
        "width_usdt": round(upper - lower, 4),
        "width_pct_of_market": round((upper - lower) / current_price * 100.0, 4),
        "grids": int(grids),
        "grid_spacing_usdt": round(spacing, 6),
        "quantity_per_grid_eth_est": round(qty, 8),
        "avg_order_notional_usdt_est": round(qty * current_price, 4),
        "buy_intervals": int(qinfo["buy_intervals"]),
        "sell_intervals": int(qinfo["sell_intervals"]),
        "net_profit_per_grid_pct_min": round(min_net, 5),
        "net_profit_per_grid_pct_max": round(max_net, 5),

        "expected_paired_grid_profit_usdt_raw": round(
            statistics.fmean(raw_profits), 6
        ),
        "median_paired_grid_profit_usdt_raw": round(
            statistics.median(raw_profits), 6
        ),
        "expected_paired_grid_profit_usdt_scaled_provisional": round(
            statistics.fmean(scaled_profits), 6
        ),
        "median_paired_grid_profit_usdt_scaled_provisional": round(
            statistics.median(scaled_profits), 6
        ),

        "expected_paired_rounds_raw": round(
            statistics.fmean(raw_rounds), 6
        ),
        "median_paired_rounds_raw": round(
            statistics.median(raw_rounds), 6
        ),
        "expected_paired_rounds_scaled_provisional": round(
            statistics.fmean(scaled_rounds), 6
        ),
        "median_paired_rounds_scaled_provisional": round(
            statistics.median(scaled_rounds), 6
        ),

        "p_zero_paired_rounds_pct": _prob_zero(raw_rounds),
        "p_paired_rounds_ge_1_pct": _prob_ge(raw_rounds, 1),
        "p_paired_rounds_ge_2_pct": _prob_ge(raw_rounds, 2),
        "p_paired_rounds_ge_5_pct": _prob_ge(raw_rounds, 5),
        "p_paired_rounds_ge_10_pct": _prob_ge(raw_rounds, 10),
        "p_paired_rounds_ge_25_pct": _prob_ge(raw_rounds, 25),
        "p_paired_rounds_ge_50_pct": _prob_ge(raw_rounds, 50),
        "p90_paired_rounds_raw": round(float(_percentile(raw_rounds, 0.90)), 6),
        "p95_paired_rounds_raw": round(float(_percentile(raw_rounds, 0.95)), 6),
        "max_paired_rounds_raw": round(max(raw_rounds), 6),

        "expected_seed_sells": round(statistics.fmean(seed_sells), 6),
        "median_seed_sells": round(statistics.median(seed_sells), 6),
        "expected_seed_inventory_realized_pnl_usdt": round(
            statistics.fmean(seed_realized), 6
        ),

        "escape_probability_pct": round(sum(escapes) / n * 100.0, 4),
        "lower_escape_probability_pct": round(sum(lows) / n * 100.0, 4),
        "upper_escape_probability_pct": round(sum(ups) / n * 100.0, 4),
        "p_total_pnl_positive_pct": round(sum(x > 0 for x in pnls) / n * 100.0, 4),
        "expected_total_pnl_usdt": round(statistics.fmean(pnls), 6),
        "p20_total_pnl_usdt": round(base.percentile(pnls, 0.20), 6),
        "p10_total_pnl_usdt": round(base.percentile(pnls, 0.10), 6),
        "sample_n": n,

        "profit_per_expected_paired_round_usdt_raw": _safe_ratio(
            statistics.fmean(raw_profits), statistics.fmean(raw_rounds)
        ),
        "seed_equity_usdt": round(float(qinfo["equity_usdt"]), 6),
        "qty_by_active_notional": round(float(qinfo["qty_by_active_notional"]), 10),
        "qty_by_equity": round(float(qinfo["qty_by_equity"]), 10),
        "calibration_status": "PROVISIONAL_LEGACY_SCALE_REUSED_NOT_VALIDATED_FOR_PAIRED_MODEL",
    }


def _paired_current_band_sweep(
    geo: dict[str, Any],
) -> list[dict[str, Any]]:
    if not _LIVE_CONTEXT:
        raise SystemExit("Live evaluation context missing for paired-cycle sweep")

    current = ((geo.get("benchmarks") or {}).get("current") or {})
    if not current:
        raise SystemExit("Current benchmark missing for paired-cycle sweep")

    lower = float(current["lower_usdt"])
    upper = float(current["upper_usdt"])
    ctx = _LIVE_CONTEXT
    out: list[dict[str, Any]] = []

    for grids in range(DIAGNOSTIC_MIN_GRIDS, DIAGNOSTIC_MAX_GRIDS + 1):
        row = _paired_rebalanced_evaluate(
            lower,
            upper,
            grids,
            ctx["current_price"],
            ctx["state"],
            ctx["fee_rate"],
            ctx["active_notional_proxy"],
            ctx["paths"],
            ctx["profit_scale"],
            ctx["rounds_scale"],
        )
        if row is None:
            continue
        row["standard_risk_eligible"] = _risk_eligible(row, geo)
        out.append(row)

    return out


def _first_seed_sell_synthetic_profit_signature(
    legacy_rows: list[dict[str, Any]],
    geo: dict[str, Any],
) -> dict[str, Any]:
    """
    Quantify the exact legacy first-seed-sell accounting effect on the lowest
    tested grid count. This is intentionally transparent and auditable.
    """
    if not legacy_rows or not _LIVE_CONTEXT:
        return {}

    row = min(legacy_rows, key=lambda r: int(r["grids"]))
    grids = int(row["grids"])
    lower = float(row["lower_usdt"])
    upper = float(row["upper_usdt"])
    current_price = float(_LIVE_CONTEXT["current_price"])
    fee_rate = float(_LIVE_CONTEXT["fee_rate"])
    profit_scale = float(_LIVE_CONTEXT["profit_scale"])
    qty = float(row["quantity_per_grid_eth_est"])

    lines = base.sim.grid_lines(lower, upper, grids)
    states = base.sim.initial_states(lines, current_price)
    sell_idx = [idx for idx, st in enumerate(states) if st == "sell"]

    if not sell_idx:
        return {}

    idx = sell_idx[0]
    synthetic_raw = base.sim.interval_net_profit_usdt(
        lines[idx], lines[idx + 1], qty, fee_rate
    )
    synthetic_scaled = synthetic_raw * profit_scale
    legacy_median = float(row.get("median_grid_profit_usdt") or 0.0)
    abs_diff = abs(legacy_median - synthetic_scaled)
    tolerance = max(0.01, abs(legacy_median) * 0.02)

    return {
        "grid_count": grids,
        "interval_lower_usdt": round(lines[idx], 6),
        "interval_upper_usdt": round(lines[idx + 1], 6),
        "start_market_price_usdt": round(current_price, 6),
        "quantity_per_grid_eth": round(qty, 8),
        "synthetic_full_interval_profit_raw_usdt": round(synthetic_raw, 6),
        "synthetic_full_interval_profit_scaled_usdt": round(synthetic_scaled, 6),
        "legacy_median_grid_profit_usdt": round(legacy_median, 6),
        "absolute_difference_usdt": round(abs_diff, 6),
        "matches_legacy_median_within_2pct_or_1cent": abs_diff <= tolerance,
        "interpretation": (
            "If this matches the legacy median, the median reported grid profit "
            "is dominated by crediting an initial seeded sell with a full grid "
            "spread even though no replay buy occurred at the interval lower line."
        ),
    }


def _paired_cycle_correction_audit(
    geo: dict[str, Any],
    legacy_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    low_grid_audit: dict[str, Any],
) -> dict[str, Any]:
    eligible = [
        r for r in paired_rows if r.get("standard_risk_eligible") is True
    ] or paired_rows

    paired_expected_champion = (
        max(
            eligible,
            key=lambda r: float(
                r.get("expected_paired_grid_profit_usdt_raw") or 0.0
            ),
        )
        if eligible
        else None
    )
    paired_median_champion = (
        max(
            eligible,
            key=lambda r: float(
                r.get("median_paired_grid_profit_usdt_raw") or 0.0
            ),
        )
        if eligible
        else None
    )

    signature = _first_seed_sell_synthetic_profit_signature(legacy_rows, geo)

    production_selected = ((geo.get("benchmarks") or {}).get("selected") or {})
    production_grid = int(production_selected.get("grids") or 0)

    legacy_by_grid = {int(r["grids"]): r for r in legacy_rows}
    paired_by_grid = {int(r["grids"]): r for r in paired_rows}

    production_legacy = legacy_by_grid.get(production_grid)
    production_paired = paired_by_grid.get(production_grid)

    inflation_ratio = None
    if production_legacy and production_paired:
        legacy_profit = float(
            production_legacy.get("expected_grid_profit_usdt") or 0.0
        )
        paired_profit = float(
            production_paired.get("expected_paired_grid_profit_usdt_scaled_provisional")
            or 0.0
        )
        if paired_profit > 1e-12:
            inflation_ratio = round(legacy_profit / paired_profit, 6)
        elif legacy_profit > 0:
            inflation_ratio = None

    seed_bias_detected = bool(
        signature.get("matches_legacy_median_within_2pct_or_1cent")
    )

    codes = []
    if seed_bias_detected:
        codes.append("INITIAL_SEED_SELL_FULL_SPREAD_CREDIT_CONFIRMED")
    if low_grid_audit.get("production_selected_at_min_boundary") is True:
        codes.append("PRODUCTION_SELECTION_AT_LOW_GRID_BOUNDARY")
    if low_grid_audit.get("legacy_best_below_production_minimum"):
        codes.append("LEGACY_OBJECTIVE_PREFERS_BELOW_PRODUCTION_MINIMUM")

    provisional_differs = (
        paired_expected_champion is not None
        and int(paired_expected_champion["grids"]) != production_grid
    )
    if provisional_differs:
        codes.append("PAIRED_CYCLE_CHAMPION_DIFFERS_FROM_PRODUCTION_SELECTION")

    safety_block = bool(
        seed_bias_detected
        and low_grid_audit.get("production_selected_at_min_boundary") is True
    )
    if safety_block:
        codes.append("KEEP_CURRENT_SAFETY_BLOCK_RECOMMENDED")

    return {
        "diagnosis_codes": codes,
        "seed_profit_bias_detected": seed_bias_detected,
        "synthetic_seed_profit_signature": signature,
        "production_selected_grid_count": production_grid,
        "production_legacy_current_band": production_legacy,
        "production_paired_current_band": production_paired,
        "production_legacy_vs_paired_scaled_profit_ratio": inflation_ratio,
        "provisional_paired_expected_profit_champion": paired_expected_champion,
        "provisional_paired_median_profit_champion": paired_median_champion,
        "provisional_champion_differs_from_production": provisional_differs,
        "safety_block_recommended": safety_block,
        "safety_blocker_code": (
            "GRID_PROFIT_INITIAL_SEED_ACCOUNTING_BIAS"
            if safety_block else None
        ),
        "paired_cycle_semantics": (
            "Only sell events that follow an actual replay buy in the same "
            "interval count as grid-profit cycles. Initial seeded-ETH sells are "
            "inventory conversion and contribute only to total P&L, not grid profit."
        ),
        "calibration_warning": (
            "Current legacy profit/round calibration scales are reused only as "
            "a provisional magnitude bridge. The paired-cycle model must receive "
            "its own historical/prospective calibration before operational promotion."
        ),
        "operational_override": False,
    }

def _write_density_output() -> None:
    if not v3.GEO_PATH.is_file():
        raise SystemExit("Phase 4D geometry output missing after v4.3 capture")

    geo = json.loads(v3.GEO_PATH.read_text(encoding="utf-8"))
    legacy_rows, rebalance_rows = _integer_current_band_sweeps(geo)
    joint = _joint_live_best_by_grid(geo)
    audit = _bias_audit(geo, legacy_rows, rebalance_rows, joint)
    headline = _headline(legacy_rows, rebalance_rows, joint)

    paired_rows = _paired_current_band_sweep(geo)
    paired_audit = _paired_cycle_correction_audit(
        geo, legacy_rows, paired_rows, audit
    )

    source = geo.get("source_state") or {}
    integration = geo.get("execution_resolution_integration") or {}

    payload = {
        "schema": "pionex_grid_density_sweep_v1",
        "version": "4.3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_DIAGNOSTIC_ONLY",
        "scope": {
            "platform": "Pionex",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "horizon_h": 24,
            "diagnostic_grid_count_range": [
                DIAGNOSTIC_MIN_GRIDS, DIAGNOSTIC_MAX_GRIDS
            ],
            "production_min_grid_count_unchanged": PRODUCTION_MIN_GRIDS,
        },
        "source_state": {
            "captured_at_utc": source.get("captured_at_utc"),
            "captured_at_local": source.get("captured_at_local"),
            "current_price_usdt": source.get("latest_market_price_usdt"),
            "geometry_market_price_source": source.get(
                "geometry_market_price_source"
            ),
        },
        "execution_resolution": {
            "execution_replay_resolution": integration.get(
                "execution_replay_resolution"
            ),
            "policy_status": integration.get("policy_status"),
            "execution_resolution_integrated": integration.get(
                "execution_resolution_integrated"
            ),
        },
        "method": {
            "live_candidate_capture": (
                "Only current/live Phase 4D candidate evaluations are captured. "
                "Historical calibration reconstruction is explicitly excluded."
            ),
            "current_band_isolation": (
                "Explicit density sweeps hold the live lower/upper bounds "
                "constant and vary only grid count."
            ),
            "legacy_sizing": (
                "Existing normal Phase 4D hypothetical sizing using current "
                "pre-edit ETH/USDT holdings + observed utilization + preserved "
                "active-order-notional cap."
            ),
            "rebalanced_counterfactual": (
                "Post-edit rebalanced sizing using the same total equity and "
                "active-order-notional cap, removing only pre-edit asset-split bias."
            ),
            "paired_cycle_correction": (
                "Initial seeded ETH sells do not earn grid-cycle profit. Only "
                "sell events preceded by an actual replay buy in that interval "
                "count as paired grid-profit cycles."
            ),
            "spacing_formula": "width_usdt / (grids - 1)",
            "operational_effect": (
                "DIAGNOSTIC_ONLY; runner may apply KEEP_CURRENT safety blocker "
                "when confirmed seed-profit accounting bias affects a boundary selection."
            ),
        },
        "live_candidate_count_captured": len(_CAPTURED_LIVE),
        "live_grid_counts_captured": sorted({
            int(r["grids"]) for r in _CAPTURED_LIVE.values()
        }),
        "current_benchmark": ((geo.get("benchmarks") or {}).get("current") or {}),
        "production_selected_geometry": ((geo.get("benchmarks") or {}).get("selected") or {}),
        "headline": headline,
        "low_grid_pressure_audit": audit,
        "paired_cycle_correction": paired_audit,
        "joint_live_best_by_grid_count": joint,
        "legacy_current_band_integer_sweep": legacy_rows,
        "rebalanced_current_band_integer_sweep": rebalance_rows,
        "paired_cycle_current_band_integer_sweep": paired_rows,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    paired_champ = (
        paired_audit.get("provisional_paired_expected_profit_champion") or {}
    )

    print("\n=== GRID-DENSITY / PAIRED-CYCLE DIAGNOSTIC v4.3 ===")
    print("Live candidates captured:", len(_CAPTURED_LIVE))
    print("Low-grid diagnosis:", audit.get("diagnosis_codes"))
    print(
        "Seed-profit diagnosis:",
        paired_audit.get("diagnosis_codes"),
    )
    print(
        "Legacy current-band champion:",
        (headline.get("legacy_current_band_expected_profit_champion") or {}).get("grids"),
        "grids",
    )
    print(
        "Provisional paired-cycle champion:",
        paired_champ.get("grids"),
        "grids",
    )
    print(
        "Safety block recommended:",
        paired_audit.get("safety_block_recommended"),
    )
    print("Written:", OUT_PATH)

def main() -> None:
    v3.integration.main = _capturing_integration_main
    try:
        v3.main()
    finally:
        v3.integration.main = _ORIGINAL_INTEGRATION_MAIN

    _write_density_output()


if __name__ == "__main__":
    main()
