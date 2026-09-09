#!/usr/bin/env python3
"""
Phase 4D v4.2 — live-only grid-density sweet-spot and low-grid-bias diagnostic.

This adapter wraps the validated v3.1 Phase 4D path and adds a diagnostic layer.
It does NOT alter the existing operational actionability decision.

Fixes vs v4.0
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


def _write_density_output() -> None:
    if not v3.GEO_PATH.is_file():
        raise SystemExit("Phase 4D geometry output missing after v4.2 capture")

    geo = json.loads(v3.GEO_PATH.read_text(encoding="utf-8"))
    legacy_rows, rebalance_rows = _integer_current_band_sweeps(geo)
    joint = _joint_live_best_by_grid(geo)
    audit = _bias_audit(geo, legacy_rows, rebalance_rows, joint)
    headline = _headline(legacy_rows, rebalance_rows, joint)

    source = geo.get("source_state") or {}
    integration = geo.get("execution_resolution_integration") or {}

    payload = {
        "schema": "pionex_grid_density_sweep_v1",
        "version": "4.2",
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
                "Both explicit density sweeps hold the live lower/upper bounds "
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
            "spacing_formula": "width_usdt / (grids - 1)",
            "operational_effect": "NONE",
        },
        "live_candidate_count_captured": len(_CAPTURED_LIVE),
        "live_grid_counts_captured": sorted({
            int(r["grids"]) for r in _CAPTURED_LIVE.values()
        }),
        "current_benchmark": ((geo.get("benchmarks") or {}).get("current") or {}),
        "production_selected_geometry": ((geo.get("benchmarks") or {}).get("selected") or {}),
        "headline": headline,
        "low_grid_pressure_audit": audit,
        "joint_live_best_by_grid_count": joint,
        "legacy_current_band_integer_sweep": legacy_rows,
        "rebalanced_current_band_integer_sweep": rebalance_rows,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== GRID-DENSITY / LOW-GRID-BIAS DIAGNOSTIC v4.2 ===")
    print("Live candidates captured:", len(_CAPTURED_LIVE))
    print("Diagnosis:", audit.get("diagnosis_codes"))
    print(
        "Legacy current-band champion:",
        (headline.get("legacy_current_band_expected_profit_champion") or {}).get("grids"),
        "grids",
    )
    print(
        "Rebalanced current-band champion:",
        (headline.get("rebalanced_current_band_expected_profit_champion") or {}).get("grids"),
        "grids",
    )
    print("Operational override: False")
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
