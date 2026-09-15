#!/usr/bin/env python3
"""
Phase 4D v4.4 — paired-cycle calibration + density plateau promotion-readiness.

Architecture
------------
This is a thin adapter over v4.3. It deliberately does NOT rewrite or delete the
v4.3 diagnostic. Instead it:

1. runs the existing v4.3 pipeline unchanged;
2. reconstructs an independent walk-forward calibration for the corrected
   paired buy->sell cycle model using historical manual Pionex state windows;
3. applies those paired-specific scales to the v4.3 fixed-current-band density
   sweep;
4. identifies a robust near-peak density plateau rather than selecting the
   single noisiest expected-profit argmax;
5. annotates platform execution feasibility without inventing a universal
   Pionex minimum-order value.

This remains decision support only. It does not place orders or edit Pionex.
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


OUT_PATH = Path("data/diagnostics/pionex_grid_density_sweep_v1.json")
CONSTRAINTS_PATH = Path("data/pionex/pionex_execution_constraints_v1.json")

VERSION = "4.4"

PAIRED_CAL_MIN_TOTAL_WINDOWS = 8
PAIRED_CAL_MIN_TRAIN_WINDOWS = 6
PAIRED_CAL_MIN_VALIDATION_WINDOWS = 2
PAIRED_CAL_MAX_VALIDATION_WINDOWS = 6

PLATEAU_EXPECTED_PROFIT_FLOOR_RATIO = 0.95
MIN_PAIRED_EXPECTED_GAIN_VS_CURRENT_PCT = 5.0
MIN_PAIRED_MEDIAN_GAIN_VS_CURRENT_PCT = 0.0

EXACT_VALIDATION_PRICE_TOL_USDT = 5.0


def _f(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v in (None, ""):
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _pct_gain(new: Any, old: Any) -> Optional[float]:
    a = _f(new)
    b = _f(old)
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return round((a / b - 1.0) * 100.0, 4)


def _mae(actual: list[float], predicted: list[float]) -> Optional[float]:
    if not actual or len(actual) != len(predicted):
        return None
    return statistics.fmean(abs(a - p) for a, p in zip(actual, predicted))


def _mape(actual: list[float], predicted: list[float]) -> Optional[float]:
    pairs = [(a, p) for a, p in zip(actual, predicted) if abs(a) > 1e-9]
    if not pairs:
        return None
    return statistics.fmean(abs(a - p) / abs(a) for a, p in pairs) * 100.0


def _paired_observed_prediction(
    start: dict,
    current_price: float,
    fee_rate: float,
    paths: list[dict],
) -> Optional[dict]:
    """
    Predict the historical live bot exactly as observed at the start snapshot.

    Unlike the hypothetical density sweep, this does NOT rebalance the start
    portfolio. It uses the recorded quantity/grid and balances so the calibration
    target matches the historical live bot as closely as the available snapshots
    permit.
    """
    lower = float(start["lower_limit_usdt"])
    upper = float(start["upper_limit_usdt"])
    grids = int(float(start["grids"]))
    qty = float(start["quantity_per_grid_eth"])
    eth0 = float(start.get("eth_holdings") or 0.0)
    usdt0 = float(start.get("usdt_holdings") or 0.0)

    if qty <= 0 or not paths:
        return None

    profits: list[float] = []
    rounds: list[float] = []

    for item in paths:
        a = v4._paired_simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "ohlc"
        )
        b = v4._paired_simulate_portfolio_path(
            item["candles"], current_price, lower, upper, grids, qty,
            fee_rate, eth0, usdt0, "olhc"
        )
        profits.append(
            (
                float(a["paired_grid_profit_usdt"])
                + float(b["paired_grid_profit_usdt"])
            ) / 2.0
        )
        rounds.append(
            (float(a["paired_rounds"]) + float(b["paired_rounds"])) / 2.0
        )

    if not profits:
        return None

    return {
        "predicted_paired_profit_24h_raw_usdt": statistics.fmean(profits),
        "predicted_paired_rounds_24h_raw": statistics.fmean(rounds),
        "median_paired_profit_24h_raw_usdt": statistics.median(profits),
        "median_paired_rounds_24h_raw": statistics.median(rounds),
        "analogue_paths": len(paths),
    }


def _paired_walk_forward_calibration() -> dict:
    """
    Reconstruct paired-cycle forecasts at historical manual Pionex state times.

    The promoted Phase 4D execution adapter leaves base.mapped_paths pointed at
    the active replay resolution. Therefore, after v4.3 completes, this function
    reuses the same 5-minute path cache/policy when 5-minute execution is active.
    """
    states = base.phase4c.read_manual_states()
    master = base.sim.build_master_candles()
    features = base.sim.load_eth_features()

    profile = base.sim.load_json(base.PROFILE_PATH)
    fee_pct = float(
        profile.get("fee_model", {}).get(
            "standard_public_spot_fee_pct_per_fill_reference", 0.05
        )
    )
    fee_rate = fee_pct / 100.0

    windows: list[dict] = []

    for start, end in zip(states, states[1:]):
        if not base.same_geometry(start, end):
            continue

        start_dt = start["_dt"]
        end_dt = end["_dt"]
        elapsed_h = (end_dt - start_dt).total_seconds() / 3600.0
        if not (base.CAL_MIN_H <= elapsed_h <= base.CAL_MAX_H):
            continue

        feat = base.nearest_feature(features, int(start_dt.timestamp()))
        if feat is None:
            continue

        mature_then = [
            r for r in features
            if int(r["ts"]) + base.sim.HORIZON_H * 3600 <= int(feat["ts"])
        ]
        independent_then = base.sim.greedy_independent(mature_then)

        try:
            analogs, _ = base.select_analogs(
                independent_then,
                feat,
                master,
                n=min(base.FINAL_ANALOG_N, len(independent_then)),
            )
        except SystemExit:
            continue

        current_price = float(start["current_price_usdt"])

        try:
            paths = base.mapped_paths(analogs, master, current_price)
        except Exception as exc:
            print(
                "Paired calibration path build failed for",
                start_dt.isoformat(),
                ":",
                exc,
            )
            continue

        if len(paths) < 24:
            continue

        pred = _paired_observed_prediction(
            start, current_price, fee_rate, paths
        )
        if not pred:
            continue

        p0 = float(start["grid_profit_usdt"])
        p1 = float(end["grid_profit_usdt"])
        r0 = float(start["rounds_total"])
        r1 = float(end["rounds_total"])

        actual_profit_24 = (p1 - p0) * 24.0 / elapsed_h
        actual_rounds_24 = (r1 - r0) * 24.0 / elapsed_h

        if actual_profit_24 < -1e-9 or actual_rounds_24 < -1e-9:
            continue

        pred_profit = float(pred["predicted_paired_profit_24h_raw_usdt"])
        pred_rounds = float(pred["predicted_paired_rounds_24h_raw"])

        windows.append({
            "start_utc": start_dt.isoformat(),
            "end_utc": end_dt.isoformat(),
            "elapsed_h": round(elapsed_h, 4),
            "grids": int(float(start["grids"])),
            "lower_usdt": float(start["lower_limit_usdt"]),
            "upper_usdt": float(start["upper_limit_usdt"]),
            "current_price_usdt": current_price,
            "quantity_per_grid_eth": float(start["quantity_per_grid_eth"]),
            "analogue_paths": int(pred["analogue_paths"]),
            "predicted_paired_profit_24h_raw_usdt": round(pred_profit, 6),
            "actual_grid_profit_24h_equiv_usdt": round(actual_profit_24, 6),
            "predicted_paired_rounds_24h_raw": round(pred_rounds, 6),
            "actual_rounds_24h_equiv": round(actual_rounds_24, 6),
            "profit_ratio": (
                round(actual_profit_24 / pred_profit, 6)
                if pred_profit > 1e-12 else None
            ),
            "rounds_ratio": (
                round(actual_rounds_24 / pred_rounds, 6)
                if pred_rounds > 1e-12 else None
            ),
        })

    windows.sort(key=lambda w: w["start_utc"])
    n = len(windows)

    validation_n = 0
    if n >= PAIRED_CAL_MIN_TOTAL_WINDOWS:
        validation_n = min(
            PAIRED_CAL_MAX_VALIDATION_WINDOWS,
            max(PAIRED_CAL_MIN_VALIDATION_WINDOWS, n // 4),
        )
        if n - validation_n < PAIRED_CAL_MIN_TRAIN_WINDOWS:
            validation_n = max(0, n - PAIRED_CAL_MIN_TRAIN_WINDOWS)

    training = windows[:-validation_n] if validation_n else windows
    validation = windows[-validation_n:] if validation_n else []

    profit_ratios = [
        float(w["profit_ratio"])
        for w in training
        if w.get("profit_ratio") is not None and float(w["profit_ratio"]) > 0
    ]
    rounds_ratios = [
        float(w["rounds_ratio"])
        for w in training
        if w.get("rounds_ratio") is not None and float(w["rounds_ratio"]) > 0
    ]

    observed_profit_scale = (
        statistics.median(profit_ratios) if profit_ratios else None
    )
    observed_rounds_scale = (
        statistics.median(rounds_ratios) if rounds_ratios else None
    )

    can_apply = (
        len(training) >= PAIRED_CAL_MIN_TRAIN_WINDOWS
        and observed_profit_scale is not None
        and observed_rounds_scale is not None
    )

    profit_scale = (
        min(base.CAL_SCALE_HI, max(base.CAL_SCALE_LO, observed_profit_scale))
        if can_apply else 1.0
    )
    rounds_scale = (
        min(base.CAL_SCALE_HI, max(base.CAL_SCALE_LO, observed_rounds_scale))
        if can_apply else 1.0
    )

    holdout_ready = (
        can_apply
        and len(validation) >= PAIRED_CAL_MIN_VALIDATION_WINDOWS
    )

    status = (
        "PAIRED_CALIBRATION_ACTIVE_WITH_HOLDOUT"
        if holdout_ready
        else (
            "PAIRED_CALIBRATION_EARLY_ACTIVE_NO_HOLDOUT"
            if can_apply
            else "PAIRED_CALIBRATION_EARLY_EVIDENCE_NOT_APPLIED"
        )
    )

    validation_metrics = {
        "validation_windows": len(validation),
        "profit_mae_usdt": None,
        "profit_mape_pct": None,
        "rounds_mae": None,
        "rounds_mape_pct": None,
    }

    if validation:
        a_profit = [float(w["actual_grid_profit_24h_equiv_usdt"]) for w in validation]
        p_profit = [
            float(w["predicted_paired_profit_24h_raw_usdt"]) * profit_scale
            for w in validation
        ]
        a_rounds = [float(w["actual_rounds_24h_equiv"]) for w in validation]
        p_rounds = [
            float(w["predicted_paired_rounds_24h_raw"]) * rounds_scale
            for w in validation
        ]
        validation_metrics = {
            "validation_windows": len(validation),
            "profit_mae_usdt": round(_mae(a_profit, p_profit), 6),
            "profit_mape_pct": round(_mape(a_profit, p_profit), 4)
                if _mape(a_profit, p_profit) is not None else None,
            "rounds_mae": round(_mae(a_rounds, p_rounds), 6),
            "rounds_mape_pct": round(_mape(a_rounds, p_rounds), 4)
                if _mape(a_rounds, p_rounds) is not None else None,
        }

    return {
        "schema": "pionex_paired_cycle_calibration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "calibration_ready": bool(holdout_ready),
        "method": (
            "Walk-forward reconstruction at historical manual Pionex state times. "
            "Only 24h-mature analogue paths available at each historical start are "
            "used. Predictions use observed historical quantity/grid and balances, "
            "corrected paired buy->sell cycle accounting, and the Phase 4D promoted "
            "execution replay resolution. Scale is estimated on earlier windows and "
            "reported against a chronologically later holdout."
        ),
        "important_limitation": (
            "Manual snapshots do not preserve every per-grid open-order state. "
            "Historical interval states are reconstructed from price and geometry, "
            "matching the existing Phase 4D calibration convention."
        ),
        "actual_rounds_semantics_assumption": (
            "Pionex rounds_total is treated as the realised completed-grid-round "
            "counter, consistent with the legacy calibration. If Pionex changes "
            "that UI/API semantic, this calibration must be revisited."
        ),
        "execution_replay_resolution": (
            ((json.loads(OUT_PATH.read_text(encoding="utf-8"))
              .get("execution_resolution") or {})
             .get("execution_replay_resolution"))
            if OUT_PATH.is_file() else None
        ),
        "evaluated_windows": n,
        "training_windows": len(training),
        "validation_windows": len(validation),
        "minimum_training_windows": PAIRED_CAL_MIN_TRAIN_WINDOWS,
        "minimum_validation_windows": PAIRED_CAL_MIN_VALIDATION_WINDOWS,
        "observed_median_profit_ratio_training": (
            round(observed_profit_scale, 6)
            if observed_profit_scale is not None else None
        ),
        "observed_median_rounds_ratio_training": (
            round(observed_rounds_scale, 6)
            if observed_rounds_scale is not None else None
        ),
        "profit_scale_applied": round(profit_scale, 6),
        "rounds_scale_applied": round(rounds_scale, 6),
        "scale_bounds": [base.CAL_SCALE_LO, base.CAL_SCALE_HI],
        "holdout_validation": validation_metrics,
        "windows": windows,
    }


def _load_constraints() -> dict:
    if not CONSTRAINTS_PATH.is_file():
        return {
            "schema": "pionex_execution_constraints_v1",
            "status": "MISSING_LIVE_VALIDATION_POLICY",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "min_order_notional_usdt": None,
            "min_quantity_eth": None,
            "validated_candidate": None,
        }
    try:
        return json.loads(CONSTRAINTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema": "pionex_execution_constraints_v1",
            "status": "INVALID_POLICY_FILE",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "min_order_notional_usdt": None,
            "min_quantity_eth": None,
            "validated_candidate": None,
        }


def _matches_validated_candidate(row: dict, validated: dict) -> bool:
    if not validated or validated.get("accepted_by_pionex_ui") is not True:
        return False
    try:
        return (
            int(row["grids"]) == int(validated["grids"])
            and abs(float(row["lower_usdt"]) - float(validated["lower_usdt"]))
                <= EXACT_VALIDATION_PRICE_TOL_USDT
            and abs(float(row["upper_usdt"]) - float(validated["upper_usdt"]))
                <= EXACT_VALIDATION_PRICE_TOL_USDT
        )
    except Exception:
        return False


def _execution_feasibility(row: dict, constraints: dict) -> dict:
    failures: list[str] = []
    checks: list[str] = []

    min_notional = _f(constraints.get("min_order_notional_usdt"))
    if min_notional is not None:
        checks.append("MIN_ORDER_NOTIONAL")
        if float(row.get("avg_order_notional_usdt_est") or 0.0) + 1e-9 < min_notional:
            failures.append("BELOW_CONFIGURED_MIN_ORDER_NOTIONAL")

    min_qty = _f(constraints.get("min_quantity_eth"))
    if min_qty is not None:
        checks.append("MIN_QUANTITY_ETH")
        if float(row.get("quantity_per_grid_eth_est") or 0.0) + 1e-12 < min_qty:
            failures.append("BELOW_CONFIGURED_MIN_QUANTITY_ETH")

    max_grids = constraints.get("max_grids")
    if max_grids not in (None, ""):
        checks.append("MAX_GRIDS")
        if int(row["grids"]) > int(max_grids):
            failures.append("ABOVE_CONFIGURED_MAX_GRIDS")

    validated = constraints.get("validated_candidate") or {}
    exact_ui_validation = _matches_validated_candidate(row, validated)

    if failures:
        status = "FAIL_KNOWN_CONSTRAINT"
    elif exact_ui_validation:
        status = "PASS_EXACT_PIONEX_UI_VALIDATION"
    elif checks:
        status = "PASS_KNOWN_STATIC_CHECKS_LIVE_UI_STILL_REQUIRED"
    else:
        status = "UNKNOWN_REQUIRES_LIVE_PIONEX_SETUP_VALIDATION"

    return {
        "status": status,
        "known_checks_applied": checks,
        "known_failures": failures,
        "exact_candidate_accepted_by_pionex_ui": exact_ui_validation,
        "policy_status": constraints.get("status"),
        "note": (
            "Pionex Spot Grid minimum investment/order feasibility is dynamic "
            "with pair, range, grid count and order rules. A candidate is not "
            "operationally promoted without exact live Pionex setup validation."
        ),
    }


def _apply_paired_calibration(rows: list[dict], calibration: dict, constraints: dict) -> None:
    profit_scale = float(calibration.get("profit_scale_applied") or 1.0)
    rounds_scale = float(calibration.get("rounds_scale_applied") or 1.0)
    status = calibration.get("status")

    for row in rows:
        ep = float(row.get("expected_paired_grid_profit_usdt_raw") or 0.0)
        mp = float(row.get("median_paired_grid_profit_usdt_raw") or 0.0)
        er = float(row.get("expected_paired_rounds_raw") or 0.0)
        mr = float(row.get("median_paired_rounds_raw") or 0.0)

        row["expected_paired_grid_profit_usdt_calibrated"] = round(ep * profit_scale, 6)
        row["median_paired_grid_profit_usdt_calibrated"] = round(mp * profit_scale, 6)
        row["expected_paired_rounds_calibrated"] = round(er * rounds_scale, 6)
        row["median_paired_rounds_calibrated"] = round(mr * rounds_scale, 6)
        row["paired_calibration_status"] = status
        row["execution_feasibility"] = _execution_feasibility(row, constraints)


def _contiguous_plateau(
    eligible: list[dict],
    expected_field: str,
    peak: dict,
    floor_ratio: float,
) -> list[dict]:
    threshold = float(peak[expected_field]) * floor_ratio
    near = sorted(
        [r for r in eligible if float(r.get(expected_field) or 0.0) >= threshold],
        key=lambda r: int(r["grids"]),
    )
    if not near:
        return [peak]

    groups: list[list[dict]] = []
    current: list[dict] = []
    for row in near:
        if not current or int(row["grids"]) == int(current[-1]["grids"]) + 1:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    if current:
        groups.append(current)

    peak_grid = int(peak["grids"])
    for group in groups:
        if any(int(r["grids"]) == peak_grid for r in group):
            return group
    return [peak]


def _compact_candidate(row: Optional[dict]) -> Optional[dict]:
    if not row:
        return None
    return {
        "lower_usdt": row.get("lower_usdt"),
        "upper_usdt": row.get("upper_usdt"),
        "grids": row.get("grids"),
        "grid_spacing_usdt": row.get("grid_spacing_usdt"),
        "quantity_per_grid_eth_est": row.get("quantity_per_grid_eth_est"),
        "avg_order_notional_usdt_est": row.get("avg_order_notional_usdt_est"),
        "expected_paired_grid_profit_usdt_calibrated": row.get(
            "expected_paired_grid_profit_usdt_calibrated"
        ),
        "median_paired_grid_profit_usdt_calibrated": row.get(
            "median_paired_grid_profit_usdt_calibrated"
        ),
        "expected_paired_rounds_calibrated": row.get(
            "expected_paired_rounds_calibrated"
        ),
        "median_paired_rounds_calibrated": row.get(
            "median_paired_rounds_calibrated"
        ),
        "expected_paired_grid_profit_usdt_raw": row.get(
            "expected_paired_grid_profit_usdt_raw"
        ),
        "median_paired_grid_profit_usdt_raw": row.get(
            "median_paired_grid_profit_usdt_raw"
        ),
        "expected_paired_rounds_raw": row.get("expected_paired_rounds_raw"),
        "median_paired_rounds_raw": row.get("median_paired_rounds_raw"),
        "p_zero_paired_rounds_pct": row.get("p_zero_paired_rounds_pct"),
        "p_paired_rounds_ge_10_pct": row.get("p_paired_rounds_ge_10_pct"),
        "p_paired_rounds_ge_25_pct": row.get("p_paired_rounds_ge_25_pct"),
        "p_paired_rounds_ge_50_pct": row.get("p_paired_rounds_ge_50_pct"),
        "escape_probability_pct": row.get("escape_probability_pct"),
        "p_total_pnl_positive_pct": row.get("p_total_pnl_positive_pct"),
        "expected_total_pnl_usdt": row.get("expected_total_pnl_usdt"),
        "standard_risk_eligible": row.get("standard_risk_eligible"),
        "execution_feasibility": row.get("execution_feasibility"),
    }


def _plateau_selector(rows: list[dict], calibration: dict) -> dict:
    eligible = [
        r for r in rows
        if r.get("standard_risk_eligible") is True
        and (r.get("execution_feasibility") or {}).get("status")
            != "FAIL_KNOWN_CONSTRAINT"
    ]
    if not eligible:
        return {
            "status": "NO_ELIGIBLE_PAIRED_DENSITY_CANDIDATES",
            "operational_override": False,
        }

    expected_field = (
        "expected_paired_grid_profit_usdt_calibrated"
        if calibration.get("calibration_ready") is True
        else "expected_paired_grid_profit_usdt_raw"
    )
    median_field = (
        "median_paired_grid_profit_usdt_calibrated"
        if calibration.get("calibration_ready") is True
        else "median_paired_grid_profit_usdt_raw"
    )

    peak = max(
        eligible,
        key=lambda r: (
            float(r.get(expected_field) or 0.0),
            float(r.get(median_field) or 0.0),
        ),
    )
    plateau = _contiguous_plateau(
        eligible,
        expected_field,
        peak,
        PLATEAU_EXPECTED_PROFIT_FLOOR_RATIO,
    )

    # Robust selector: within the contiguous near-peak expected-profit plateau,
    # prefer the strongest median profit. Then prefer fewer dead paths, stronger
    # activity, and finally larger order notional as an execution-robustness tie
    # breaker.
    selected = max(
        plateau,
        key=lambda r: (
            float(r.get(median_field) or 0.0),
            -float(r.get("p_zero_paired_rounds_pct") or 0.0),
            float(r.get("expected_paired_rounds_raw") or 0.0),
            float(r.get("avg_order_notional_usdt_est") or 0.0),
        ),
    )

    by_grid = {int(r["grids"]): r for r in rows}
    current = by_grid.get(22)
    if current is None:
        # Use exact live grid count when it differs from the historical 22-grid
        # convention.
        try:
            current_grids = int(
                json.loads(OUT_PATH.read_text(encoding="utf-8"))
                .get("current_benchmark", {})
                .get("grids")
            )
            current = by_grid.get(current_grids)
        except Exception:
            current = None

    expected_gain = _pct_gain(
        selected.get(expected_field),
        current.get(expected_field) if current else None,
    )
    median_gain = _pct_gain(
        selected.get(median_field),
        current.get(median_field) if current else None,
    )

    selected_feas = selected.get("execution_feasibility") or {}
    platform_validated = (
        selected_feas.get("status") == "PASS_EXACT_PIONEX_UI_VALIDATION"
    )

    research_candidate = (
        calibration.get("calibration_ready") is True
        and expected_gain is not None
        and expected_gain >= MIN_PAIRED_EXPECTED_GAIN_VS_CURRENT_PCT
        and median_gain is not None
        and median_gain >= MIN_PAIRED_MEDIAN_GAIN_VS_CURRENT_PCT
    )

    return {
        "status": (
            "CALIBRATED_RESEARCH_CANDIDATE"
            if research_candidate
            else "RESEARCH_ONLY_NOT_PROMOTION_READY"
        ),
        "selection_policy": {
            "expected_profit_plateau_floor_ratio": PLATEAU_EXPECTED_PROFIT_FLOOR_RATIO,
            "plateau_definition": (
                "Contiguous integer grid-count region containing the paired "
                "expected-profit champion where every candidate retains at least "
                "95% of peak expected paired profit."
            ),
            "robust_choice_rule": (
                "Within that plateau, maximise median paired grid profit; "
                "tie-break on lower zero-cycle probability, then higher paired "
                "activity, then larger estimated order notional."
            ),
        },
        "score_fields": {
            "expected": expected_field,
            "median": median_field,
        },
        "expected_profit_champion": _compact_candidate(peak),
        "plateau_grid_min": min(int(r["grids"]) for r in plateau),
        "plateau_grid_max": max(int(r["grids"]) for r in plateau),
        "plateau_candidate_count": len(plateau),
        "plateau_grid_counts": [int(r["grids"]) for r in plateau],
        "robust_plateau_selection": _compact_candidate(selected),
        "current_band_live_density": _compact_candidate(current),
        "selected_expected_profit_gain_vs_current_pct": expected_gain,
        "selected_median_profit_gain_vs_current_pct": median_gain,
        "paired_calibration_ready": calibration.get("calibration_ready"),
        "exact_pionex_candidate_validation": platform_validated,
        "promotion_gate": (
            "EXACT_PIONEX_UI_VALIDATION_PASSED"
            if platform_validated
            else "AWAITING_EXACT_PIONEX_LIVE_SETUP_VALIDATION"
        ),
        "operational_override": False,
    }


def _annotate_v44() -> None:
    if not OUT_PATH.is_file():
        raise SystemExit("v4.3 density output missing before v4.4 annotation")

    density = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    if density.get("schema") != "pionex_grid_density_sweep_v1":
        raise SystemExit(f"Unexpected density schema: {density.get('schema')}")

    calibration = _paired_walk_forward_calibration()
    constraints = _load_constraints()

    paired_rows = list(density.get("paired_cycle_current_band_integer_sweep") or [])
    if not paired_rows:
        raise SystemExit("v4.3 paired-cycle sweep missing")

    _apply_paired_calibration(paired_rows, calibration, constraints)
    plateau = _plateau_selector(paired_rows, calibration)

    density["version"] = VERSION
    density["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    density["paired_cycle_current_band_integer_sweep"] = paired_rows

    method = density.get("method") or {}
    method["paired_specific_calibration"] = (
        "v4.4 reconstructs the corrected paired model on historical manual "
        "Pionex state windows, estimates scaling on earlier windows and reports "
        "chronologically later holdout validation."
    )
    method["density_selection"] = (
        "v4.4 uses a robust 95%-of-peak expected-profit plateau selector rather "
        "than a single-grid argmax."
    )
    method["platform_feasibility"] = (
        "No universal Pionex Spot Grid minimum is invented. Known configured "
        "constraints are checked offline; exact candidate promotion still "
        "requires validation in the live Pionex setup/edit interface."
    )
    density["method"] = method

    correction = density.get("paired_cycle_correction") or {}
    correction["paired_calibration_warning"] = None
    correction["paired_specific_calibration"] = {
        k: v for k, v in calibration.items() if k != "windows"
    }
    correction["calibration_windows"] = calibration.get("windows")
    density["paired_cycle_correction"] = correction

    density["promotion_readiness"] = {
        "schema": "pionex_paired_density_promotion_v1",
        "version": VERSION,
        "paired_calibration": calibration,
        "execution_constraints": constraints,
        "plateau_selector": plateau,
        "cross_run_stability": {
            "status": "POPULATED_BY_FULL_RUNNER",
            "stable": False,
        },
        "paired_joint_geometry_status": (
            "NOT_YET_PROMOTED: v4.4 promotion-readiness applies to density on "
            "the current live band only. Centre/width remain a separate geometry "
            "problem until the corrected paired model is extended to the joint "
            "geometry search."
        ),
        "operational_effect": "NONE_DIRECTLY",
    }

    OUT_PATH.write_text(json.dumps(density, indent=2) + "\n", encoding="utf-8")

    selected = (
        (density.get("promotion_readiness") or {})
        .get("plateau_selector", {})
        .get("robust_plateau_selection")
        or {}
    )
    print("\n=== PAIRED DENSITY PROMOTION-READINESS v4.4 ===")
    print("Paired calibration:", calibration.get("status"))
    print("Calibration windows:", calibration.get("evaluated_windows"))
    print("Holdout:", calibration.get("holdout_validation"))
    print("Robust plateau selection:", selected.get("grids"), "grids")
    print(
        "Plateau:",
        plateau.get("plateau_grid_min"),
        "to",
        plateau.get("plateau_grid_max"),
    )
    print("Promotion gate:", plateau.get("promotion_gate"))
    print("Operational override: False")


def main() -> None:
    # Run the complete validated v4.3 layer first.
    v4.main()
    # Then add paired-specific calibration / plateau promotion-readiness.
    _annotate_v44()


if __name__ == "__main__":
    main()
