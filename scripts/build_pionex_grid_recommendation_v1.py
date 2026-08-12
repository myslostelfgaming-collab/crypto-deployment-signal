#!/usr/bin/env python3

"""Phase 4C: prospective calibration + continuous-ish ETH/USDT Pionex band optimizer.

This remains diagnostic / decision-support only. It never connects to Pionex and never
places or edits orders.
"""

import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import build_pionex_grid_simulator_v1 as sim

PROFILE_PATH = os.path.join("data", "pionex", "pionex_grid_profile_v1.json")
STATE_PATH = os.path.join("data", "pionex", "manual_grid_states_v1.csv")
LEDGER_PATH = os.path.join("data", "pionex", "grid_recommendations_v1.csv")
OUT_RECOMMENDATION = os.path.join("data", "diagnostics", "pionex_grid_recommendation_v1.json")
OUT_CALIBRATION = os.path.join("data", "diagnostics", "pionex_grid_calibration_v1.json")

ATR_ANALOG_N = 60
RISK_CAP_ESCAPE_PCT = 10.0
COARSE_STEP_USDT = 5.0
REFINE_STEP_USDT = 0.5
REFINE_RADIUS_USDT = 6.0
TOP_COARSE_TO_REFINE = 4
MIN_GAIN_TO_MOVE_USDT = 0.02
MIN_RISK_REDUCTION_TO_MOVE_PP = 2.5
PRACTICAL_ROUND_USDT = 5.0
MIN_CALIBRATION_WINDOWS = 5
CALIBRATION_INTERVAL_MIN_H = 18.0
CALIBRATION_INTERVAL_MAX_H = 30.0

LEDGER_FIELDS = [
    "generated_at_utc",
    "feature_ts_utc",
    "state_captured_at_utc",
    "market_price_usdt",
    "state_lower_usdt",
    "state_upper_usdt",
    "state_grids",
    "state_quantity_per_grid_eth",
    "state_grid_profit_usdt",
    "state_rounds_total",
    "current_atr14_pct",
    "calibration_profit_scale",
    "calibration_rounds_scale",
    "current_expected_profit_usdt",
    "current_expected_rounds",
    "current_escape_probability_pct",
    "raw_optimal_shift_usdt",
    "raw_optimal_lower_usdt",
    "raw_optimal_upper_usdt",
    "raw_optimal_expected_profit_usdt",
    "raw_optimal_escape_probability_pct",
    "recommended_action",
    "recommended_shift_usdt",
    "practical_shift_usdt",
    "recommended_lower_usdt",
    "recommended_upper_usdt",
    "recommended_expected_profit_usdt",
    "recommended_expected_rounds",
    "recommended_escape_probability_pct",
    "recommended_p_profit_ge_0_25_pct",
    "recommended_p_profit_ge_0_50_pct",
    "recommendation_status",
]


def f(v, default: Optional[float] = None) -> Optional[float]:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except Exception:
        return default


def i(v, default: Optional[int] = None) -> Optional[int]:
    try:
        if v in (None, ""):
            return default
        return int(float(v))
    except Exception:
        return default


def parse_iso(v: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def read_manual_states() -> List[dict]:
    if not os.path.isfile(STATE_PATH):
        return []
    rows = []
    with open(STATE_PATH, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            dt = parse_iso(r.get("captured_at_utc") or "")
            if dt is None:
                continue
            if (r.get("pair") or "").upper() not in {"ETH/USDT", "ETH-USDT"}:
                continue
            rows.append({**r, "_dt": dt})
    rows.sort(key=lambda r: r["_dt"])
    return rows


def read_ledger() -> List[dict]:
    if not os.path.isfile(LEDGER_PATH):
        return []
    with open(LEDGER_PATH, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def same_grid(a: dict, b: dict, tol: float = 1e-6) -> bool:
    pairs = [
        (f(a.get("state_lower_usdt")), f(b.get("lower_limit_usdt"))),
        (f(a.get("state_upper_usdt")), f(b.get("upper_limit_usdt"))),
        (f(a.get("state_quantity_per_grid_eth")), f(b.get("quantity_per_grid_eth"))),
    ]
    if i(a.get("state_grids")) != i(b.get("grids")):
        return False
    for x, y in pairs:
        if x is None or y is None or abs(x - y) > tol:
            return False
    return True


def build_calibration(states: List[dict], ledger: List[dict]) -> dict:
    windows = []
    for rec in ledger:
        start_dt = parse_iso(rec.get("state_captured_at_utc") or "")
        if start_dt is None:
            continue
        pred_profit = f(rec.get("current_expected_profit_usdt"))
        pred_rounds = f(rec.get("current_expected_rounds"))
        start_profit = f(rec.get("state_grid_profit_usdt"))
        start_rounds = f(rec.get("state_rounds_total"))
        if None in (pred_profit, pred_rounds, start_profit, start_rounds):
            continue
        eligible = []
        for st in states:
            end_dt = st["_dt"]
            elapsed_h = (end_dt - start_dt).total_seconds() / 3600.0
            if elapsed_h < CALIBRATION_INTERVAL_MIN_H or elapsed_h > CALIBRATION_INTERVAL_MAX_H:
                continue
            if not same_grid(rec, st):
                continue
            end_profit = f(st.get("grid_profit_usdt"))
            end_rounds = f(st.get("rounds_total"))
            if end_profit is None or end_rounds is None:
                continue
            eligible.append((abs(elapsed_h - 24.0), st, elapsed_h, end_profit, end_rounds))
        if not eligible:
            continue
        _, st, elapsed_h, end_profit, end_rounds = sorted(eligible, key=lambda x: x[0])[0]
        actual_profit_24 = (end_profit - start_profit) * 24.0 / elapsed_h
        actual_rounds_24 = (end_rounds - start_rounds) * 24.0 / elapsed_h
        if actual_profit_24 < -1e-9 or actual_rounds_24 < -1e-9:
            continue
        profit_ratio = actual_profit_24 / pred_profit if pred_profit and pred_profit > 0 else None
        rounds_ratio = actual_rounds_24 / pred_rounds if pred_rounds and pred_rounds > 0 else None
        windows.append({
            "start_utc": start_dt.isoformat(),
            "end_utc": st["_dt"].isoformat(),
            "elapsed_h": round(elapsed_h, 4),
            "predicted_profit_24h_usdt": round(pred_profit, 6),
            "actual_profit_24h_equiv_usdt": round(actual_profit_24, 6),
            "predicted_rounds_24h": round(pred_rounds, 6),
            "actual_rounds_24h_equiv": round(actual_rounds_24, 6),
            "profit_ratio": round(profit_ratio, 6) if profit_ratio is not None else None,
            "rounds_ratio": round(rounds_ratio, 6) if rounds_ratio is not None else None,
        })

    # Dedupe windows by start timestamp because reruns can append the same market state.
    dedup = {}
    for w in windows:
        dedup[w["start_utc"]] = w
    windows = [dedup[k] for k in sorted(dedup)]

    profit_ratios = [w["profit_ratio"] for w in windows if w["profit_ratio"] is not None and 0 < w["profit_ratio"] < 5]
    round_ratios = [w["rounds_ratio"] for w in windows if w["rounds_ratio"] is not None and 0 < w["rounds_ratio"] < 5]
    enough = len(windows) >= MIN_CALIBRATION_WINDOWS
    profit_scale = statistics.median(profit_ratios) if enough and profit_ratios else 1.0
    rounds_scale = statistics.median(round_ratios) if enough and round_ratios else 1.0
    # Avoid one early odd window blowing up the decision engine.
    profit_scale = min(1.75, max(0.50, profit_scale))
    rounds_scale = min(1.75, max(0.50, rounds_scale))

    return {
        "schema": "pionex_grid_calibration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_ACTIVE" if enough else "EARLY_NO_CALIBRATION_APPLIED",
        "minimum_windows_before_apply": MIN_CALIBRATION_WINDOWS,
        "evaluated_windows": len(windows),
        "profit_scale_applied": round(profit_scale, 6),
        "rounds_scale_applied": round(rounds_scale, 6),
        "method": "Median actual/predicted 24h-equivalent ratio over 18-30h windows with unchanged grid geometry; clipped to [0.50, 1.75].",
        "windows": windows[-30:],
    }


def calibrated_cases(cases: List[dict], profit_scale: float, rounds_scale: float) -> List[dict]:
    out = []
    for c in cases:
        x = dict(c)
        for key in ("profit_conservative", "profit_mid", "profit_optimistic"):
            x[key] = x[key] * profit_scale
        for key in ("rounds_conservative", "rounds_mid", "rounds_optimistic"):
            x[key] = x[key] * rounds_scale
        out.append(x)
    return out


def model_profile(profile_raw: dict, state: dict, market_price: float) -> dict:
    eth = f(state.get("eth_holdings"), 0.0) or 0.0
    usdt = f(state.get("usdt_holdings"), 0.0) or 0.0
    equity = eth * market_price + usdt
    return {
        "current_equity_usdt": equity,
        "original_investment_usdt": f(state.get("investment_usdt"), 0.0) or 0.0,
        "dollar_thresholds": [float(x) for x in profile_raw["objective"]["report_dollar_thresholds_usdt_24h"]],
        "return_thresholds": [float(x) for x in profile_raw["objective"]["report_thresholds_return_pct_24h"]],
    }


def make_ref(state: dict, market_price: float) -> dict:
    return {
        "lower_limit_usdt": f(state.get("lower_limit_usdt")),
        "upper_limit_usdt": f(state.get("upper_limit_usdt")),
        "current_price_usdt": market_price,
        "grids": i(state.get("grids")),
        "quantity_per_grid_eth": f(state.get("quantity_per_grid_eth")),
    }


def feasible_shift_bounds(ref: dict) -> Tuple[float, float]:
    # Require current market price to remain inside the shifted band.
    current = float(ref["current_price_usdt"])
    lower = float(ref["lower_limit_usdt"])
    upper = float(ref["upper_limit_usdt"])
    return current - upper, current - lower


def frange(lo: float, hi: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step")
    start = math.ceil(lo / step) * step
    vals = []
    x = start
    while x <= hi + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


def candidate_key(c: dict) -> float:
    return round(float(c["shift_usdt"]), 6)


def evaluate_shift(shift: float, ref: dict, fee_rate: float, analogs: List[dict], master: Dict[int, List[float]],
                   profile: dict, profit_scale: float, rounds_scale: float) -> Optional[dict]:
    cfg = sim.make_cfg(ref, fee_rate, shift)
    if cfg is None:
        return None
    cases = [sim.simulate_historical_case(r, master, cfg) for r in analogs]
    cases = [x for x in cases if x is not None]
    cases = calibrated_cases(cases, profit_scale, rounds_scale)
    if not cases:
        return None
    s = sim.candidate_summary(cases, cfg, profile)
    expected_profit = s["grid_profit_usdt"]["midpoint"].get("mean")
    median_profit = s["grid_profit_usdt"]["midpoint"].get("median")
    expected_rounds = s["rounds_midpoint"].get("mean")
    return {
        "shift_usdt": round(shift, 6),
        "lower_usdt": s["configuration"]["lower_usdt"],
        "upper_usdt": s["configuration"]["upper_usdt"],
        "expected_grid_profit_usdt": expected_profit,
        "median_grid_profit_usdt": median_profit,
        "expected_rounds": expected_rounds,
        "escape_probability_pct": s["range_escape_probability_pct"],
        "lower_escape_probability_pct": s["lower_escape_probability_pct"],
        "upper_escape_probability_pct": s["upper_escape_probability_pct"],
        "p_profit_ge_0_25_pct": ((s["probability_grid_profit_ge_dollar_threshold"].get("0.25") or {}).get("midpoint_pct")),
        "p_profit_ge_0_50_pct": ((s["probability_grid_profit_ge_dollar_threshold"].get("0.5") or {}).get("midpoint_pct")),
        "full_summary": s,
    }


def pareto_frontier(candidates: List[dict]) -> List[dict]:
    out = []
    for c in candidates:
        dominated = False
        for d in candidates:
            if d is c:
                continue
            p_c = c["expected_grid_profit_usdt"] or -1e9
            p_d = d["expected_grid_profit_usdt"] or -1e9
            r_c = c["escape_probability_pct"] if c["escape_probability_pct"] is not None else 1e9
            r_d = d["escape_probability_pct"] if d["escape_probability_pct"] is not None else 1e9
            if p_d >= p_c and r_d <= r_c and (p_d > p_c + 1e-12 or r_d < r_c - 1e-12):
                dominated = True
                break
        if not dominated:
            out.append(c)
    return sorted(out, key=lambda x: (x["escape_probability_pct"], -(x["expected_grid_profit_usdt"] or 0)))


def slim(c: Optional[dict]) -> Optional[dict]:
    if c is None:
        return None
    return {k: v for k, v in c.items() if k != "full_summary"}


def nearest_candidate(candidates: List[dict], shift: float) -> dict:
    return min(candidates, key=lambda c: abs(c["shift_usdt"] - shift))


def select_recommendation(candidates: List[dict]) -> dict:
    base = nearest_candidate(candidates, 0.0)
    profit_max = max(candidates, key=lambda x: (x["expected_grid_profit_usdt"] or -1e9, -(x["escape_probability_pct"] or 1e9)))
    base_risk = base["escape_probability_pct"] if base["escape_probability_pct"] is not None else RISK_CAP_ESCAPE_PCT
    # Never recommend *more* escape risk than the current band, and if the current
    # band is already risky, force the optimizer back under the absolute ceiling.
    effective_risk_cap = min(RISK_CAP_ESCAPE_PCT, base_risk)
    eligible = [c for c in candidates if c["escape_probability_pct"] is not None and c["escape_probability_pct"] <= effective_risk_cap + 1e-9]
    risk_opt = max(eligible, key=lambda x: (x["expected_grid_profit_usdt"] or -1e9, -(x["escape_probability_pct"] or 1e9), -abs(x["shift_usdt"]))) if eligible else base
    risk_min = min(candidates, key=lambda x: (x["escape_probability_pct"] if x["escape_probability_pct"] is not None else 1e9, -(x["expected_grid_profit_usdt"] or 0)))

    gain = (risk_opt["expected_grid_profit_usdt"] or 0) - (base["expected_grid_profit_usdt"] or 0)
    risk_change = (risk_opt["escape_probability_pct"] or 0) - (base["escape_probability_pct"] or 0)
    material = gain >= MIN_GAIN_TO_MOVE_USDT or risk_change <= -MIN_RISK_REDUCTION_TO_MOVE_PP
    if abs(risk_opt["shift_usdt"]) < REFINE_STEP_USDT / 2 or not material:
        action = "KEEP_CURRENT"
        chosen = base
    else:
        action = "SHIFT_UP" if risk_opt["shift_usdt"] > 0 else "SHIFT_DOWN"
        chosen = risk_opt

    practical_shift = round(chosen["shift_usdt"] / PRACTICAL_ROUND_USDT) * PRACTICAL_ROUND_USDT
    practical = nearest_candidate(candidates, practical_shift)
    return {
        "base_current": base,
        "raw_profit_max": profit_max,
        "risk_constrained_optimum": risk_opt,
        "minimum_escape_candidate": risk_min,
        "selected": chosen,
        "selected_action": action,
        "practical": practical,
        "expected_gain_vs_current_usdt": round((chosen["expected_grid_profit_usdt"] or 0) - (base["expected_grid_profit_usdt"] or 0), 6),
        "escape_risk_change_vs_current_pp": round((chosen["escape_probability_pct"] or 0) - (base["escape_probability_pct"] or 0), 6),
        "effective_escape_risk_cap_pct": round(effective_risk_cap, 6),
        "material_move_rule": {
            "minimum_expected_profit_gain_usdt": MIN_GAIN_TO_MOVE_USDT,
            "or_minimum_escape_risk_reduction_pp": MIN_RISK_REDUCTION_TO_MOVE_PP,
        },
    }


def append_ledger(row: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    existing = read_ledger()
    # Idempotent reruns for the same market feature and source state.
    key = (str(row.get("feature_ts_utc")), str(row.get("state_captured_at_utc")))
    kept = [r for r in existing if (str(r.get("feature_ts_utc")), str(r.get("state_captured_at_utc"))) != key]
    kept.append({k: row.get(k, "") for k in LEDGER_FIELDS})
    kept.sort(key=lambda r: (str(r.get("feature_ts_utc", "")), str(r.get("state_captured_at_utc", ""))))
    with open(LEDGER_PATH, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(kept)


def main() -> None:
    if not os.path.isfile(PROFILE_PATH):
        raise SystemExit(f"Missing {PROFILE_PATH}")
    states = read_manual_states()
    if not states:
        raise SystemExit(f"No manual Pionex states in {STATE_PATH}")
    profile_raw = sim.load_json(PROFILE_PATH)
    latest_state = states[-1]

    master = sim.build_master_candles()
    features = sim.load_eth_features()
    if not features:
        raise SystemExit("No ETH features")
    latest_feature = features[-1]
    market_price = float(latest_feature["entry_close"])
    current_atr = float(latest_feature["atr14_pct"])

    matured = [r for r in features if sim.forward_24h(master, r["ts"]) is not None]
    independent = sim.greedy_independent(matured)
    analogs = sorted(independent, key=lambda r: abs(math.log(r["atr14_pct"] / current_atr)))[: min(ATR_ANALOG_N, len(independent))]
    if len(analogs) < 30:
        raise SystemExit(f"Too few ATR analogues: {len(analogs)}")

    previous_ledger = read_ledger()
    calibration = build_calibration(states, previous_ledger)
    profit_scale = float(calibration["profit_scale_applied"])
    rounds_scale = float(calibration["rounds_scale_applied"])

    fee_pct = float(profile_raw.get("fee_model", {}).get("standard_public_spot_fee_pct_per_fill_reference", 0.05))
    fee_rate = fee_pct / 100.0
    ref = make_ref(latest_state, market_price)
    if None in (ref["lower_limit_usdt"], ref["upper_limit_usdt"], ref["grids"], ref["quantity_per_grid_eth"]):
        raise SystemExit("Latest Pionex state is missing required grid geometry")
    prof = model_profile(profile_raw, latest_state, market_price)

    lo_shift, hi_shift = feasible_shift_bounds(ref)
    # Keep a tiny buffer so floating-point equality at the edge does not dominate the optimizer.
    lo_search = lo_shift + 0.01
    hi_search = hi_shift - 0.01
    coarse_shifts = frange(lo_search, hi_search, COARSE_STEP_USDT)
    coarse_shifts.append(0.0)
    coarse = {}
    for shift in sorted(set(coarse_shifts)):
        c = evaluate_shift(shift, ref, fee_rate, analogs, master, prof, profit_scale, rounds_scale)
        if c:
            coarse[candidate_key(c)] = c

    # Refine around several coarse candidates so the answer is not constrained to $50 or even $5 increments.
    coarse_rank = sorted(coarse.values(), key=lambda x: (-(x["expected_grid_profit_usdt"] or -1e9), x["escape_probability_pct"] if x["escape_probability_pct"] is not None else 1e9))
    seeds = coarse_rank[:TOP_COARSE_TO_REFINE]
    risk_seeds = sorted(coarse.values(), key=lambda x: (x["escape_probability_pct"] if x["escape_probability_pct"] is not None else 1e9, -(x["expected_grid_profit_usdt"] or 0)))[:2]
    refine_shifts = {0.0}
    for seed in seeds + risk_seeds:
        a = max(lo_search, seed["shift_usdt"] - REFINE_RADIUS_USDT)
        b = min(hi_search, seed["shift_usdt"] + REFINE_RADIUS_USDT)
        refine_shifts.update(frange(a, b, REFINE_STEP_USDT))

    evaluated = dict(coarse)
    for shift in sorted(refine_shifts):
        k = round(shift, 6)
        if k in evaluated:
            continue
        c = evaluate_shift(shift, ref, fee_rate, analogs, master, prof, profit_scale, rounds_scale)
        if c:
            evaluated[candidate_key(c)] = c
    candidates = sorted(evaluated.values(), key=lambda x: x["shift_usdt"])

    selection = select_recommendation(candidates)
    frontier = pareto_frontier(candidates)

    # Robustness check for a small set of important candidates against all 24h-spaced history.
    important_shifts = {
        0.0,
        selection["raw_profit_max"]["shift_usdt"],
        selection["risk_constrained_optimum"]["shift_usdt"],
        selection["minimum_escape_candidate"]["shift_usdt"],
        selection["selected"]["shift_usdt"],
        selection["practical"]["shift_usdt"],
    }
    robustness = []
    for shift in sorted(important_shifts):
        cfg = sim.make_cfg(ref, fee_rate, shift)
        if cfg is None:
            continue
        cases = [sim.simulate_historical_case(r, master, cfg) for r in independent]
        cases = [x for x in cases if x is not None]
        cases = calibrated_cases(cases, profit_scale, rounds_scale)
        s = sim.candidate_summary(cases, cfg, prof)
        robustness.append({
            "shift_usdt": shift,
            "expected_profit_usdt_all_history": s["grid_profit_usdt"]["midpoint"].get("mean"),
            "escape_probability_pct_all_history": s["range_escape_probability_pct"],
            "sample_n": s["sample_n"],
        })

    chosen = selection["selected"]
    practical = selection["practical"]
    status = "PROSPECTIVE_DIAGNOSTIC_ONLY"
    if calibration["evaluated_windows"] >= MIN_CALIBRATION_WINDOWS:
        status = "PROSPECTIVE_CALIBRATED_DIAGNOSTIC"

    out = {
        "schema": "pionex_grid_recommendation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": {"platform": "Pionex", "pair": "ETH/USDT", "horizon_h": 24, "bot_type": "Spot Grid"},
        "source_state": {
            "captured_at_utc": latest_state.get("captured_at_utc"),
            "captured_at_local": latest_state.get("captured_at_local"),
            "state_price_usdt": f(latest_state.get("current_price_usdt")),
            "latest_market_price_usdt": market_price,
            "lower_limit_usdt": f(latest_state.get("lower_limit_usdt")),
            "upper_limit_usdt": f(latest_state.get("upper_limit_usdt")),
            "width_usdt": round((f(latest_state.get("upper_limit_usdt")) or 0) - (f(latest_state.get("lower_limit_usdt")) or 0), 6),
            "grids": i(latest_state.get("grids")),
            "quantity_per_grid_eth": f(latest_state.get("quantity_per_grid_eth")),
        },
        "market_regime": {
            "feature_ts_utc": latest_feature["ts"],
            "current_atr14_pct": current_atr,
            "atr_analogue_rows": len(analogs),
            "independent_history_rows": len(independent),
        },
        "calibration": {k: v for k, v in calibration.items() if k != "windows"},
        "optimizer": {
            "shift_both_bounds_together": True,
            "width_preserved": True,
            "grid_count_preserved": True,
            "coarse_step_usdt": COARSE_STEP_USDT,
            "refine_step_usdt": REFINE_STEP_USDT,
            "feasible_shift_min_usdt": round(lo_shift, 6),
            "feasible_shift_max_usdt": round(hi_shift, 6),
            "evaluated_candidate_count": len(candidates),
            "absolute_escape_risk_cap_pct": RISK_CAP_ESCAPE_PCT,
            "effective_escape_risk_cap_pct": selection["effective_escape_risk_cap_pct"],
            "precision_note": "Raw optimum is searched to $0.50 resolution around promising regions. A practical $5-rounded alternative is also reported because sub-dollar precision is not yet statistically decision-grade.",
            "objective_note": "Primary selection maximises calibrated expected 24h grid profit subject to the escape-risk cap and a material-improvement rule. Pareto frontier is reported rather than pretending profit/risk is a single objectively weighted score.",
        },
        "decision": {
            "action": selection["selected_action"],
            "raw_selected_shift_usdt": chosen["shift_usdt"],
            "raw_selected_lower_usdt": chosen["lower_usdt"],
            "raw_selected_upper_usdt": chosen["upper_usdt"],
            "practical_shift_usdt": practical["shift_usdt"],
            "practical_lower_usdt": practical["lower_usdt"],
            "practical_upper_usdt": practical["upper_usdt"],
            "expected_gain_vs_current_usdt": selection["expected_gain_vs_current_usdt"],
            "escape_risk_change_vs_current_pp": selection["escape_risk_change_vs_current_pp"],
            "warning": "Decision support only. Inventory/trend P&L risk is not yet included in the optimization objective; Phase 4D will add it.",
        },
        "benchmarks": {
            "current": slim(selection["base_current"]),
            "raw_profit_max": slim(selection["raw_profit_max"]),
            "risk_constrained_optimum": slim(selection["risk_constrained_optimum"]),
            "minimum_escape_candidate": slim(selection["minimum_escape_candidate"]),
            "selected": slim(selection["selected"]),
            "practical": slim(selection["practical"]),
        },
        "pareto_frontier": [slim(c) for c in frontier],
        "all_history_robustness": robustness,
        "prospective_calibration_windows": calibration["windows"],
        "next_phase": {
            "phase": "4D",
            "name": "Inventory/trend-risk + daily decision head",
            "requirements": [
                "continue appending Pionex screenshots/states, ideally near the same time daily and immediately after any configuration change",
                "add trend/inventory P&L risk to the objective rather than optimizing grid profit alone",
                "validate whether ATR-only analogue selection should be augmented with path-shape/trend features",
            ],
        },
    }

    os.makedirs(os.path.dirname(OUT_RECOMMENDATION), exist_ok=True)
    with open(OUT_RECOMMENDATION, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(OUT_CALIBRATION, "w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=2)

    ledger_row = {
        "generated_at_utc": out["generated_at_utc"],
        "feature_ts_utc": latest_feature["ts"],
        "state_captured_at_utc": latest_state.get("captured_at_utc") or "",
        "market_price_usdt": market_price,
        "state_lower_usdt": f(latest_state.get("lower_limit_usdt")),
        "state_upper_usdt": f(latest_state.get("upper_limit_usdt")),
        "state_grids": i(latest_state.get("grids")),
        "state_quantity_per_grid_eth": f(latest_state.get("quantity_per_grid_eth")),
        "state_grid_profit_usdt": f(latest_state.get("grid_profit_usdt")),
        "state_rounds_total": i(latest_state.get("rounds_total")),
        "current_atr14_pct": current_atr,
        "calibration_profit_scale": profit_scale,
        "calibration_rounds_scale": rounds_scale,
        "current_expected_profit_usdt": selection["base_current"]["expected_grid_profit_usdt"],
        "current_expected_rounds": selection["base_current"]["expected_rounds"],
        "current_escape_probability_pct": selection["base_current"]["escape_probability_pct"],
        "raw_optimal_shift_usdt": selection["risk_constrained_optimum"]["shift_usdt"],
        "raw_optimal_lower_usdt": selection["risk_constrained_optimum"]["lower_usdt"],
        "raw_optimal_upper_usdt": selection["risk_constrained_optimum"]["upper_usdt"],
        "raw_optimal_expected_profit_usdt": selection["risk_constrained_optimum"]["expected_grid_profit_usdt"],
        "raw_optimal_escape_probability_pct": selection["risk_constrained_optimum"]["escape_probability_pct"],
        "recommended_action": selection["selected_action"],
        "recommended_shift_usdt": chosen["shift_usdt"],
        "practical_shift_usdt": practical["shift_usdt"],
        "recommended_lower_usdt": chosen["lower_usdt"],
        "recommended_upper_usdt": chosen["upper_usdt"],
        "recommended_expected_profit_usdt": chosen["expected_grid_profit_usdt"],
        "recommended_expected_rounds": chosen["expected_rounds"],
        "recommended_escape_probability_pct": chosen["escape_probability_pct"],
        "recommended_p_profit_ge_0_25_pct": chosen["p_profit_ge_0_25_pct"],
        "recommended_p_profit_ge_0_50_pct": chosen["p_profit_ge_0_50_pct"],
        "recommendation_status": status,
    }
    append_ledger(ledger_row)

    print("Phase 4C recommendation written:", OUT_RECOMMENDATION)
    print("Calibration:", calibration["status"], "windows=", calibration["evaluated_windows"], "profit_scale=", profit_scale, "rounds_scale=", rounds_scale)
    print("Market price:", market_price, "ATR14%:", current_atr, "analogues:", len(analogs))
    print("Candidates evaluated:", len(candidates), "Pareto points:", len(frontier))
    print("Decision:", out["decision"])
    print("Selected metrics:", slim(chosen))


if __name__ == "__main__":
    main()
