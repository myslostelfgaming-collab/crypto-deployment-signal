#!/usr/bin/env python3
"""Phase 4D: ETH/USDT Pionex grid-geometry optimizer.

Diagnostic / decision-support only. Never connects to Pionex or places orders.

This phase extends 4C from centre-shift optimisation to a joint search over:
  * band centre,
  * band width,
  * grid count.

It also augments ATR conditioning with short-horizon path-shape / choppiness
features and reports a simple inventory-aware total-P&L simulation using the
latest manually captured Pionex balances.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import build_pionex_grid_simulator_v1 as sim
import build_pionex_grid_recommendation_v1 as phase4c

PROFILE_PATH = os.path.join("data", "pionex", "pionex_grid_profile_v1.json")
STATE_PATH = os.path.join("data", "pionex", "manual_grid_states_v1.csv")
OUT_PATH = os.path.join("data", "diagnostics", "pionex_grid_geometry_optimizer_v1.json")
CAL_PATH = os.path.join("data", "diagnostics", "pionex_grid_calibration_v2.json")
LEDGER_PATH = os.path.join("data", "pionex", "grid_geometry_recommendations_v1.csv")

# Analogue selection: preserve the proven ATR signal, then refine on path shape.
ATR_STAGE1_N = 120
FINAL_ANALOG_N = 60
TRAILING_STATE_H = 24

# Geometry search. Width is expressed as a fraction of current market price so
# the optimizer remains usable as ETH changes price over time.
MIN_WIDTH_PCT = 6.0
MAX_WIDTH_PCT = 19.0
COARSE_WIDTH_STEP_PCT = 2.0
REFINE_WIDTH_STEP_PCT = 0.5
COARSE_CENTER_STEP_USDT = 20.0
REFINE_CENTER_STEP_USDT = 2.5
CENTER_SEARCH_PCT = 4.0
COARSE_GRID_COUNTS = tuple(range(10, 81, 5))
REFINE_GRID_RADIUS = 2
TOP_COARSE_TO_REFINE = 4

# Net profit/grid must remain meaningfully above round-trip fees. This is a
# modelling safety margin, not a claim about Pionex's live minimum.
MIN_NET_PROFIT_GRID_PCT_FLOOR = 0.12
FEE_BUFFER_PP = 0.00

# Risk / action gates.
ABS_ESCAPE_RISK_CAP_PCT = 10.0
MAX_POSITIVE_PNL_PROB_DROP_PP = 5.0
MAX_P20_TOTAL_PNL_WORSEN_USDT = 0.50
MIN_EXPECTED_GRID_PROFIT_GAIN_USDT = 0.02
MIN_ESCAPE_RISK_REDUCTION_PP = 2.5
PRACTICAL_PRICE_ROUND_USDT = 5.0

# Calibration v2 reconstructs historical predictions at each manual state using
# only target-matured historical paths available at that time.
CAL_MIN_H = 14.0
CAL_MAX_H = 30.0
CAL_MIN_WINDOWS_TO_APPLY = 3
CAL_SCALE_LO = 0.35
CAL_SCALE_HI = 1.50

LEDGER_FIELDS = [
    "generated_at_utc", "feature_ts_utc", "state_captured_at_utc",
    "market_price_usdt", "state_age_h", "current_atr14_pct",
    "current_choppiness_efficiency", "current_reversal_rate",
    "current_path_length_pct", "calibration_windows",
    "profit_scale_applied", "rounds_scale_applied",
    "current_lower_usdt", "current_upper_usdt", "current_width_usdt",
    "current_grids", "current_qty_eth", "current_expected_profit_usdt",
    "current_expected_rounds", "current_escape_probability_pct",
    "recommended_action", "recommended_lower_usdt", "recommended_upper_usdt",
    "recommended_width_usdt", "recommended_grids", "recommended_qty_eth",
    "recommended_expected_profit_usdt", "recommended_expected_rounds",
    "recommended_escape_probability_pct", "recommended_p_total_pnl_positive_pct",
    "recommended_p20_total_pnl_usdt", "practical_lower_usdt",
    "practical_upper_usdt", "practical_grids", "confidence", "status",
]


def f(v, default: Optional[float] = None) -> Optional[float]:
    try:
        if v in (None, ""):
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def i(v, default: Optional[int] = None) -> Optional[int]:
    try:
        if v in (None, ""):
            return default
        return int(float(v))
    except Exception:
        return default


def pct_return(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return (b / a - 1.0) * 100.0


def trailing_candles(master: Dict[int, List[float]], ts: int, hours: int = TRAILING_STATE_H) -> Optional[List[List[float]]]:
    out = []
    for k in range(hours - 1, -1, -1):
        c = master.get(ts - 3600 * k)
        if c is None:
            return None
        out.append(c)
    return out


def path_shape_metrics(master: Dict[int, List[float]], row: dict) -> Optional[dict]:
    w = trailing_candles(master, int(row["ts"]))
    if not w or len(w) < 4:
        return None
    closes = [float(c[4]) for c in w]
    highs = [float(c[2]) for c in w]
    lows = [float(c[3]) for c in w]
    rets = []
    signs = []
    for a, b in zip(closes, closes[1:]):
        r = pct_return(a, b)
        rets.append(r)
        if r > 1e-12:
            signs.append(1)
        elif r < -1e-12:
            signs.append(-1)
        else:
            signs.append(0)
    path_length = sum(abs(x) for x in rets)
    signed_move = pct_return(closes[0], closes[-1])
    efficiency = abs(signed_move) / path_length if path_length > 1e-12 else 0.0
    nz = [x for x in signs if x != 0]
    sign_changes = sum(1 for a, b in zip(nz, nz[1:]) if a != b)
    reversal_rate = sign_changes / max(1, len(nz) - 1)
    entry = float(row.get("entry_close") or closes[-1])
    range_pct = (max(highs) - min(lows)) / entry * 100.0 if entry > 0 else 0.0
    return {
        "path_length_pct": path_length,
        "signed_move_pct": signed_move,
        "efficiency_ratio": min(1.0, max(0.0, efficiency)),
        "reversal_rate": min(1.0, max(0.0, reversal_rate)),
        "range_pct": range_pct,
    }


def median_abs_dev(vals: List[float]) -> float:
    if not vals:
        return 1.0
    med = statistics.median(vals)
    mad = statistics.median(abs(v - med) for v in vals)
    return max(1e-6, 1.4826 * mad)


def enrich_rows(rows: Iterable[dict], master: Dict[int, List[float]]) -> List[dict]:
    out = []
    for r in rows:
        m = path_shape_metrics(master, r)
        if m is None:
            continue
        out.append({**r, "shape": m})
    return out


def select_analogs(independent: List[dict], current_row: dict, master: Dict[int, List[float]], n: int = FINAL_ANALOG_N) -> Tuple[List[dict], dict]:
    current_shape = path_shape_metrics(master, current_row)
    if current_shape is None:
        raise SystemExit("Current market state lacks a complete 24h trailing path")
    enriched = enrich_rows(independent, master)
    if len(enriched) < 30:
        raise SystemExit(f"Too few path-shape rows: {len(enriched)}")
    cur_atr = float(current_row["atr14_pct"])
    stage1 = sorted(enriched, key=lambda r: abs(math.log(float(r["atr14_pct"]) / cur_atr)))[: min(ATR_STAGE1_N, len(enriched))]
    metrics = ["efficiency_ratio", "reversal_rate", "path_length_pct", "signed_move_pct"]
    scales = {k: median_abs_dev([float(r["shape"][k]) for r in stage1]) for k in metrics}

    def shape_distance(r: dict) -> float:
        s = r["shape"]
        return math.sqrt(sum(((float(s[k]) - float(current_shape[k])) / scales[k]) ** 2 for k in metrics))

    ranked = sorted(stage1, key=lambda r: (shape_distance(r), abs(math.log(float(r["atr14_pct"]) / cur_atr))))
    chosen = ranked[: min(n, len(ranked))]
    meta = {
        "method": "Two-stage: nearest ATR14 states first, then robust-normalized Euclidean distance on 24h efficiency, reversal rate, path length and signed move.",
        "atr_stage1_n": len(stage1),
        "final_n": len(chosen),
        "current_shape": {k: round(float(v), 6) for k, v in current_shape.items()},
        "robust_scales": {k: round(float(v), 6) for k, v in scales.items()},
    }
    return chosen, meta


def percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    return sim.percentile(vals, p)


def initial_order_requirements(lines: List[float], start_price: float, fee_rate: float) -> Tuple[int, float, int, float]:
    states = sim.initial_states(lines, start_price)
    sell_intervals = [idx for idx, st in enumerate(states) if st == "sell"]
    buy_intervals = [idx for idx, st in enumerate(states) if st == "buy"]
    # Per unit ETH quantity, how much USDT is needed to seed all buy-side orders?
    buy_usdt_per_eth_qty = sum(lines[idx] * (1.0 + fee_rate) for idx in buy_intervals)
    return len(sell_intervals), float(len(sell_intervals)), len(buy_intervals), buy_usdt_per_eth_qty


def observed_utilization_factor(state: dict, market_price: float, fee_rate: float) -> float:
    lower = float(state["lower_limit_usdt"])
    upper = float(state["upper_limit_usdt"])
    grids = int(state["grids"])
    qty = float(state["quantity_per_grid_eth"])
    eth = float(state.get("eth_holdings") or 0.0)
    usdt = float(state.get("usdt_holdings") or 0.0)
    lines = sim.grid_lines(lower, upper, grids)
    sell_n, sell_per_qty, _, buy_per_qty = initial_order_requirements(lines, market_price, fee_rate)
    max_by_eth = eth / sell_per_qty if sell_n > 0 and sell_per_qty > 0 else float("inf")
    max_by_usdt = usdt / buy_per_qty if buy_per_qty > 0 else float("inf")
    max_uniform = min(max_by_eth, max_by_usdt)
    if not math.isfinite(max_uniform) or max_uniform <= 0:
        return 1.0
    return min(1.0, max(0.10, qty / max_uniform))




def active_order_notional_proxy(state: dict, current_price: float) -> float:
    lower = float(state["lower_limit_usdt"])
    upper = float(state["upper_limit_usdt"])
    grids = int(float(state["grids"]))
    qty = float(state["quantity_per_grid_eth"])
    lines = sim.grid_lines(lower, upper, grids)
    states = sim.initial_states(lines, current_price)
    trigger_sum = 0.0
    for idx, st in enumerate(states):
        trigger_sum += lines[idx + 1] if st == "sell" else lines[idx]
    return qty * trigger_sum


def candidate_quantity(state: dict, current_price: float, lower: float, upper: float, grids: int,
                       fee_rate: float, utilization: float, active_notional_proxy: float) -> Optional[dict]:
    eth = float(state.get("eth_holdings") or 0.0)
    usdt = float(state.get("usdt_holdings") or 0.0)
    lines = sim.grid_lines(lower, upper, grids)
    states = sim.initial_states(lines, current_price)
    sell_idx = [idx for idx, st in enumerate(states) if st == "sell"]
    buy_idx = [idx for idx, st in enumerate(states) if st == "buy"]
    if len(sell_idx) < 2 or len(buy_idx) < 2:
        return None
    max_by_eth = eth / len(sell_idx) if sell_idx else float("inf")
    buy_usdt_per_qty = sum(lines[idx] * (1.0 + fee_rate) for idx in buy_idx)
    max_by_usdt = usdt / buy_usdt_per_qty if buy_usdt_per_qty > 0 else float("inf")
    max_uniform = min(max_by_eth, max_by_usdt)
    if not math.isfinite(max_uniform) or max_uniform <= 0:
        return None
    trigger_sum = sum(lines[idx + 1] if states[idx] == "sell" else lines[idx] for idx in range(len(states)))
    max_by_active_notional = active_notional_proxy / trigger_sum if trigger_sum > 0 else 0.0
    qty = min(max_uniform * utilization, max_by_active_notional)
    if qty <= 0:
        return None
    avg_order_notional = qty * current_price
    return {
        "qty": qty,
        "max_uniform_qty": max_uniform,
        "sell_intervals": len(sell_idx),
        "buy_intervals": len(buy_idx),
        "avg_order_notional_usdt": avg_order_notional,
    }


def net_grid_profit_bounds(lower: float, upper: float, grids: int, fee_rate: float) -> Tuple[float, float]:
    lines = sim.grid_lines(lower, upper, grids)
    vals = [sim.interval_net_profit_pct(lines[j], lines[j + 1], fee_rate) for j in range(len(lines) - 1)]
    return min(vals), max(vals)


def process_segment_portfolio(a: float, b: float, lines: List[float], states: List[str], qty: float,
                              fee_rate: float, balances: dict) -> Tuple[int, float]:
    rounds = 0
    grid_profit = 0.0
    if b > a:
        for idx in range(len(lines) - 1):
            trigger = lines[idx + 1]
            if a < trigger <= b and states[idx] == "sell":
                if balances["eth"] + 1e-12 < qty:
                    continue
                balances["eth"] -= qty
                balances["usdt"] += qty * trigger * (1.0 - fee_rate)
                rounds += 1
                grid_profit += sim.interval_net_profit_usdt(lines[idx], lines[idx + 1], qty, fee_rate)
                states[idx] = "buy"
    elif b < a:
        for idx in range(len(lines) - 2, -1, -1):
            trigger = lines[idx]
            if b <= trigger < a and states[idx] == "buy":
                cost = qty * trigger * (1.0 + fee_rate)
                if balances["usdt"] + 1e-9 < cost:
                    continue
                balances["usdt"] -= cost
                balances["eth"] += qty
                states[idx] = "sell"
    return rounds, grid_profit


def simulate_portfolio_path(mapped_candles: List[List[float]], current_price: float, lower: float, upper: float,
                            grids: int, qty: float, fee_rate: float, eth0: float, usdt0: float, mode: str) -> dict:
    lines = sim.grid_lines(lower, upper, grids)
    states = sim.initial_states(lines, current_price)
    balances = {"eth": eth0, "usdt": usdt0}
    start_equity = eth0 * current_price + usdt0
    rounds = 0
    grid_profit = 0.0
    lower_escape = False
    upper_escape = False
    prev = current_price
    for c in mapped_candles:
        _, o, h, l, cl, _ = c
        lower_escape = lower_escape or l < lower
        upper_escape = upper_escape or h > upper
        pts = [o, h, l, cl] if mode == "ohlc" else [o, l, h, cl]
        pts = [prev] + pts
        for a, b in zip(pts, pts[1:]):
            r, p = process_segment_portfolio(a, b, lines, states, qty, fee_rate, balances)
            rounds += r
            grid_profit += p
        prev = cl
    end_equity = balances["usdt"] + balances["eth"] * prev
    return {
        "rounds": rounds,
        "grid_profit_usdt": grid_profit,
        "total_pnl_usdt": end_equity - start_equity,
        "end_equity_usdt": end_equity,
        "end_eth": balances["eth"],
        "end_usdt": balances["usdt"],
        "lower_escape": lower_escape,
        "upper_escape": upper_escape,
        "any_escape": lower_escape or upper_escape,
        "end_price": prev,
    }


def mapped_paths(rows: List[dict], master: Dict[int, List[float]], current_price: float) -> List[dict]:
    out = []
    for r in rows:
        fwd = sim.forward_24h(master, int(r["ts"]))
        if fwd is None:
            continue
        out.append({"row": r, "candles": sim.map_historical_path_to_current(float(r["entry_close"]), fwd, current_price)})
    return out


def calibration_scales(calibration: dict) -> Tuple[float, float]:
    if calibration.get("status") != "CALIBRATION_ACTIVE":
        return 1.0, 1.0
    return float(calibration["profit_scale_applied"]), float(calibration["rounds_scale_applied"])


def evaluate_geometry(lower: float, upper: float, grids: int, current_price: float, state: dict,
                      fee_rate: float, utilization: float, active_notional_proxy: float, paths: List[dict], profit_scale: float,
                      rounds_scale: float) -> Optional[dict]:
    if upper <= lower or not (lower < current_price < upper):
        return None
    qinfo = candidate_quantity(state, current_price, lower, upper, grids, fee_rate, utilization, active_notional_proxy)
    if qinfo is None:
        return None
    min_net, max_net = net_grid_profit_bounds(lower, upper, grids, fee_rate)
    min_required = max(MIN_NET_PROFIT_GRID_PCT_FLOOR, fee_rate * 100.0 * 2.0 + FEE_BUFFER_PP)
    if min_net < min_required:
        return None

    eth0 = float(state.get("eth_holdings") or 0.0)
    usdt0 = float(state.get("usdt_holdings") or 0.0)
    profits, rounds, pnls = [], [], []
    escapes, low_esc, up_esc = [], [], []
    for item in paths:
        a = simulate_portfolio_path(item["candles"], current_price, lower, upper, grids, qinfo["qty"], fee_rate, eth0, usdt0, "ohlc")
        b = simulate_portfolio_path(item["candles"], current_price, lower, upper, grids, qinfo["qty"], fee_rate, eth0, usdt0, "olhc")
        profits.append((a["grid_profit_usdt"] + b["grid_profit_usdt"]) / 2.0 * profit_scale)
        rounds.append((a["rounds"] + b["rounds"]) / 2.0 * rounds_scale)
        pnls.append((a["total_pnl_usdt"] + b["total_pnl_usdt"]) / 2.0)
        escapes.append(a["any_escape"] or b["any_escape"])
        low_esc.append(a["lower_escape"] or b["lower_escape"])
        up_esc.append(a["upper_escape"] or b["upper_escape"])
    if not profits:
        return None
    n = len(profits)
    return {
        "lower_usdt": round(lower, 4),
        "upper_usdt": round(upper, 4),
        "center_usdt": round((lower + upper) / 2.0, 4),
        "width_usdt": round(upper - lower, 4),
        "width_pct_of_market": round((upper - lower) / current_price * 100.0, 4),
        "grids": grids,
        "quantity_per_grid_eth_est": round(qinfo["qty"], 8),
        "avg_order_notional_usdt_est": round(qinfo["avg_order_notional_usdt"], 4),
        "buy_intervals": qinfo["buy_intervals"],
        "sell_intervals": qinfo["sell_intervals"],
        "net_profit_per_grid_pct_min": round(min_net, 5),
        "net_profit_per_grid_pct_max": round(max_net, 5),
        "expected_grid_profit_usdt": round(statistics.fmean(profits), 6),
        "median_grid_profit_usdt": round(statistics.median(profits), 6),
        "expected_rounds": round(statistics.fmean(rounds), 6),
        "median_rounds": round(statistics.median(rounds), 6),
        "escape_probability_pct": round(sum(bool(x) for x in escapes) / n * 100.0, 4),
        "lower_escape_probability_pct": round(sum(bool(x) for x in low_esc) / n * 100.0, 4),
        "upper_escape_probability_pct": round(sum(bool(x) for x in up_esc) / n * 100.0, 4),
        "p_grid_profit_ge_0_25_pct": sim.probability_ge(profits, 0.25),
        "p_grid_profit_ge_0_50_pct": sim.probability_ge(profits, 0.50),
        "p_total_pnl_positive_pct": round(sum(x > 0 for x in pnls) / n * 100.0, 4),
        "expected_total_pnl_usdt": round(statistics.fmean(pnls), 6),
        "p20_total_pnl_usdt": round(percentile(pnls, 0.20), 6),
        "p10_total_pnl_usdt": round(percentile(pnls, 0.10), 6),
        "sample_n": n,
    }


def geometry_key(c: dict) -> Tuple[float, float, int]:
    return (round(float(c["center_usdt"]), 3), round(float(c["width_usdt"]), 3), int(c["grids"]))


def candidate_from_params(center: float, width: float, grids: int, *args) -> Optional[dict]:
    return evaluate_geometry(center - width / 2.0, center + width / 2.0, grids, *args)


def pareto_frontier(candidates: List[dict]) -> List[dict]:
    out = []
    for c in candidates:
        dominated = False
        for d in candidates:
            if c is d:
                continue
            better_profit = d["expected_grid_profit_usdt"] >= c["expected_grid_profit_usdt"] - 1e-12
            lower_escape = d["escape_probability_pct"] <= c["escape_probability_pct"] + 1e-12
            better_tail = d["p20_total_pnl_usdt"] >= c["p20_total_pnl_usdt"] - 1e-12
            strict = (d["expected_grid_profit_usdt"] > c["expected_grid_profit_usdt"] + 1e-12 or
                      d["escape_probability_pct"] < c["escape_probability_pct"] - 1e-12 or
                      d["p20_total_pnl_usdt"] > c["p20_total_pnl_usdt"] + 1e-12)
            if better_profit and lower_escape and better_tail and strict:
                dominated = True
                break
        if not dominated:
            out.append(c)
    return sorted(out, key=lambda x: (x["escape_probability_pct"], -x["expected_grid_profit_usdt"], -x["p20_total_pnl_usdt"]))


def nearest_feature(features: List[dict], ts: int, max_gap_h: float = 4.0) -> Optional[dict]:
    eligible = [r for r in features if int(r["ts"]) <= ts]
    if not eligible:
        return None
    r = eligible[-1]
    if ts - int(r["ts"]) > max_gap_h * 3600:
        return None
    return r


def same_geometry(a: dict, b: dict, tol: float = 1e-6) -> bool:
    return (
        abs(float(a["lower_limit_usdt"]) - float(b["lower_limit_usdt"])) <= tol and
        abs(float(a["upper_limit_usdt"]) - float(b["upper_limit_usdt"])) <= tol and
        int(float(a["grids"])) == int(float(b["grids"])) and
        abs(float(a["quantity_per_grid_eth"]) - float(b["quantity_per_grid_eth"])) <= tol
    )


def reconstruct_calibration(states: List[dict], features: List[dict], master: Dict[int, List[float]], fee_rate: float) -> dict:
    windows = []
    for start, end in zip(states, states[1:]):
        if not same_geometry(start, end):
            continue
        start_dt = start["_dt"]
        end_dt = end["_dt"]
        elapsed_h = (end_dt - start_dt).total_seconds() / 3600.0
        if not (CAL_MIN_H <= elapsed_h <= CAL_MAX_H):
            continue
        feat = nearest_feature(features, int(start_dt.timestamp()))
        if feat is None:
            continue
        # At the time of the prediction, only paths whose 24h target was already mature are eligible.
        mature_then = [r for r in features if int(r["ts"]) + sim.HORIZON_H * 3600 <= int(feat["ts"])]
        independent_then = sim.greedy_independent(mature_then)
        try:
            analogs, _ = select_analogs(independent_then, feat, master, n=min(FINAL_ANALOG_N, len(independent_then)))
        except SystemExit:
            continue
        current_price = float(start["current_price_usdt"])
        paths = mapped_paths(analogs, master, current_price)
        utilization = observed_utilization_factor(start, current_price, fee_rate)
        active_proxy = active_order_notional_proxy(start, current_price)
        base = evaluate_geometry(float(start["lower_limit_usdt"]), float(start["upper_limit_usdt"]), int(float(start["grids"])), current_price, start, fee_rate, utilization, active_proxy, paths, 1.0, 1.0)
        if not base:
            continue
        p0 = float(start["grid_profit_usdt"])
        p1 = float(end["grid_profit_usdt"])
        r0 = float(start["rounds_total"])
        r1 = float(end["rounds_total"])
        actual_profit_24 = (p1 - p0) * 24.0 / elapsed_h
        actual_rounds_24 = (r1 - r0) * 24.0 / elapsed_h
        if actual_profit_24 < -1e-9 or actual_rounds_24 < -1e-9:
            continue
        pred_profit = float(base["expected_grid_profit_usdt"])
        pred_rounds = float(base["expected_rounds"])
        windows.append({
            "start_utc": start_dt.isoformat(), "end_utc": end_dt.isoformat(), "elapsed_h": round(elapsed_h, 4),
            "predicted_profit_24h_usdt": pred_profit, "actual_profit_24h_equiv_usdt": round(actual_profit_24, 6),
            "predicted_rounds_24h": pred_rounds, "actual_rounds_24h_equiv": round(actual_rounds_24, 6),
            "profit_ratio": round(actual_profit_24 / pred_profit, 6) if pred_profit > 0 else None,
            "rounds_ratio": round(actual_rounds_24 / pred_rounds, 6) if pred_rounds > 0 else None,
        })
    p_ratios = [w["profit_ratio"] for w in windows if w["profit_ratio"] is not None and w["profit_ratio"] > 0]
    r_ratios = [w["rounds_ratio"] for w in windows if w["rounds_ratio"] is not None and w["rounds_ratio"] > 0]
    observed_profit_scale = statistics.median(p_ratios) if p_ratios else None
    observed_rounds_scale = statistics.median(r_ratios) if r_ratios else None
    active = len(windows) >= CAL_MIN_WINDOWS_TO_APPLY
    profit_scale = min(CAL_SCALE_HI, max(CAL_SCALE_LO, observed_profit_scale)) if active and observed_profit_scale is not None else 1.0
    rounds_scale = min(CAL_SCALE_HI, max(CAL_SCALE_LO, observed_rounds_scale)) if active and observed_rounds_scale is not None else 1.0
    return {
        "schema": "pionex_grid_calibration_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CALIBRATION_ACTIVE" if active else "EARLY_EVIDENCE_NOT_APPLIED",
        "minimum_windows_before_apply": CAL_MIN_WINDOWS_TO_APPLY,
        "evaluated_windows": len(windows),
        "observed_median_profit_ratio": round(observed_profit_scale, 6) if observed_profit_scale is not None else None,
        "observed_median_rounds_ratio": round(observed_rounds_scale, 6) if observed_rounds_scale is not None else None,
        "profit_scale_applied": round(profit_scale, 6),
        "rounds_scale_applied": round(rounds_scale, 6),
        "method": "Walk-forward reconstruction at manual Pionex state times; only 24h-mature historical analogue paths available by each start time are used. Scale applies after >=3 usable 14-30h same-geometry windows.",
        "windows": windows,
    }


def select_geometry(candidates: List[dict], current: dict) -> dict:
    base_escape = current["escape_probability_pct"]
    effective_escape_cap = min(ABS_ESCAPE_RISK_CAP_PCT, base_escape)
    eligible = []
    for c in candidates:
        if c["escape_probability_pct"] > effective_escape_cap + 1e-9:
            continue
        if c["p_total_pnl_positive_pct"] < current["p_total_pnl_positive_pct"] - MAX_POSITIVE_PNL_PROB_DROP_PP:
            continue
        if c["p20_total_pnl_usdt"] < current["p20_total_pnl_usdt"] - MAX_P20_TOTAL_PNL_WORSEN_USDT:
            continue
        eligible.append(c)
    if not eligible:
        eligible = [current]
    raw = max(eligible, key=lambda c: (c["expected_grid_profit_usdt"], -c["escape_probability_pct"], c["p20_total_pnl_usdt"]))
    gain = raw["expected_grid_profit_usdt"] - current["expected_grid_profit_usdt"]
    risk_change = raw["escape_probability_pct"] - current["escape_probability_pct"]
    rounds_gain = raw["expected_rounds"] - current["expected_rounds"]
    # Activity is diagnostic only. A geometry change must earn materially more
    # expected USDT grid profit per 24h, or materially reduce escape risk.
    material = (gain >= MIN_EXPECTED_GRID_PROFIT_GAIN_USDT or risk_change <= -MIN_ESCAPE_RISK_REDUCTION_PP)
    changed = (abs(raw["lower_usdt"] - current["lower_usdt"]) > 0.01 or abs(raw["upper_usdt"] - current["upper_usdt"]) > 0.01 or raw["grids"] != current["grids"])
    chosen = raw if material and changed else current
    action = "KEEP_CURRENT"
    if chosen is not current:
        centre_delta = chosen["center_usdt"] - current["center_usdt"]
        width_delta = chosen["width_usdt"] - current["width_usdt"]
        grid_delta = chosen["grids"] - current["grids"]
        parts = []
        if abs(centre_delta) >= 2.5:
            parts.append("SHIFT_UP" if centre_delta > 0 else "SHIFT_DOWN")
        if abs(width_delta) >= 2.5:
            parts.append("WIDEN" if width_delta > 0 else "NARROW")
        if grid_delta:
            parts.append("MORE_GRIDS" if grid_delta > 0 else "FEWER_GRIDS")
        action = "+".join(parts) or "CHANGE_GEOMETRY"
    return {
        "current": current,
        "raw_selected": raw,
        "selected": chosen,
        "action": action,
        "expected_gain_vs_current_usdt": round(chosen["expected_grid_profit_usdt"] - current["expected_grid_profit_usdt"], 6),
        "rounds_gain_vs_current": round(chosen["expected_rounds"] - current["expected_rounds"], 6),
        "escape_change_vs_current_pp": round(chosen["escape_probability_pct"] - current["escape_probability_pct"], 6),
        "effective_escape_cap_pct": round(effective_escape_cap, 6),
        "eligible_count": len(eligible),
    }


def practicalize(c: dict, candidates: List[dict]) -> dict:
    lower = round(c["lower_usdt"] / PRACTICAL_PRICE_ROUND_USDT) * PRACTICAL_PRICE_ROUND_USDT
    upper = round(c["upper_usdt"] / PRACTICAL_PRICE_ROUND_USDT) * PRACTICAL_PRICE_ROUND_USDT
    target = min(candidates, key=lambda x: abs(x["lower_usdt"] - lower) + abs(x["upper_usdt"] - upper) + 2.0 * abs(x["grids"] - c["grids"]))
    return target


def append_ledger(row: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    existing = []
    if os.path.isfile(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
    key = (str(row.get("feature_ts_utc")), str(row.get("state_captured_at_utc")))
    existing = [r for r in existing if (str(r.get("feature_ts_utc")), str(r.get("state_captured_at_utc"))) != key]
    existing.append({k: row.get(k, "") for k in LEDGER_FIELDS})
    existing.sort(key=lambda r: (str(r.get("feature_ts_utc", "")), str(r.get("state_captured_at_utc", ""))))
    with open(LEDGER_PATH, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        w.writeheader(); w.writerows(existing)


def main() -> None:
    profile = sim.load_json(PROFILE_PATH)
    states = phase4c.read_manual_states()
    if not states:
        raise SystemExit("No manual Pionex states")
    state = states[-1]
    master = sim.build_master_candles()
    features = sim.load_eth_features()
    if not features:
        raise SystemExit("No ETH feature rows")
    current_feature = features[-1]
    current_price = float(current_feature["entry_close"])
    current_atr = float(current_feature["atr14_pct"])
    fee_pct = float(profile.get("fee_model", {}).get("standard_public_spot_fee_pct_per_fill_reference", 0.05))
    fee_rate = fee_pct / 100.0

    calibration = reconstruct_calibration(states, features, master, fee_rate)
    profit_scale, rounds_scale = calibration_scales(calibration)

    matured = [r for r in features if sim.forward_24h(master, int(r["ts"])) is not None]
    independent = sim.greedy_independent(matured)
    analogs, analogue_meta = select_analogs(independent, current_feature, master)
    paths = mapped_paths(analogs, master, current_price)
    if len(paths) < 30:
        raise SystemExit(f"Too few mapped analogue paths: {len(paths)}")

    utilization = observed_utilization_factor(state, current_price, fee_rate)
    active_notional = active_order_notional_proxy(state, current_price)
    current_lower = float(state["lower_limit_usdt"])
    current_upper = float(state["upper_limit_usdt"])
    current_grids = int(float(state["grids"]))
    current = evaluate_geometry(current_lower, current_upper, current_grids, current_price, state, fee_rate, utilization, active_notional, paths, profit_scale, rounds_scale)
    if current is None:
        raise SystemExit("Current Pionex geometry failed model feasibility checks")

    # Coarse joint search.
    market = current_price
    center_offsets = []
    lim = market * CENTER_SEARCH_PCT / 100.0
    x = -lim
    while x <= lim + 1e-9:
        center_offsets.append(x)
        x += COARSE_CENTER_STEP_USDT
    center_offsets.append(current["center_usdt"] - market)
    widths = []
    w = MIN_WIDTH_PCT
    while w <= MAX_WIDTH_PCT + 1e-9:
        widths.append(market * w / 100.0)
        w += COARSE_WIDTH_STEP_PCT
    widths.append(current["width_usdt"])
    grids_list = sorted(set(COARSE_GRID_COUNTS + (current_grids,)))

    evaluated: Dict[Tuple[float, float, int], dict] = {geometry_key(current): current}
    for off in sorted(set(round(v, 6) for v in center_offsets)):
        center = market + off
        for width in widths:
            for grids in grids_list:
                c = candidate_from_params(center, width, grids, current_price, state, fee_rate, utilization, active_notional, paths, profit_scale, rounds_scale)
                if c:
                    evaluated[geometry_key(c)] = c

    # Rank coarse candidates with the same transparent risk constraints used for final selection.
    coarse_candidates = list(evaluated.values())
    coarse_selection = select_geometry(coarse_candidates, current)
    eligible_for_refine = sorted(
        [c for c in coarse_candidates if c["escape_probability_pct"] <= coarse_selection["effective_escape_cap_pct"] + 1e-9],
        key=lambda c: (-c["expected_grid_profit_usdt"], c["escape_probability_pct"], -c["p20_total_pnl_usdt"]),
    )[:TOP_COARSE_TO_REFINE]

    # Local refinement around the strongest coarse regions.
    for seed in eligible_for_refine:
        centers = [seed["center_usdt"] + d for d in (-7.5, -5, -2.5, 0, 2.5, 5, 7.5)]
        widths_ref = [seed["width_usdt"] + market * p / 100.0 for p in (-0.5, 0, 0.5)]
        grids_ref = range(max(10, seed["grids"] - REFINE_GRID_RADIUS), min(80, seed["grids"] + REFINE_GRID_RADIUS) + 1)
        for center in centers:
            for width in widths_ref:
                if width <= 0:
                    continue
                for grids in grids_ref:
                    c = candidate_from_params(center, width, grids, current_price, state, fee_rate, utilization, active_notional, paths, profit_scale, rounds_scale)
                    if c:
                        evaluated[geometry_key(c)] = c

    candidates = list(evaluated.values())
    selection = select_geometry(candidates, current)
    selected = selection["selected"]
    practical = practicalize(selected, candidates)
    frontier = pareto_frontier(candidates)

    state_dt = state["_dt"]
    feature_dt = datetime.fromtimestamp(int(current_feature["ts"]), tz=timezone.utc)
    state_age_h = (datetime.now(timezone.utc) - state_dt).total_seconds() / 3600.0
    market_gap_from_state_h = (feature_dt - state_dt).total_seconds() / 3600.0
    confidence = "LOW"
    if state_age_h <= 6 and calibration["evaluated_windows"] >= CAL_MIN_WINDOWS_TO_APPLY:
        confidence = "MEDIUM"
    elif state_age_h <= 6:
        confidence = "LOW_MEDIUM"

    min_required = max(MIN_NET_PROFIT_GRID_PCT_FLOOR, fee_pct * 2.0 + FEE_BUFFER_PP)
    out = {
        "schema": "pionex_grid_geometry_optimizer_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_DIAGNOSTIC_ONLY",
        "scope": {"platform": "Pionex", "pair": "ETH/USDT", "bot_type": "Spot Grid", "horizon_h": 24},
        "source_state": {
            "captured_at_utc": state.get("captured_at_utc"), "captured_at_local": state.get("captured_at_local"),
            "state_age_h_at_run": round(state_age_h, 4), "state_price_usdt": f(state.get("current_price_usdt")),
            "latest_market_price_usdt": current_price, "market_feature_minus_state_h": round(market_gap_from_state_h, 4),
            "eth_holdings": f(state.get("eth_holdings")), "usdt_holdings": f(state.get("usdt_holdings")),
        },
        "market_regime": {
            "feature_ts_utc": int(current_feature["ts"]), "current_atr14_pct": current_atr,
            "analogue_selection": analogue_meta, "independent_history_rows": len(independent),
        },
        "calibration": {k: v for k, v in calibration.items() if k != "windows"},
        "geometry_model": {
            "joint_variables": ["centre", "width", "grid_count"],
            "optimization_objective": "Maximise expected_grid_profit_usdt over the 24h horizon among candidates that pass the existing risk constraints. Expected rounds/activity are diagnostic only and never make a change actionable by themselves.",
            "grid_count_search": {"min": 10, "max": 80, "coarse_step": 5, "refine_radius": REFINE_GRID_RADIUS},
            "quantity_sizing": "Uniform ETH quantity/grid is rescaled from the latest observed balances using the same utilization ratio as the live bot, subject to buy-side USDT and sell-side ETH seeding requirements.",
            "portfolio_model": "Historical analogue paths replay buys/sells against the latest captured ETH and USDT holdings and mark remaining inventory to the 24h end price.",
            "min_net_profit_per_grid_pct_constraint": round(min_required, 6),
            "net_profit_floor_semantics": "The interval profit metric is already net of modeled trading fees; 0.12% is therefore a true net-profit/grid floor.",
            "platform_minimum_warning": "Pionex's live minimum investment/order requirement depends on pair, range and grid count. Offline candidates still require confirmation in the Pionex edit screen before use.",
            "candidate_count": len(candidates), "pareto_count": len(frontier),
            "current_utilization_factor_est": round(utilization, 6),
            "active_order_notional_proxy_usdt_preserved": round(active_notional, 6),
        },
        "risk_policy": {
            "absolute_escape_cap_pct": ABS_ESCAPE_RISK_CAP_PCT,
            "effective_escape_cap_pct": selection["effective_escape_cap_pct"],
            "do_not_reduce_total_pnl_positive_probability_by_more_than_pp": MAX_POSITIVE_PNL_PROB_DROP_PP,
            "do_not_worsen_p20_total_pnl_by_more_than_usdt": MAX_P20_TOTAL_PNL_WORSEN_USDT,
            "material_change_rule": {
                "expected_grid_profit_gain_usdt": MIN_EXPECTED_GRID_PROFIT_GAIN_USDT,
                "or_escape_risk_reduction_pp": MIN_ESCAPE_RISK_REDUCTION_PP,
            },
        },
        "decision": {
            "action": selection["action"], "confidence": confidence,
            "expected_gain_vs_current_usdt": selection["expected_gain_vs_current_usdt"],
            "rounds_gain_vs_current": selection["rounds_gain_vs_current"],
            "escape_change_vs_current_pp": selection["escape_change_vs_current_pp"],
            "practical_note": "Price bounds are rounded to nearby $5 levels only for usability; the raw selected geometry remains the research result.",
            "warning": "Decision support only; use Pionex's live edit screen to confirm the setup remains valid for the current bot balance and platform minimums.",
        },
        "benchmarks": {"current": current, "raw_selected": selection["raw_selected"], "selected": selected, "practical": practical},
        "pareto_frontier_top": frontier[:30],
        "calibration_windows": calibration["windows"],
        "next": {
            "collect": "Continue daily screenshots and capture immediately after any range/grid-count change.",
            "promotion_gate": "Do not treat geometry recommendations as decision-grade until prospective profit/round calibration is active and stable across multiple market regimes.",
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(CAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=2)

    append_ledger({
        "generated_at_utc": out["generated_at_utc"], "feature_ts_utc": int(current_feature["ts"]),
        "state_captured_at_utc": state.get("captured_at_utc") or "", "market_price_usdt": current_price,
        "state_age_h": round(state_age_h, 4), "current_atr14_pct": current_atr,
        "current_choppiness_efficiency": analogue_meta["current_shape"]["efficiency_ratio"],
        "current_reversal_rate": analogue_meta["current_shape"]["reversal_rate"],
        "current_path_length_pct": analogue_meta["current_shape"]["path_length_pct"],
        "calibration_windows": calibration["evaluated_windows"], "profit_scale_applied": profit_scale,
        "rounds_scale_applied": rounds_scale, "current_lower_usdt": current["lower_usdt"],
        "current_upper_usdt": current["upper_usdt"], "current_width_usdt": current["width_usdt"],
        "current_grids": current["grids"], "current_qty_eth": current["quantity_per_grid_eth_est"],
        "current_expected_profit_usdt": current["expected_grid_profit_usdt"], "current_expected_rounds": current["expected_rounds"],
        "current_escape_probability_pct": current["escape_probability_pct"], "recommended_action": selection["action"],
        "recommended_lower_usdt": selected["lower_usdt"], "recommended_upper_usdt": selected["upper_usdt"],
        "recommended_width_usdt": selected["width_usdt"], "recommended_grids": selected["grids"],
        "recommended_qty_eth": selected["quantity_per_grid_eth_est"], "recommended_expected_profit_usdt": selected["expected_grid_profit_usdt"],
        "recommended_expected_rounds": selected["expected_rounds"], "recommended_escape_probability_pct": selected["escape_probability_pct"],
        "recommended_p_total_pnl_positive_pct": selected["p_total_pnl_positive_pct"], "recommended_p20_total_pnl_usdt": selected["p20_total_pnl_usdt"],
        "practical_lower_usdt": practical["lower_usdt"], "practical_upper_usdt": practical["upper_usdt"],
        "practical_grids": practical["grids"], "confidence": confidence, "status": out["status"],
    })

    print("Phase 4D geometry optimizer written:", OUT_PATH)
    print("Manual states:", len(states), "Calibration:", calibration["status"], "windows=", calibration["evaluated_windows"])
    print("Analogue state:", analogue_meta["current_shape"])
    print("Candidates:", len(candidates), "Pareto:", len(frontier))
    print("Current:", current)
    print("Decision:", out["decision"])
    print("Selected:", selected)
    print("Practical:", practical)


if __name__ == "__main__":
    main()
