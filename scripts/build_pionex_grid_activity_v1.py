#!/usr/bin/env python3
"""
Phase 4D.2 — Pionex grid activity and actionability layer.

This phase intentionally sits *above* the Phase 4D research optimizer.

It does not replace the 4D geometry search. Instead it:
  - preserves the exact observed live quantity/grid for the current benchmark;
  - measures stale-state risk and blocks operational changes when the latest
    Pionex screenshot is too old;
  - reconstructs observed grid activity between manual screenshots, including
    net unmatched inventory legs where the holdings delta supports that inference;
  - reports nearest waiting buy/sell triggers from the live geometry;
  - converts the raw 4D selected bounds into literal $5-rounded practical bounds;
  - consumes the optional Phase 4D.1 execution-resolution policy when present;
  - applies explicit prospective gates before any research recommendation is
    considered operationally actionable.

Diagnostic / decision support only. Never connects to Pionex or places orders.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import build_pionex_grid_simulator_v1 as sim

STATE_PATH = Path("data/pionex/manual_grid_states_v1.csv")
GEOMETRY_PATH = Path("data/diagnostics/pionex_grid_geometry_optimizer_v1.json")
CAL_PATH = Path("data/diagnostics/pionex_grid_calibration_v2.json")
EXEC_POLICY_PATH = Path("data/pionex/execution_resolution_policy_v1.json")

OUT_PATH = Path("data/diagnostics/pionex_grid_actionability_v1.json")
ACTIVITY_LEDGER_PATH = Path("data/pionex/grid_activity_windows_v1.csv")

FRESH_STATE_MAX_H = 6.0
ACTIVITY_INTEGER_TOL_GRID_UNITS = 0.18
PRACTICAL_ROUND_USDT = 5.0
MIN_CAL_WINDOWS_FOR_OPERATIONAL_ACTION = 3
MIN_EXEC_WINDOWS_FOR_OPERATIONAL_ACTION = 3

ACTIVITY_FIELDS = [
    "start_utc", "end_utc", "elapsed_h", "same_geometry",
    "start_price_usdt", "end_price_usdt",
    "completed_rounds_delta", "grid_profit_delta_usdt",
    "eth_delta", "usdt_delta", "quantity_per_grid_eth",
    "net_inventory_grid_units_raw", "net_inventory_grid_units_inferred",
    "inventory_inference_status", "estimated_fill_count_lower_bound",
    "end_reported_rounds_24h",
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


def parse_dt(v: str) -> datetime:
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)


def read_states() -> List[dict]:
    with STATE_PATH.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            dt = parse_dt(r["captured_at_utc"])
        except Exception:
            continue
        x = dict(r)
        x["_dt"] = dt
        x["_ts"] = int(dt.timestamp())
        out.append(x)
    out.sort(key=lambda r: r["_ts"])
    return out


def same_geometry(a: dict, b: dict) -> bool:
    for key in ("lower_limit_usdt", "upper_limit_usdt", "grids", "quantity_per_grid_eth"):
        av, bv = f(a.get(key)), f(b.get(key))
        if av is None or bv is None or abs(av - bv) > 1e-9:
            return False
    return True


def infer_activity(a: dict, b: dict) -> dict:
    qty = f(a.get("quantity_per_grid_eth"))
    eth_a, eth_b = f(a.get("eth_holdings")), f(b.get("eth_holdings"))
    usdt_a, usdt_b = f(a.get("usdt_holdings")), f(b.get("usdt_holdings"))
    rounds_a, rounds_b = i(a.get("rounds_total")), i(b.get("rounds_total"))
    gp_a, gp_b = f(a.get("grid_profit_usdt")), f(b.get("grid_profit_usdt"))

    elapsed_h = (b["_dt"] - a["_dt"]).total_seconds() / 3600.0
    completed = (rounds_b - rounds_a) if rounds_a is not None and rounds_b is not None else None
    gp_delta = (gp_b - gp_a) if gp_a is not None and gp_b is not None else None
    eth_delta = (eth_b - eth_a) if eth_a is not None and eth_b is not None else None
    usdt_delta = (usdt_b - usdt_a) if usdt_a is not None and usdt_b is not None else None

    raw_units = None
    inferred_units = None
    inference_status = "UNAVAILABLE"
    if qty and qty > 0 and eth_delta is not None:
        raw_units = eth_delta / qty
        nearest = int(round(raw_units))
        if abs(raw_units - nearest) <= ACTIVITY_INTEGER_TOL_GRID_UNITS:
            inferred_units = nearest
            if nearest > 0:
                inference_status = "NET_UNMATCHED_BUY_LEGS_INCREASE"
            elif nearest < 0:
                inference_status = "NET_UNMATCHED_SELL_LEGS_INCREASE"
            else:
                inference_status = "NO_NET_OPEN_LEG_CHANGE"
        else:
            inference_status = "HOLDINGS_DELTA_NOT_CLOSE_TO_INTEGER_GRID_UNITS"

    lower_bound_fills = None
    if completed is not None and completed >= 0:
        # Every completed Pionex round requires at least one buy + one sell.
        # Add only the absolute *net* unmatched inventory-unit change. This is
        # deliberately a lower bound, not a claim about total fills.
        lower_bound_fills = 2 * completed
        if inferred_units is not None:
            lower_bound_fills += abs(inferred_units)

    return {
        "start_utc": a.get("captured_at_utc"),
        "end_utc": b.get("captured_at_utc"),
        "elapsed_h": round(elapsed_h, 4),
        "same_geometry": same_geometry(a, b),
        "start_price_usdt": f(a.get("current_price_usdt")),
        "end_price_usdt": f(b.get("current_price_usdt")),
        "completed_rounds_delta": completed,
        "grid_profit_delta_usdt": round(gp_delta, 8) if gp_delta is not None else None,
        "eth_delta": round(eth_delta, 8) if eth_delta is not None else None,
        "usdt_delta": round(usdt_delta, 8) if usdt_delta is not None else None,
        "quantity_per_grid_eth": qty,
        "net_inventory_grid_units_raw": round(raw_units, 6) if raw_units is not None else None,
        "net_inventory_grid_units_inferred": inferred_units,
        "inventory_inference_status": inference_status,
        "estimated_fill_count_lower_bound": lower_bound_fills,
        "end_reported_rounds_24h": i(b.get("rounds_24h")),
    }


def write_activity(rows: List[dict]) -> None:
    ACTIVITY_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVITY_LEDGER_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ACTIVITY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ACTIVITY_FIELDS})


def nearest_waiting_triggers(state: dict) -> dict:
    lower = float(state["lower_limit_usdt"])
    upper = float(state["upper_limit_usdt"])
    grids = int(float(state["grids"]))
    price = float(state["current_price_usdt"])

    lines = sim.grid_lines(lower, upper, grids)
    states = sim.initial_states(lines, price)

    buy_triggers = [lines[idx] for idx, s in enumerate(states) if s == "buy" and lines[idx] < price]
    sell_triggers = [lines[idx + 1] for idx, s in enumerate(states) if s == "sell" and lines[idx + 1] > price]

    buy = max(buy_triggers) if buy_triggers else None
    sell = min(sell_triggers) if sell_triggers else None

    return {
        "nearest_waiting_buy_usdt": round(buy, 4) if buy is not None else None,
        "nearest_waiting_buy_distance_pct": round((buy / price - 1.0) * 100.0, 4) if buy else None,
        "nearest_waiting_sell_usdt": round(sell, 4) if sell is not None else None,
        "nearest_waiting_sell_distance_pct": round((sell / price - 1.0) * 100.0, 4) if sell else None,
        "grid_spacing_usdt": round((upper - lower) / (grids - 1), 6),
    }


def literal_practical_geometry(selected: dict) -> dict:
    lower = f(selected.get("lower_usdt"))
    upper = f(selected.get("upper_usdt"))
    grids = i(selected.get("grids"))
    if lower is None or upper is None or grids is None:
        return {}
    pl = round(lower / PRACTICAL_ROUND_USDT) * PRACTICAL_ROUND_USDT
    pu = round(upper / PRACTICAL_ROUND_USDT) * PRACTICAL_ROUND_USDT
    if pu <= pl:
        return {}
    return {
        "lower_usdt": pl,
        "upper_usdt": pu,
        "center_usdt": round((pl + pu) / 2.0, 4),
        "width_usdt": round(pu - pl, 4),
        "grids": grids,
        "metrics_recomputed_for_rounded_bounds": False,
        "note": (
            "Literal usability rounding only. Research metrics belong to the raw selected geometry; "
            "Pionex live edit validation is required before any use."
        ),
    }


def load_optional_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None



def geometry_execution_integration_gate(geometry: dict, exec_gate_result: dict) -> dict:
    integration = geometry.get("execution_resolution_integration") or {}
    expected = str(exec_gate_result.get("active_execution_resolution") or "1hour")
    actual = str(integration.get("execution_replay_resolution") or "")
    integrated = integration.get("execution_resolution_integrated") is True

    # If Phase 4D.1 has promoted a resolution, the geometry metrics themselves
    # must have been generated using that exact execution replay resolution.
    passed = integrated and actual == expected
    return {
        "passed": passed,
        "expected_execution_resolution": expected,
        "geometry_execution_replay_resolution": actual or None,
        "integration_metadata_available": bool(integration),
        "policy_status_seen_by_geometry": integration.get("policy_status"),
    }


def execution_gate(exec_policy: Optional[dict]) -> dict:
    if not exec_policy:
        return {
            "available": False,
            "passed": False,
            "status": "EXECUTION_RESOLUTION_VALIDATION_NOT_AVAILABLE",
            "active_execution_resolution": "1hour",
            "windows_evaluated": 0,
        }

    status = str(exec_policy.get("status") or "")
    windows = int(exec_policy.get("windows_evaluated") or 0)
    active = str(exec_policy.get("active_execution_resolution") or "1hour")
    passed = (
        windows >= MIN_EXEC_WINDOWS_FOR_OPERATIONAL_ACTION
        and status in {"PROMOTION_READY", "KEEP_1H_EXECUTION_RESOLUTION"}
    )
    return {
        "available": True,
        "passed": passed,
        "status": status,
        "active_execution_resolution": active,
        "windows_evaluated": windows,
    }


def main() -> None:
    states = read_states()
    if not states:
        raise SystemExit("No manual Pionex states")
    latest = states[-1]

    geometry = load_optional_json(GEOMETRY_PATH)
    if not geometry:
        raise SystemExit("Phase 4D geometry diagnostic missing; run Phase 4D first")
    calibration = load_optional_json(CAL_PATH) or {}
    exec_policy = load_optional_json(EXEC_POLICY_PATH)

    now = datetime.now(timezone.utc)
    state_age_h = (now - latest["_dt"]).total_seconds() / 3600.0
    fresh = state_age_h <= FRESH_STATE_MAX_H

    activity = [infer_activity(a, b) for a, b in zip(states, states[1:])]
    write_activity(activity)

    benches = geometry.get("benchmarks") or {}
    current_model = benches.get("current") or {}
    selected = benches.get("selected") or {}
    raw_selected = benches.get("raw_selected") or selected
    research_decision = geometry.get("decision") or {}

    observed_qty = f(latest.get("quantity_per_grid_eth"))
    model_qty = f(current_model.get("quantity_per_grid_eth_est"))
    qty_diff_pct = None
    if observed_qty and model_qty:
        qty_diff_pct = (model_qty / observed_qty - 1.0) * 100.0

    cal_windows = int(calibration.get("evaluated_windows") or 0)
    cal_active = str(calibration.get("status") or "") == "CALIBRATION_ACTIVE"
    cal_passed = cal_active and cal_windows >= MIN_CAL_WINDOWS_FOR_OPERATIONAL_ACTION

    egate = execution_gate(exec_policy)
    gexec_gate = geometry_execution_integration_gate(geometry, egate)
    geometry_changed = str(research_decision.get("action") or "KEEP_CURRENT") != "KEEP_CURRENT"
    research_confidence = str(research_decision.get("confidence") or "LOW")
    confidence_passed = research_confidence in {"LOW_MEDIUM", "MEDIUM", "HIGH"}

    blockers = []
    if not fresh:
        blockers.append(f"STALE_PIONEX_STATE_GT_{FRESH_STATE_MAX_H:g}H")
    if not cal_passed:
        blockers.append("PROSPECTIVE_CALIBRATION_NOT_ACTIVE")
    if not egate["passed"]:
        blockers.append("EXECUTION_RESOLUTION_NOT_PROSPECTIVELY_VALIDATED")
    if egate["passed"] and not gexec_gate["passed"]:
        blockers.append("GEOMETRY_NOT_REPLAYED_AT_PROMOTED_EXECUTION_RESOLUTION")
    if geometry_changed and not confidence_passed:
        blockers.append("RESEARCH_GEOMETRY_CONFIDENCE_TOO_LOW")

    operationally_actionable = geometry_changed and not blockers
    if operationally_actionable:
        operational_action = research_decision.get("action")
    else:
        operational_action = "KEEP_CURRENT"

    latest_activity = activity[-1] if activity else None
    triggers = nearest_waiting_triggers(latest)
    practical = literal_practical_geometry(raw_selected)

    out = {
        "schema": "pionex_grid_actionability_v1",
        "generated_at_utc": now.isoformat(),
        "status": "PROSPECTIVE_DECISION_SUPPORT_ONLY",
        "scope": {
            "platform": "Pionex",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "role": "Operational gate above Phase 4D research geometry",
        },
        "latest_observed_state": {
            "captured_at_utc": latest.get("captured_at_utc"),
            "captured_at_local": latest.get("captured_at_local"),
            "state_age_h": round(state_age_h, 4),
            "fresh_for_operational_use": fresh,
            "freshness_limit_h": FRESH_STATE_MAX_H,
            "current_price_usdt": f(latest.get("current_price_usdt")),
            "lower_usdt": f(latest.get("lower_limit_usdt")),
            "upper_usdt": f(latest.get("upper_limit_usdt")),
            "grids": i(latest.get("grids")),
            "observed_quantity_per_grid_eth": observed_qty,
            "eth_holdings": f(latest.get("eth_holdings")),
            "usdt_holdings": f(latest.get("usdt_holdings")),
            "rounds_24h_reported": i(latest.get("rounds_24h")),
            "rounds_total": i(latest.get("rounds_total")),
            "grid_profit_usdt": f(latest.get("grid_profit_usdt")),
            **triggers,
        },
        "current_benchmark_quantity_audit": {
            "observed_live_quantity_per_grid_eth": observed_qty,
            "phase4d_estimated_quantity_per_grid_eth": model_qty,
            "phase4d_estimate_minus_observed_pct": round(qty_diff_pct, 4) if qty_diff_pct is not None else None,
            "policy": (
                "The live/current benchmark quantity is the observed Pionex value. "
                "Rescaled quantity estimates apply only to hypothetical candidate geometries."
            ),
        },
        "latest_observed_activity_window": latest_activity,
        "activity_model": {
            "completed_round_definition": "One completed Pionex round is treated as at least one buy plus one sell.",
            "net_open_leg_inference": (
                "ETH holdings delta divided by observed quantity/grid; integer inference is accepted only "
                f"within ±{ACTIVITY_INTEGER_TOL_GRID_UNITS:.2f} grid units."
            ),
            "estimated_fill_count_is_lower_bound": True,
            "windows_logged": len(activity),
        },
        "research_geometry": {
            "research_action": research_decision.get("action"),
            "research_confidence": research_confidence,
            "raw_selected": raw_selected,
            "literal_practical_rounding": practical,
        },
        "prospective_gates": {
            "state_freshness": {"passed": fresh, "state_age_h": round(state_age_h, 4)},
            "calibration": {
                "passed": cal_passed,
                "status": calibration.get("status"),
                "evaluated_windows": cal_windows,
                "minimum_windows": MIN_CAL_WINDOWS_FOR_OPERATIONAL_ACTION,
            },
            "execution_resolution": egate,
            "geometry_execution_integration": gexec_gate,
            "research_confidence": {
                "passed": confidence_passed if geometry_changed else True,
                "value": research_confidence,
            },
        },
        "operational_decision": {
            "action": operational_action,
            "actionable_geometry_change": operationally_actionable,
            "blockers": blockers,
            "note": (
                "Research output remains visible even when blocked. Operational action defaults to KEEP_CURRENT "
                "until every prospective gate passes."
            ),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Phase 4D.2 actionability written:", OUT_PATH)
    print("Latest state age h:", round(state_age_h, 4), "fresh=", fresh)
    print("Latest observed activity:", latest_activity)
    print("Nearest triggers:", triggers)
    print("Execution gate:", egate)
    print("Geometry execution integration gate:", gexec_gate)
    print("Calibration:", calibration.get("status"), "windows=", cal_windows)
    print("Research action:", research_decision.get("action"), research_confidence)
    print("Operational action:", operational_action, "blockers=", blockers)


if __name__ == "__main__":
    main()
