#!/usr/bin/env python3
"""Run the complete Pionex Phase 4D decision stack in one process.

Order is intentionally fixed:
  1. Phase 4D.1 execution-resolution validation
  2. Phase 4D joint grid-geometry optimisation
  3. Phase 4D.2 activity/actionability gate

The individual builders remain the source of truth and can still be run or
edited independently for debugging. This orchestrator only sequences them,
checks lineage/integration, and writes a compact final decision summary.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANUAL_STATES = ROOT / "data/pionex/manual_grid_states_v1.csv"
EXEC_DIAG = ROOT / "data/diagnostics/pionex_execution_resolution_v1.json"
EXEC_POLICY = ROOT / "data/pionex/execution_resolution_policy_v1.json"
GEOMETRY_DIAG = ROOT / "data/diagnostics/pionex_grid_geometry_optimizer_v1.json"
CALIBRATION_DIAG = ROOT / "data/diagnostics/pionex_grid_calibration_v2.json"
ACTIONABILITY_DIAG = ROOT / "data/diagnostics/pionex_grid_actionability_v1.json"
FULL_DECISION_DIAG = ROOT / "data/diagnostics/pionex_full_decision_v1.json"


def fail(message: str) -> None:
    raise SystemExit(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"Required output missing or empty: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_manual_state() -> dict[str, str]:
    if not MANUAL_STATES.is_file():
        fail(f"Manual state file missing: {MANUAL_STATES.relative_to(ROOT)}")
    with MANUAL_STATES.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), "Manual Pionex state ledger is empty")
    return rows[-1]


def run_builder(label: str, relative_script: str) -> None:
    script = ROOT / relative_script
    require(script.is_file(), f"Missing builder for {label}: {relative_script}")
    print(f"\n=== {label} ===")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def gate_snapshot(gates: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    compact: dict[str, Any] = {}
    pass_values: list[bool] = []
    for name, payload in gates.items():
        if not isinstance(payload, dict):
            compact[name] = payload
            continue
        row: dict[str, Any] = {}
        for key in (
            "passed", "available", "status", "value", "state_age_h",
            "evaluated_windows", "minimum_windows", "active_execution_resolution",
            "expected_execution_resolution", "geometry_execution_replay_resolution",
        ):
            if key in payload:
                row[key] = payload[key]
        compact[name] = row
        if "passed" in payload:
            pass_values.append(payload.get("passed") is True)
    return compact, bool(pass_values) and all(pass_values)


def main() -> None:
    state = latest_manual_state()
    state_utc = str(state.get("captured_at_utc") or "")
    state_local = str(state.get("captured_at_local") or "")
    require(state_utc, "Latest manual state lacks captured_at_utc")

    print("Pionex full Phase 4D pipeline")
    print("Latest manual state:", state_local or state_utc)

    run_builder("Phase 4D.1 - execution resolution", "scripts/build_pionex_execution_resolution_v1.py")
    exec_diag = load_json(EXEC_DIAG)
    policy = load_json(EXEC_POLICY)
    require(exec_diag.get("schema") == "pionex_execution_resolution_v1", "Unexpected Phase 4D.1 diagnostic schema")
    require(policy.get("schema") == "pionex_execution_resolution_policy_v1", "Unexpected Phase 4D.1 policy schema")
    active_resolution = str(policy.get("active_execution_resolution") or "1hour")

    run_builder("Phase 4D - geometry optimisation", "scripts/build_pionex_grid_geometry_optimizer_v2.py")
    geometry = load_json(GEOMETRY_DIAG)
    calibration = load_json(CALIBRATION_DIAG)
    require(geometry.get("schema") == "pionex_grid_geometry_optimizer_v1", "Unexpected Phase 4D geometry schema")
    require(calibration.get("schema") == "pionex_grid_calibration_v2", "Unexpected Phase 4D calibration schema")

    geo_source = geometry.get("source_state") or {}
    geo_integration = geometry.get("execution_resolution_integration") or {}
    geo_resolution = str(geo_integration.get("execution_replay_resolution") or "")

    require(str(geo_source.get("captured_at_utc") or "") == state_utc,
            f"Geometry used state {geo_source.get('captured_at_utc')} instead of latest manual state {state_utc}")
    require(geo_integration.get("execution_resolution_integrated") is True,
            f"Geometry did not integrate Phase 4D.1 policy: {geo_integration}")
    require(geo_resolution == active_resolution,
            f"Geometry replay resolution {geo_resolution} != fresh policy {active_resolution}")
    require(str(calibration.get("execution_replay_resolution") or "") == active_resolution,
            "Calibration execution resolution does not match fresh Phase 4D.1 policy")

    run_builder("Phase 4D.2 - activity and actionability", "scripts/build_pionex_grid_activity_v1.py")
    actionability = load_json(ACTIONABILITY_DIAG)
    require(actionability.get("schema") == "pionex_grid_actionability_v1", "Unexpected Phase 4D.2 schema")

    latest_observed = actionability.get("latest_observed_state") or {}
    gates = actionability.get("prospective_gates") or {}
    operational = actionability.get("operational_decision") or {}
    research = actionability.get("research_geometry") or {}

    require(str(latest_observed.get("captured_at_utc") or "") == state_utc,
            f"Actionability used state {latest_observed.get('captured_at_utc')} instead of latest manual state {state_utc}")
    require(operational.get("action") is not None, "Operational decision is missing")

    exec_gate = gates.get("execution_resolution") or {}
    integration_gate = gates.get("geometry_execution_integration") or {}
    if exec_gate.get("available") is True:
        require(str(exec_gate.get("active_execution_resolution") or active_resolution) == active_resolution,
                "Phase 4D.2 execution gate does not match the fresh Phase 4D.1 policy")
    if integration_gate:
        require(str(integration_gate.get("geometry_execution_replay_resolution") or active_resolution) == active_resolution,
                "Phase 4D.2 geometry integration does not match the fresh Phase 4D.1 policy")

    compact_gates, all_gates_passed = gate_snapshot(gates)
    benches = geometry.get("benchmarks") or {}
    current_geometry = benches.get("current") or {}
    selected_geometry = benches.get("selected") or {}

    integrity = {
        "latest_manual_state_used_by_geometry": str(geo_source.get("captured_at_utc") or "") == state_utc,
        "latest_manual_state_used_by_actionability": str(latest_observed.get("captured_at_utc") or "") == state_utc,
        "geometry_consumed_fresh_execution_policy": geo_resolution == active_resolution,
        "calibration_consumed_fresh_execution_policy": str(calibration.get("execution_replay_resolution") or "") == active_resolution,
        "actionability_execution_gate_matches_policy": (
            exec_gate.get("available") is not True
            or str(exec_gate.get("active_execution_resolution") or active_resolution) == active_resolution
        ),
    }
    require(all(integrity.values()), f"Full-pipeline integrity check failed: {integrity}")

    summary = {
        "schema": "pionex_full_decision_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_DECISION_SUPPORT_ONLY",
        "pipeline_order": [
            "Phase 4D.1 execution resolution",
            "Phase 4D geometry optimisation",
            "Phase 4D.2 actionability",
        ],
        "source_state": {
            "captured_at_utc": state_utc,
            "captured_at_local": state_local,
            "current_price_usdt": latest_observed.get("current_price_usdt"),
            "lower_usdt": latest_observed.get("lower_usdt"),
            "upper_usdt": latest_observed.get("upper_usdt"),
            "grids": latest_observed.get("grids"),
            "observed_quantity_per_grid_eth": latest_observed.get("observed_quantity_per_grid_eth"),
            "state_age_h_at_final_gate": latest_observed.get("state_age_h"),
            "fresh_for_operational_use": latest_observed.get("fresh_for_operational_use"),
        },
        "execution_resolution": {
            "status": policy.get("status"),
            "active_execution_resolution": active_resolution,
            "research_best_resolution": policy.get("research_best_resolution"),
            "windows_evaluated": policy.get("windows_evaluated"),
            "candidate_improvement_vs_1hour_pct": policy.get("candidate_improvement_vs_1hour_pct"),
        },
        "calibration": {
            "status": calibration.get("status"),
            "evaluated_windows": calibration.get("evaluated_windows"),
            "profit_scale_applied": calibration.get("profit_scale_applied"),
            "rounds_scale_applied": calibration.get("rounds_scale_applied"),
        },
        "research_decision": {
            "action": (geometry.get("decision") or {}).get("action"),
            "confidence": (geometry.get("decision") or {}).get("confidence"),
            "current_geometry": current_geometry,
            "selected_geometry": selected_geometry,
            "research_geometry_from_actionability": research,
        },
        "prospective_gates": compact_gates,
        "all_prospective_gates_passed": all_gates_passed,
        "operational_decision": operational,
        "pipeline_integrity": integrity,
    }

    FULL_DECISION_DIAG.parent.mkdir(parents=True, exist_ok=True)
    FULL_DECISION_DIAG.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== PIONEX FULL DECISION ===")
    print(f"Source state: {state_local or state_utc}")
    print("Execution resolution:", active_resolution,
          f"({policy.get('status')}, windows={policy.get('windows_evaluated')})")
    print("Calibration:", calibration.get("status"),
          f"windows={calibration.get('evaluated_windows')}",
          f"profit_scale={calibration.get('profit_scale_applied')}",
          f"rounds_scale={calibration.get('rounds_scale_applied')}")
    print("Research:", (geometry.get("decision") or {}).get("action"),
          f"confidence={(geometry.get('decision') or {}).get('confidence')}")
    print("All prospective gates passed:", all_gates_passed)
    print("Operational decision:", operational.get("action"))
    print("Actionable geometry change:", operational.get("actionable_geometry_change"))
    print("Blockers:", operational.get("blockers"))
    print("Summary written:", FULL_DECISION_DIAG.relative_to(ROOT))


if __name__ == "__main__":
    main()
