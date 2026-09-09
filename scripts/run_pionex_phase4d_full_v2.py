#!/usr/bin/env python3
"""
Full Phase 4D runner v2 — v4.3 paired-cycle safety integration.

Routes geometry through build_pionex_grid_geometry_optimizer_v4.py and activity
through v2. The legacy research output remains visible, but a confirmed
initial-seed grid-profit accounting bias can block operational geometry changes.

Why the blocker exists
----------------------
The legacy simulator can credit an initial seeded-ETH sell with a full interval
grid profit even when no replay buy occurred at the interval lower line. This
structurally favours very wide / low-density grids. Until the corrected paired
buy->sell model is independently calibrated, the safe operational action is
KEEP_CURRENT when that bias is confirmed and the production optimizer is sitting
on its lower grid-count boundary.
"""

from __future__ import annotations

import json

import run_pionex_phase4d_full_v1 as legacy


_original_run_builder = legacy.run_builder
DENSITY_DIAG = legacy.ROOT / "data/diagnostics/pionex_grid_density_sweep_v1.json"
ACTIONABILITY_DIAG = legacy.ROOT / "data/diagnostics/pionex_grid_actionability_v1.json"

SAFETY_BLOCKER = "GRID_PROFIT_INITIAL_SEED_ACCOUNTING_BIAS"


def _run_builder(label: str, relative_script: str) -> None:
    if relative_script == "scripts/build_pionex_grid_geometry_optimizer_v2.py":
        relative_script = "scripts/build_pionex_grid_geometry_optimizer_v4.py"
    elif relative_script == "scripts/build_pionex_grid_activity_v1.py":
        relative_script = "scripts/build_pionex_grid_activity_v2.py"
    _original_run_builder(label, relative_script)


def _load_density() -> dict:
    if not DENSITY_DIAG.is_file():
        raise SystemExit(
            "Grid-density sweep missing after Phase 4D v4.3 geometry run: "
            f"{DENSITY_DIAG.relative_to(legacy.ROOT)}"
        )
    density = json.loads(DENSITY_DIAG.read_text(encoding="utf-8"))
    if density.get("schema") != "pionex_grid_density_sweep_v1":
        raise SystemExit(f"Unexpected density schema: {density.get('schema')}")
    if density.get("version") != "4.3":
        raise SystemExit(f"Expected density v4.3, got {density.get('version')}")
    return density


def _operational_block_payload(op: dict, correction: dict) -> dict:
    out = dict(op or {})
    blockers = list(out.get("blockers") or [])
    if SAFETY_BLOCKER not in blockers:
        blockers.append(SAFETY_BLOCKER)

    previous_action = out.get("action")
    previous_actionable = out.get("actionable_geometry_change")

    out["pre_safety_block_action"] = previous_action
    out["pre_safety_block_actionable_geometry_change"] = previous_actionable
    out["action"] = "KEEP_CURRENT"
    out["actionable_geometry_change"] = False
    out["blockers"] = blockers
    out["paired_cycle_safety_block"] = {
        "applied": True,
        "blocker": SAFETY_BLOCKER,
        "reason": (
            "Legacy grid-profit accounting credits initial seeded sells with a "
            "full grid spread without a replay buy. Production selection is at "
            "the lower grid-count boundary. Keep current geometry until the "
            "paired-cycle model is independently calibrated and promoted."
        ),
        "provisional_paired_expected_profit_champion": correction.get(
            "provisional_paired_expected_profit_champion"
        ),
    }
    out["note"] = (
        "Research output remains visible. Operational geometry changes are "
        "blocked by v4.3 paired-cycle safety validation."
    )
    return out


def _annotate_and_apply_safety() -> None:
    density = _load_density()
    if not legacy.FULL_DECISION_DIAG.is_file():
        raise SystemExit("Full Phase 4D decision missing before density annotation")

    full = json.loads(
        legacy.FULL_DECISION_DIAG.read_text(encoding="utf-8")
    )

    full["grid_density_sweep"] = {
        "schema": density.get("schema"),
        "version": density.get("version"),
        "status": density.get("status"),
        "scope": density.get("scope"),
        "source_state": density.get("source_state"),
        "execution_resolution": density.get("execution_resolution"),
        "method": density.get("method"),
        "live_candidate_count_captured": density.get("live_candidate_count_captured"),
        "live_grid_counts_captured": density.get("live_grid_counts_captured"),
        "current_benchmark": density.get("current_benchmark"),
        "production_selected_geometry": density.get("production_selected_geometry"),
        "headline": density.get("headline"),
        "low_grid_pressure_audit": density.get("low_grid_pressure_audit"),
        "paired_cycle_correction": density.get("paired_cycle_correction"),
        "joint_live_best_by_grid_count": density.get("joint_live_best_by_grid_count"),
        "legacy_current_band_integer_sweep": density.get(
            "legacy_current_band_integer_sweep"
        ),
        "rebalanced_current_band_integer_sweep": density.get(
            "rebalanced_current_band_integer_sweep"
        ),
        "paired_cycle_current_band_integer_sweep": density.get(
            "paired_cycle_current_band_integer_sweep"
        ),
        "operational_effect": "DIAGNOSTIC_ONLY_UNLESS_SAFETY_BLOCK_APPLIED",
    }

    correction = density.get("paired_cycle_correction") or {}
    should_block = correction.get("safety_block_recommended") is True

    if should_block:
        full["operational_decision"] = _operational_block_payload(
            full.get("operational_decision") or {},
            correction,
        )
        full["grid_density_sweep"]["operational_effect"] = (
            "KEEP_CURRENT_SAFETY_BLOCK_APPLIED"
        )

        if ACTIONABILITY_DIAG.is_file():
            actionability = json.loads(
                ACTIONABILITY_DIAG.read_text(encoding="utf-8")
            )
            actionability["operational_decision"] = _operational_block_payload(
                actionability.get("operational_decision") or {},
                correction,
            )
            actionability["paired_cycle_validation"] = {
                "schema": density.get("schema"),
                "version": density.get("version"),
                "seed_profit_bias_detected": correction.get(
                    "seed_profit_bias_detected"
                ),
                "safety_block_recommended": True,
                "blocker": SAFETY_BLOCKER,
            }
            ACTIONABILITY_DIAG.write_text(
                json.dumps(actionability, indent=2) + "\n",
                encoding="utf-8",
            )

    full["paired_cycle_validation"] = {
        "seed_profit_bias_detected": correction.get("seed_profit_bias_detected"),
        "safety_block_recommended": correction.get("safety_block_recommended"),
        "safety_block_applied": should_block,
        "blocker": SAFETY_BLOCKER if should_block else None,
    }

    legacy.FULL_DECISION_DIAG.write_text(
        json.dumps(full, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    legacy.run_builder = _run_builder
    legacy.main()
    _annotate_and_apply_safety()

    density = _load_density()
    correction = density.get("paired_cycle_correction") or {}
    paired = correction.get("provisional_paired_expected_profit_champion") or {}

    print("\n=== GRID-DENSITY / PAIRED-CYCLE SAFETY SUMMARY ===")
    print("Seed-profit bias detected:", correction.get("seed_profit_bias_detected"))
    print(
        "Provisional paired-cycle champion:",
        paired.get("grids"),
        "grids / raw paired profit",
        paired.get("expected_paired_grid_profit_usdt_raw"),
    )
    print("Safety block recommended:", correction.get("safety_block_recommended"))
    print(
        "Final operational action:",
        json.loads(
            legacy.FULL_DECISION_DIAG.read_text(encoding="utf-8")
        ).get("operational_decision", {}).get("action"),
    )


if __name__ == "__main__":
    main()
