#!/usr/bin/env python3
"""
Full Phase 4D runner v2.

Routes the geometry stage through v4.2, which retains the validated v3.1
live-price/out-of-grid logic and adds diagnostic-only grid-density / sizing-bias
analysis. Activity remains v2.

The density diagnostic does NOT override operational actionability.

Persistence design
------------------
The automated workflow already persists pionex_full_decision_v1.json. To avoid
requiring a workflow YAML change, this runner embeds the useful density sweep
inside the full-decision JSON after the legacy pipeline completes. The standalone
pionex_grid_density_sweep_v1.json is still written locally/artifact-ready.
"""

from __future__ import annotations

import json

import run_pionex_phase4d_full_v1 as legacy


_original_run_builder = legacy.run_builder
DENSITY_DIAG = legacy.ROOT / "data/diagnostics/pionex_grid_density_sweep_v1.json"


def _run_builder(label: str, relative_script: str) -> None:
    if relative_script == "scripts/build_pionex_grid_geometry_optimizer_v2.py":
        relative_script = "scripts/build_pionex_grid_geometry_optimizer_v4.py"
    elif relative_script == "scripts/build_pionex_grid_activity_v1.py":
        relative_script = "scripts/build_pionex_grid_activity_v2.py"
    _original_run_builder(label, relative_script)


def _annotate_full_decision() -> None:
    if not DENSITY_DIAG.is_file():
        raise SystemExit(
            "Grid-density sweep missing after Phase 4D v4.2 geometry run: "
            f"{DENSITY_DIAG.relative_to(legacy.ROOT)}"
        )
    if not legacy.FULL_DECISION_DIAG.is_file():
        raise SystemExit("Full Phase 4D decision missing before density annotation")

    density = json.loads(DENSITY_DIAG.read_text(encoding="utf-8"))
    full = json.loads(
        legacy.FULL_DECISION_DIAG.read_text(encoding="utf-8")
    )

    if density.get("schema") != "pionex_grid_density_sweep_v1":
        raise SystemExit(
            f"Unexpected density schema: {density.get('schema')}"
        )

    # Persist the full useful diagnostic through a file the existing automated
    # workflow already stages and commits. This intentionally avoids making a
    # workflow-YAML edit a prerequisite for the research layer.
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
        "joint_live_best_by_grid_count": density.get("joint_live_best_by_grid_count"),
        "legacy_current_band_integer_sweep": density.get(
            "legacy_current_band_integer_sweep"
        ),
        "rebalanced_current_band_integer_sweep": density.get(
            "rebalanced_current_band_integer_sweep"
        ),
        "operational_effect": "NONE_DIAGNOSTIC_ONLY",
    }

    legacy.FULL_DECISION_DIAG.write_text(
        json.dumps(full, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    legacy.run_builder = _run_builder
    legacy.main()
    _annotate_full_decision()

    density = json.loads(DENSITY_DIAG.read_text(encoding="utf-8"))
    headline = density.get("headline") or {}
    audit = density.get("low_grid_pressure_audit") or {}
    legacy_champ = headline.get("legacy_current_band_expected_profit_champion") or {}
    reb_champ = headline.get("rebalanced_current_band_expected_profit_champion") or {}

    print("\n=== GRID-DENSITY SWEET-SPOT / BIAS SUMMARY ===")
    print(
        "Legacy current-band champion:",
        legacy_champ.get("grids"),
        "grids /",
        legacy_champ.get("expected_grid_profit_usdt"),
        "USDT expected 24h grid profit",
    )
    print(
        "Rebalanced current-band champion:",
        reb_champ.get("grids"),
        "grids /",
        reb_champ.get("expected_grid_profit_usdt"),
        "USDT expected 24h grid profit",
    )
    print("Diagnosis codes:", audit.get("diagnosis_codes"))
    print("Production action unchanged:", True)


if __name__ == "__main__":
    main()
