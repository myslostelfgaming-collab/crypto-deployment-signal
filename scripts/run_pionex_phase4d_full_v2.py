#!/usr/bin/env python3
"""
Full Phase 4D runner v2.

Reuses the existing validated full-pipeline integrity checks, substituting only:
- geometry v4 for live-price/out-of-grid recovery + density sweet-spot capture;
- activity v2 for correct above/below-grid waiting-trigger state.

The density sweep remains diagnostic-only and does not override the operational
actionability decision.
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
            "Grid-density sweep missing after Phase 4D v4 geometry run: "
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

    full["grid_density_sweep"] = {
        "schema": density.get("schema"),
        "status": density.get("status"),
        "source_state": density.get("source_state"),
        "execution_resolution": density.get("execution_resolution"),
        "captured_candidate_count": density.get("captured_candidate_count"),
        "captured_grid_counts": density.get("captured_grid_counts"),
        "headline": density.get("headline"),
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
    expected = headline.get("expected_profit_champion") or {}
    band = headline.get("current_band_expected_profit_champion") or {}
    fifty = headline.get("fifty_plus_rounds") or {}

    print("\n=== GRID-DENSITY SWEET-SPOT SUMMARY ===")
    print(
        "Expected-profit champion:",
        expected.get("grids"),
        "grids /",
        expected.get("expected_grid_profit_usdt"),
        "USDT expected 24h grid profit",
    )
    print(
        "Current-band champion:",
        band.get("grids"),
        "grids /",
        band.get("expected_grid_profit_usdt"),
        "USDT expected 24h grid profit",
    )
    print(
        "Maximum P(raw rounds >= 50):",
        fifty.get("maximum_p_raw_rounds_ge_50_pct"),
        "%",
    )
    print("Operational override: False")


if __name__ == "__main__":
    main()
