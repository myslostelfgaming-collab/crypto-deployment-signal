#!/usr/bin/env python3
"""
Full Phase 4D runner v2 — v4.4 paired-density promotion-readiness + safety.

Routes geometry through build_pionex_grid_geometry_optimizer_v5.py, which runs
v4.3 unchanged and adds v4.4 paired-specific calibration and a robust density
plateau selector.

Safety principle
----------------
v4.4 still does NOT automatically promote a new grid density. However, when the
legacy production optimizer is pinned to its minimum grid-count boundary and
the independently calibrated paired model materially contradicts that choice,
the legacy geometry change is blocked and the operational action becomes
KEEP_CURRENT.

This is a one-way safety gate: corrected paired evidence may block a suspect
legacy change, but it may not itself edit/promote a new live Pionex geometry.
"""

from __future__ import annotations

import json
import statistics

import run_pionex_phase4d_full_v1 as legacy


_original_run_builder = legacy.run_builder
DENSITY_DIAG = legacy.ROOT / "data/diagnostics/pionex_grid_density_sweep_v1.json"
ACTIONABILITY_DIAG = legacy.ROOT / "data/diagnostics/pionex_grid_actionability_v1.json"

SEED_BIAS_BLOCKER = "GRID_PROFIT_INITIAL_SEED_ACCOUNTING_BIAS"
PAIRED_CONFLICT_BLOCKER = "PAIRED_CALIBRATED_DENSITY_CONFLICT"

HISTORY_LIMIT = 12
STABILITY_MIN_RUNS = 3
STABILITY_MAX_SELECTED_GRID_SPAN = 6


def _run_builder(label: str, relative_script: str) -> None:
    if relative_script == "scripts/build_pionex_grid_geometry_optimizer_v2.py":
        relative_script = "scripts/build_pionex_grid_geometry_optimizer_v5.py"
    elif relative_script == "scripts/build_pionex_grid_activity_v1.py":
        relative_script = "scripts/build_pionex_grid_activity_v2.py"
    _original_run_builder(label, relative_script)


def _load_previous_history() -> list[dict]:
    if not legacy.FULL_DECISION_DIAG.is_file():
        return []
    try:
        old = json.loads(legacy.FULL_DECISION_DIAG.read_text(encoding="utf-8"))
        return list(old.get("paired_density_history") or [])
    except Exception:
        return []


def _load_density() -> dict:
    if not DENSITY_DIAG.is_file():
        raise SystemExit(
            "Grid-density sweep missing after Phase 4D v4.4 geometry run: "
            f"{DENSITY_DIAG.relative_to(legacy.ROOT)}"
        )
    density = json.loads(DENSITY_DIAG.read_text(encoding="utf-8"))
    if density.get("schema") != "pionex_grid_density_sweep_v1":
        raise SystemExit(f"Unexpected density schema: {density.get('schema')}")
    if density.get("version") != "4.4":
        raise SystemExit(f"Expected density v4.4, got {density.get('version')}")
    return density


def _history_entry(density: dict) -> dict:
    promo = density.get("promotion_readiness") or {}
    plateau = promo.get("plateau_selector") or {}
    selected = plateau.get("robust_plateau_selection") or {}
    peak = plateau.get("expected_profit_champion") or {}
    cal = promo.get("paired_calibration") or {}
    source = density.get("source_state") or {}

    return {
        "captured_at_utc": source.get("captured_at_utc"),
        "generated_at_utc": density.get("generated_at_utc"),
        "market_price_usdt": source.get("current_price_usdt"),
        "live_lower_usdt": (density.get("current_benchmark") or {}).get("lower_usdt"),
        "live_upper_usdt": (density.get("current_benchmark") or {}).get("upper_usdt"),
        "live_grids": (density.get("current_benchmark") or {}).get("grids"),
        "expected_profit_peak_grids": peak.get("grids"),
        "robust_plateau_selected_grids": selected.get("grids"),
        "plateau_grid_min": plateau.get("plateau_grid_min"),
        "plateau_grid_max": plateau.get("plateau_grid_max"),
        "selected_expected_profit_gain_vs_current_pct": plateau.get(
            "selected_expected_profit_gain_vs_current_pct"
        ),
        "selected_median_profit_gain_vs_current_pct": plateau.get(
            "selected_median_profit_gain_vs_current_pct"
        ),
        "paired_calibration_status": cal.get("status"),
        "paired_calibration_ready": cal.get("calibration_ready"),
        "promotion_gate": plateau.get("promotion_gate"),
    }


def _update_history(previous: list[dict], density: dict) -> tuple[list[dict], dict]:
    entry = _history_entry(density)
    captured = entry.get("captured_at_utc")

    history = [
        h for h in previous
        if h.get("captured_at_utc") and h.get("captured_at_utc") != captured
    ]
    history.append(entry)
    history = history[-HISTORY_LIMIT:]

    usable = [
        h for h in history
        if h.get("robust_plateau_selected_grids") is not None
        and h.get("paired_calibration_ready") is True
    ]
    recent = usable[-STABILITY_MIN_RUNS:]

    selected_values = [
        int(h["robust_plateau_selected_grids"]) for h in recent
    ]
    peak_values = [
        int(h["expected_profit_peak_grids"]) for h in recent
        if h.get("expected_profit_peak_grids") is not None
    ]

    selected_span = (
        max(selected_values) - min(selected_values)
        if selected_values else None
    )
    stable = (
        len(recent) >= STABILITY_MIN_RUNS
        and selected_span is not None
        and selected_span <= STABILITY_MAX_SELECTED_GRID_SPAN
    )

    overlap_min = None
    overlap_max = None
    if recent:
        mins = [int(h["plateau_grid_min"]) for h in recent if h.get("plateau_grid_min") is not None]
        maxs = [int(h["plateau_grid_max"]) for h in recent if h.get("plateau_grid_max") is not None]
        if mins and maxs:
            overlap_min = max(mins)
            overlap_max = min(maxs)

    overlap_exists = (
        overlap_min is not None
        and overlap_max is not None
        and overlap_min <= overlap_max
    )

    stability = {
        "status": (
            "STABLE_RECENT_PLATEAU"
            if stable and overlap_exists
            else (
                "ACCUMULATING_EVIDENCE"
                if len(recent) < STABILITY_MIN_RUNS
                else "UNSTABLE_OR_NONOVERLAPPING_PLATEAU"
            )
        ),
        "stable": bool(stable and overlap_exists),
        "minimum_runs": STABILITY_MIN_RUNS,
        "runs_considered": len(recent),
        "selected_grid_values": selected_values,
        "selected_grid_span": selected_span,
        "selected_grid_median": (
            statistics.median(selected_values) if selected_values else None
        ),
        "expected_peak_grid_values": peak_values,
        "plateau_overlap_grid_min": overlap_min,
        "plateau_overlap_grid_max": overlap_max,
        "plateau_overlap_exists": overlap_exists,
        "rule": (
            "Require at least 3 paired-calibrated runs; robust selected grid "
            "counts must span <=6 grids and their 95%-of-peak plateaus must overlap."
        ),
    }
    return history, stability


def _paired_conflict_evidence(density: dict) -> dict:
    promo = density.get("promotion_readiness") or {}
    cal = promo.get("paired_calibration") or {}
    plateau = promo.get("plateau_selector") or {}
    selected = plateau.get("robust_plateau_selection") or {}

    low = density.get("low_grid_pressure_audit") or {}
    correction = density.get("paired_cycle_correction") or {}
    production_paired = correction.get("production_paired_current_band") or {}
    production_grid = correction.get("production_selected_grid_count")

    selected_grids = selected.get("grids")
    production_zero = production_paired.get("p_zero_paired_rounds_pct")

    selected_profit = selected.get("expected_paired_grid_profit_usdt_calibrated")
    production_raw = production_paired.get("expected_paired_grid_profit_usdt_raw")
    profit_scale = float(cal.get("profit_scale_applied") or 1.0)
    production_profit_calibrated = (
        float(production_raw) * profit_scale
        if production_raw is not None else None
    )

    gain_pct = None
    if (
        selected_profit is not None
        and production_profit_calibrated is not None
        and production_profit_calibrated > 1e-12
    ):
        gain_pct = (
            float(selected_profit) / production_profit_calibrated - 1.0
        ) * 100.0

    conflict = bool(
        cal.get("calibration_ready") is True
        and low.get("production_selected_at_min_boundary") is True
        and production_grid is not None
        and selected_grids is not None
        and int(selected_grids) >= int(production_grid) + 5
        and (
            (production_zero is not None and float(production_zero) >= 25.0)
            or (gain_pct is not None and gain_pct >= 5.0)
        )
    )

    return {
        "conflict": conflict,
        "paired_calibration_ready": cal.get("calibration_ready"),
        "production_selected_at_min_boundary": low.get(
            "production_selected_at_min_boundary"
        ),
        "production_selected_grids": production_grid,
        "paired_robust_plateau_selected_grids": selected_grids,
        "production_p_zero_paired_rounds_pct": production_zero,
        "paired_selected_expected_profit_gain_vs_production_low_grid_pct": (
            round(gain_pct, 4) if gain_pct is not None else None
        ),
        "reason": (
            "The calibrated paired-cycle model materially contradicts a legacy "
            "selection pinned to the minimum grid-count boundary."
        ),
    }


def _operational_block_payload(
    op: dict,
    blocker: str,
    reason: str,
    details: dict,
) -> dict:
    out = dict(op or {})
    blockers = list(out.get("blockers") or [])
    if blocker not in blockers:
        blockers.append(blocker)

    # Preserve the first pre-safety action if multiple independent blockers apply.
    out.setdefault("pre_safety_block_action", out.get("action"))
    out.setdefault(
        "pre_safety_block_actionable_geometry_change",
        out.get("actionable_geometry_change"),
    )

    out["action"] = "KEEP_CURRENT"
    out["actionable_geometry_change"] = False
    out["blockers"] = blockers

    safety = dict(out.get("paired_cycle_safety") or {})
    safety[blocker] = {
        "applied": True,
        "reason": reason,
        "details": details,
    }
    out["paired_cycle_safety"] = safety
    out["note"] = (
        "Research output remains visible. Suspect legacy geometry changes are "
        "blocked; v4.4 does not automatically promote a paired-density candidate."
    )
    return out


def _annotate_and_apply_safety(
    density: dict,
    history: list[dict],
    stability: dict,
) -> None:
    if not legacy.FULL_DECISION_DIAG.is_file():
        raise SystemExit("Full Phase 4D decision missing before v4.4 annotation")

    promo = density.get("promotion_readiness") or {}
    promo["cross_run_stability"] = stability

    plateau = promo.get("plateau_selector") or {}
    selected = plateau.get("robust_plateau_selection") or {}
    selected_feas = selected.get("execution_feasibility") or {}

    if (
        promo.get("paired_calibration", {}).get("calibration_ready") is True
        and stability.get("stable") is True
        and selected_feas.get("status") == "PASS_EXACT_PIONEX_UI_VALIDATION"
    ):
        promo["promotion_status"] = "READY_FOR_MANUAL_PROMOTION_REVIEW"
    elif (
        promo.get("paired_calibration", {}).get("calibration_ready") is True
        and stability.get("stable") is True
    ):
        promo["promotion_status"] = "AWAITING_EXACT_PIONEX_LIVE_SETUP_VALIDATION"
    elif promo.get("paired_calibration", {}).get("calibration_ready") is True:
        promo["promotion_status"] = "ACCUMULATING_CROSS_RUN_STABILITY_EVIDENCE"
    else:
        promo["promotion_status"] = "PAIRED_CALIBRATION_NOT_READY"

    density["promotion_readiness"] = promo
    DENSITY_DIAG.write_text(json.dumps(density, indent=2) + "\n", encoding="utf-8")

    full = json.loads(legacy.FULL_DECISION_DIAG.read_text(encoding="utf-8"))

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
        "promotion_readiness": density.get("promotion_readiness"),
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
        "operational_effect": "SAFETY_BLOCK_ONLY_NO_AUTOMATIC_PAIRED_PROMOTION",
    }
    full["paired_density_history"] = history

    correction = density.get("paired_cycle_correction") or {}
    seed_should_block = correction.get("safety_block_recommended") is True
    conflict = _paired_conflict_evidence(density)

    op = full.get("operational_decision") or {}

    if seed_should_block:
        op = _operational_block_payload(
            op,
            SEED_BIAS_BLOCKER,
            (
                "Legacy grid-profit accounting shows confirmed initial seeded-sell "
                "full-spread bias at a lower-boundary production selection."
            ),
            {
                "seed_profit_bias_detected": correction.get(
                    "seed_profit_bias_detected"
                ),
                "provisional_paired_expected_profit_champion": correction.get(
                    "provisional_paired_expected_profit_champion"
                ),
            },
        )

    if conflict.get("conflict") is True:
        op = _operational_block_payload(
            op,
            PAIRED_CONFLICT_BLOCKER,
            conflict["reason"],
            conflict,
        )

    full["operational_decision"] = op

    blockers = []
    if seed_should_block:
        blockers.append(SEED_BIAS_BLOCKER)
    if conflict.get("conflict") is True:
        blockers.append(PAIRED_CONFLICT_BLOCKER)

    full["paired_cycle_validation"] = {
        "version": "4.4",
        "paired_calibration_ready": promo.get("paired_calibration", {}).get(
            "calibration_ready"
        ),
        "cross_run_stability": stability,
        "promotion_status": promo.get("promotion_status"),
        "seed_profit_bias_detected": correction.get("seed_profit_bias_detected"),
        "paired_density_conflict": conflict,
        "safety_block_applied": bool(blockers),
        "blockers": blockers,
        "automatic_paired_promotion_allowed": False,
    }

    if ACTIONABILITY_DIAG.is_file():
        actionability = json.loads(ACTIONABILITY_DIAG.read_text(encoding="utf-8"))
        actionability["paired_cycle_validation"] = full["paired_cycle_validation"]
        if blockers:
            actionability["operational_decision"] = op
        ACTIONABILITY_DIAG.write_text(
            json.dumps(actionability, indent=2) + "\n",
            encoding="utf-8",
        )

    legacy.FULL_DECISION_DIAG.write_text(
        json.dumps(full, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    previous_history = _load_previous_history()

    legacy.run_builder = _run_builder
    legacy.main()

    density = _load_density()
    history, stability = _update_history(previous_history, density)
    _annotate_and_apply_safety(density, history, stability)

    final = json.loads(legacy.FULL_DECISION_DIAG.read_text(encoding="utf-8"))
    promo = (final.get("grid_density_sweep") or {}).get("promotion_readiness") or {}
    plateau = promo.get("plateau_selector") or {}
    selected = plateau.get("robust_plateau_selection") or {}

    print("\n=== GRID-DENSITY / PAIRED-CYCLE v4.4 SUMMARY ===")
    print(
        "Paired calibration:",
        (promo.get("paired_calibration") or {}).get("status"),
    )
    print(
        "Robust plateau selection:",
        selected.get("grids"),
        "grids",
    )
    print(
        "Cross-run stability:",
        (promo.get("cross_run_stability") or {}).get("status"),
    )
    print("Promotion status:", promo.get("promotion_status"))
    print(
        "Final operational action:",
        (final.get("operational_decision") or {}).get("action"),
    )
    print(
        "Safety blockers:",
        (final.get("paired_cycle_validation") or {}).get("blockers"),
    )


if __name__ == "__main__":
    main()
