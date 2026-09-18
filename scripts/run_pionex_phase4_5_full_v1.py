#!/usr/bin/env python3
"""
Phase 4.5 full runner.

Runs the existing v4.4 full runner, but routes its geometry stage through the
Phase 4.5 joint paired optimiser (v6). After the legacy/actionability pipeline
finishes, this runner embeds the new joint paired result and maintains its own
cross-run stability history.

Safety is one-way:
- Phase 4.5 may block a suspect legacy geometry change.
- Phase 4.5 may NOT automatically promote its own candidate.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import run_pionex_phase4d_full_v2 as v44


ROOT = v44.legacy.ROOT
FULL_DECISION = v44.legacy.FULL_DECISION_DIAG
ACTIONABILITY = v44.ACTIONABILITY_DIAG
JOINT_DIAG = ROOT / "data/diagnostics/pionex_joint_paired_geometry_v1.json"

JOINT_BLOCKER = "PAIRED_JOINT_GEOMETRY_CONFLICT"

HISTORY_LIMIT = 12
STABILITY_MIN_RUNS = 3
STABILITY_MAX_CENTER_OFFSET_SPAN_PP = 1.5
STABILITY_MAX_WIDTH_SPAN_PP = 3.0
STABILITY_MAX_GRID_SPAN = 10

_ORIGINAL_V44_BUILDER = v44._run_builder


def _run_builder(label: str, relative_script: str) -> None:
    if relative_script == "scripts/build_pionex_grid_geometry_optimizer_v2.py":
        v44._original_run_builder(
            label, "scripts/build_pionex_grid_geometry_optimizer_v6.py"
        )
        return
    _ORIGINAL_V44_BUILDER(label, relative_script)


def _previous_joint_history() -> list[dict]:
    if not FULL_DECISION.is_file():
        return []
    try:
        old = json.loads(FULL_DECISION.read_text(encoding="utf-8"))
        return list(old.get("phase4_5_joint_history") or [])
    except Exception:
        return []


def _load_joint() -> dict:
    if not JOINT_DIAG.is_file():
        raise SystemExit(
            "Phase 4.5 joint diagnostic missing: "
            "data/diagnostics/pionex_joint_paired_geometry_v1.json"
        )
    joint = json.loads(JOINT_DIAG.read_text(encoding="utf-8"))
    if joint.get("schema") != "pionex_joint_paired_geometry_v1":
        raise SystemExit(f"Unexpected Phase 4.5 schema: {joint.get('schema')}")
    if joint.get("version") != "4.5":
        raise SystemExit(f"Unexpected Phase 4.5 version: {joint.get('version')}")
    return joint


def _history_entry(joint: dict) -> dict:
    source = joint.get("source_state") or {}
    selection = joint.get("selection") or {}
    chosen = selection.get("robust_joint_selection") or {}
    practical = joint.get("practical_candidate_exact_recompute") or {}

    candidate = practical if (
        practical.get("status") == "ROUNDED_GEOMETRY_RECOMPUTED_AND_ELIGIBLE"
    ) else chosen

    return {
        "captured_at_utc": source.get("captured_at_utc"),
        "generated_at_utc": joint.get("generated_at_utc"),
        "market_price_usdt": source.get("current_price_usdt"),
        "selection_status": selection.get("status"),
        "research_action": selection.get("research_action"),
        "lower_usdt": candidate.get("lower_usdt"),
        "upper_usdt": candidate.get("upper_usdt"),
        "center_usdt": candidate.get("center_usdt"),
        "center_offset_pct_of_market":
            candidate.get("center_offset_pct_of_market"),
        "width_pct_of_market": candidate.get("width_pct_of_market"),
        "grids": candidate.get("grids"),
        "expected_profit_gain_vs_current_pct":
            selection.get("expected_profit_gain_vs_current_pct"),
        "median_profit_gain_vs_current_pct":
            selection.get("median_profit_gain_vs_current_pct"),
        "escape_reduction_vs_current_pp":
            selection.get("escape_reduction_vs_current_pp"),
        "exact_pionex_ui_validation": (
            (candidate.get("execution_feasibility") or {})
            .get("exact_candidate_accepted_by_pionex_ui")
        ),
    }


def _update_history(
    previous: list[dict],
    joint: dict,
) -> tuple[list[dict], dict]:
    entry = _history_entry(joint)
    captured = entry.get("captured_at_utc")

    history = [
        h for h in previous
        if h.get("captured_at_utc")
        and h.get("captured_at_utc") != captured
    ]
    history.append(entry)
    history = history[-HISTORY_LIMIT:]

    usable = [
        h for h in history
        if h.get("selection_status") == "MATERIAL_RESEARCH_CANDIDATE"
        and h.get("center_offset_pct_of_market") is not None
        and h.get("width_pct_of_market") is not None
        and h.get("grids") is not None
    ]
    recent = usable[-STABILITY_MIN_RUNS:]

    centers = [
        float(h["center_offset_pct_of_market"]) for h in recent
    ]
    widths = [float(h["width_pct_of_market"]) for h in recent]
    grids = [int(h["grids"]) for h in recent]

    center_span = max(centers) - min(centers) if centers else None
    width_span = max(widths) - min(widths) if widths else None
    grid_span = max(grids) - min(grids) if grids else None

    stable = bool(
        len(recent) >= STABILITY_MIN_RUNS
        and center_span is not None
        and center_span <= STABILITY_MAX_CENTER_OFFSET_SPAN_PP
        and width_span is not None
        and width_span <= STABILITY_MAX_WIDTH_SPAN_PP
        and grid_span is not None
        and grid_span <= STABILITY_MAX_GRID_SPAN
    )

    return history, {
        "status": (
            "STABLE_RECENT_JOINT_GEOMETRY"
            if stable else (
                "ACCUMULATING_JOINT_EVIDENCE"
                if len(recent) < STABILITY_MIN_RUNS
                else "UNSTABLE_JOINT_GEOMETRY"
            )
        ),
        "stable": stable,
        "minimum_runs": STABILITY_MIN_RUNS,
        "runs_considered": len(recent),
        "center_offset_pct_values": centers,
        "center_offset_span_pp": center_span,
        "width_pct_values": widths,
        "width_span_pp": width_span,
        "grid_values": grids,
        "grid_span": grid_span,
        "median_center_offset_pct": (
            statistics.median(centers) if centers else None
        ),
        "median_width_pct": (
            statistics.median(widths) if widths else None
        ),
        "median_grids": (
            statistics.median(grids) if grids else None
        ),
        "rule": (
            "Require >=3 material Phase 4.5 runs with centre-offset span <=1.5 "
            "percentage points, width span <=3.0 percentage points, and grid "
            "span <=10."
        ),
    }


def _joint_conflict(joint: dict) -> dict:
    comp = joint.get("comparison_to_legacy_optimizer") or {}
    selection = joint.get("selection") or {}

    gain = comp.get("robust_expected_profit_gain_vs_legacy_pct")
    zero = comp.get("legacy_zero_cycle_probability_pct")
    legacy_risk = comp.get("legacy_standard_risk_eligible")

    conflict = bool(
        selection.get("material_research_candidate") is True
        and (
            legacy_risk is False
            or (gain is not None and float(gain) >= 5.0)
            or (zero is not None and float(zero) >= 25.0)
        )
    )

    return {
        "conflict": conflict,
        "material_joint_candidate":
            selection.get("material_research_candidate"),
        "joint_research_action": selection.get("research_action"),
        "robust_expected_profit_gain_vs_legacy_pct": gain,
        "legacy_zero_cycle_probability_pct": zero,
        "legacy_standard_risk_eligible": legacy_risk,
        "reason": (
            "The corrected joint paired-cycle optimiser materially contradicts "
            "the legacy centre/width/grid selection."
        ),
    }


def _block_operational(op: dict, conflict: dict) -> dict:
    out = dict(op or {})
    blockers = list(out.get("blockers") or [])
    if JOINT_BLOCKER not in blockers:
        blockers.append(JOINT_BLOCKER)

    out.setdefault("pre_phase4_5_action", out.get("action"))
    out.setdefault(
        "pre_phase4_5_actionable_geometry_change",
        out.get("actionable_geometry_change"),
    )

    out["action"] = "KEEP_CURRENT"
    out["actionable_geometry_change"] = False
    out["blockers"] = blockers

    safety = dict(out.get("paired_cycle_safety") or {})
    safety[JOINT_BLOCKER] = {
        "applied": True,
        "reason": conflict["reason"],
        "details": conflict,
    }
    out["paired_cycle_safety"] = safety
    out["note"] = (
        "Phase 4.5 is research-only. Corrected joint paired evidence may block "
        "a suspect legacy change but cannot automatically promote a new bot."
    )
    return out


def _annotate(
    joint: dict,
    history: list[dict],
    stability: dict,
) -> None:
    if not FULL_DECISION.is_file():
        raise SystemExit("Full decision missing before Phase 4.5 annotation")

    full = json.loads(FULL_DECISION.read_text(encoding="utf-8"))
    conflict = _joint_conflict(joint)

    selection = joint.get("selection") or {}
    practical = joint.get("practical_candidate_exact_recompute") or {}
    candidate = practical if (
        practical.get("status") == "ROUNDED_GEOMETRY_RECOMPUTED_AND_ELIGIBLE"
    ) else (selection.get("robust_joint_selection") or {})

    exact_ui = (
        (candidate.get("execution_feasibility") or {})
        .get("exact_candidate_accepted_by_pionex_ui") is True
    )

    if (
        stability.get("stable") is True
        and selection.get("material_research_candidate") is True
        and exact_ui
    ):
        promotion_status = "READY_FOR_MANUAL_PHASE4_5_PROMOTION_REVIEW"
    elif (
        stability.get("stable") is True
        and selection.get("material_research_candidate") is True
    ):
        promotion_status = "AWAITING_EXACT_PIONEX_LIVE_SETUP_VALIDATION"
    elif selection.get("material_research_candidate") is True:
        promotion_status = "ACCUMULATING_JOINT_STABILITY_EVIDENCE"
    else:
        promotion_status = "NO_MATERIAL_JOINT_CHANGE"

    compact_joint = {
        "schema": joint.get("schema"),
        "version": joint.get("version"),
        "status": joint.get("status"),
        "source_state": joint.get("source_state"),
        "execution_resolution": joint.get("execution_resolution"),
        "paired_calibration": joint.get("paired_calibration"),
        "calibration_quality": joint.get("calibration_quality"),
        "current_status_quo_paired_benchmark":
            joint.get("current_status_quo_paired_benchmark"),
        "current_geometry_fresh_rebalance_counterfactual":
            joint.get("current_geometry_fresh_rebalance_counterfactual"),
        "risk_policy": joint.get("risk_policy"),
        "search": joint.get("search"),
        "selection": selection,
        "practical_candidate_exact_recompute": practical,
        "practical_candidate_status":
            joint.get("practical_candidate_status"),
        "comparison_to_legacy_optimizer":
            joint.get("comparison_to_legacy_optimizer"),
        "candidate_counts": joint.get("candidate_counts"),
        "cross_run_stability": stability,
        "promotion_status": promotion_status,
        "operational_effect":
            "SAFETY_BLOCK_ONLY_NO_AUTOMATIC_JOINT_PROMOTION",
    }

    full["phase4_5_joint_paired_geometry"] = compact_joint
    full["phase4_5_joint_history"] = history

    if conflict.get("conflict") is True:
        full["operational_decision"] = _block_operational(
            full.get("operational_decision") or {},
            conflict,
        )

    validation = dict(full.get("paired_cycle_validation") or {})
    validation["phase4_5"] = {
        "joint_conflict": conflict,
        "cross_run_stability": stability,
        "promotion_status": promotion_status,
        "exact_pionex_ui_validation": exact_ui,
        "automatic_joint_promotion_allowed": False,
    }
    full["paired_cycle_validation"] = validation

    FULL_DECISION.write_text(
        json.dumps(full, indent=2) + "\n",
        encoding="utf-8",
    )

    if ACTIONABILITY.is_file():
        actionability = json.loads(
            ACTIONABILITY.read_text(encoding="utf-8")
        )
        actionability["phase4_5_joint_paired_geometry"] = {
            "selection": selection,
            "practical_candidate_exact_recompute": practical,
            "cross_run_stability": stability,
            "promotion_status": promotion_status,
            "joint_conflict": conflict,
            "automatic_joint_promotion_allowed": False,
        }
        actionability["operational_decision"] = full.get(
            "operational_decision"
        )
        ACTIONABILITY.write_text(
            json.dumps(actionability, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    previous = _previous_joint_history()

    # v4.4 main refers to its module-global _run_builder at runtime, so replacing
    # that function here safely changes only the geometry builder target.
    v44._run_builder = _run_builder
    v44.main()

    joint = _load_joint()
    history, stability = _update_history(previous, joint)
    _annotate(joint, history, stability)

    final = json.loads(FULL_DECISION.read_text(encoding="utf-8"))
    phase45 = final.get("phase4_5_joint_paired_geometry") or {}
    selection = phase45.get("selection") or {}
    chosen = selection.get("robust_joint_selection") or {}

    print("\n=== PHASE 4.5 FULL DECISION SUMMARY ===")
    print("Selection status:", selection.get("status"))
    print("Research action:", selection.get("research_action"))
    print(
        "Joint candidate:",
        chosen.get("lower_usdt"),
        "to",
        chosen.get("upper_usdt"),
        "/",
        chosen.get("grids"),
        "grids",
    )
    print("Joint stability:", stability.get("status"))
    print("Promotion:", phase45.get("promotion_status"))
    print(
        "Operational:",
        (final.get("operational_decision") or {}).get("action"),
    )
    print(
        "Blockers:",
        (final.get("operational_decision") or {}).get("blockers"),
    )


if __name__ == "__main__":
    main()
