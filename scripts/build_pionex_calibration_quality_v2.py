#!/usr/bin/env python3
"""Add Phase 4.5 calibration-v2 evidence to current diagnostics.

Backward-compatible by design:
- legacy calibration fields are retained unchanged;
- existing research/operational actions are not rewritten;
- new confidence/evidence fields are additive only;
- no Pionex endpoint is called.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pionex_calibration_quality_v2 as q

ROOT = Path(__file__).resolve().parents[1]
DENSITY = ROOT / "data/diagnostics/pionex_grid_density_sweep_v1.json"
JOINT = ROOT / "data/diagnostics/pionex_joint_paired_geometry_v1.json"
FULL = ROOT / "data/diagnostics/pionex_full_decision_v1.json"
ACTIONABILITY = ROOT / "data/diagnostics/pionex_grid_actionability_v1.json"
OUT = ROOT / "data/diagnostics/pionex_calibration_quality_v2.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _repo_context():
    try:
        import build_pionex_grid_geometry_optimizer_v1 as base
    except Exception as exc:
        return None, None, f"base import unavailable: {exc}"
    try:
        states = base.phase4c.read_manual_states()
        audit = q.audit_state_pair_exclusions(states, min_h=base.CAL_MIN_H, max_h=base.CAL_MAX_H)
    except Exception as exc:
        audit = None
        audit_note = f"state audit unavailable: {exc}"
    else:
        audit_note = None

    try:
        master = base.sim.build_master_candles()
        features = base.sim.load_eth_features()

        def enrich(window: dict[str, Any]) -> dict[str, Any]:
            raw = str(window.get("start_utc") or "")
            try:
                ts = int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
            except Exception:
                return {}
            feat = base.nearest_feature(features, ts)
            if feat is None:
                return {}
            out = dict(base.path_shape_metrics(master, feat) or {})
            for key in ("atr14_pct", "atr_pct", "atr14_percent"):
                if feat.get(key) not in (None, ""):
                    out["atr14_pct"] = float(feat[key])
                    break
            return out
    except Exception:
        enrich = None
    return audit, enrich, audit_note


def _estimate(calibration: dict[str, Any], candidate: dict[str, Any], source: dict[str, Any], enricher) -> dict[str, Any] | None:
    if not candidate or candidate.get("lower_usdt") is None or candidate.get("upper_usdt") is None or candidate.get("grids") is None:
        return None
    target = {
        "start_utc": source.get("captured_at_utc"),
        "lower_usdt": candidate.get("lower_usdt"),
        "upper_usdt": candidate.get("upper_usdt"),
        "grids": candidate.get("grids"),
        "current_price_usdt": source.get("current_price_usdt"),
        "quantity_per_grid_eth": candidate.get("quantity_per_grid_eth_est") or candidate.get("quantity_per_grid_eth"),
    }
    if enricher:
        target.update(enricher(target) or {})
    factors = q.target_factors(calibration, target)
    raw_profit = candidate.get("expected_paired_grid_profit_usdt_raw")
    raw_rounds = candidate.get("expected_paired_rounds_raw")
    out = {
        "method": factors.get("method"),
        "cohort": factors.get("cohort"),
        "rounds_scale": factors.get("rounds_scale"),
        "productivity_scale": factors.get("productivity_scale"),
        "profit_scale": factors.get("profit_scale"),
    }
    if raw_profit is not None:
        out["expected_paired_grid_profit_usdt_v2"] = round(float(raw_profit) * float(factors.get("profit_scale") or 1.0), 6)
    if raw_rounds is not None:
        out["expected_paired_rounds_v2"] = round(float(raw_rounds) * float(factors.get("rounds_scale") or 1.0), 6)
    return out


def main() -> None:
    density = _load(DENSITY)
    joint = _load(JOINT)
    legacy = ((density.get("promotion_readiness") or {}).get("paired_calibration") or {})
    if not legacy.get("windows"):
        raise SystemExit("paired calibration windows missing")

    audit, enricher, audit_note = _repo_context()
    exclusions = (audit or {}).get("geometry_change_exclusions") or []
    upgraded = q.upgrade_calibration(
        legacy,
        exclusions=exclusions,
        feature_enricher=enricher,
        scale_bounds=tuple(legacy.get("scale_bounds") or (0.35, 1.50)),
    )
    v2 = upgraded["calibration_v2"]
    v2["state_pair_audit"] = audit or {"status": "UNAVAILABLE", "note": audit_note}
    _write(OUT, v2)

    quality = upgraded["calibration_quality_v2"]
    source = joint.get("source_state") or {}
    confidence = {
        "forecast_confidence": quality.get("forecast_confidence"),
        "directional_signal_reliable": quality.get("directional_signal_reliable"),
        "magnitude_estimate_reliable": quality.get("magnitude_estimate_reliable"),
        "calibration_status": quality.get("status"),
        "policy": "CALIBRATION_V2_MAY_DOWNGRADE_CONFIDENCE_BUT_CANNOT_PROMOTE_A_LIVE_CHANGE",
    }

    # Preserve legacy fields and add v2 evidence alongside them.
    joint["calibration_quality_v2"] = quality
    joint["forecast_calibration_confidence"] = confidence
    joint["calibration_v2_report_path"] = "data/diagnostics/pionex_calibration_quality_v2.json"

    benchmark = joint.get("current_status_quo_paired_benchmark") or {}
    estimate = _estimate(upgraded, benchmark, source, enricher)
    if estimate:
        benchmark["calibration_v2_estimate"] = estimate
        joint["current_status_quo_paired_benchmark"] = benchmark

    selection = joint.get("selection") or {}
    chosen = selection.get("robust_joint_selection") or {}
    estimate = _estimate(upgraded, chosen, source, enricher)
    if estimate:
        chosen["calibration_v2_estimate"] = estimate
        selection["robust_joint_selection"] = chosen
        joint["selection"] = selection

    practical = joint.get("practical_candidate_exact_recompute") or {}
    estimate = _estimate(upgraded, practical, source, enricher)
    if estimate:
        practical["calibration_v2_estimate"] = estimate
        joint["practical_candidate_exact_recompute"] = practical
    _write(JOINT, joint)

    if FULL.is_file():
        full = _load(FULL)
        full["calibration_quality_v2"] = quality
        full["forecast_calibration_confidence"] = confidence
        phase45 = full.get("phase4_5_joint_paired_geometry") or {}
        phase45["calibration_quality_v2"] = quality
        phase45["forecast_calibration_confidence"] = confidence
        phase45["calibration_v2_report_path"] = "data/diagnostics/pionex_calibration_quality_v2.json"
        full["phase4_5_joint_paired_geometry"] = phase45
        validation = dict(full.get("paired_cycle_validation") or {})
        p45 = dict(validation.get("phase4_5") or {})
        p45["calibration_v2"] = {
            "status": quality.get("status"),
            "forecast_confidence": quality.get("forecast_confidence"),
            "directional_signal_reliable": quality.get("directional_signal_reliable"),
            "magnitude_estimate_reliable": quality.get("magnitude_estimate_reliable"),
            "automatic_joint_promotion_allowed": False,
        }
        validation["phase4_5"] = p45
        full["paired_cycle_validation"] = validation
        _write(FULL, full)

    if ACTIONABILITY.is_file():
        actionability = _load(ACTIONABILITY)
        actionability["calibration_quality_v2"] = quality
        actionability["forecast_calibration_confidence"] = confidence
        _write(ACTIONABILITY, actionability)

    print("Calibration v2 status:", quality.get("status"))
    print("Forecast confidence:", confidence)
    print("Report:", OUT)


if __name__ == "__main__":
    main()
