#!/usr/bin/env python3
"""
Phase 4D v4 — grid-density sweet-spot diagnostic adapter.

This wraps the validated Phase 4D v3.1 live-price / edge-recovery adapter and
captures every geometry evaluation already performed by that run. No second
candidate sweep is required.

Purpose
-------
The Phase 4D optimizer already searches 10..80 grids, but its persisted output
only exposes the selected geometry and Pareto frontier. That makes it difficult
to answer the operationally important question:

    At what grid density does frequency x net profit per completed round
    maximise robust 24h grid profit?

v4 persists the full density evidence needed to answer that question, including:

* best feasible geometry for each grid count,
* best standard-risk-eligible geometry for each grid count,
* an apples-to-apples density curve on the CURRENT live price band,
* raw and calibrated round distributions,
* P(0 rounds), P(1+), P(2+), P(5+), P(10+), P(25+), P(50+),
* expected/median/max rounds,
* expected grid profit and median grid profit,
* profit per expected round,
* boundary-optimum warnings.

Safety
------
Diagnostic / decision-support only.
No Pionex write endpoint is used.
The existing Phase 4D actionability decision is NOT replaced or overridden.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_pionex_grid_geometry_optimizer_v1 as base
import build_pionex_grid_geometry_optimizer_v3 as v3

OUT_PATH = Path("data/diagnostics/pionex_grid_density_sweep_v1.json")
ROUND_THRESHOLDS = (1, 2, 5, 10, 25, 50)

_CAPTURED: dict[tuple[float, float, int], dict[str, Any]] = {}
_DISTRIBUTIONS: dict[tuple[float, float, int], dict[str, list[float]]] = {}

_ORIGINAL_INTEGRATION_MAIN = v3.integration.main


def _key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        round(float(row["lower_usdt"]), 4),
        round(float(row["upper_usdt"]), 4),
        int(row["grids"]),
    )


def _prob_ge(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return round(sum(float(v) >= threshold for v in values) / len(values) * 100.0, 4)


def _prob_zero(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(abs(float(v)) < 1e-12 for v in values) / len(values) * 100.0, 4)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _safe_ratio(a: Any, b: Any) -> float | None:
    try:
        aa = float(a)
        bb = float(b)
    except Exception:
        return None
    if abs(bb) < 1e-12:
        return None
    return round(aa / bb, 6)


def _instrumented_evaluator(evaluator):
    """Capture per-path rounds/profits while preserving the exact v3.1 evaluator."""
    def wrapped(
        lower,
        upper,
        grids,
        current_price,
        state,
        fee_rate,
        utilization,
        active_notional_proxy,
        paths,
        profit_scale,
        rounds_scale,
    ):
        original_simulate = base.simulate_portfolio_path
        sim_results: list[tuple[float, float]] = []

        def capture_simulation(*args, **kwargs):
            result = original_simulate(*args, **kwargs)
            sim_results.append(
                (
                    float(result.get("rounds") or 0.0),
                    float(result.get("grid_profit_usdt") or 0.0),
                )
            )
            return result

        base.simulate_portfolio_path = capture_simulation
        try:
            result = evaluator(
                lower,
                upper,
                grids,
                current_price,
                state,
                fee_rate,
                utilization,
                active_notional_proxy,
                paths,
                profit_scale,
                rounds_scale,
            )
        finally:
            base.simulate_portfolio_path = original_simulate

        if result is None:
            return None

        k = _key(result)
        _CAPTURED[k] = dict(result)

        if sim_results and len(sim_results) % 2 == 0:
            raw_rounds: list[float] = []
            calibrated_rounds: list[float] = []
            calibrated_profits: list[float] = []

            for idx in range(0, len(sim_results), 2):
                a_rounds, a_profit = sim_results[idx]
                b_rounds, b_profit = sim_results[idx + 1]
                raw = (a_rounds + b_rounds) / 2.0
                raw_rounds.append(raw)
                calibrated_rounds.append(raw * float(rounds_scale))
                calibrated_profits.append(
                    ((a_profit + b_profit) / 2.0) * float(profit_scale)
                )

            _DISTRIBUTIONS[k] = {
                "raw_rounds": raw_rounds,
                "calibrated_rounds": calibrated_rounds,
                "calibrated_grid_profits": calibrated_profits,
            }

        return result

    return wrapped


def _capturing_integration_main() -> None:
    production_evaluator = base.evaluate_geometry
    base.evaluate_geometry = _instrumented_evaluator(production_evaluator)
    try:
        _ORIGINAL_INTEGRATION_MAIN()
    finally:
        base.evaluate_geometry = production_evaluator


def _risk_eligible(candidate: dict[str, Any], geo: dict[str, Any]) -> bool:
    current = ((geo.get("benchmarks") or {}).get("current") or {})
    policy = geo.get("risk_policy") or {}
    if not current:
        return False

    try:
        cap = float(policy["effective_escape_cap_pct"])
        max_prob_drop = float(
            policy["do_not_reduce_total_pnl_positive_probability_by_more_than_pp"]
        )
        max_p20_worsen = float(
            policy["do_not_worsen_p20_total_pnl_by_more_than_usdt"]
        )
        return (
            float(candidate["escape_probability_pct"]) <= cap + 1e-9
            and float(candidate["p_total_pnl_positive_pct"])
            >= float(current["p_total_pnl_positive_pct"]) - max_prob_drop
            and float(candidate["p20_total_pnl_usdt"])
            >= float(current["p20_total_pnl_usdt"]) - max_p20_worsen
        )
    except Exception:
        return False


def _enrich(candidate: dict[str, Any], geo: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    k = _key(candidate)
    dist = _DISTRIBUTIONS.get(k) or {}
    raw_rounds = list(dist.get("raw_rounds") or [])
    cal_rounds = list(dist.get("calibrated_rounds") or [])
    profits = list(dist.get("calibrated_grid_profits") or [])

    grids = int(row["grids"])
    width = float(row["upper_usdt"]) - float(row["lower_usdt"])
    row["grid_spacing_usdt"] = round(width / grids, 6)
    row["standard_risk_eligible"] = _risk_eligible(row, geo)
    row["profit_per_expected_round_usdt"] = _safe_ratio(
        row.get("expected_grid_profit_usdt"), row.get("expected_rounds")
    )
    row["profit_per_median_round_usdt"] = _safe_ratio(
        row.get("median_grid_profit_usdt"), row.get("median_rounds")
    )

    if raw_rounds:
        row["round_distribution"] = {
            "sample_n": len(raw_rounds),
            "p_zero_rounds_pct": _prob_zero(raw_rounds),
            "expected_raw_rounds": round(statistics.fmean(raw_rounds), 6),
            "median_raw_rounds": round(statistics.median(raw_rounds), 6),
            "p90_raw_rounds": round(float(_percentile(raw_rounds, 0.90)), 6),
            "p95_raw_rounds": round(float(_percentile(raw_rounds, 0.95)), 6),
            "max_raw_rounds": round(max(raw_rounds), 6),
            "expected_calibrated_rounds": round(statistics.fmean(cal_rounds), 6),
            "median_calibrated_rounds": round(statistics.median(cal_rounds), 6),
            "max_calibrated_rounds": round(max(cal_rounds), 6),
            "raw_probability_thresholds_pct": {
                f"p_rounds_ge_{n}": _prob_ge(raw_rounds, n)
                for n in ROUND_THRESHOLDS
            },
            "calibrated_probability_thresholds_pct": {
                f"p_rounds_ge_{n}": _prob_ge(cal_rounds, n)
                for n in ROUND_THRESHOLDS
            },
        }

    if profits:
        row["profit_distribution"] = {
            "p_zero_grid_profit_pct": _prob_zero(profits),
            "p_positive_grid_profit_pct": round(
                sum(p > 0 for p in profits) / len(profits) * 100.0, 4
            ),
            "expected_grid_profit_usdt_reconstructed": round(
                statistics.fmean(profits), 6
            ),
            "median_grid_profit_usdt_reconstructed": round(
                statistics.median(profits), 6
            ),
            "p90_grid_profit_usdt": round(float(_percentile(profits, 0.90)), 6),
            "max_grid_profit_usdt": round(max(profits), 6),
        }

    return row


def _best(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get(field) or 0.0))


def _group_density_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts = sorted({int(r["grids"]) for r in rows})
    for grids in counts:
        group = [r for r in rows if int(r["grids"]) == grids]
        eligible = [r for r in group if r.get("standard_risk_eligible") is True]
        out.append(
            {
                "grids": grids,
                "feasible_candidate_count": len(group),
                "standard_risk_eligible_candidate_count": len(eligible),
                "best_feasible_by_expected_profit": _best(
                    group, "expected_grid_profit_usdt"
                ),
                "best_standard_risk_eligible_by_expected_profit": _best(
                    eligible, "expected_grid_profit_usdt"
                ),
            }
        )
    return out


def _current_band_curve(
    rows: list[dict[str, Any]],
    geo: dict[str, Any],
) -> list[dict[str, Any]]:
    current = ((geo.get("benchmarks") or {}).get("current") or {})
    if not current:
        return []

    lower = float(current["lower_usdt"])
    upper = float(current["upper_usdt"])
    requested_counts = sorted(
        set(int(x) for x in base.COARSE_GRID_COUNTS)
        | {int(current["grids"])}
    )

    curve: list[dict[str, Any]] = []
    for grids in requested_counts:
        matches = [
            r
            for r in rows
            if int(r["grids"]) == grids
            and abs(float(r["lower_usdt"]) - lower) <= 0.01
            and abs(float(r["upper_usdt"]) - upper) <= 0.01
        ]
        best = _best(matches, "expected_grid_profit_usdt")
        curve.append(
            {
                "grids": grids,
                "feasible": best is not None,
                "geometry": best,
            }
        )
    return curve


def _headline(
    rows: list[dict[str, Any]],
    current_band: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [r for r in rows if r.get("standard_risk_eligible") is True]
    if not eligible:
        eligible = rows

    expected = _best(eligible, "expected_grid_profit_usdt")
    median = _best(eligible, "median_grid_profit_usdt")
    activity = _best(eligible, "expected_rounds")

    band_rows = [
        x["geometry"]
        for x in current_band
        if x.get("feasible") and x.get("geometry")
    ]
    band_eligible = [
        r for r in band_rows if r.get("standard_risk_eligible") is True
    ] or band_rows

    band_expected = _best(band_eligible, "expected_grid_profit_usdt")
    band_activity = _best(band_eligible, "expected_rounds")

    p50_rows = []
    for row in eligible:
        probs = (
            (row.get("round_distribution") or {})
            .get("raw_probability_thresholds_pct") or {}
        )
        p50_rows.append((float(probs.get("p_rounds_ge_50") or 0.0), row))
    p50_champion = max(p50_rows, key=lambda x: x[0])[1] if p50_rows else None
    max_p50 = max((x[0] for x in p50_rows), default=0.0)

    near_peak: list[dict[str, Any]] = []
    if expected:
        floor = float(expected["expected_grid_profit_usdt"]) * 0.90
        near_peak = [
            r
            for r in eligible
            if float(r["expected_grid_profit_usdt"]) >= floor
        ]
        near_peak.sort(
            key=lambda r: (
                float(
                    ((r.get("round_distribution") or {}).get(
                        "p_zero_rounds_pct"
                    ))
                    if (r.get("round_distribution") or {}).get(
                        "p_zero_rounds_pct"
                    ) is not None
                    else 100.0
                ),
                -float(r.get("median_grid_profit_usdt") or 0.0),
                -float(r.get("expected_grid_profit_usdt") or 0.0),
            )
        )

    counts = sorted({int(r["grids"]) for r in rows})
    expected_at_boundary = (
        expected is not None
        and counts
        and int(expected["grids"]) in {counts[0], counts[-1]}
    )

    return {
        "expected_profit_champion": expected,
        "median_profit_champion": median,
        "activity_champion": activity,
        "current_band_expected_profit_champion": band_expected,
        "current_band_activity_champion": band_activity,
        "fifty_plus_rounds": {
            "any_analogue_support_for_50_plus_rounds": max_p50 > 0.0,
            "maximum_p_raw_rounds_ge_50_pct": round(max_p50, 4),
            "champion": p50_champion,
        },
        "near_peak_90pct_expected_profit_low_dead_day_candidates": near_peak[:10],
        "expected_profit_optimum_on_tested_grid_boundary": expected_at_boundary,
        "operational_override": False,
    }


def _write_density_output() -> None:
    if not v3.GEO_PATH.is_file():
        raise SystemExit("Phase 4D geometry output missing after v4 capture")

    geo = json.loads(v3.GEO_PATH.read_text(encoding="utf-8"))
    rows = [_enrich(r, geo) for r in _CAPTURED.values()]
    rows.sort(
        key=lambda r: (
            int(r["grids"]),
            float(r["center_usdt"]),
            float(r["width_usdt"]),
        )
    )

    joint_curve = _group_density_curve(rows)
    current_band = _current_band_curve(rows, geo)
    headline = _headline(rows, current_band)

    source = geo.get("source_state") or {}
    integration = geo.get("execution_resolution_integration") or {}

    payload = {
        "schema": "pionex_grid_density_sweep_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_DIAGNOSTIC_ONLY",
        "scope": {
            "platform": "Pionex",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "horizon_h": 24,
            "objective": (
                "Expose the grid-density sweet spot for net 24h grid profit "
                "across the same candidates already evaluated by Phase 4D."
            ),
        },
        "source_state": {
            "captured_at_utc": source.get("captured_at_utc"),
            "captured_at_local": source.get("captured_at_local"),
            "current_price_usdt": source.get("latest_market_price_usdt"),
            "geometry_market_price_source": source.get(
                "geometry_market_price_source"
            ),
        },
        "execution_resolution": {
            "execution_replay_resolution": integration.get(
                "execution_replay_resolution"
            ),
            "policy_status": integration.get("policy_status"),
            "execution_resolution_integrated": integration.get(
                "execution_resolution_integrated"
            ),
        },
        "method": {
            "candidate_capture": (
                "Captures the exact candidate evaluations performed by the "
                "Phase 4D v3.1 live-price/recovery run; no second sweep."
            ),
            "density_isolation": (
                "current_band_density_curve holds the live lower/upper bounds "
                "constant and changes only grid count."
            ),
            "joint_optimisation_view": (
                "joint_best_by_grid_count allows centre and width to vary and "
                "reports the best candidate observed for each grid count."
            ),
            "round_thresholds": list(ROUND_THRESHOLDS),
            "zero_round_probability": (
                "P(0 rounds) is measured on raw analogue replays. Zero is "
                "unchanged by calibration scaling."
            ),
            "platform_minimums": (
                "Offline sizing/profit feasibility only. Pionex live "
                "minimum-order/investment validation is still required."
            ),
            "operational_effect": (
                "NONE. Existing Phase 4D actionability remains authoritative "
                "until density evidence is reviewed and explicitly promoted."
            ),
        },
        "captured_candidate_count": len(rows),
        "captured_grid_counts": sorted({int(r["grids"]) for r in rows}),
        "current_benchmark": ((geo.get("benchmarks") or {}).get("current") or {}),
        "joint_best_by_grid_count": joint_curve,
        "current_band_density_curve": current_band,
        "headline": headline,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== PIONEX GRID-DENSITY SWEET-SPOT DIAGNOSTIC ===")
    print("Candidates captured:", len(rows))
    print("Grid counts captured:", payload["captured_grid_counts"])
    print(
        "Expected-profit champion:",
        (headline.get("expected_profit_champion") or {}).get("grids"),
        "grids",
    )
    print(
        "Current-band champion:",
        (headline.get("current_band_expected_profit_champion") or {}).get(
            "grids"
        ),
        "grids",
    )
    print(
        "Max P(raw rounds >= 50):",
        (headline.get("fifty_plus_rounds") or {}).get(
            "maximum_p_raw_rounds_ge_50_pct"
        ),
        "%",
    )
    print("Density diagnostic written:", OUT_PATH)


def main() -> None:
    v3.integration.main = _capturing_integration_main
    try:
        v3.main()
    finally:
        v3.integration.main = _ORIGINAL_INTEGRATION_MAIN

    _write_density_output()


if __name__ == "__main__":
    main()
