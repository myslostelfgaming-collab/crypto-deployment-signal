# Pionex Phase 4.5 Calibration v2 Upgrade

Research/read-only calibration upgrade for `crypto-deployment-signal`.

## Safety invariants

- No Pionex write endpoint is added.
- Existing Phase 4.5 geometry/action fields remain authoritative and unchanged.
- Legacy calibration fields are retained unchanged.
- Calibration v2 is additive and can downgrade/qualify forecast confidence only.
- `automatic_joint_promotion_allowed` remains `false`.

## What changes

1. MAPE is retained as a secondary guarded diagnostic.
2. WAPE becomes the main percentage-type error metric.
3. MAE, median absolute error, bias, RMSE and recency-weighted metrics are added for profit and paired rounds.
4. Activity windows are classified as `NORMAL_ACTIVITY`, `LOW_ACTIVITY`, or `ZERO_ACTIVITY` using thresholds derived from fit-only history.
5. Geometry-change state pairs are explicitly audited as `CALIBRATION_WINDOW_EXCLUDED_GEOMETRY_CHANGE` instead of silently disappearing.
6. Calibration uses chronological fit -> holdout -> prospective partitions.
7. A transparent cohort hierarchy prefers exact geometry, then similar geometry+regime, similar geometry, same market regime, then global history.
8. Profit calibration is decomposed into activity/rounds scaling and conditional profit-per-cycle productivity scaling.
9. Recency weighting is deterministic (30-day half-life, 0.25 floor by default) and both weighted/unweighted metrics are emitted.
10. Multi-metric quality statuses replace MAPE-only interpretation.
11. Decision diagnostics receive additive `forecast_calibration_confidence`, `directional_signal_reliable`, and `magnitude_estimate_reliable` fields.

## Files to add

- `scripts/pionex_calibration_quality_v2.py`
- `scripts/replay_pionex_calibration_v2.py`
- `scripts/build_pionex_calibration_quality_v2.py`
- `scripts/run_pionex_phase4_6_full_v1.py`
- `tests/test_pionex_calibration_quality_v2.py`

## Workflow change

In `.github/workflows/pionex-automated-phase4d-ci.yml`, replace:

```bash
python scripts/run_pionex_phase4_5_full_v1.py
```

with:

```bash
python scripts/run_pionex_phase4_6_full_v1.py
```

Then add validation that `data/diagnostics/pionex_calibration_quality_v2.json` exists and that:

- `schema == pionex_paired_cycle_calibration_v2`
- `primary_percentage_metric == WAPE`
- `secondary_percentage_metric == GUARDED_MAPE_NORMAL_ACTIVITY_ONLY`
- `forecast_calibration_confidence` exists in the joint/full diagnostics
- `automatic_joint_promotion_allowed` remains `false`

Add `data/diagnostics/pionex_calibration_quality_v2.json` to the uploaded/committed diagnostic bundle.

## Current historical replay findings (2026-11-04 repo data)

The current repository replay contains **63 usable same-geometry performance windows**. The current canonical state-pair audit also records **3 explicit geometry-change exclusions** with reason `CALIBRATION_WINDOW_EXCLUDED_GEOMETRY_CHANGE`; they are retained as audit evidence rather than silently disappearing.

Activity classification on the 63 usable windows:

- normal: 54
- low activity: 9
- zero activity: 0
- low/zero share: 14.29%

Chronological governance split:

- fit: 54 windows
- holdout: 6 windows
- newest prospective: 3 windows
- random shuffle: disabled

### Champion/challenger result

The geometry/regime-aware two-stage calibration is now a **challenger**, not an automatic replacement. It can only replace the legacy global calibration when it Pareto-dominates the legacy model on the holdout: profit WAPE, profit MAE, rounds WAPE and rounds MAE must all be non-degrading, and at least one must improve. Prospective windows are never used to select the winner.

On the current 6-window holdout:

Legacy global champion:

- Profit MAE: 0.132288 USDT/day
- Profit WAPE: 44.8391%
- Profit bias: -0.112408 USDT/day
- Rounds MAE: 8.000630/day
- Rounds WAPE: 42.8742%
- Rounds bias: -6.654035/day

Geometry/regime challenger:

- Profit MAE: 0.135033 USDT/day
- Profit WAPE: 45.7693%
- Rounds MAE: 8.232259/day
- Rounds WAPE: 44.1155%

The challenger therefore degrades holdout profit WAPE/MAE by about **2.07%** and rounds WAPE/MAE by about **2.90%**. The gate correctly selects:

```json
{
  "selected_model": "LEGACY_GLOBAL_MEDIAN",
  "challenger_promoted": false,
  "gate": "PARETO_HOLDOUT_NO_DEGRADATION",
  "prospective_used_for_selection": false
}
```

The newest 3 prospective windows are still reported separately. The cohort challenger reaches about **38.17% profit WAPE** and **38.17% rounds WAPE** there, which is useful evidence but is deliberately not allowed to override the holdout governance decision.

The exact newest-six replay also does not justify promotion: cohort v2 is about **7.43% worse on profit WAPE/MAE** and **7.68% worse on rounds WAPE/MAE** than the refit global comparator.

This is the intended calibration behavior: regime/geometry complexity must earn promotion with out-of-sample evidence. The system does not tune thresholds or suppress warnings to make the challenger look better.

### Current confidence policy

Current quality remains:

```json
{
  "status": "CALIBRATION_ACCEPTABLE",
  "forecast_confidence": "MEDIUM",
  "directional_signal_reliable": true,
  "magnitude_estimate_reliable": false
}
```

That means the direction of a geometry recommendation may still be useful, but an exact uplift such as `+17%` should not be treated as numerically precise. Calibration quality alone cannot promote a live bot change, and `automatic_joint_promotion_allowed` remains `false`.

## Recommended primary metric

Use **profit WAPE** as the primary percentage calibration metric, paired with **profit MAE in USDT/day**. For activity, use **rounds WAPE + rounds MAE**. Bias is a mandatory directional diagnostic. Guarded MAPE remains useful as a secondary diagnostic only.
