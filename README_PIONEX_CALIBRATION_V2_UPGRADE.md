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

## Current historical replay findings (2026-10-05 data)

The current paired history contains 43 usable same-geometry performance windows. Inspection of the canonical state history identifies 15 explicit structural geometry-change transitions; the runtime state-pair audit records these separately rather than silently dropping them.

Activity classification on the 43 usable windows:

- normal: 35
- low activity: 8
- zero activity: 0

Fit-derived low-activity thresholds in the offline replay were `rounds <= 2.5` or `profit <= $0.08/day` (with near-zero activity treated separately). These are derived from the lower tail of fit-only history with conservative caps, not hard-coded as performance targets.

### Exact latest-six legacy validation span

Legacy global-scalar replay:

- Profit MAE: 0.096081 USDT/day
- Profit WAPE: 52.6814%
- Profit bias: +0.053474 USDT/day (overprediction)
- Rounds MAE: 7.740838/day
- Rounds WAPE: 47.3076%
- Rounds bias: +3.790924/day (overprediction)

Calibration v2 strict walk-forward replay on the same six windows:

- Profit MAE: 0.084122 USDT/day
- Profit WAPE: 46.1242%
- Profit guarded MAPE: 30.5768% over normal-activity windows
- Profit bias: +0.038103 USDT/day
- Rounds MAE: 6.877605/day
- Rounds WAPE: 42.0321%
- Rounds guarded MAPE: 30.4052% over normal-activity windows
- Rounds bias: +2.623418/day

Improvement versus the same legacy validation span:

- Profit WAPE/MAE: ~12.45% improvement
- Rounds WAPE/MAE: ~11.15% improvement
- Profit overprediction bias reduced by ~28.7%
- Rounds overprediction bias reduced by ~30.8%

The current evidence therefore supports geometry-aware/two-stage calibration as a real improvement, but not a precise magnitude forecaster. Current policy should be approximately:

```json
{
  "forecast_confidence": "MEDIUM",
  "directional_signal_reliable": true,
  "magnitude_estimate_reliable": false
}
```

The full volatility/path-regime enrichment is implemented but could not be replayed outside the repository workflow because the generated feature/candle context is not available in this local patch environment. Its incremental benefit must therefore be measured honestly in CI before any claim that *regime* awareness itself improves holdout accuracy. Until then, the demonstrated uplift is specifically from geometry-aware cohorting + recency weighting + activity/productivity decomposition.

## Recommended primary metric

Use **profit WAPE** as the primary percentage calibration metric, paired with **profit MAE in USDT/day**. For activity, use **rounds WAPE + rounds MAE**. Bias is a mandatory directional diagnostic. Guarded MAPE remains useful as a secondary diagnostic only.
