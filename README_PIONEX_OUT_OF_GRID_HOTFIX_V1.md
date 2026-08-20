# Pionex Out-of-Grid Hotfix v1

## Failure diagnosed

Phase 4D's legacy `evaluate_geometry` returns `None` unless:

`lower < current_market_price < upper`

The live bot escaped above `2300`, so the current geometry benchmark became
"infeasible" and Phase 4D aborted with:

`Current Pionex geometry failed model feasibility checks`

This is a modelling edge case, not a Pionex API failure.

## Fix

The hotfix adds wrappers rather than rewriting the validated core scripts.

- Correct mature grid state above the upper bound: all grid intervals wait to buy
  on re-entry.
- Evaluate the already-escaped current bot using its actual quantity and balances.
- Evaluate candidate recovery ranges using a post-edit rebalance proxy that
  preserves active grid notional and current mark-to-market equity.
- When the live range is already escaped, select an in-grid recovery candidate
  from the minimum escape-risk region rather than comparing it as if the old grid
  were still an ordinary active candidate.
- Preserve all historical in-grid calibration behaviour.
- Keep all existing Phase 4D prospective gates and integrity checks.

## Files

Add:
- `scripts/pionex_out_of_grid_support_v1.py`
- `scripts/build_pionex_grid_geometry_optimizer_v3.py`
- `scripts/build_pionex_grid_activity_v2.py`
- `scripts/run_pionex_phase4d_full_v2.py`

Replace:
- `.github/workflows/pionex-automated-phase4d-ci.yml`

Then manually run `Pionex Automated Phase 4D Decision CI`.

## Safety

The Pionex API remains read-only. The recovery geometry is decision support.
Candidate sizing assumes that editing the Pionex range can rebalance the bot's
active grid capital. Always confirm the quantity/grid and platform validation
shown in Pionex before applying a suggested range.
