# Phase 4D v4.2 — density sweet-spot + low-grid-bias diagnostic

Upload/replace these two files in the repo:

1. `scripts/build_pionex_grid_geometry_optimizer_v4.py`
2. `scripts/run_pionex_phase4d_full_v2.py`

Then run:

**Actions → Pionex Automated Phase 4D Decision CI → Run workflow**

No workflow YAML edit is required for this revision.

## What this fixes

### A. v4.0 calibration contamination

The first density diagnostic captured historical `reconstruct_calibration()`
evaluations as if they were current live candidates. That is why a geometry such
as `$2185–$2585 / 21` could appear as the "live champion" even though its centre
was outside the current live centre-search band.

v4.2 disables capture during calibration reconstruction and only records the
current live candidate search.

### B. Grid spacing formula

The prior diagnostic displayed spacing as `width / grids`.

The simulator actually models Pionex grid count as N price levels with N-1
intervals, so v4.2 uses:

`spacing = width / (grids - 1)`

### C. Explicit lower-bound test

Production Phase 4D still searches from 10 grids upward.

The diagnostic now tests **5 through 80 grids** on the CURRENT live band.
Counts 5–9 are diagnostic-only. They are never promoted into the production
selection.

If the legacy objective prefers 5–9 grids, we have direct evidence that the
optimizer is pressing against its configured lower boundary rather than finding
an interior optimum.

### D. Sizing-bias investigation

The production normal-candidate model sizes hypothetical geometries from the
current pre-edit ETH/USDT holdings plus the observed utilization and active-order
notional cap.

That can disadvantage denser hypothetical grids because the current holdings
were arranged for the live geometry.

v4.2 therefore runs the same 5–80 current-band sweep twice:

1. `legacy_current_band_integer_sweep`
   - existing production sizing semantics;

2. `rebalanced_current_band_integer_sweep`
   - post-edit rebalance counterfactual;
   - same total equity;
   - same active-order-notional cap;
   - removes only the pre-edit ETH/USDT split constraint.

This tells us whether "10 grids keeps winning" is:
- a genuine frequency × profit/cycle optimum,
- lower-bound pressure from the objective,
- a current-asset-split sizing artefact,
- or some combination.

### E. Persistence without workflow edits

The standalone file is still written:

`data/diagnostics/pionex_grid_density_sweep_v1.json`

But the runner now embeds the useful full density diagnostic into:

`data/diagnostics/pionex_full_decision_v1.json`

The existing automated workflow already stages and persists that file, so the
density evidence will survive the run even if the standalone JSON is not added
to the workflow's `git add` list.

## Expected new diagnostic sections

Inside `pionex_full_decision_v1.json`:

`grid_density_sweep.low_grid_pressure_audit`

Look for codes such as:
- `LOW_GRID_BOUNDARY_PRESSURE_PRESENT`
- `OBJECTIVE_PREFERS_BELOW_PRODUCTION_MINIMUM`
- `PRE_EDIT_ASSET_SPLIT_HAS_MATERIAL_DENSITY_EFFECT`

Also inspect:

- `grid_density_sweep.headline`
- `grid_density_sweep.joint_live_best_by_grid_count`
- `grid_density_sweep.legacy_current_band_integer_sweep`
- `grid_density_sweep.rebalanced_current_band_integer_sweep`

## Acceptance checks

After the workflow completes:

1. Existing full Phase 4D pipeline remains green.
2. `pionex_full_decision_v1.json` contains `grid_density_sweep.version == "4.2"`.
3. Density `source_state.captured_at_utc` equals the fresh API state used by the
   full decision.
4. `legacy_current_band_integer_sweep` includes feasible integer counts below 10
   where the model permits them.
5. `rebalanced_current_band_integer_sweep` is present.
6. `grid_spacing_usdt` for the live 22-grid / $440-wide geometry is approximately
   `440 / 21 = 20.95238`, not 20.0.
7. `operational_effect` remains `NONE_DIAGNOSTIC_ONLY`.
8. Existing Pionex operational decision is NOT changed by this diagnostic.

## What we will read after the run

The most important outputs are:

- Does the legacy curve peak below 10?
- Does the rebalanced curve peak at a materially different density?
- How quickly does quantity/grid fall as density rises?
- How quickly do completed rounds rise?
- Where does `expected_grid_profit_usdt` actually peak?
- Where does `median_grid_profit_usdt` peak?
- What is P(0 rounds) at each density?
- Is there any realistic support for 25+ or 50+ rounds/day?
- Is the profit peak sharp or a broad plateau?

Do not change the live bot based only on the v4.2 diagnostic. First inspect one
fresh successful run and compare the two sizing curves.
