# Phase 4D v4.4 — paired calibration + robust density plateau

## Upload these files

Add:

1. `scripts/build_pionex_grid_geometry_optimizer_v5.py`
2. `data/pionex/pionex_execution_constraints_v1.json`

Replace:

3. `scripts/run_pionex_phase4d_full_v2.py`

Do **not** delete or replace `scripts/build_pionex_grid_geometry_optimizer_v4.py`.
v5 intentionally wraps the existing v4.3 implementation so rollback/audit remain
easy.

Then run:

**Actions → Pionex Automated Phase 4D Decision CI → Run workflow**

No workflow YAML change is required.

---

## What v4.4 adds

### 1. Independent paired-cycle walk-forward calibration

v4.3 correctly changed the accounting semantics but still reused the legacy
profit/round scale.

v4.4 reconstructs the corrected paired model at historical manual Pionex state
windows using:

- the historical live range;
- historical observed quantity/grid;
- historical ETH/USDT balances;
- only analogue paths that were already mature at that historical time;
- the Phase 4D promoted replay resolution (5-minute when promoted);
- only true replay buy → sell paired cycles as grid profit.

Earlier windows estimate the calibration scale. Chronologically later windows
are held out and reported separately.

Expected output:

`grid_density_sweep.promotion_readiness.paired_calibration`

Key fields:

- `status`
- `calibration_ready`
- `training_windows`
- `validation_windows`
- `profit_scale_applied`
- `rounds_scale_applied`
- `holdout_validation`

### 2. Robust plateau selector

v4.4 stops treating the single highest expected-profit grid count as a magic
integer.

It finds the contiguous grid-count region containing the expected-profit peak
where every candidate retains at least **95% of peak expected paired profit**.

Inside that plateau it selects:

1. highest **median paired profit**;
2. then lower P(0 paired cycles);
3. then higher paired activity;
4. then larger estimated order notional.

This should naturally prefer a robust point in the broad ~35–46 region when the
data continues to support it, rather than oscillating between e.g. 37 and 39.

Expected output:

`grid_density_sweep.promotion_readiness.plateau_selector`

### 3. Cross-run stability history

The runner preserves the previous full-decision history and appends each fresh
v4.4 result.

Stability requires:

- at least 3 paired-calibrated runs;
- robust selected grids spanning no more than 6 grids;
- overlapping 95%-of-peak plateaus.

Output:

`paired_density_history`

and

`grid_density_sweep.promotion_readiness.cross_run_stability`

### 4. Pionex execution validation policy

Pionex does not publish one universal Spot Grid minimum that can safely be
hard-coded for every pair/range/grid combination.

The new config:

`data/pionex/pionex_execution_constraints_v1.json`

therefore starts with dynamic live validation required.

Known constraints can be added later, but an exact candidate is only marked
live-validated when `validated_candidate.accepted_by_pionex_ui` is explicitly
recorded after testing that exact setup in Pionex.

**Do not fill the example validation values until an actual Pionex setup has
been checked.**

### 5. Stronger low-grid safety gate

v4.3 only blocked the legacy recommendation when its narrow synthetic
seed-profit detector fired.

v4.4 adds a second, broader safety condition.

If:

- paired calibration is ready;
- legacy production selection is pinned to the minimum grid-count boundary;
- the robust paired plateau is materially denser;
- and either the low-grid paired candidate has >=25% zero-cycle risk or the
  calibrated paired plateau beats it by >=5% expected paired profit;

then the operational legacy geometry change is blocked with:

`PAIRED_CALIBRATED_DENSITY_CONFLICT`

and operational action becomes:

`KEEP_CURRENT`

This is intentionally **one-way**.

The paired model may block a suspect legacy change, but v4.4 will **not**
automatically move the bot to the paired candidate.

---

## Important scope limit

v4.4 promotion-readiness is for **density on the current live band**.

It does not yet promote a corrected paired-cycle joint optimisation of:

- centre,
- width,
- grid count.

So a result such as "46 grids" means "46 grids on the current range being swept",
not automatically "use the legacy optimizer's proposed new range with 46 grids."

That joint paired-geometry step should come only after the density model is
calibrated and stable.

---

## Acceptance checks after the run

Confirm:

1. `pionex_full_decision_v1.json` contains
   `grid_density_sweep.version == "4.4"`.
2. `promotion_readiness.paired_calibration.evaluated_windows > 0`.
3. When enough history exists,
   `paired_calibration.status == "PAIRED_CALIBRATION_ACTIVE_WITH_HOLDOUT"`.
4. `plateau_selector.expected_profit_champion` is present.
5. `plateau_selector.robust_plateau_selection` is present.
6. `paired_density_history` contains the latest capture.
7. The first run will normally say `ACCUMULATING_EVIDENCE` because v4.3 history
   did not yet contain the v4.4 plateau selector.
8. If calibrated paired evidence materially contradicts the legacy 10-grid
   boundary selection, final operational action should be `KEEP_CURRENT` with
   blocker `PAIRED_CALIBRATED_DENSITY_CONFLICT`.
9. `automatic_paired_promotion_allowed` must remain `false`.
10. No Pionex write endpoint is called.

---

## After the first successful run

Send ChatGPT the repo again. We should inspect:

- paired training/holdout calibration scales;
- holdout profit and rounds error;
- expected-profit plateau bounds;
- robust plateau grid count;
- current-vs-plateau calibrated gains;
- whether the low-grid conflict safety blocker fired;
- the order-notional estimate of the plateau candidate;
- cross-run stability as it accumulates over subsequent automated runs.
