# Phase 4.5 — Joint Paired-Cycle Geometry Optimiser

This is the next development stage after v4.4.

## Upload

### Add
1. `scripts/build_pionex_grid_geometry_optimizer_v6.py`
2. `scripts/run_pionex_phase4_5_full_v1.py`

### Replace
3. `.github/workflows/pionex-automated-phase4d-ci.yml`

Do not delete or replace:
- `build_pionex_grid_geometry_optimizer_v4.py`
- `build_pionex_grid_geometry_optimizer_v5.py`
- `run_pionex_phase4d_full_v2.py`

Phase 4.5 intentionally wraps those validated layers.

After upload, run:

**Actions → Pionex Automated Phase 4.5 Decision CI → Run workflow**

---

## What Phase 4.5 solves

v4.4 only optimises grid density on the current live band.

Phase 4.5 jointly searches:
- centre
- width
- grid count

using:
- paired buy→sell grid-profit accounting;
- paired-specific walk-forward calibration;
- the promoted 5-minute replay;
- actual live bot equity/state;
- fresh post-edit rebalance sizing for changed candidates;
- the same risk constraints used by the existing Phase 4D architecture.

It remains research-only.

---

## Important v4.4 hotfix included

The v4.4 plateau selector originally tried grid count 22 first when defining
`current_band_live_density`.

The live bot later changed to 40 grids, so gain-vs-current percentages could be
anchored to a hypothetical 22-grid row.

Phase 4.5 repairs the current run before v4.4 history is updated:

`phase4_5_hotfix.v44_current_live_density_anchor_repaired = true`

Phase 4.5 itself does not use the old contaminated v4.4 stability history.

---

## New diagnostic

`data/diagnostics/pionex_joint_paired_geometry_v1.json`

Schema:

`pionex_joint_paired_geometry_v1`

Version:

`4.5`

Key sections:

- `current_status_quo_paired_benchmark`
- `current_geometry_fresh_rebalance_counterfactual`
- `paired_calibration`
- `calibration_quality`
- `risk_policy`
- `search`
- `selection`
- `practical_candidate_exact_recompute`
- `comparison_to_legacy_optimizer`
- `best_eligible_by_grid_count`
- `near_peak_candidates_top100`

---

## Search logic

Coarse search uses the established Phase 4D envelope:
- centre: ±4% around market, $20 coarse spacing;
- width: 6%–19% of market, 2 percentage-point coarse spacing;
- grids: 10–80 in steps of 5.

The strongest four corrected paired candidates are locally refined using:
- centre ±$5 in $2.50 increments;
- width ±0.5 percentage points;
- grids ±3.

Every changed candidate is sized as a hypothetical post-edit bot using:
- the same total live equity;
- the preserved active-order-notional budget;
- a fresh rebalance seed.

The status quo is evaluated separately using the actual live:
- quantity/grid;
- ETH balance;
- USDT balance.

---

## Robust selector

Phase 4.5 does NOT simply choose one expected-profit argmax.

It keeps candidates:
1. passing risk and known execution checks;
2. within 95% of peak calibrated expected paired grid profit;
3. then within 95% of the best median paired grid profit in that near-peak set.

It prefers a candidate surrounded by the greatest number of other near-peak
geometries, then:
- lower escape risk;
- smaller change from the current bot;
- higher median/expected paired profit;
- larger order notional.

This is intended to favour a broad joint plateau rather than a fragile
one-coordinate maximum.

---

## Practical rounding is now exact

Older output sometimes displayed rounded $5 bounds while leaving metrics from
the unrounded candidate.

Phase 4.5 rounds the selected bounds to the nearest $5 and SIMULATES THAT
ROUNDED GEOMETRY AGAIN.

The workflow fails if a practical candidate is emitted without:

`metrics_recomputed_for_rounded_bounds = true`

---

## Cross-run stability

`pionex_full_decision_v1.json` now accumulates:

`phase4_5_joint_history`

The newest three material joint selections are called stable only if:
- centre-offset span <= 1.5 percentage points;
- width span <= 3.0 percentage points;
- grid-count span <= 10.

This is deliberately broader than v4.4 density-only stability because Phase 4.5
is solving three variables at once.

---

## Safety

Phase 4.5 can add:

`PAIRED_JOINT_GEOMETRY_CONFLICT`

when its corrected joint evidence materially contradicts the legacy
centre/width/grid recommendation.

That blocker can force/retain:

`KEEP_CURRENT`

But:

`automatic_joint_promotion_allowed = false`

is a hard invariant.

Phase 4.5 cannot automatically move the live bot.

A changed geometry still needs:
1. repeated/stable joint evidence;
2. exact live Pionex setup validation;
3. manual review.

---

## Calibration caution

The paired calibration is statistically active, but recent holdout percentage
errors have been large.

Phase 4.5 therefore records:

`calibration_quality`

and raises:

`CAUTION_HIGH_HOLDOUT_PERCENTAGE_ERROR`

when profit or rounds holdout MAPE exceeds 100%.

This does not stop research ranking, but it prevents us from pretending that
small dollar differences are precise.

---

## Acceptance checks after first run

Check that:

1. workflow passes;
2. `pionex_joint_paired_geometry_v1.json` exists;
3. version is `4.5`;
4. its source capture matches the latest Pionex API state;
5. `current_status_quo_paired_benchmark.grids` equals the ACTUAL live bot;
6. `phase4_5_hotfix.v44_current_live_density_anchor_repaired == true`;
7. a joint search candidate count is reported;
8. practical candidate metrics are recomputed if a practical candidate exists;
9. full decision contains `phase4_5_joint_paired_geometry`;
10. `phase4_5_joint_history` gets the current run;
11. `automatic_joint_promotion_allowed == false`;
12. operational output remains manual/safety-gated.

After the run, send ChatGPT back to the repo. The main questions will be:
- does corrected paired geometry want the band centred lower/higher?
- does it want a different width?
- what grid count belongs on that new band?
- is the result materially better than staying on the live bot?
- what does the legacy 10-grid candidate look like under corrected paired
  accounting?
- does the same joint region recur over the next several automated runs?
