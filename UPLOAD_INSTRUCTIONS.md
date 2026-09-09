# Phase 4D v4.3 — paired-cycle accounting correction

## Upload / replace exactly these two files

1. `scripts/build_pionex_grid_geometry_optimizer_v4.py`
2. `scripts/run_pionex_phase4d_full_v2.py`

Then run:

**Actions → Pionex Automated Phase 4D Decision CI → Run workflow**

No workflow YAML edit is required.

---

## Why v4.3 is necessary

The v4.2 run confirmed that the production optimizer is genuinely pressing
against its minimum grid-count boundary:

- production search minimum: 10 grids;
- production selection: 10 grids;
- diagnostic legacy current-band optimum: 5 grids.

The more important finding is an accounting defect in the legacy prospective
grid-profit simulator.

For hypothetical/reconfigured grids, intervals above the current market price
start with seeded ETH and are immediately marked ready to sell. When the first
sell triggers, the legacy simulator books the *entire interval spread* as grid
profit even though no replay buy occurred at that interval's lower grid line.

On the 5-grid live-band diagnostic:

- range: $2190–$2630;
- levels: $2190, $2300, $2410, $2520, $2630;
- market at the run: about $2510.94;
- first seeded sell: $2520;
- legacy accounting credits profit as though the ETH was bought at $2410.

Using the model's own quantity, fee and calibration scale, that single synthetic
full-spread credit equals the reported legacy median grid profit to rounding.
That is direct evidence that the low-grid objective is being structurally
inflated by initial seed accounting.

---

## What the new paired-cycle model does

For the fresh/reconfigured diagnostic:

- initial ETH above market is tagged `seed_sell`;
- its first sale is inventory conversion;
- that sale affects total portfolio P&L normally;
- it does **not** earn grid-cycle profit;
- after an interval executes a real replay buy, it becomes `paired_sell`;
- only the subsequent sell counts as a completed grid cycle and earns
  paired grid profit.

This is the metric we actually intended when asking whether many small completed
cycles can beat one large cycle.

The corrected diagnostic again sweeps every integer grid count from 5 through 80
on the current live band.

New persisted section in `pionex_full_decision_v1.json`:

`grid_density_sweep.paired_cycle_correction`

and the full curve:

`grid_density_sweep.paired_cycle_current_band_integer_sweep`

Key fields include:

- `expected_paired_grid_profit_usdt_raw`
- `median_paired_grid_profit_usdt_raw`
- `expected_paired_rounds_raw`
- `median_paired_rounds_raw`
- `p_zero_paired_rounds_pct`
- `p_paired_rounds_ge_10_pct`
- `p_paired_rounds_ge_25_pct`
- `p_paired_rounds_ge_50_pct`
- `expected_seed_sells`
- `expected_seed_inventory_realized_pnl_usdt`

The old calibration scales are also shown provisionally for magnitude
comparability, but the paired model is **not yet independently calibrated**.
The raw paired-profit ranking is the main research signal for this run.

---

## Safety behaviour

v4.3 does NOT silently promote the paired model to operational authority.

However, if BOTH are true:

1. the initial-seed full-spread accounting bias is numerically confirmed; and
2. production selection is sitting on the minimum grid-count boundary,

then the final runner applies:

`GRID_PROFIT_INITIAL_SEED_ACCOUNTING_BIAS`

and changes only the final operational recommendation to:

`KEEP_CURRENT`

with:

`actionable_geometry_change = false`

The legacy research recommendation remains visible for audit.

The runner patches both:

- `data/diagnostics/pionex_full_decision_v1.json`
- `data/diagnostics/pionex_grid_actionability_v1.json`

Both are already persisted by the existing automated workflow.

---

## Acceptance checks after the run

1. Workflow succeeds.
2. `pionex_full_decision_v1.json` contains:
   - `grid_density_sweep.version == "4.3"`
   - `grid_density_sweep.paired_cycle_correction`
   - `grid_density_sweep.paired_cycle_current_band_integer_sweep`
3. Density source state matches the fresh API state.
4. `synthetic_seed_profit_signature` reports whether the legacy median matches
   the synthetic initial seeded-sell profit.
5. If confirmed while production remains at the minimum boundary:
   - final operational action is `KEEP_CURRENT`;
   - blocker includes `GRID_PROFIT_INITIAL_SEED_ACCOUNTING_BIAS`.
6. Legacy research output is still visible and unchanged.
7. No Pionex write endpoint is used.

---

## What to inspect next

After one successful v4.3 run, compare the paired-cycle curve at roughly:

5, 10, 15, 20, 22, 30, 40, 50, 60, 70, and the highest feasible density.

We want to see:

- where raw paired grid profit peaks;
- whether the optimum is an interior density rather than a boundary;
- where median paired profit peaks;
- how P(0 paired cycles) changes;
- whether 25+ / 50+ completed paired cycles are genuinely plausible;
- whether high-density profitability is being limited mainly by order size,
  fees, or insufficient volatility.

Only after that should the paired-cycle model be independently calibrated and
considered for operational promotion.
