# Phase 4D — ETH/USDT Pionex Grid Geometry Optimizer

Phase 4D extends the Phase 4C centre-shift optimizer so the model can jointly choose:

- band centre;
- total band width;
- grid count.

It remains **diagnostic / decision-support only**. It never connects to Pionex and never changes the live bot.

## What changed

1. **ATR + path-shape conditioning**
   - ATR14 remains the first-stage volatility filter.
   - Similar historical states are then re-ranked using 24h path efficiency, reversal rate, realised path length and signed move.
   - This is intended to distinguish quiet/choppy movement from equally low-ATR but one-directional movement.

2. **Joint geometry search**
   - The search varies centre, width and number of grids together.
   - A coarse search is followed by local refinement around the strongest risk-constrained regions.
   - `KEEP_CURRENT` remains a valid result.

3. **Capital-normalised grid quantity**
   - Candidate quantity/grid is not held artificially fixed when grid count changes.
   - The model preserves the live bot's approximate active-order notional and respects the latest captured ETH/USDT balances.

4. **Fee-aware constraint**
   - Candidate spacing must leave fee-deducted profit/grid meaningfully above round-trip fees.
   - Pionex's live minimum investment/order requirement still has to be confirmed in the edit screen because it depends on pair, range, grid count and current platform rules.

5. **Inventory-aware total P&L diagnostic**
   - Historical analogue paths replay grid buys/sells using the latest captured ETH and USDT balances.
   - The optimizer reports expected total P&L, probability total P&L is positive, and downside P20/P10 outcomes in addition to grid profit and escape risk.

6. **Prospective calibration v2**
   - Manual screenshot states are paired into 14–30h same-geometry windows.
   - For each starting state, the model reconstructs what it could have predicted using only already-mature 24h historical paths.
   - Calibration is not applied until at least 3 usable windows exist.

## New prospective states included

The supplied `manual_grid_states_v1.csv` appends the two later screenshots already supplied in chat:

- 2026-08-13 04:23 SAST — 1,750–2,000, 30 grids, 4,265 total rounds, grid profit 172.34 USDT.
- 2026-08-14 04:16 SAST — 1,750–2,000, 30 grids, 4,271 total rounds, grid profit 172.47 USDT.

These preserve the no-backfill rule: only settings actually observed in screenshots are recorded.

## Files

- `scripts/build_pionex_grid_geometry_optimizer_v1.py`
- `.github/workflows/phase4d-grid-geometry-ci.yml`
- `data/pionex/manual_grid_states_v1.csv` (updated with the two observed states)

Generated on run:

- `data/diagnostics/pionex_grid_geometry_optimizer_v1.json`
- `data/diagnostics/pionex_grid_calibration_v2.json`
- `data/pionex/grid_geometry_recommendations_v1.csv`

## Run

After uploading/committing the three source/input files:

**Actions → Phase 4D Pionex Grid Geometry CI → Run workflow**

The workflow validates the output, uploads an artifact and persists the prospective Phase 4D diagnostics/ledger back to `main`.
