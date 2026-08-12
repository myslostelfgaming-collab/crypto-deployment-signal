# Phase 4C — ETH/USDT Pionex Grid Recommendation + Prospective Calibration

## Purpose
Phase 4C turns Phase 4B from a fixed-shift comparison into a daily shift optimizer for the user's existing Pionex ETH/USDT Spot Grid bot.

It preserves the user's operating policy:
- shift lower and upper limits together;
- preserve band width;
- preserve grid count;
- maximise expected 24h grid profit without accepting more range-escape risk than the current band (and never above a 10% absolute ceiling).

## Shift search
The model is **not restricted to $50 or $100 moves**.

It searches the full feasible shift interval while keeping current ETH price inside the grid:
1. coarse scan at $5 increments;
2. refinement around promising profit/risk regions at $0.50 increments.

The JSON therefore reports a raw mathematical shift (for example +$28.50) and also a $5-rounded practical alternative. The practical rounding is not a model constraint; it is an acknowledgement that sub-dollar precision is not yet statistically justified.

## Decision rule
The primary risk-constrained candidate:
- may not have higher range-escape probability than the current band;
- may never exceed 10% modeled range-escape probability;
- maximises calibrated expected 24h grid profit within that risk set.

To avoid pointless churn, the system keeps the current grid unless the proposed move improves expected 24h profit by at least $0.02 **or** reduces escape risk by at least 2.5 percentage points.

The full profit-vs-risk Pareto frontier is retained in the diagnostic output.

## Prospective calibration
`data/pionex/grid_recommendations_v1.csv` is a persistent prospective ledger.

When future manual Pionex snapshots are added, Phase 4C looks for 18–30 hour windows where grid geometry remained unchanged and compares:
- predicted vs actual fee-deducted grid profit;
- predicted vs actual completed rounds.

After at least 5 usable windows it applies the median actual/predicted scale factor, clipped to [0.50, 1.75]. Before that, calibration remains 1.0 and the status remains early/diagnostic.

For the cleanest calibration, capture a Pionex screenshot at roughly the same time each day. If changing the grid, a screenshot immediately after the change creates a clean new starting state.

## Files
- `scripts/build_pionex_grid_recommendation_v1.py`
- `.github/workflows/phase4c-grid-recommendation-ci.yml`

Generated/persisted by the workflow:
- `data/diagnostics/pionex_grid_recommendation_v1.json`
- `data/diagnostics/pionex_grid_calibration_v1.json`
- `data/pionex/grid_recommendations_v1.csv`

## Safety / scope
- diagnostic decision support only;
- no Pionex API connection;
- no order placement;
- no automatic bot edits;
- Phase 4C optimises **grid profit + range-escape risk** only.

Inventory/trend P&L risk is intentionally deferred to Phase 4D.
