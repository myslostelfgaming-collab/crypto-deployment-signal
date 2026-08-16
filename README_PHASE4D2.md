# Phase 4D.2 — Pionex Grid Activity & Actionability

Phase 4D.2 is the **operational gate above the Phase 4D research optimiser**.

It does not place trades or connect to Pionex.

## Why this layer exists

Two issues became clear during prospective validation:

1. The Phase 4D research optimiser can use a stale screenshot state even while the market data is current.
2. `0 completed rounds` does not necessarily mean `0 grid fills`. A buy can occur and remain unmatched until its corresponding sell completes.

Phase 4D.2 makes those distinctions explicit.

## Improvements

### 1. Fresh-state hard gate

A geometry change is never operationally actionable when the newest Pionex screenshot is older than **6 hours**.

The research result is still recorded for study, but:

`operational_action = KEEP_CURRENT`

### 2. Exact live baseline quantity

For the current bot benchmark, the authoritative quantity/grid is the value observed in Pionex.

Candidate geometries may estimate a different quantity, but that estimate is not allowed to overwrite the real baseline.

### 3. Open-leg / grid-activity accounting

For every consecutive manual screenshot pair, Phase 4D.2 records:

- completed round delta;
- grid-profit delta;
- ETH and USDT holdings delta;
- ETH delta expressed in grid units;
- whether the holdings delta is close enough to an integer grid quantity to infer a net unmatched buy/sell leg;
- a conservative lower bound on total fills.

This catches states such as:

> 0 completed rounds in 24h, but one net buy leg is still waiting to sell.

### 4. Nearest waiting grid triggers

The diagnostic reports the nearest waiting:

- buy trigger;
- sell trigger;
- percentage distance to each.

### 5. Literal practical rounding

The raw research geometry remains untouched.

For usability, a separate practical display literally rounds lower/upper bounds to the nearest **$5**. It no longer pretends a nearby unrounded candidate is the rounded value.

Metrics are **not** falsely transferred to the rounded geometry.

### 6. Phase 4D.1 integration

If `data/pionex/execution_resolution_policy_v1.json` exists, Phase 4D.2 reads it.

A research geometry change can become operationally actionable only after:

- current Pionex state ≤ 6h old;
- prospective calibration is active with ≥3 windows;
- execution-resolution validation has ≥3 windows and has resolved to either:
  - `PROMOTION_READY`, or
  - `KEEP_1H_EXECUTION_RESOLUTION`;
- research geometry confidence is at least `LOW_MEDIUM`.

Until all gates pass, the bot remains:

`KEEP_CURRENT`

## New 16 Aug prospective state

The supplied `manual_grid_states_v1.csv` adds:

- 2026-08-16 04:28 SAST
- current price 1884.44
- grid profit 172.54 USDT
- total rounds 4274
- 24h completed rounds 0
- ETH 0.04467
- USDT 259.14
- quantity/grid 0.00318 ETH
- unchanged 1750–2000 / 30-grid geometry.

The gap from 14 Aug is too long for the strict 14–30h calibration rule, but it remains a valid observed-activity window and becomes the anchor for the next daily calibration pair.

## Files

Upload/replace:

- `scripts/build_pionex_grid_activity_v1.py`
- `.github/workflows/phase4d2-grid-actionability-ci.yml`
- `data/pionex/manual_grid_states_v1.csv`

Then run:

**Actions → Phase 4D.2 Pionex Grid Activity & Actionability CI → Run workflow**

Phase 4D.1 is optional at first run, but its execution-resolution policy is required before a geometry change can ever pass the operational gate.
