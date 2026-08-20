# Pionex Read-Only API Collector v1

This is a validation-stage collector for the live ETH/USDT Spot Grid bot.

It calls only three GET endpoints:

- `GET /api/v1/bot/orders`
- `GET /api/v1/bot/orders/spotGrid/order`
- `GET /api/v1/market/tickers`

The first two require **Bot reading**. The script contains no Bot trading request and never prints or stores the API key/secret. It also strips `userId`, `keyId`, and raw bot IDs before persisting data to the public repository.

## Outputs

- `data/pionex/api_bot_state_latest_v1.json`
- `data/pionex/api_grid_states_v1.csv`
- `data/diagnostics/pionex_api_parity_v1.json`

The first version intentionally does **not** modify `manual_grid_states_v1.csv`. We first compare API vs UI and inspect any undocumented response-key names. This protects the existing prospective calibration history.

## Round-count limitation

The current official Spot Grid schema documents range, grids, per-grid volume, bot balances, grid profit, realized profit, fees, average cost, TP/SL/trigger/reinvest and related fields, but does not document rolling 24h rounds, lifetime rounds, or average transactions/day.

The collector therefore records the names (not values) of any undocumented fields returned by Pionex. If useful round-count fields appear, we can investigate them. Otherwise the next test is whether ordinary read-only trade fills include bot-account fills.

## Install and test

Copy these two files to the exact repo paths:

- `scripts/pull_pionex_grid_state_v1.py`
- `.github/workflows/pionex-api-snapshot-ci.yml`

Then run:

**Actions → Pionex Read-Only API Snapshot CI → Run workflow**

After the first successful run, inspect `data/diagnostics/pionex_api_parity_v1.json` before promoting the API as the primary calibration-state source.
