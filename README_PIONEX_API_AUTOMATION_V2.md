# Pionex API Automation v2

This promotes the validated read-only Pionex Bot API to the primary live-state source while preserving the historical screenshot dataset as an immutable audit trail.

## Data layers

### Raw API ledger
`data/pionex/api_grid_states_v1.csv`

Captured hourly. It is **not** fed directly into calibration.

### Historical manual archive
`data/pionex/manual_grid_states_v1.csv`

Retained unchanged. It contains the screenshot observations that established the first prospective calibration windows.

### Canonical model history
`data/pionex/model_grid_states_v1.csv`

On first run, this is seeded from the historical manual archive. Thereafter API observations are added only when:

- the range / grid count / quantity per grid changes; or
- at least 23 hours have elapsed since the previous canonical state.

This preserves useful 14–30h calibration windows.

### Runtime Phase 4D history
`data/pionex/runtime_grid_states_v1.csv`

Canonical history plus at most one fresh latest API state. The automated workflow temporarily supplies this file at the legacy Phase 4D state path, runs Phase 4D, and restores the original manual file immediately afterward.

No Phase 4D core model script is changed in v2.

## Confirmed UI/API mappings

- `exchangeOrderPairedCount` = Pionex Transaction tab `History` completed rounds.
- Range, grid count, quantity/grid, balances, grid profit, start price and settings match the UI.
- Current P&L is reconstructed mark-to-market:
  `ETH balance * ETHUSDT price + USDT balance - investment`.
- Trend P&L = total P&L - grid profit.
- `trx24h` is **not** the UI 24h completed-round count.

Rolling 24h rounds are derived from paired-count differences once the hourly API ledger has enough history.

## Workflows

### Pionex Read-Only API Snapshot CI
Runs hourly at minute 22 UTC and can also be dispatched manually.

### Pionex Automated Phase 4D Decision CI
Runs every 3 hours at 04:42, 07:42, 10:42, 13:42, 16:42, 19:42, 22:42 and 01:42 SAST, and can also be dispatched manually.

It:
1. captures a fresh API state;
2. updates canonical/runtime model histories;
3. temporarily substitutes the runtime state for the hard-coded legacy state path;
4. runs the complete existing Phase 4D pipeline;
5. restores the manual archive;
6. validates that Phase 4D actually consumed the fresh API timestamp;
7. persists API/model/decision outputs.

## Safety

The Pionex collector remains GET-only and uses the Bot-reading key. No Bot-trading endpoint is called.

The original manual screenshot CSV is checked for mutations after every automated full run and the workflow fails if it changed.

## Install

Replace:
- `.github/workflows/pionex-api-snapshot-ci.yml`

Add:
- `scripts/build_pionex_model_state_v1.py`
- `.github/workflows/pionex-automated-phase4d-ci.yml`

Optional:
- `README_PIONEX_API_AUTOMATION_V2.md`

Then manually run `Pionex Automated Phase 4D Decision CI` once. If it passes, both schedules can be left enabled.
