# Phase 3 — Luno Price Predict Head

Phase 3 separates the Luno competition objective from the general directional/grid research pipeline.

## Objective

Luno Price Predict is scored by **absolute BTC USD price error at a fixed settlement timestamp**, not by directional accuracy.

This module therefore treats the problem as:

`minimise | forecast BTC USD price - settlement BTC USD price |`

The existing Similarity V2 model remains untouched.

## What this phase adds

### 1. Luno-native price proxy collector

`scripts/build_luno_price_proxy_v1.py`

Every workflow run queries Luno's public ticker API and records:

- XBTZAR
- USDTZAR
- USDCZAR
- XBTUSDT when available
- XBTUSDC when available
- a robust median Luno BTC/USD proxy
- the existing market BTC-USDT reference
- the basis between the Luno proxy and the market reference

Outputs:

- `data/luno/luno_price_proxy_latest_v1.json`
- `data/luno/luno_price_proxy_history_v1.csv`

The proxy is **not** labelled as the true competition settlement rate. Luno Price Predict uses Luno's internal market rate, which may differ from exchange/app prices.

### 2. Exact-clock historical tournament

`scripts/build_luno_point_tournament_v1.py`

Instead of assuming generic 48h and 336h labels exactly match the competition clock, the backtest uses:

- weekly: latest repo BTC snapshot before Wednesday 12:00 UTC -> settlement proxy nearest Friday 12:00 UTC
- monthly: latest repo BTC snapshot before day 14 12:00 UTC -> settlement proxy nearest day 28 12:00 UTC

Historical settlement truth is the repo's KuCoin BTC-USDT price proxy, not the exact Luno internal rate.

Output:

- `data/diagnostics/luno_point_tournament_v1.json`

Current historical proxy result from the supplied August repo:

| Challenge | Model | n | MAE USD |
|---|---|---:|---:|
| Weekly | Persistence/current price | 16 | 1535.45 |
| Weekly | Similarity V2 | 11 | 2215.84 |
| Weekly | Full 48h mean reversion | 16 | 2468.30 |
| Weekly | Shrunk 48h mean reversion | 16 | 1620.65 |
| Monthly | Persistence/current price | 4 | 3231.20 |
| Monthly | Similarity V2 | 3 | 3872.72 |

Persistence remains champion. The monthly sample is far too small for promotion decisions.

The backtest also confirms a target-alignment issue: the actual competition clock differs from our generic horizon by roughly an hour because forecasts are made before the 12:00 UTC submission cutoff.

### 3. Prospective Luno competition shadow

`scripts/build_luno_shadow_v1.py`

While a weekly/monthly competition is open, the script keeps replacing the shadow entry with the **latest snapshot before cutoff**. After the cutoff, that entry is frozen.

It records these candidates:

- current market price / persistence
- Luno-proxy persistence
- Similarity V2 applied to market price
- Similarity V2 applied to Luno proxy
- 48h mean-reversion price candidate

For now, the recommended candidate remains **market persistence**. The Luno proxy is a calibration challenger until exact Luno results prove it is better aligned.

Outputs:

- `data/luno/luno_shadow_entries_v1.csv`
- `data/luno/luno_shadow_latest_v1.json`
- `data/luno/luno_shadow_performance_v1.json`
- `data/luno/luno_manual_results_v1.csv`

Matured entries are automatically scored against the repo's BTC-USDT settlement proxy.

If an exact Luno recorded settlement price becomes available, add it to `luno_manual_results_v1.csv`. The script will then separately score market persistence, Luno-proxy persistence and the similarity candidates against the exact Luno result.

## Workflow change

The normal hourly workflow remains at `07 * * * *` UTC.

Phase 3 adds a second daily run at:

`37 11 * * *` UTC

On Luno cutoff days this gives the system a fresher pre-deadline candidate than the normal 11:07 UTC run while retaining some buffer before 12:00 UTC.

## AI handoff

`data/ai_handoff_v1.json` now includes:

`luno_price_predict_state`

with:

- latest Luno price proxy and basis
- exact-clock point-price tournament
- prospective Luno shadow state
- prospective Luno performance

## Promotion rule

Do not promote Similarity V2, mean reversion, the Luno price proxy, or any future challenger merely because it wins a small historical sample.

The Luno head should only move away from persistence after prospective evidence demonstrates lower absolute price error.
