# Phase 4A — ETH/Pionex Grid Tournament v1

Scope: ETH/USDT Spot Grid on Pionex, 24-hour horizon.

This package is diagnostic only. It does not modify the live directional/Luno model and it does not place trades.

Files:
- scripts/build_grid_tournament_v1.py
- data/pionex/pionex_grid_profile_v1.json
- data/pionex/manual_grid_states_v1.csv
- .github/workflows/phase4-grid-ci.yml

Install:
1. Copy the files into the same repo-relative paths on main.
2. Commit them.
3. In GitHub Actions, run `Phase 4 ETH Pionex Grid CI` manually.
4. The workflow validates the tournament and uploads `grid-tournament-v1` as an artifact.

Phase 4A tests ETH 24h realised-range predictors using walk-forward calibration with outcome maturity and a 24h-spaced independence diagnostic.

Phase 4B, after Phase 4A validation, will add the Pionex path/grid simulator and probability of achieving specified 24h grid-return thresholds.
