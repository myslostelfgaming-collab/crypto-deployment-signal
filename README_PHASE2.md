# Crypto Prediction Tool — Phase 2: Model Tournament + Prospective Shadow Test

## Purpose

Phase 2 does **not** replace Similarity Forecast V2. It installs a diagnostic tournament and starts a forward-only shadow scoreboard so that simple challengers can prove themselves prospectively before any live model switch.

## Historical findings at installation

Using 7,605 evaluated feature-matched predictions and a horizon-spaced cleaner sample:

### Cleaner directional tournament

| Horizon | Similarity V2 | 48h Mean Reversion | Best current transparent challenger |
|---|---:|---:|---|
| 24h | 46.86% | **56.52%** | 48h mean reversion |
| 48h | 48.54% | **59.22%** | 48h mean reversion |
| 168h | 44.83% | 44.83% | SMA24 trend 65.52% (n=29; too small) |
| 336h | 42.86% | 64.29% | 48h mean reversion (n=14; far too small) |

Short horizons combined (24h + 48h):
- Similarity V2: about **47.4%**
- 48h mean reversion: about **57.4%**

The long-horizon independent samples are too small to promote anything.

### Adaptive selector result

The first walk-forward regime selector **failed**:
- eligible 24h/48h cleaner tests: n=230
- adaptive selector: **44.35%**
- V2 on same rows: **47.83%**
- fixed 48h mean reversion on same rows: **55.65%**

This is intentionally preserved in the output. It is evidence that switching among recently successful rules can overfit/noise-chase.

## New files

### `scripts/build_model_tournament_v1.py`
Runs the historical transparent-rule tournament. Candidate rules:
- Similarity V2
- 24h momentum
- 48h momentum
- 24h mean reversion
- 48h mean reversion
- SMA24 trend
- SMA48 trend
- SMA24 reversion
- SMA48 reversion

It also classifies a simple contemporaneous regime:
- `trend_up`
- `trend_down`
- `range`

The adaptive selector is walk-forward-safe: it may only learn from outcomes whose target timestamp had already matured before the test prediction timestamp.

### `scripts/validate_model_tournament_v1.py`
Fails the workflow if the tournament output is missing or structurally invalid.

### `scripts/build_shadow_challengers_v1.py`
Starts a **forward-only** challenger log. It does not backfill historical calls.

On the first GitHub run it creates:
- `data/model/shadow_predictions_v1.csv`
- `data/model/shadow_challenger_performance_v1.json`

Each new hourly batch records these directional calls before their outcomes mature:
- Similarity V2
- 48h mean reversion
- 24h mean reversion
- SMA24 trend
- SMA48 trend

Later runs automatically score old shadow rows when `predictions_v1.csv` has matured them.

### `data/diagnostics/model_tournament_v1.json`
Current historical tournament output included for reference. The workflow regenerates it hourly.

### Updated integration files
- `.github/workflows/deployment-signal.yml`
- `scripts/build_ai_handoff_v1.py`
- `scripts/build_repo_diagnostics.py`

The AI handoff now contains:
- `prediction_audit_state`
- `model_tournament_state`
- `shadow_challenger_state`

## Important interpretation rule

Historical discovery and prospective confirmation are different things.

The 48h mean-reversion rule is the current **historical benchmark**, but it should not replace V2 yet because it was discovered using the same historical dataset on which it looks strong.

The shadow log exists to answer the next question cleanly:

> After 7 August 2026, does the simple mean-reversion edge continue when the calls are recorded before the outcomes happen?

## Promotion policy

No automatic live switch is implemented.

Current provisional short-horizon guardrail:
- at least 100 eligible cleaner/prospective observations;
- >55% directional accuracy;
- beat Similarity V2;
- beat the fixed 48h mean-reversion benchmark if proposing a more complex model;
- survive more than one market regime.

## Install

Copy the contents of this package into the repository root, preserving folders, then commit/push.

On the next workflow run confirm that these are created/updated:

```text
data/diagnostics/model_tournament_v1.json
data/model/shadow_predictions_v1.csv
data/model/shadow_challenger_performance_v1.json
data/ai_handoff_v1.json
```

The live Similarity V2 prediction formula and Actionability V1 formula are unchanged in this phase.
