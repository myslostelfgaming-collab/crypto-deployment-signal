# Crypto Prediction Tool — Phase 1 Audit Upgrade

This patch is diagnostic-only. It does **not** change the live V2 prediction formula.

## Files to replace/add

- `.github/workflows/deployment-signal.yml`
- `scripts/audit_predictions_v2.py` (new)
- `scripts/build_ai_handoff_v1.py`
- `scripts/build_repo_diagnostics.py`
- `data/diagnostics/prediction_audit_v2.json` (current baseline output; future Actions runs regenerate it)

## What the audit adds

- Direction confusion matrix and balanced accuracy
- Majority-class and inverse-model benchmarks
- Asset × horizon performance
- Confidence and analogue-quality performance
- Conviction thresholds (tests whether larger predicted moves are actually more reliable)
- Less-overlapping evaluation sample
- Simple momentum / mean-reversion / SMA baseline tournament
- Monthly short-horizon benchmark
- Paired V2 vs 48h mean-reversion comparison on the less-overlapping 24h+48h sample
- Compact audit block in `ai_handoff_v1.json`

## Baseline findings from the 2026-08-07 repository snapshot

- 7,605 evaluated predictions
- V2 overall direction accuracy: ~46.61%
- Majority-class baseline: ~50.76%
- Less-overlapping V2 accuracy: ~47.03%
- 24h V2: ~51.25%
- 48h V2: ~48.58%
- 168h V2: ~42.75%
- 336h V2: ~43.21%
- On the less-overlapping 24h+48h sample:
  - V2: ~47.42%
  - simple 48h mean-reversion rule: ~57.10%
  - paired exact p-value: ~0.0095
- Accuracy falls as predicted magnitude rises:
  - all calls: ~46.61%
  - |prediction| >= 1%: ~43.83%
  - |prediction| >= 2%: ~41.37%
  - |prediction| >= 3%: ~38.05%

## Recommended next phase

Do not invert V2 globally. The long-horizon anti-signal is regime-dependent and was especially severe during May 2026.

Next build a V3 experimental backtester with separate objectives:

1. Short-horizon/grid head: range + regime + mean-reversion/trend gating.
2. Luno head: minimise BTC absolute price error at 48h and 336h, beginning with persistence as the benchmark.
3. Only promote a new model after walk-forward testing beats the relevant benchmark.
