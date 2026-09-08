# Grid-density sweet-spot repo update

## Why this change

The existing Phase 4D optimizer already searches grid counts from 10 to 80, but
the persisted diagnostics do not expose the full density curve. That means a
10-grid expected-value winner can obscure whether 30–80 grids would produce many
more profitable cycles, and it does not expose P(0 rounds) or P(50+ rounds).

This patch adds a diagnostic-only v4 adapter. It captures the exact candidates
already evaluated by the current v3.1 live-price/recovery + promoted 5-minute
replay pipeline, so there is no second optimizer sweep and no change to the live
actionability decision.

## Apply these changes

### 1. Add

`scripts/build_pionex_grid_geometry_optimizer_v4.py`

Use the supplied file unchanged.

### 2. Replace

`scripts/run_pionex_phase4d_full_v2.py`

with the supplied replacement file.

The only behavioural routing change is:

- geometry v2 request -> `build_pionex_grid_geometry_optimizer_v4.py`
- activity v1 request -> existing activity v2

After the legacy full pipeline finishes, the runner adds a compact
`grid_density_sweep` block to `pionex_full_decision_v1.json`.

### 3. Update `.github/workflows/pionex-automated-phase4d-ci.yml`

In **Validate immutable manual archive and full outputs**, add:

```bash
test -s data/diagnostics/pionex_grid_density_sweep_v1.json
```

In **Upload automated Phase 4D bundle**, add:

```yaml
data/diagnostics/pionex_grid_density_sweep_v1.json
```

In **Persist API, state-model, and Phase 4D outputs**, add:

```bash
data/diagnostics/pionex_grid_density_sweep_v1.json \
```

to the `git add` list.

Recommended validation inside the existing Python validation step:

```python
density = json.loads(
    Path("data/diagnostics/pionex_grid_density_sweep_v1.json").read_text(
        encoding="utf-8"
    )
)
if density.get("schema") != "pionex_grid_density_sweep_v1":
    raise SystemExit(f"Unexpected density schema: {density.get('schema')}")

if (
    (density.get("source_state") or {}).get("captured_at_utc")
    != latest_source.get("captured_at_utc")
):
    raise SystemExit("Density sweep did not use the latest API runtime state")

if (density.get("headline") or {}).get("operational_override") is not False:
    raise SystemExit("Density sweep must remain diagnostic-only")
```

### 4. Do NOT change actionability yet

The first pass is evidence collection only. Do not make the density champion
operational and do not replace the current Phase 4D selection rule yet.

## New output

`data/diagnostics/pionex_grid_density_sweep_v1.json`

It contains:

- `joint_best_by_grid_count`
  - best feasible and best standard-risk-eligible geometry for each grid count;
- `current_band_density_curve`
  - holds the current live lower/upper bounds fixed and changes ONLY grid count;
- per-candidate:
  - grid spacing,
  - expected/median grid profit,
  - expected/median rounds,
  - profit per expected round,
  - P(0 rounds),
  - P(1+), P(2+), P(5+), P(10+), P(25+), P(50+) raw rounds,
  - calibrated equivalents,
  - p90/p95/max raw rounds,
  - positive/zero grid-profit probabilities;
- headline:
  - expected-profit champion,
  - median-profit champion,
  - activity champion,
  - current-band expected-profit champion,
  - current-band activity champion,
  - strongest evidence for 50+ rounds,
  - candidates within 90% of peak expected profit sorted toward fewer dead days,
  - warning if the expected-profit optimum is still on a tested grid-count boundary.

## Acceptance checks

Run:

```bash
python -m py_compile \
  scripts/build_pionex_grid_geometry_optimizer_v4.py \
  scripts/run_pionex_phase4d_full_v2.py
```

Then run the automated Phase 4D workflow or its equivalent full runner.

The run must satisfy:

1. Existing Phase 4D outputs still validate.
2. `pionex_grid_density_sweep_v1.json` exists and is non-empty.
3. Its source state equals the fresh API runtime state used by the full decision.
4. `captured_grid_counts` includes the coarse 10, 15, 20 ... 80 search counts
   wherever they are feasible; refined neighbouring counts may also appear.
5. `current_band_density_curve` contains the live grid count and attempts the
   full coarse density series on the same price band.
6. `headline.operational_override == false`.
7. Existing `pionex_grid_actionability_v1.json` semantics are unchanged.
8. `pionex_full_decision_v1.json` gains a `grid_density_sweep` block.

## What to inspect after the first successful run

Do not focus only on the champion. Compare, for the CURRENT price band:

- grids,
- spacing,
- expected grid profit,
- median grid profit,
- P(0 rounds),
- expected raw rounds,
- p90/p95/max rounds,
- P(10+),
- P(25+),
- P(50+).

Then compare that curve with the joint-best-by-grid-count curve.

The immediate research question is:

> Does expected net 24h profit peak at low density because of rare chunky
> outcomes, or is there a higher-frequency density where many smaller cycles
> produce greater and more robust total profit?

Only after reviewing that evidence should the density layer be considered for
promotion into the operational gate.
