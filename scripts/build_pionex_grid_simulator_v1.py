#!/usr/bin/env python3

import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from glob import glob
from typing import Dict, List, Optional, Tuple

HISTORY_ROOT = os.path.join('data', 'history')
FEATURES_PATH = os.path.join('data', 'features', 'features_v1.csv')
PROFILE_PATH = os.path.join('data', 'pionex', 'pionex_grid_profile_v1.json')
OUT_PATH = os.path.join('data', 'diagnostics', 'pionex_grid_simulator_v1.json')

ASSET = 'ETH-USDT'
CANDLE_KEY = 'eth_usdt_1h'
HORIZON_H = 24
INDEPENDENCE_H = 24
ATR_ANALOG_N = 60


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_float(v) -> Optional[float]:
    try:
        if v in (None, ''):
            return None
        return float(v)
    except Exception:
        return None


def extract_candles(snapshot: dict) -> List[List[float]]:
    # History logger stores compact candles as [ts, open, high, low, close, volume].
    candles = snapshot.get('signal', {}).get('candles', {}).get(CANDLE_KEY)
    if candles is None:
        candles = snapshot.get('candles', {}).get(CANDLE_KEY, [])
    out = []
    if not isinstance(candles, list):
        return out
    for c in candles:
        if not isinstance(c, list) or len(c) < 6:
            continue
        try:
            ts = int(c[0])
            o, h, l, cl, v = map(float, c[1:6])
        except Exception:
            continue
        # Cheap schema guard: a valid OHLC candle must contain O and C inside [L,H].
        if h < max(o, cl) - 1e-9 or l > min(o, cl) + 1e-9 or h < l:
            continue
        out.append([ts, o, h, l, cl, v])
    out.sort(key=lambda x: x[0])
    return out


def build_master_candles() -> Dict[int, List[float]]:
    master: Dict[int, List[float]] = {}
    for path in sorted(glob(os.path.join(HISTORY_ROOT, '*', '*.json'))):
        try:
            snap = load_json(path)
        except Exception:
            continue
        for c in extract_candles(snap):
            master.setdefault(int(c[0]), c)
    return master


def load_eth_features() -> List[dict]:
    rows = []
    with open(FEATURES_PATH, 'r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('asset') != ASSET:
                continue
            try:
                ts = int(float(r.get('entry_ts_utc') or 0))
            except Exception:
                continue
            atr = to_float(r.get('atr14_pct'))
            entry = to_float(r.get('entry_close'))
            if not ts or atr is None or atr <= 0 or entry is None or entry <= 0:
                continue
            rows.append({
                'ts': ts,
                'entry_close': entry,
                'atr14_pct': atr,
                'range_24h_pct': to_float(r.get('range_24h_pct')),
                'range_48h_pct': to_float(r.get('range_48h_pct')),
                'published_at_utc': r.get('published_at_utc') or '',
            })
    # Dedupe identical market-state timestamps, preferring the latest published row.
    dedup: Dict[int, dict] = {}
    for r in rows:
        old = dedup.get(r['ts'])
        if old is None or r['published_at_utc'] >= old['published_at_utc']:
            dedup[r['ts']] = r
    return [dedup[k] for k in sorted(dedup)]


def forward_24h(master: Dict[int, List[float]], entry_ts: int) -> Optional[List[List[float]]]:
    out = []
    for k in range(1, HORIZON_H + 1):
        c = master.get(entry_ts + 3600 * k)
        if c is None:
            return None
        out.append(c)
    return out


def greedy_independent(rows: List[dict], spacing_h: int = INDEPENDENCE_H) -> List[dict]:
    out = []
    last_ts = None
    spacing = spacing_h * 3600
    for r in sorted(rows, key=lambda x: x['ts']):
        if last_ts is None or r['ts'] - last_ts >= spacing:
            out.append(r)
            last_ts = r['ts']
    return out


def grid_lines(lower: float, upper: float, grids: int) -> List[float]:
    # The supplied Pionex screen is exactly matched by treating "30 grids" as
    # 30 displayed grid price levels, hence 29 equal arithmetic intervals.
    if grids < 2 or upper <= lower:
        raise ValueError('Invalid grid geometry')
    step = (upper - lower) / (grids - 1)
    return [lower + i * step for i in range(grids)]


def interval_net_profit_usdt(buy: float, sell: float, qty_eth: float, fee_rate: float) -> float:
    gross = qty_eth * (sell - buy)
    fees = fee_rate * qty_eth * (buy + sell)
    return gross - fees


def interval_net_profit_pct(buy: float, sell: float, fee_rate: float) -> float:
    # Quantity cancels from the percentage return on the buy-side notional.
    if buy <= 0:
        return 0.0
    return ((sell - buy) - fee_rate * (buy + sell)) / buy * 100.0


def initial_states(lines: List[float], start_price: float) -> List[str]:
    # Each interval is either waiting for its lower buy ('buy') or upper sell ('sell').
    # In a mature Pionex grid, intervals above the current pivot have seeded ETH and
    # wait to sell; intervals below the pivot wait to buy back after prior sells.
    pivot = 0
    for i, level in enumerate(lines):
        if level <= start_price:
            pivot = i
        else:
            break
    pivot = min(pivot, len(lines) - 2)
    states = []
    for i in range(len(lines) - 1):
        states.append('buy' if i < pivot else 'sell')
    return states


def process_segment(a: float, b: float, lines: List[float], states: List[str], qty: float, fee_rate: float) -> Tuple[int, float]:
    rounds = 0
    profit = 0.0
    if b > a:
        # Upward: sell triggers occur at interval upper boundaries.
        for i in range(len(lines) - 1):
            trigger = lines[i + 1]
            if a < trigger <= b and states[i] == 'sell':
                rounds += 1
                profit += interval_net_profit_usdt(lines[i], lines[i + 1], qty, fee_rate)
                states[i] = 'buy'
    elif b < a:
        # Downward: buy triggers occur at interval lower boundaries.
        for i in range(len(lines) - 2, -1, -1):
            trigger = lines[i]
            if b <= trigger < a and states[i] == 'buy':
                states[i] = 'sell'
    return rounds, profit


def simulate_path(mapped_candles: List[List[float]], start_price: float, lower: float, upper: float,
                  grids: int, qty: float, fee_rate: float, mode: str) -> dict:
    lines = grid_lines(lower, upper, grids)
    states = initial_states(lines, start_price)
    rounds = 0
    profit = 0.0
    lower_escape = False
    upper_escape = False
    min_price = start_price
    max_price = start_price
    prev = start_price

    for c in mapped_candles:
        _, o, h, l, cl, _ = c
        min_price = min(min_price, l)
        max_price = max(max_price, h)
        if l < lower:
            lower_escape = True
        if h > upper:
            upper_escape = True

        if mode == 'ohlc':
            pts = [o, h, l, cl]
        elif mode == 'olhc':
            pts = [o, l, h, cl]
        else:
            raise ValueError(mode)

        # Include any gap from previous close to current open.
        pts = [prev] + pts
        for a, b in zip(pts, pts[1:]):
            r, p = process_segment(a, b, lines, states, qty, fee_rate)
            rounds += r
            profit += p
        prev = cl

    return {
        'rounds': rounds,
        'grid_profit_usdt': profit,
        'lower_escape': lower_escape,
        'upper_escape': upper_escape,
        'any_escape': lower_escape or upper_escape,
        'min_price': min_price,
        'max_price': max_price,
        'end_price': prev,
        'sell_ready_intervals_end': sum(1 for x in states if x == 'sell'),
    }


def map_historical_path_to_current(entry_close: float, fwd: List[List[float]], current_price: float) -> List[List[float]]:
    scale = current_price / entry_close
    out = []
    for c in fwd:
        ts, o, h, l, cl, v = c
        out.append([ts, o * scale, h * scale, l * scale, cl * scale, v])
    return out


def simulate_historical_case(row: dict, master: Dict[int, List[float]], cfg: dict) -> Optional[dict]:
    fwd = forward_24h(master, row['ts'])
    if not fwd:
        return None
    mapped = map_historical_path_to_current(row['entry_close'], fwd, cfg['current_price'])
    a = simulate_path(mapped, cfg['current_price'], cfg['lower'], cfg['upper'], cfg['grids'], cfg['qty'], cfg['fee_rate'], 'ohlc')
    b = simulate_path(mapped, cfg['current_price'], cfg['lower'], cfg['upper'], cfg['grids'], cfg['qty'], cfg['fee_rate'], 'olhc')
    lo_profit = min(a['grid_profit_usdt'], b['grid_profit_usdt'])
    hi_profit = max(a['grid_profit_usdt'], b['grid_profit_usdt'])
    lo_rounds = min(a['rounds'], b['rounds'])
    hi_rounds = max(a['rounds'], b['rounds'])
    return {
        'ts': row['ts'],
        'atr14_pct': row['atr14_pct'],
        'profit_conservative': lo_profit,
        'profit_optimistic': hi_profit,
        'profit_mid': (lo_profit + hi_profit) / 2.0,
        'rounds_conservative': lo_rounds,
        'rounds_optimistic': hi_rounds,
        'rounds_mid': (lo_rounds + hi_rounds) / 2.0,
        'any_escape': a['any_escape'] or b['any_escape'],
        'lower_escape': a['lower_escape'] or b['lower_escape'],
        'upper_escape': a['upper_escape'] or b['upper_escape'],
        'end_return_pct': ((a['end_price'] / cfg['current_price']) - 1.0) * 100.0,
    }


def percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    x = (len(s) - 1) * p
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


def dist_summary(vals: List[float]) -> dict:
    if not vals:
        return {'n': 0}
    return {
        'n': len(vals),
        'mean': round(statistics.fmean(vals), 6),
        'median': round(statistics.median(vals), 6),
        'p20': round(percentile(vals, 0.20), 6),
        'p50': round(percentile(vals, 0.50), 6),
        'p80': round(percentile(vals, 0.80), 6),
        'p90': round(percentile(vals, 0.90), 6),
        'p95': round(percentile(vals, 0.95), 6),
    }


def probability_ge(vals: List[float], threshold: float) -> Optional[float]:
    if not vals:
        return None
    return round(sum(v >= threshold for v in vals) / len(vals) * 100.0, 4)


def candidate_summary(cases: List[dict], cfg: dict, profile: dict) -> dict:
    profits_c = [x['profit_conservative'] for x in cases]
    profits_m = [x['profit_mid'] for x in cases]
    profits_o = [x['profit_optimistic'] for x in cases]
    rounds_m = [x['rounds_mid'] for x in cases]
    current_equity = profile['current_equity_usdt']
    original = profile['original_investment_usdt']

    dollar_thresholds = profile['dollar_thresholds']
    return_thresholds = profile['return_thresholds']

    by_dollar = {}
    for t in dollar_thresholds:
        by_dollar[str(t)] = {
            'conservative_pct': probability_ge(profits_c, t),
            'midpoint_pct': probability_ge(profits_m, t),
            'optimistic_pct': probability_ge(profits_o, t),
        }

    by_return = {'current_bot_equity': {}, 'original_investment_usdt': {}}
    for r in return_thresholds:
        th_equity = current_equity * r / 100.0
        th_original = original * r / 100.0
        by_return['current_bot_equity'][str(r)] = {
            'threshold_usdt': round(th_equity, 6),
            'midpoint_probability_pct': probability_ge(profits_m, th_equity),
        }
        by_return['original_investment_usdt'][str(r)] = {
            'threshold_usdt': round(th_original, 6),
            'midpoint_probability_pct': probability_ge(profits_m, th_original),
        }

    n = len(cases)
    return {
        'configuration': {
            'lower_usdt': round(cfg['lower'], 6),
            'upper_usdt': round(cfg['upper'], 6),
            'width_usdt': round(cfg['upper'] - cfg['lower'], 6),
            'grids': cfg['grids'],
            'quantity_per_grid_eth': cfg['qty'],
            'fee_rate_pct_per_fill': round(cfg['fee_rate'] * 100.0, 6),
        },
        'sample_n': n,
        'grid_profit_usdt': {
            'conservative': dist_summary(profits_c),
            'midpoint': dist_summary(profits_m),
            'optimistic': dist_summary(profits_o),
        },
        'rounds_midpoint': dist_summary(rounds_m),
        'probability_grid_profit_ge_dollar_threshold': by_dollar,
        'probability_grid_return_ge_pct': by_return,
        'range_escape_probability_pct': round(sum(x['any_escape'] for x in cases) / n * 100.0, 4) if n else None,
        'lower_escape_probability_pct': round(sum(x['lower_escape'] for x in cases) / n * 100.0, 4) if n else None,
        'upper_escape_probability_pct': round(sum(x['upper_escape'] for x in cases) / n * 100.0, 4) if n else None,
        'end_return_pct': dist_summary([x['end_return_pct'] for x in cases]),
    }


def make_cfg(ref: dict, fee_rate: float, shift: float) -> Optional[dict]:
    lower = float(ref['lower_limit_usdt']) + shift
    upper = float(ref['upper_limit_usdt']) + shift
    current = float(ref['current_price_usdt'])
    if not (lower <= current <= upper):
        return None
    return {
        'lower': lower,
        'upper': upper,
        'current_price': current,
        'grids': int(ref['grids']),
        'qty': float(ref['quantity_per_grid_eth']),
        'fee_rate': fee_rate,
        'shift_usdt': shift,
    }


def main():
    if not os.path.isfile(FEATURES_PATH):
        raise SystemExit(f'Missing {FEATURES_PATH}')
    if not os.path.isfile(PROFILE_PATH):
        raise SystemExit(f'Missing {PROFILE_PATH}')

    profile_raw = load_json(PROFILE_PATH)
    ref = profile_raw['current_reference_state']
    fee_pct = float(profile_raw.get('fee_model', {}).get('standard_public_spot_fee_pct_per_fill_reference', 0.05))
    fee_rate = fee_pct / 100.0

    current_equity = float(ref['eth_holdings']) * float(ref['current_price_usdt']) + float(ref['usdt_holdings'])
    model_profile = {
        'current_equity_usdt': current_equity,
        'original_investment_usdt': float(ref['investment_usdt']),
        'dollar_thresholds': [float(x) for x in profile_raw['objective']['report_dollar_thresholds_usdt_24h']],
        'return_thresholds': [float(x) for x in profile_raw['objective']['report_thresholds_return_pct_24h']],
    }

    master = build_master_candles()
    features = load_eth_features()
    if not features:
        raise SystemExit('No ETH features')
    latest = features[-1]

    matured = []
    for r in features:
        if forward_24h(master, r['ts']) is not None:
            matured.append(r)
    independent = greedy_independent(matured)

    # Current-regime analogues: use the Phase 4A historical champion signal (ATR14%) only.
    current_atr = latest['atr14_pct']
    analog_pool = sorted(independent, key=lambda r: abs(math.log(r['atr14_pct'] / current_atr)))
    analogs = analog_pool[: min(ATR_ANALOG_N, len(analog_pool))]

    shifts = sorted(set([0.0] + [float(x) for x in profile_raw['adjustment_policy'].get('typical_shift_usdt', [50, 100])] +
                        [-float(x) for x in profile_raw['adjustment_policy'].get('typical_shift_usdt', [50, 100])]))

    candidates = []
    for shift in shifts:
        cfg = make_cfg(ref, fee_rate, shift)
        if cfg is None:
            continue
        all_cases = [simulate_historical_case(r, master, cfg) for r in independent]
        all_cases = [x for x in all_cases if x is not None]
        analog_cases = [simulate_historical_case(r, master, cfg) for r in analogs]
        analog_cases = [x for x in analog_cases if x is not None]
        candidates.append({
            'shift_usdt': shift,
            'all_independent_history': candidate_summary(all_cases, cfg, model_profile),
            'current_atr_regime': candidate_summary(analog_cases, cfg, model_profile),
        })

    # Validate that our Pionex arithmetic-grid and fee interpretation reproduces the app's displayed profit/grid.
    base_cfg = make_cfg(ref, fee_rate, 0.0)
    lines = grid_lines(base_cfg['lower'], base_cfg['upper'], base_cfg['grids'])
    profit_pcts = [interval_net_profit_pct(lines[i], lines[i + 1], fee_rate) for i in range(len(lines) - 1)]
    displayed = [float(x) for x in ref['profit_per_grid_pct_fee_deducted']]
    model_min, model_max = min(profit_pcts), max(profit_pcts)
    grid_math_pass = abs(model_min - displayed[0]) <= 0.02 and abs(model_max - displayed[1]) <= 0.02

    # Rank current-regime candidate bands. Profit first, but require range-escape awareness rather than hiding it.
    ranking = []
    for c in candidates:
        s = c['current_atr_regime']
        ranking.append({
            'shift_usdt': c['shift_usdt'],
            'lower_usdt': s['configuration']['lower_usdt'],
            'upper_usdt': s['configuration']['upper_usdt'],
            'expected_grid_profit_usdt_midpoint': s['grid_profit_usdt']['midpoint'].get('mean'),
            'median_grid_profit_usdt_midpoint': s['grid_profit_usdt']['midpoint'].get('median'),
            'expected_rounds_midpoint': s['rounds_midpoint'].get('mean'),
            'range_escape_probability_pct': s['range_escape_probability_pct'],
            'p_profit_ge_0_50_usdt_pct': (s['probability_grid_profit_ge_dollar_threshold'].get('0.5') or {}).get('midpoint_pct'),
        })
    ranking.sort(key=lambda x: (-(x['expected_grid_profit_usdt_midpoint'] or -1e9), x['range_escape_probability_pct'] or 1e9))

    out = {
        'schema': 'pionex_grid_simulator_v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'DIAGNOSTIC_ONLY',
        'scope': {'platform': 'Pionex', 'pair': 'ETH/USDT', 'horizon_h': 24, 'bot_type': 'Spot Grid'},
        'methodology': {
            'historical_path_mapping': 'Each historical ETH 24h OHLC path is converted to returns from its entry price and remapped onto the current ETH price, so today\'s exact dollar grid geometry can be simulated.',
            'intrahour_uncertainty': 'Hourly candles do not reveal whether high or low happened first. Both O-H-L-C and O-L-H-C paths are simulated; conservative/midpoint/optimistic results are reported.',
            'steady_state_order_model': 'Intervals below the starting pivot wait to buy; intervals at/above the pivot are seeded and wait to sell, matching a mature running grid rather than a brand-new bot.',
            'grid_count_interpretation': '30 Pionex grids treated as 30 arithmetic price levels / 29 intervals because that exactly reproduces the supplied screen levels and displayed 0.33%-0.39% fee-deducted profit/grid.',
            'conditioning': f'Current-regime probabilities use the {ATR_ANALOG_N} independent historical states nearest to current ATR14%; Phase 4A identified ATR14 as the best simple 24h range signal.',
            'independence': 'Greedy 24h-spaced historical start times reduce hourly pseudo-replication.',
            'no_live_trading': True,
        },
        'data': {
            'master_eth_candles': len(master),
            'feature_rows_eth': len(features),
            'matured_24h_rows': len(matured),
            'independent_rows': len(independent),
            'atr_analog_rows': len(analogs),
            'current_feature_ts_utc': latest['ts'],
            'current_atr14_pct': current_atr,
        },
        'current_pionex_state': {
            'current_price_usdt': float(ref['current_price_usdt']),
            'lower_limit_usdt': float(ref['lower_limit_usdt']),
            'upper_limit_usdt': float(ref['upper_limit_usdt']),
            'grids': int(ref['grids']),
            'quantity_per_grid_eth': float(ref['quantity_per_grid_eth']),
            'current_equity_usdt_approx': round(current_equity, 6),
            'original_investment_usdt': float(ref['investment_usdt']),
            'rounds_24h_observed_at_capture': int(ref['rounds_24h']),
        },
        'grid_math_validation': {
            'fee_pct_per_fill_assumed': fee_pct,
            'model_profit_per_grid_pct_min': round(model_min, 6),
            'model_profit_per_grid_pct_max': round(model_max, 6),
            'pionex_displayed_profit_per_grid_pct': displayed,
            'matches_display_within_0_02pp': grid_math_pass,
        },
        'candidate_shift_policy': {'shift_both_bounds_together': True, 'candidate_shifts_usdt': [c['shift_usdt'] for c in candidates]},
        'candidates': candidates,
        'current_atr_regime_ranking_by_expected_grid_profit': ranking,
        'next_phase': {
            'phase': '4C',
            'name': 'Prospective Pionex return calibration',
            'requirements': [
                'append daily/manual Pionex state screenshots',
                'compare simulated vs observed 24h rounds and fee-deducted grid profit',
                'calibrate path/round-count bias before using probabilities as decision-grade',
                'then combine grid-profit probability with inventory/trend PnL risk',
            ],
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('Pionex grid simulator written:', OUT_PATH)
    print('Data:', out['data'])
    print('Grid math validation:', out['grid_math_validation'])
    print('Current ATR-regime ranking:')
    for r in ranking:
        print(' ', r)


if __name__ == '__main__':
    main()
