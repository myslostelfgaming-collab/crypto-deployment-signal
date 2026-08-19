#!/usr/bin/env python3
"""
Phase 4D.1 — Pionex execution-resolution validation.

Purpose
-------
Keep the main forecasting/regime layer hourly, but determine the coarsest
execution-candle resolution that reproduces the user's real Pionex Spot Grid
behaviour well enough for geometry optimisation.

The hard data ceiling is 5 minutes. We explicitly do NOT infer sub-5-minute
path ordering.

This script:
  1. downloads/publicly backfills KuCoin ETH-USDT 5-minute spot candles;
  2. aggregates those same candles to 15-minute and 1-hour paths;
  3. replays the exact observed Pionex geometry over prospective screenshot
     windows;
  4. compares predicted rounds/grid-profit with actual Pionex deltas;
  5. ranks 1hour vs 15min vs 5min;
  6. writes a promotion policy, but does not alter Phase 4D geometry logic yet.

No orders are placed and no Pionex API connection is used.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

import build_pionex_grid_simulator_v1 as sim

STATE_PATH = Path("data/pionex/manual_grid_states_v1.csv")
PROFILE_PATH = Path("data/pionex/pionex_grid_profile_v1.json")

CANDLE_DIR = Path("data/pionex/execution_candles")
CANDLE_5M_PATH = CANDLE_DIR / "ETH-USDT_5min.csv"
OUT_PATH = Path("data/diagnostics/pionex_execution_resolution_v1.json")
WINDOWS_PATH = Path("data/pionex/execution_resolution_windows_v1.csv")
POLICY_PATH = Path("data/pionex/execution_resolution_policy_v1.json")

SYMBOL = "ETH-USDT"
BASE_INTERVAL_SEC = 300
RESOLUTIONS = {
    "1hour": 3600,
    "15min": 900,
    "5min": 300,
}

# Public KuCoin endpoints. UTA is primary because its current spot schema is
# [start, open, high, low, close, volume, amount]. Legacy is a fallback and
# uses [start, open, close, high, low, volume, amount].
UTA_URL = "https://api.kucoin.com/api/ua/v1/market/kline"
LEGACY_URL = "https://api.kucoin.com/api/v1/market/candles"

MAX_RECORDS_PER_REQUEST = 1500
REQUEST_TIMEOUT_SEC = 20
REQUEST_RETRIES = 4
REQUEST_SLEEP_SEC = 0.30

CAL_MIN_H = 14.0
CAL_MAX_H = 30.0
MIN_WINDOWS_TO_PROMOTE = 3
MIN_SCORE_IMPROVEMENT_VS_1H_PCT = 10.0
NEAR_BEST_TOLERANCE_PCT = 5.0

# We deliberately make 5m the finest permissible execution assumption.
HARD_EXECUTION_FLOOR = "5min"

CANDLE_FIELDS = [
    "ts_utc", "open", "high", "low", "close", "volume", "amount", "source"
]

WINDOW_FIELDS = [
    "start_utc", "end_utc", "elapsed_h", "lower_usdt", "upper_usdt", "grids",
    "qty_eth", "start_price_usdt", "end_price_usdt",
    "actual_rounds", "actual_grid_profit_usdt",
    "resolution", "candles_used", "edge_gap_start_min", "edge_gap_end_min",
    "pred_rounds_conservative", "pred_rounds_mid", "pred_rounds_optimistic",
    "pred_profit_conservative_usdt", "pred_profit_mid_usdt",
    "pred_profit_optimistic_usdt",
    "rounds_abs_error_mid", "profit_abs_error_mid_usdt",
    "actual_rounds_within_path_ambiguity", "actual_profit_within_path_ambiguity",
]


def safe_float(v, default: Optional[float] = None) -> Optional[float]:
    try:
        if v in (None, ""):
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def safe_int(v, default: Optional[int] = None) -> Optional[int]:
    try:
        if v in (None, ""):
            return default
        return int(float(v))
    except Exception:
        return default


def parse_dt(v: str) -> datetime:
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)


def floor_ts(ts: int, step: int) -> int:
    return (ts // step) * step


def ceil_ts(ts: int, step: int) -> int:
    return ((ts + step - 1) // step) * step


def read_states() -> List[dict]:
    with STATE_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            dt = parse_dt(r["captured_at_utc"])
        except Exception:
            continue
        r = dict(r)
        r["_dt"] = dt
        r["_ts"] = int(dt.timestamp())
        out.append(r)
    out.sort(key=lambda r: r["_ts"])
    return out


def same_geometry(a: dict, b: dict) -> bool:
    keys = ("lower_limit_usdt", "upper_limit_usdt", "grids", "quantity_per_grid_eth")
    for k in keys:
        av = safe_float(a.get(k))
        bv = safe_float(b.get(k))
        if av is None or bv is None or abs(av - bv) > 1e-9:
            return False
    return True


def prospective_windows(states: List[dict]) -> List[Tuple[dict, dict]]:
    out = []
    for a, b in zip(states, states[1:]):
        elapsed = (b["_dt"] - a["_dt"]).total_seconds() / 3600.0
        if CAL_MIN_H <= elapsed <= CAL_MAX_H and same_geometry(a, b):
            if safe_float(a.get("grid_profit_usdt")) is None or safe_float(b.get("grid_profit_usdt")) is None:
                continue
            if safe_int(a.get("rounds_total")) is None or safe_int(b.get("rounds_total")) is None:
                continue
            out.append((a, b))
    return out


def request_json(url: str, params: dict) -> dict:
    last_err = None
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SEC)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("code")) != "200000":
                raise RuntimeError(f"KuCoin response code {payload.get('code')}: {payload}")
            return payload
        except Exception as exc:
            last_err = exc
            if attempt + 1 < REQUEST_RETRIES:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"KuCoin request failed after {REQUEST_RETRIES} attempts: {last_err}")


def parse_uta_rows(data: Iterable[list]) -> List[List[float]]:
    out = []
    for row in data or []:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            ts = int(float(row[0]))
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
            v = float(row[5])
            amount = float(row[6]) if len(row) > 6 else 0.0
        except Exception:
            continue
        if h < max(o, c) - 1e-9 or l > min(o, c) + 1e-9 or h < l:
            continue
        out.append([ts, o, h, l, c, v, amount, "kucoin_uta"])
    return out


def parse_legacy_rows(data: Iterable[list]) -> List[List[float]]:
    out = []
    for row in data or []:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            ts = int(float(row[0]))
            o = float(row[1])
            c = float(row[2])
            h = float(row[3])
            l = float(row[4])
            v = float(row[5])
            amount = float(row[6]) if len(row) > 6 else 0.0
        except Exception:
            continue
        if h < max(o, c) - 1e-9 or l > min(o, c) + 1e-9 or h < l:
            continue
        out.append([ts, o, h, l, c, v, amount, "kucoin_legacy"])
    return out


def fetch_chunk(start_ts: int, end_ts: int) -> List[List[float]]:
    # Primary current UTA endpoint.
    uta_params = {
        "tradeType": "SPOT",
        "symbol": SYMBOL,
        "interval": "5min",
        "startAt": start_ts,
        "endAt": end_ts,
    }
    try:
        payload = request_json(UTA_URL, uta_params)
        rows = parse_uta_rows(payload.get("data") or [])
        if rows:
            return rows
    except Exception as uta_err:
        print("UTA 5m fetch failed; trying legacy:", uta_err)

    # Fallback legacy public spot endpoint.
    legacy_params = {
        "symbol": SYMBOL,
        "type": "5min",
        "startAt": start_ts,
        "endAt": end_ts,
    }
    payload = request_json(LEGACY_URL, legacy_params)
    rows = parse_legacy_rows(payload.get("data") or [])
    if not rows:
        raise RuntimeError(f"KuCoin returned no usable 5m candles for {start_ts}..{end_ts}")
    return rows


def fetch_5m_range(start_ts: int, end_ts: int) -> List[List[float]]:
    start_ts = floor_ts(start_ts, BASE_INTERVAL_SEC)
    end_ts = floor_ts(end_ts, BASE_INTERVAL_SEC)
    if end_ts < start_ts:
        return []

    # 1500 records maximum. Use 1490 buckets per request for inclusive-end safety.
    chunk_span = BASE_INTERVAL_SEC * 1489
    dedup: Dict[int, List[float]] = {}
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(end_ts, cursor + chunk_span)
        rows = fetch_chunk(cursor, chunk_end)
        for r in rows:
            ts = int(r[0])
            if start_ts <= ts <= end_ts:
                dedup[ts] = r
        cursor = chunk_end + BASE_INTERVAL_SEC
        time.sleep(REQUEST_SLEEP_SEC)
    return [dedup[k] for k in sorted(dedup)]


def load_existing_5m() -> Dict[int, List[float]]:
    if not CANDLE_5M_PATH.is_file():
        return {}
    out = {}
    with CANDLE_5M_PATH.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                ts = int(r["ts_utc"])
                o, h, l, c = map(float, [r["open"], r["high"], r["low"], r["close"]])
                v = float(r.get("volume") or 0.0)
                amount = float(r.get("amount") or 0.0)
            except Exception:
                continue
            if h < max(o, c) - 1e-9 or l > min(o, c) + 1e-9 or h < l:
                continue
            out[ts] = [ts, o, h, l, c, v, amount, r.get("source") or "persisted"]
    return out


def write_5m(candles: Dict[int, List[float]]) -> None:
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    with CANDLE_5M_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANDLE_FIELDS)
        w.writeheader()
        for ts in sorted(candles):
            r = candles[ts]
            w.writerow({
                "ts_utc": int(r[0]),
                "open": f"{r[1]:.10f}".rstrip("0").rstrip("."),
                "high": f"{r[2]:.10f}".rstrip("0").rstrip("."),
                "low": f"{r[3]:.10f}".rstrip("0").rstrip("."),
                "close": f"{r[4]:.10f}".rstrip("0").rstrip("."),
                "volume": f"{r[5]:.12f}".rstrip("0").rstrip("."),
                "amount": f"{r[6]:.12f}".rstrip("0").rstrip("."),
                "source": r[7],
            })


def update_5m_history(states: List[dict]) -> Dict[int, List[float]]:
    if not states:
        raise SystemExit("No manual Pionex states available")
    target_start = floor_ts(states[0]["_ts"], BASE_INTERVAL_SEC) - BASE_INTERVAL_SEC
    # Last *completed* 5-minute candle start.
    now_ts = int(datetime.now(timezone.utc).timestamp())
    target_end = floor_ts(now_ts, BASE_INTERVAL_SEC) - BASE_INTERVAL_SEC

    existing = load_existing_5m()
    fetched = []

    if not existing:
        fetched.extend(fetch_5m_range(target_start, target_end))
    else:
        first = min(existing)
        last = max(existing)
        if first > target_start:
            fetched.extend(fetch_5m_range(target_start, first - BASE_INTERVAL_SEC))
        # Refetch a small overlap so finalized recent candles replace partial old rows.
        fetch_from = max(target_start, last - 2 * BASE_INTERVAL_SEC)
        if fetch_from <= target_end:
            fetched.extend(fetch_5m_range(fetch_from, target_end))

    for row in fetched:
        existing[int(row[0])] = row

    # Keep only the relevant prospective era; this is an execution-validation
    # store, not a full market archive yet.
    existing = {ts: r for ts, r in existing.items() if target_start <= ts <= target_end}
    write_5m(existing)
    return existing


def aggregate(base: Dict[int, List[float]], interval_sec: int) -> Dict[int, List[float]]:
    if interval_sec == BASE_INTERVAL_SEC:
        return {ts: list(r[:7]) for ts, r in base.items()}

    ratio = interval_sec // BASE_INTERVAL_SEC
    groups: Dict[int, List[List[float]]] = {}
    for ts, r in base.items():
        bucket = floor_ts(ts, interval_sec)
        groups.setdefault(bucket, []).append(r)

    out = {}
    for bucket, rows in groups.items():
        rows.sort(key=lambda r: r[0])
        expected_ts = [bucket + BASE_INTERVAL_SEC * k for k in range(ratio)]
        actual_ts = [int(r[0]) for r in rows]
        # Do not invent missing sub-candles. Only use complete aggregates.
        if actual_ts != expected_ts:
            continue
        o = float(rows[0][1])
        h = max(float(r[2]) for r in rows)
        l = min(float(r[3]) for r in rows)
        c = float(rows[-1][4])
        v = sum(float(r[5]) for r in rows)
        amount = sum(float(r[6]) for r in rows)
        out[bucket] = [bucket, o, h, l, c, v, amount]
    return out


def grid_replay(candles: Dict[int, List[float]], interval_sec: int, start: dict, end: dict,
                mode: str, fee_rate: float) -> dict:
    start_ts = start["_ts"]
    end_ts = end["_ts"]
    lower = float(start["lower_limit_usdt"])
    upper = float(start["upper_limit_usdt"])
    grids = int(float(start["grids"]))
    qty = float(start["quantity_per_grid_eth"])
    start_price = float(start["current_price_usdt"])
    end_price = float(end["current_price_usdt"])

    lines = sim.grid_lines(lower, upper, grids)
    states = sim.initial_states(lines, start_price)

    # Only complete candles fully inside the screenshot-to-screenshot window.
    first_full = ceil_ts(start_ts, interval_sec)
    last_start = floor_ts(end_ts - interval_sec, interval_sec)
    selected = [
        candles[ts] for ts in sorted(candles)
        if first_full <= ts <= last_start and ts + interval_sec <= end_ts
    ]

    rounds = 0
    profit = 0.0
    prev = start_price

    for c in selected:
        _, o, h, l, cl, *_ = c
        pts = [o, h, l, cl] if mode == "ohlc" else [o, l, h, cl]
        pts = [prev] + pts
        for a, b in zip(pts, pts[1:]):
            rr, pp = sim.process_segment(a, b, lines, states, qty, fee_rate)
            rounds += rr
            profit += pp
        prev = cl

    # Anchor the final partial interval to the actual screenshot end-price. We
    # know the endpoint, but deliberately do not invent unseen intrabar extrema.
    rr, pp = sim.process_segment(prev, end_price, lines, states, qty, fee_rate)
    rounds += rr
    profit += pp

    first_start = selected[0][0] if selected else None
    last_end = selected[-1][0] + interval_sec if selected else None
    edge_start = ((first_start - start_ts) / 60.0) if first_start is not None else None
    edge_end = ((end_ts - last_end) / 60.0) if last_end is not None else None

    return {
        "rounds": rounds,
        "profit_usdt": profit,
        "candles_used": len(selected),
        "edge_gap_start_min": edge_start,
        "edge_gap_end_min": edge_end,
    }


def within(x: float, a: float, b: float, eps: float = 1e-9) -> bool:
    lo, hi = sorted([a, b])
    return lo - eps <= x <= hi + eps


def compare_window(start: dict, end: dict, by_resolution: Dict[str, Dict[int, List[float]]],
                   fee_rate: float) -> List[dict]:
    actual_rounds = int(float(end["rounds_total"])) - int(float(start["rounds_total"]))
    actual_profit = float(end["grid_profit_usdt"]) - float(start["grid_profit_usdt"])
    elapsed_h = (end["_dt"] - start["_dt"]).total_seconds() / 3600.0

    out = []
    for name, interval_sec in RESOLUTIONS.items():
        data = by_resolution[name]
        a = grid_replay(data, interval_sec, start, end, "ohlc", fee_rate)
        b = grid_replay(data, interval_sec, start, end, "olhc", fee_rate)

        rlo, rhi = sorted([a["rounds"], b["rounds"]])
        plo, phi = sorted([a["profit_usdt"], b["profit_usdt"]])
        rmid = (rlo + rhi) / 2.0
        pmid = (plo + phi) / 2.0

        out.append({
            "start_utc": start["captured_at_utc"],
            "end_utc": end["captured_at_utc"],
            "elapsed_h": round(elapsed_h, 4),
            "lower_usdt": float(start["lower_limit_usdt"]),
            "upper_usdt": float(start["upper_limit_usdt"]),
            "grids": int(float(start["grids"])),
            "qty_eth": float(start["quantity_per_grid_eth"]),
            "start_price_usdt": float(start["current_price_usdt"]),
            "end_price_usdt": float(end["current_price_usdt"]),
            "actual_rounds": actual_rounds,
            "actual_grid_profit_usdt": round(actual_profit, 8),
            "resolution": name,
            "candles_used": min(a["candles_used"], b["candles_used"]),
            "edge_gap_start_min": round(float(a["edge_gap_start_min"] or 0.0), 4),
            "edge_gap_end_min": round(float(a["edge_gap_end_min"] or 0.0), 4),
            "pred_rounds_conservative": rlo,
            "pred_rounds_mid": rmid,
            "pred_rounds_optimistic": rhi,
            "pred_profit_conservative_usdt": round(plo, 8),
            "pred_profit_mid_usdt": round(pmid, 8),
            "pred_profit_optimistic_usdt": round(phi, 8),
            "rounds_abs_error_mid": round(abs(rmid - actual_rounds), 8),
            "profit_abs_error_mid_usdt": round(abs(pmid - actual_profit), 8),
            "actual_rounds_within_path_ambiguity": within(actual_rounds, rlo, rhi),
            "actual_profit_within_path_ambiguity": within(actual_profit, plo, phi, eps=0.005),
        })
    return out


def write_windows(rows: List[dict]) -> None:
    WINDOWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WINDOWS_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in WINDOW_FIELDS})


def resolution_metrics(rows: List[dict]) -> dict:
    grouped: Dict[str, List[dict]] = {k: [] for k in RESOLUTIONS}
    for r in rows:
        grouped[r["resolution"]].append(r)

    out = {}
    for name, rr in grouped.items():
        if not rr:
            continue
        profit_errors = [float(r["profit_abs_error_mid_usdt"]) for r in rr]
        rounds_errors = [float(r["rounds_abs_error_mid"]) for r in rr]
        profit_rel = []
        rounds_rel = []
        for r in rr:
            actual_p = abs(float(r["actual_grid_profit_usdt"]))
            actual_r = abs(float(r["actual_rounds"]))
            profit_rel.append(float(r["profit_abs_error_mid_usdt"]) / max(actual_p, 0.02))
            rounds_rel.append(float(r["rounds_abs_error_mid"]) / max(actual_r, 1.0))

        # Equal weight to dollars and activity. This is only a resolution-ranking
        # score; it is not a trading objective.
        score = statistics.fmean(profit_rel + rounds_rel)
        out[name] = {
            "windows": len(rr),
            "profit_mae_usdt": round(statistics.fmean(profit_errors), 8),
            "rounds_mae": round(statistics.fmean(rounds_errors), 8),
            "mean_profit_relative_error": round(statistics.fmean(profit_rel), 6),
            "mean_rounds_relative_error": round(statistics.fmean(rounds_rel), 6),
            "combined_relative_error_score": round(score, 6),
            "rounds_within_path_ambiguity_pct": round(
                sum(bool(r["actual_rounds_within_path_ambiguity"]) for r in rr) / len(rr) * 100.0, 4
            ),
            "profit_within_path_ambiguity_pct": round(
                sum(bool(r["actual_profit_within_path_ambiguity"]) for r in rr) / len(rr) * 100.0, 4
            ),
        }
    return out


def choose_resolution(metrics: dict, n_windows: int) -> dict:
    order_coarse_to_fine = ["1hour", "15min", "5min"]
    available = [r for r in order_coarse_to_fine if r in metrics]
    if not available:
        return {
            "status": "NO_VALID_RESOLUTION",
            "research_best_resolution": None,
            "promotion_candidate": None,
            "active_execution_resolution": "1hour",
        }

    best = min(available, key=lambda r: metrics[r]["combined_relative_error_score"])
    best_score = metrics[best]["combined_relative_error_score"]

    # Prefer the coarsest resolution statistically near the best score.
    near_best = [
        r for r in available
        if metrics[r]["combined_relative_error_score"] <= best_score * (1.0 + NEAR_BEST_TOLERANCE_PCT / 100.0)
    ]
    promotion_candidate = near_best[0] if near_best else best

    baseline = metrics.get("1hour", {}).get("combined_relative_error_score")
    cand_score = metrics[promotion_candidate]["combined_relative_error_score"]
    improvement = None
    if baseline is not None and baseline > 0:
        improvement = (baseline - cand_score) / baseline * 100.0

    enough_windows = n_windows >= MIN_WINDOWS_TO_PROMOTE
    materially_better = (
        promotion_candidate != "1hour"
        and improvement is not None
        and improvement >= MIN_SCORE_IMPROVEMENT_VS_1H_PCT
    )

    if enough_windows and materially_better:
        status = "PROMOTION_READY"
        active = promotion_candidate
    elif enough_windows:
        status = "KEEP_1H_EXECUTION_RESOLUTION"
        active = "1hour"
    else:
        status = "EARLY_EVIDENCE_NOT_PROMOTED"
        active = "1hour"

    return {
        "status": status,
        "research_best_resolution": best,
        "promotion_candidate": promotion_candidate,
        "active_execution_resolution": active,
        "windows_evaluated": n_windows,
        "minimum_windows_to_promote": MIN_WINDOWS_TO_PROMOTE,
        "candidate_improvement_vs_1hour_pct": round(improvement, 6) if improvement is not None else None,
        "minimum_required_improvement_vs_1hour_pct": MIN_SCORE_IMPROVEMENT_VS_1H_PCT,
        "coarsest_within_best_tolerance_pct": NEAR_BEST_TOLERANCE_PCT,
        "hard_execution_resolution_floor": HARD_EXECUTION_FLOOR,
        "hard_limit_note": (
            "The project will not infer sub-5-minute execution paths. Even when 5m is promoted, "
            "each 5m candle remains path-ambiguous and is replayed in both OHLC and OLHC order."
        ),
    }


def build_finalized_hourly_reference() -> Dict[int, List[float]]:
    """
    Build a validation-only hourly reference that prefers the *latest* logged
    observation of each KuCoin hour.

    Important: sim.build_master_candles() intentionally uses setdefault(), which
    preserves the first observation ever seen for a timestamp. That is useful for
    some historical/prospective workflows, but the first observation can be a
    still-forming hourly candle. Comparing finalized 5m aggregates against that
    partial hour creates false consistency failures.

    Here we overwrite repeated timestamps as history files advance, so old hours
    converge to their finalized KuCoin OHLC values. This function is used only by
    the 4D.1 schema/alignment guard; it does not alter the production master.
    """
    master: Dict[int, List[float]] = {}
    for path in sorted(glob(os.path.join(sim.HISTORY_ROOT, "*", "*.json"))):
        try:
            snap = sim.load_json(path)
        except Exception:
            continue
        for candle in sim.extract_candles(snap):
            master[int(candle[0])] = candle
    return master


def repo_hourly_consistency(agg_1h: Dict[int, List[float]]) -> dict:
    # Compare derived 1h candles with a finalized/latest historical reference.
    # This remains a schema/timestamp guard, not a market-performance test.
    try:
        master = build_finalized_hourly_reference()
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": str(exc), "n": 0}

    diffs = {"open": [], "high": [], "low": [], "close": []}
    worst = {"close_pct": -1.0, "ts": None, "derived_close": None, "reference_close": None}

    for ts, c in agg_1h.items():
        ref = master.get(ts)
        if ref is None:
            continue
        denom = max(abs(float(ref[4])), 1e-9)
        for key, idx in [("open", 1), ("high", 2), ("low", 3), ("close", 4)]:
            d = abs(float(c[idx]) - float(ref[idx])) / denom * 100.0
            diffs[key].append(d)
            if key == "close" and d > worst["close_pct"]:
                worst = {
                    "close_pct": d,
                    "ts": int(ts),
                    "derived_close": float(c[idx]),
                    "reference_close": float(ref[idx]),
                }

    n = len(diffs["close"])
    if n == 0:
        return {"status": "NO_OVERLAP", "n": 0}

    def percentile(values, p):
        if not values:
            return None
        vals = sorted(values)
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        "status": "OK",
        "reference_policy": "latest_logged_hourly_observation_per_open_timestamp",
        "n": n,
        "mean_abs_pct_diff": {
            k: round(statistics.fmean(v), 8) if v else None for k, v in diffs.items()
        },
        "p95_abs_pct_diff": {
            k: round(percentile(v, 0.95), 8) if v else None for k, v in diffs.items()
        },
        "max_abs_pct_diff": {
            k: round(max(v), 8) if v else None for k, v in diffs.items()
        },
        "worst_close": {
            "timestamp_utc": (
                datetime.fromtimestamp(worst["ts"], tz=timezone.utc).isoformat()
                if worst["ts"] is not None else None
            ),
            "derived_close": worst["derived_close"],
            "reference_close": worst["reference_close"],
            "abs_pct_diff": round(worst["close_pct"], 8) if worst["close_pct"] >= 0 else None,
        },
        "note": (
            "Validation reference prefers the latest logged form of each historical "
            "hour so finalized 5m aggregates are not compared with an early partial "
            "1h candle. Production historical master semantics are unchanged."
        ),
    }



def fetch_direct_uta_hourly(start_ts: int, end_ts: int) -> Dict[int, List[float]]:
    """
    Fetch fresh direct 1-hour KuCoin UTA candles covering the same period as the
    persisted 5-minute execution data.

    This is the authoritative schema/alignment guard for Phase 4D.1 because both
    sides of the comparison are reconstructed from the same current KuCoin API
    family rather than from the repo's incrementally captured historical archive.
    """
    step = 3600
    start_ts = floor_ts(start_ts, step)
    end_ts = floor_ts(end_ts, step)
    if end_ts < start_ts:
        return {}

    chunk_span = step * 1489
    dedup: Dict[int, List[float]] = {}
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(end_ts, cursor + chunk_span)
        params = {
            "tradeType": "SPOT",
            "symbol": SYMBOL,
            "interval": "1hour",
            "startAt": cursor,
            "endAt": chunk_end,
        }
        payload = request_json(UTA_URL, params)
        rows = parse_uta_rows(payload.get("data") or [])
        for row in rows:
            ts = int(row[0])
            if start_ts <= ts <= end_ts:
                dedup[ts] = row
        cursor = chunk_end + step
        time.sleep(REQUEST_SLEEP_SEC)

    return dedup


def direct_hourly_consistency(agg_1h: Dict[int, List[float]],
                              direct_1h: Dict[int, List[float]]) -> dict:
    """
    Compare 1h candles aggregated from KuCoin 5m data against KuCoin's fresh
    direct 1h candles.

    A schema/order error would create large systematic OHLC differences. Small
    isolated deviations can still occur from exchange-side historical revisions,
    so the workflow keeps a 0.10% maximum close-difference guard.
    """
    diffs = {"open": [], "high": [], "low": [], "close": []}
    worst = {"close_pct": -1.0, "ts": None, "derived_close": None, "reference_close": None}

    for ts, c in agg_1h.items():
        ref = direct_1h.get(ts)
        if ref is None:
            continue
        denom = max(abs(float(ref[4])), 1e-9)
        for key, idx in [("open", 1), ("high", 2), ("low", 3), ("close", 4)]:
            d = abs(float(c[idx]) - float(ref[idx])) / denom * 100.0
            diffs[key].append(d)
            if key == "close" and d > worst["close_pct"]:
                worst = {
                    "close_pct": d,
                    "ts": int(ts),
                    "derived_close": float(c[idx]),
                    "reference_close": float(ref[idx]),
                }

    n = len(diffs["close"])
    if n == 0:
        return {"status": "NO_OVERLAP", "n": 0}

    def percentile(values, p):
        vals = sorted(values)
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        "status": "OK",
        "reference_policy": "fresh_direct_kucoin_uta_1hour",
        "n": n,
        "mean_abs_pct_diff": {
            k: round(statistics.fmean(v), 8) if v else None for k, v in diffs.items()
        },
        "p95_abs_pct_diff": {
            k: round(percentile(v, 0.95), 8) if v else None for k, v in diffs.items()
        },
        "max_abs_pct_diff": {
            k: round(max(v), 8) if v else None for k, v in diffs.items()
        },
        "worst_close": {
            "timestamp_utc": (
                datetime.fromtimestamp(worst["ts"], tz=timezone.utc).isoformat()
                if worst["ts"] is not None else None
            ),
            "derived_close": worst["derived_close"],
            "reference_close": worst["reference_close"],
            "abs_pct_diff": round(worst["close_pct"], 8) if worst["close_pct"] >= 0 else None,
        },
        "note": (
            "Hard Phase 4D.1 schema/alignment guard: finalized 1h candles aggregated "
            "from current KuCoin 5m UTA history versus fresh direct KuCoin 1h UTA history."
        ),
    }


def main() -> None:
    states = read_states()
    windows = prospective_windows(states)
    if not states:
        raise SystemExit("No manual Pionex states")
    if not windows:
        raise SystemExit("No usable 14-30h same-geometry prospective windows")

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    fee_pct = safe_float(
        (profile.get("fee_model") or {}).get("standard_public_spot_fee_pct_per_fill_reference"),
        0.05,
    )
    fee_rate = float(fee_pct) / 100.0

    base = update_5m_history(states)
    if len(base) < 100:
        raise SystemExit(f"Too few 5m candles fetched/persisted: {len(base)}")

    by_resolution = {
        "5min": aggregate(base, 300),
        "15min": aggregate(base, 900),
        "1hour": aggregate(base, 3600),
    }

    comparison_rows = []
    for a, b in windows:
        comparison_rows.extend(compare_window(a, b, by_resolution, fee_rate))
    write_windows(comparison_rows)

    metrics = resolution_metrics(comparison_rows)
    policy = choose_resolution(metrics, len(windows))

    # Hard consistency guard: compare two fresh/current KuCoin UTA resolutions.
    direct_1h = fetch_direct_uta_hourly(min(base), max(base))
    consistency = direct_hourly_consistency(by_resolution["1hour"], direct_1h)

    # Diagnostic-only comparison with the repo's incrementally captured history.
    # This is retained to study archive timing/revision effects, but it must not
    # block the execution-resolution experiment.
    repo_consistency = repo_hourly_consistency(by_resolution["1hour"])

    now = datetime.now(timezone.utc).isoformat()
    out = {
        "schema": "pionex_execution_resolution_v1",
        "generated_at_utc": now,
        "status": "PROSPECTIVE_DIAGNOSTIC_ONLY",
        "scope": {
            "platform": "Pionex",
            "pair": "ETH/USDT",
            "bot_type": "Spot Grid",
            "purpose": "execution path resolution only",
        },
        "data_policy": {
            "forecast_regime_clock": "1hour",
            "execution_resolution_hard_floor": HARD_EXECUTION_FLOOR,
            "sub_5min_inference_allowed": False,
            "available_resolution_comparison": list(RESOLUTIONS.keys()),
            "kucoin_primary_endpoint": UTA_URL,
            "kucoin_fallback_endpoint": LEGACY_URL,
            "persisted_5m_rows": len(base),
            "persisted_5m_first_utc": datetime.fromtimestamp(min(base), tz=timezone.utc).isoformat(),
            "persisted_5m_last_utc": datetime.fromtimestamp(max(base), tz=timezone.utc).isoformat(),
            "aggregation_rule": "15m and 1h are built from the exact same 5m source candles; incomplete aggregates are excluded.",
        },
        "prospective_windows": {
            "usable_window_count": len(windows),
            "window_rule": f"{CAL_MIN_H:.0f}-{CAL_MAX_H:.0f}h consecutive same-geometry manual Pionex screenshots",
        },
        "resolution_metrics": metrics,
        "selection_policy": policy,
        "hourly_consistency_guard": consistency,
        "repo_hourly_consistency_reference": repo_consistency,
        "limitations": [
            "Pionex's exact internal fill timing is not observable from screenshots.",
            "Even 5-minute OHLC does not reveal sub-5-minute price ordering; both OHLC and OLHC paths are replayed.",
            "The current public fee reference remains a model input until account-specific effective fees are verified.",
            "Fine execution resolution is not promoted into Phase 4D geometry recommendations until the prospective promotion gate passes.",
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(json.dumps({
        "schema": "pionex_execution_resolution_policy_v1",
        "generated_at_utc": now,
        **policy,
    }, indent=2), encoding="utf-8")

    print("Phase 4D.1 execution-resolution validation written:", OUT_PATH)
    print("5m candles persisted:", len(base))
    print("Prospective windows:", len(windows))
    print("Metrics:", metrics)
    print("Policy:", policy)
    print("Direct KuCoin hourly consistency guard:", consistency)
    print("Repo hourly consistency reference (diagnostic only):", repo_consistency)


if __name__ == "__main__":
    main()
