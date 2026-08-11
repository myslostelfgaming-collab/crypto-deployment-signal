#!/usr/bin/env python3
"""
Luno Competition Shadow V1
==========================

Prospective, schedule-aware Luno Price Predict shadow logger.

This script never places an entry in Luno. It records what the forecasting
system would submit, keeps the latest snapshot before each official cutoff, and
automatically evaluates matured entries against the repo's BTC-USDT market
proxy. If the user later supplies the exact Luno recorded settlement price in
`data/luno/luno_manual_results_v1.csv`, the same rows are also scored against
the true competition result.

The live recommendation remains persistence until a challenger beats it in
prospective evidence.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

SIM_BTC_PATH = os.path.join("data", "model", "similarity_forecast_v2_BTC-USDT.json")
SIM_MULTI_PATH = os.path.join("data", "model", "similarity_forecast_v2_multi.json")
LUNO_PROXY_PATH = os.path.join("data", "luno", "luno_price_proxy_latest_v1.json")
HISTORY_INDEX = os.path.join("data", "history", "index.csv")
OUT_ENTRIES = os.path.join("data", "luno", "luno_shadow_entries_v1.csv")
OUT_PERF = os.path.join("data", "luno", "luno_shadow_performance_v1.json")
OUT_LATEST = os.path.join("data", "luno", "luno_shadow_latest_v1.json")
MANUAL_RESULTS = os.path.join("data", "luno", "luno_manual_results_v1.csv")

SCHEMA = "luno_shadow_v1"
PERF_SCHEMA = "luno_shadow_performance_v1"
SETTLEMENT_TOLERANCE_MIN = 90.0

FIELDS = [
    "challenge_key",
    "challenge_type",
    "cutoff_utc",
    "settlement_utc",
    "forecast_snapshot_utc",
    "minutes_before_cutoff",
    "forecast_stage",
    "reference_horizon_h",
    "hours_from_snapshot_to_settlement",
    "market_entry_price_usd",
    "luno_proxy_entry_usd",
    "luno_basis_pct",
    "similarity_predicted_return_pct",
    "ret48_pct",
    "forecast_persistence_market",
    "forecast_persistence_luno_proxy",
    "forecast_similarity_market",
    "forecast_similarity_luno_proxy",
    "forecast_mean_reversion_48h_market",
    "recommended_model",
    "recommended_submission_usd",
    "recommended_submission_rounded_usd",
    "market_proxy_status",
    "actual_market_proxy_usd",
    "actual_market_proxy_snapshot_utc",
    "actual_market_proxy_diff_minutes",
    "error_persistence_market_usd",
    "error_similarity_market_usd",
    "error_mean_reversion_48h_market_usd",
    "error_recommended_vs_market_proxy_usd",
    "luno_exact_status",
    "actual_luno_price_usd",
    "error_persistence_market_vs_luno_exact_usd",
    "error_persistence_luno_proxy_vs_luno_exact_usd",
    "error_similarity_market_vs_luno_exact_usd",
    "error_similarity_luno_proxy_vs_luno_exact_usd",
    "error_recommended_vs_luno_exact_usd",
    "updated_at_utc",
]


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_similarity() -> Optional[Dict[str, Any]]:
    direct = load_json(SIM_BTC_PATH)
    if direct:
        return direct
    multi = load_json(SIM_MULTI_PATH)
    if multi:
        return (multi.get("forecasts") or {}).get("BTC-USDT")
    return None


def blended_similarity_return(sim: Dict[str, Any], horizon: int) -> Optional[float]:
    block = (sim.get("forecast_summary_primary_top20") or {}).get(f"close_change_pct_{horizon}") or {}
    wm = safe_float(block.get("weighted_mean"))
    wmed = safe_float(block.get("weighted_median"))
    if wm is None and wmed is None:
        return None
    if wmed is None:
        return wm
    if wm is None:
        return wmed
    return 0.7 * wmed + 0.3 * wm


def current_state() -> Optional[Dict[str, Any]]:
    sim = load_similarity()
    if not sim:
        return None
    current = sim.get("current_state") or {}
    entry_price = safe_float(current.get("entry_close"))
    source_time = parse_iso(sim.get("as_of_utc") or sim.get("generated_at_utc") or "")
    if entry_price is None or source_time is None:
        return None
    features = current.get("features") or {}
    return {
        "sim": sim,
        "time": source_time,
        "market_price": entry_price,
        "ret48": safe_float(features.get("ret_48h_pct")),
    }


def challenge_specs_at(t: datetime) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    # Weekly opens Monday 07:00 UTC and closes Wednesday 12:00 UTC.
    monday = (t - timedelta(days=t.weekday())).date()
    weekly_open = datetime(monday.year, monday.month, monday.day, 7, tzinfo=timezone.utc)
    weekly_cutoff = datetime(monday.year, monday.month, monday.day, 12, tzinfo=timezone.utc) + timedelta(days=2)
    weekly_settle = weekly_cutoff + timedelta(days=2)
    if weekly_open <= t <= weekly_cutoff:
        specs.append({
            "challenge_type": "weekly",
            "cutoff": weekly_cutoff,
            "settlement": weekly_settle,
            "reference_horizon_h": 48,
        })

    # Monthly opens day 1 07:00 UTC and closes day 14 12:00 UTC.
    monthly_open = datetime(t.year, t.month, 1, 7, tzinfo=timezone.utc)
    monthly_cutoff = datetime(t.year, t.month, 14, 12, tzinfo=timezone.utc)
    monthly_settle = datetime(t.year, t.month, 28, 12, tzinfo=timezone.utc)
    if monthly_open <= t <= monthly_cutoff:
        specs.append({
            "challenge_type": "monthly",
            "cutoff": monthly_cutoff,
            "settlement": monthly_settle,
            "reference_horizon_h": 336,
        })

    return specs


def challenge_key(kind: str, settlement: datetime) -> str:
    return f"{kind}:{settlement.strftime('%Y-%m-%dT%H:%MZ')}"


def read_rows() -> List[Dict[str, str]]:
    if not os.path.isfile(OUT_ENTRIES):
        return []
    with open(OUT_ENTRIES, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(OUT_ENTRIES), exist_ok=True)
    with open(OUT_ENTRIES, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def ensure_manual_results() -> None:
    os.makedirs(os.path.dirname(MANUAL_RESULTS), exist_ok=True)
    if os.path.isfile(MANUAL_RESULTS):
        return
    with open(MANUAL_RESULTS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["challenge_key", "actual_luno_price_usd", "source_note", "recorded_at_utc"])


def load_manual_results() -> Dict[str, float]:
    ensure_manual_results()
    out: Dict[str, float] = {}
    with open(MANUAL_RESULTS, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            price = safe_float(row.get("actual_luno_price_usd"))
            key = row.get("challenge_key") or ""
            if key and price is not None:
                out[key] = price
    return out


def load_market_points() -> List[Tuple[datetime, float]]:
    points: List[Tuple[datetime, float]] = []
    if not os.path.isfile(HISTORY_INDEX):
        return points
    with open(HISTORY_INDEX, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = parse_iso(row.get("published_at_utc") or "")
            rel = row.get("history_file") or ""
            if t is None or not rel or not os.path.isfile(rel):
                continue
            try:
                with open(rel, "r", encoding="utf-8") as hf:
                    payload = json.load(hf)
                price = safe_float((payload.get("btc_usdt") or {}).get("close"))
                if price is not None:
                    points.append((t, price))
            except Exception:
                continue
    points.sort(key=lambda x: x[0])
    return points


def nearest_market(points: List[Tuple[datetime, float]], target: datetime) -> Optional[Tuple[datetime, float, float]]:
    best = None
    best_diff = None
    tolerance = timedelta(minutes=SETTLEMENT_TOLERANCE_MIN)
    for t, price in points:
        diff = abs(t - target)
        if diff > tolerance:
            continue
        if best_diff is None or diff < best_diff:
            best = (t, price)
            best_diff = diff
    if best is None or best_diff is None:
        return None
    return best[0], best[1], best_diff.total_seconds() / 60.0


def make_candidate(spec: Dict[str, Any], state: Dict[str, Any], luno_proxy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    t = state["time"]
    cutoff = spec["cutoff"]
    settlement = spec["settlement"]
    horizon = int(spec["reference_horizon_h"])
    market = float(state["market_price"])
    ret48 = state.get("ret48")
    sim_ret = blended_similarity_return(state["sim"], horizon)

    lp = None
    basis_pct = None
    if luno_proxy and luno_proxy.get("available"):
        lp = safe_float(luno_proxy.get("luno_usd_proxy"))
        basis_pct = safe_float((luno_proxy.get("basis") or {}).get("pct"))

    similarity_market = market * (1.0 + sim_ret / 100.0) if sim_ret is not None else None
    similarity_luno = lp * (1.0 + sim_ret / 100.0) if lp is not None and sim_ret is not None else None
    mr_market = market * (1.0 - ret48 / 100.0) if ret48 is not None else None

    minutes_before = (cutoff - t).total_seconds() / 60.0
    stage = "FINAL_CANDIDATE" if 0 <= minutes_before <= 90 else "EARLY_PREVIEW"

    # Persistence is deliberately champion until prospective evidence beats it.
    # The historical tournament is scored on the market BTC-USDT proxy, so we
    # do NOT assume the new Luno-exchange proxy is closer to Luno's internal
    # competition rate until exact settlement observations demonstrate that.
    recommended_model = "persistence_market"
    recommended = market

    now = datetime.now(timezone.utc).isoformat()
    return {
        "challenge_key": challenge_key(spec["challenge_type"], settlement),
        "challenge_type": spec["challenge_type"],
        "cutoff_utc": cutoff.isoformat(),
        "settlement_utc": settlement.isoformat(),
        "forecast_snapshot_utc": t.isoformat(),
        "minutes_before_cutoff": round(minutes_before, 2),
        "forecast_stage": stage,
        "reference_horizon_h": horizon,
        "hours_from_snapshot_to_settlement": round((settlement - t).total_seconds() / 3600.0, 4),
        "market_entry_price_usd": round_or_none(market, 4),
        "luno_proxy_entry_usd": round_or_none(lp, 4),
        "luno_basis_pct": round_or_none(basis_pct, 6),
        "similarity_predicted_return_pct": round_or_none(sim_ret, 6),
        "ret48_pct": round_or_none(ret48, 6),
        "forecast_persistence_market": round_or_none(market, 4),
        "forecast_persistence_luno_proxy": round_or_none(lp, 4),
        "forecast_similarity_market": round_or_none(similarity_market, 4),
        "forecast_similarity_luno_proxy": round_or_none(similarity_luno, 4),
        "forecast_mean_reversion_48h_market": round_or_none(mr_market, 4),
        "recommended_model": recommended_model,
        "recommended_submission_usd": round_or_none(recommended, 4),
        "recommended_submission_rounded_usd": int(round(recommended)),
        "market_proxy_status": "PENDING",
        "luno_exact_status": "PENDING_MANUAL_RESULT",
        "updated_at_utc": now,
    }


def upsert_candidates(rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {r.get("challenge_key") or "": dict(r) for r in rows if r.get("challenge_key")}
    for cand in candidates:
        key = cand["challenge_key"]
        old = by_key.get(key)
        if old is None:
            by_key[key] = cand
            continue
        old_t = parse_iso(old.get("forecast_snapshot_utc") or "")
        new_t = parse_iso(cand.get("forecast_snapshot_utc") or "")
        cutoff = parse_iso(cand.get("cutoff_utc") or "")
        # Latest snapshot that is still at/before cutoff wins.
        if new_t and cutoff and new_t <= cutoff and (old_t is None or new_t > old_t):
            # Preserve already evaluated truth fields only if replacing after an
            # accidental re-run; normally cutoff prevents that situation.
            by_key[key] = cand

    return sorted(by_key.values(), key=lambda r: r.get("settlement_utc") or "")


def evaluate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manual = load_manual_results()
    now = datetime.now(timezone.utc)

    # Reading every historical JSON snapshot is unnecessary on ordinary runs.
    # Only load market points when at least one settlement has matured and has
    # not already been scored.
    needs_market_eval = any(
        (parse_iso(r.get("settlement_utc") or "") or now + timedelta(days=1)) <= now
        and r.get("market_proxy_status") != "EVALUATED"
        for r in rows
    )
    points = load_market_points() if needs_market_eval else []
    latest_market_time = points[-1][0] if points else None

    for row in rows:
        settlement = parse_iso(row.get("settlement_utc") or "")
        if settlement is None:
            continue

        if settlement <= now and latest_market_time and latest_market_time >= settlement - timedelta(minutes=SETTLEMENT_TOLERANCE_MIN):
            actual = nearest_market(points, settlement)
            if actual:
                at, ap, diff = actual
                row["market_proxy_status"] = "EVALUATED"
                row["actual_market_proxy_usd"] = round(ap, 4)
                row["actual_market_proxy_snapshot_utc"] = at.isoformat()
                row["actual_market_proxy_diff_minutes"] = round(diff, 2)
                for forecast_field, error_field in [
                    ("forecast_persistence_market", "error_persistence_market_usd"),
                    ("forecast_similarity_market", "error_similarity_market_usd"),
                    ("forecast_mean_reversion_48h_market", "error_mean_reversion_48h_market_usd"),
                    ("recommended_submission_usd", "error_recommended_vs_market_proxy_usd"),
                ]:
                    fp = safe_float(row.get(forecast_field))
                    row[error_field] = round(abs(fp - ap), 4) if fp is not None else ""

        key = row.get("challenge_key") or ""
        if key in manual:
            actual_luno = manual[key]
            row["luno_exact_status"] = "EVALUATED"
            row["actual_luno_price_usd"] = round(actual_luno, 4)
            exact_pairs = [
                ("forecast_persistence_market", "error_persistence_market_vs_luno_exact_usd"),
                ("forecast_persistence_luno_proxy", "error_persistence_luno_proxy_vs_luno_exact_usd"),
                ("forecast_similarity_market", "error_similarity_market_vs_luno_exact_usd"),
                ("forecast_similarity_luno_proxy", "error_similarity_luno_proxy_vs_luno_exact_usd"),
                ("recommended_submission_usd", "error_recommended_vs_luno_exact_usd"),
            ]
            for forecast_field, error_field in exact_pairs:
                fp = safe_float(row.get(forecast_field))
                row[error_field] = round(abs(fp - actual_luno), 4) if fp is not None else ""
        elif settlement <= now:
            row["luno_exact_status"] = "AWAITING_MANUAL_RESULT"

        row["updated_at_utc"] = now.isoformat()

    return rows


def metric(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    vals = [safe_float(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return {
        "n": len(vals),
        "mae_usd": round_or_none(mean(vals) if vals else None, 4),
        "median_ae_usd": round_or_none(median(vals) if vals else None, 4),
    }


def performance_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type = {}
    for kind in ("weekly", "monthly"):
        subset = [r for r in rows if r.get("challenge_type") == kind]
        by_type[kind] = {
            "rows": len(subset),
            "market_proxy_evaluated": sum(r.get("market_proxy_status") == "EVALUATED" for r in subset),
            "luno_exact_evaluated": sum(r.get("luno_exact_status") == "EVALUATED" for r in subset),
            "market_proxy_errors": {
                "persistence_market": metric(subset, "error_persistence_market_usd"),
                "similarity_market": metric(subset, "error_similarity_market_usd"),
                "mean_reversion_48h_market": metric(subset, "error_mean_reversion_48h_market_usd"),
                "recommended": metric(subset, "error_recommended_vs_market_proxy_usd"),
            },
            "luno_exact_errors": {
                "persistence_market": metric(subset, "error_persistence_market_vs_luno_exact_usd"),
                "persistence_luno_proxy": metric(subset, "error_persistence_luno_proxy_vs_luno_exact_usd"),
                "similarity_market": metric(subset, "error_similarity_market_vs_luno_exact_usd"),
                "similarity_luno_proxy": metric(subset, "error_similarity_luno_proxy_vs_luno_exact_usd"),
                "recommended": metric(subset, "error_recommended_vs_luno_exact_usd"),
            },
        }
    return {
        "schema": PERF_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_SHADOW_ONLY",
        "rows_total": len(rows),
        "by_challenge_type": by_type,
        "truth_layers": {
            "market_proxy": "automatically evaluated against repo KuCoin BTC-USDT snapshot near settlement",
            "luno_exact": "evaluated only when actual Luno settlement is supplied in luno_manual_results_v1.csv",
        },
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ensure_manual_results()
    state = current_state()
    rows: List[Dict[str, Any]] = [dict(r) for r in read_rows()]
    candidates: List[Dict[str, Any]] = []

    if state:
        luno_proxy = load_json(LUNO_PROXY_PATH)
        for spec in challenge_specs_at(state["time"]):
            candidates.append(make_candidate(spec, state, luno_proxy))
        rows = upsert_candidates(rows, candidates)

    rows = evaluate(rows)
    write_rows(rows)
    perf = performance_payload(rows)
    write_json(OUT_PERF, perf)

    latest_rows = sorted(rows, key=lambda r: r.get("settlement_utc") or "")[-4:]
    latest_payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROSPECTIVE_SHADOW_ONLY",
        "source_state_available": state is not None,
        "new_or_updated_candidates_this_run": len(candidates),
        "recommendation_policy": (
            "Market-price persistence remains the recommended model until a challenger beats it in prospective evidence. "
            "The Luno-native proxy is logged as a calibration challenger until exact Luno settlement results show whether it improves alignment."
        ),
        "latest_challenges": latest_rows,
        "performance_summary": perf,
    }
    write_json(OUT_LATEST, latest_payload)

    print("Luno shadow rows:", len(rows), "candidates this run:", len(candidates))
    for c in candidates:
        print(
            c["challenge_type"],
            c["forecast_stage"],
            "cutoff", c["cutoff_utc"],
            "recommended", c["recommended_submission_rounded_usd"],
            c["recommended_model"],
        )


if __name__ == "__main__":
    main()
