#!/usr/bin/env python3
"""
Phase 4D v2 execution adapter.

Architecture
------------
- Keep Phase 4D's existing hourly market-state / ATR / path-shape analogue
  selection unchanged.
- Read the prospective Phase 4D.1 execution-resolution policy.
- If 5-minute execution has been promoted, replace only Phase 4D's forward
  analogue replay with KuCoin 5-minute candles.
- Recompute Phase 4D calibration and candidate geometry scores under the same
  promoted execution resolution.
- Preserve the existing Phase 4D output filenames and schema for downstream
  compatibility, while adding explicit execution-resolution integration metadata.

This script does not place orders and does not connect to Pionex.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import build_pionex_grid_geometry_optimizer_v1 as base
import build_pionex_execution_resolution_v1 as execres

EXEC_POLICY_PATH = Path("data/pionex/execution_resolution_policy_v1.json")
GEOMETRY_OUT_PATH = Path("data/diagnostics/pionex_grid_geometry_optimizer_v1.json")
CAL_OUT_PATH = Path("data/diagnostics/pionex_grid_calibration_v2.json")

SUPPORTED_EXECUTION_RESOLUTIONS = {"1hour", "5min"}
MIN_PATH_COMPLETION_RATIO = 0.80
MIN_PATHS_IF_REQUESTED_GE_30 = 24

# Cache each 24h historical analogue once per run. Calibration windows often
# reuse many of the same analogues, so this avoids repeated KuCoin requests.
_PATH_CACHE: Dict[int, Optional[List[List[float]]]] = {}
_STATS = {
    "path_requests": 0,
    "path_cache_hits": 0,
    "path_fetch_failures": 0,
    "paths_built": 0,
    "five_minute_candles_used": 0,
    "first_analogue_5m_utc": None,
    "last_analogue_5m_utc": None,
}


def load_policy() -> dict:
    if not EXEC_POLICY_PATH.is_file():
        return {
            "status": "NOT_AVAILABLE",
            "active_execution_resolution": "1hour",
            "windows_evaluated": 0,
        }
    return json.loads(EXEC_POLICY_PATH.read_text(encoding="utf-8"))


def update_time_bounds(first_ts: int, last_ts: int) -> None:
    first_iso = datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat()
    last_iso = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
    if _STATS["first_analogue_5m_utc"] is None or first_iso < _STATS["first_analogue_5m_utc"]:
        _STATS["first_analogue_5m_utc"] = first_iso
    if _STATS["last_analogue_5m_utc"] is None or last_iso > _STATS["last_analogue_5m_utc"]:
        _STATS["last_analogue_5m_utc"] = last_iso


def fetch_forward_5m(row: dict, hourly_master: Dict[int, List[float]]) -> Optional[List[List[float]]]:
    """
    Fetch the 5-minute candles spanning exactly the same 24 hourly forward
    candles that the legacy Phase 4D replay would have used.

    The legacy forward path contains hourly candles at t+1h ... t+24h.
    For each of those hourly buckets we use all twelve constituent 5m buckets,
    giving 24 * 12 = 288 execution candles.
    """
    entry_ts = int(row["ts"])

    if entry_ts in _PATH_CACHE:
        _STATS["path_cache_hits"] += 1
        cached = _PATH_CACHE[entry_ts]
        return [list(c) for c in cached] if cached is not None else None

    _STATS["path_requests"] += 1

    hourly_fwd = base.sim.forward_24h(hourly_master, entry_ts)
    if hourly_fwd is None or len(hourly_fwd) != base.sim.HORIZON_H:
        _PATH_CACHE[entry_ts] = None
        _STATS["path_fetch_failures"] += 1
        return None

    start_ts = int(hourly_fwd[0][0])
    # Include the full final hourly bucket.
    end_ts = int(hourly_fwd[-1][0]) + 3600 - execres.BASE_INTERVAL_SEC

    try:
        raw = execres.fetch_5m_range(start_ts, end_ts)
    except Exception as exc:
        print(f"5m analogue fetch failed for entry {entry_ts}: {exc}")
        _PATH_CACHE[entry_ts] = None
        _STATS["path_fetch_failures"] += 1
        return None

    candles = []
    by_ts = {}
    for r in raw:
        if len(r) < 6:
            continue
        ts = int(r[0])
        by_ts[ts] = [
            ts,
            float(r[1]),
            float(r[2]),
            float(r[3]),
            float(r[4]),
            float(r[5]),
        ]

    expected = list(range(start_ts, end_ts + 1, execres.BASE_INTERVAL_SEC))
    if len(expected) != base.sim.HORIZON_H * (3600 // execres.BASE_INTERVAL_SEC):
        raise RuntimeError(f"Unexpected expected 5m path length: {len(expected)}")

    if any(ts not in by_ts for ts in expected):
        missing = sum(ts not in by_ts for ts in expected)
        print(f"Incomplete 5m analogue path for entry {entry_ts}: missing {missing}/{len(expected)} buckets")
        _PATH_CACHE[entry_ts] = None
        _STATS["path_fetch_failures"] += 1
        return None

    candles = [by_ts[ts] for ts in expected]

    # Cheap OHLC integrity guard.
    for c in candles:
        _, o, h, l, cl, _ = c
        if h < max(o, cl) - 1e-9 or l > min(o, cl) + 1e-9 or h < l:
            raise RuntimeError(f"Invalid 5m OHLC candle in analogue path: {c}")

    _PATH_CACHE[entry_ts] = candles
    _STATS["paths_built"] += 1
    _STATS["five_minute_candles_used"] += len(candles)
    update_time_bounds(candles[0][0], candles[-1][0])
    return [list(c) for c in candles]


def mapped_paths_5m(rows: List[dict], master: Dict[int, List[float]], current_price: float) -> List[dict]:
    out = []
    requested = len(rows)

    for r in rows:
        fwd = fetch_forward_5m(r, master)
        if fwd is None:
            continue
        mapped = base.sim.map_historical_path_to_current(
            float(r["entry_close"]),
            fwd,
            current_price,
        )
        out.append({
            "row": r,
            "candles": mapped,
            "execution_resolution": "5min",
            "execution_candles": len(mapped),
        })

    if requested:
        ratio = len(out) / requested
        minimum = MIN_PATHS_IF_REQUESTED_GE_30 if requested >= 30 else max(1, math.ceil(requested * MIN_PATH_COMPLETION_RATIO))
        if ratio < MIN_PATH_COMPLETION_RATIO or len(out) < minimum:
            raise RuntimeError(
                "Too few complete 5m analogue paths: "
                f"{len(out)}/{requested} ({ratio:.1%}); "
                f"required >= {MIN_PATH_COMPLETION_RATIO:.0%} and >= {minimum} paths."
            )

    return out


def annotate_outputs(policy: dict, active: str) -> None:
    integration = {
        "policy_status": policy.get("status"),
        "policy_windows_evaluated": int(policy.get("windows_evaluated") or 0),
        "policy_research_best_resolution": policy.get("research_best_resolution"),
        "policy_candidate_improvement_vs_1hour_pct": policy.get("candidate_improvement_vs_1hour_pct"),
        "hourly_regime_and_analogue_selection_preserved": True,
        "execution_replay_resolution": active,
        "execution_resolution_integrated": True,
        "sub_5min_inference_allowed": False,
        "five_minute_hard_floor_preserved": True,
        "historical_5m_replay": {
            **_STATS,
            "cache_entries": len(_PATH_CACHE),
            "expected_candles_per_24h_path": 288 if active == "5min" else None,
            "fetch_policy": (
                "On-demand KuCoin 5m history for selected 24h analogue paths; "
                "in-memory cache only, so the repository does not accumulate a full historical 5m archive."
                if active == "5min"
                else "Not used because active execution resolution is 1hour."
            ),
        },
        "note": (
            "Market-state conditioning remains hourly. Candidate grid geometry and "
            "prospective calibration are replayed at the Phase 4D.1 promoted execution resolution."
        ),
    }

    if GEOMETRY_OUT_PATH.is_file():
        geo = json.loads(GEOMETRY_OUT_PATH.read_text(encoding="utf-8"))
        geo["execution_resolution_integration"] = integration

        model = geo.get("geometry_model") or {}
        model["market_state_resolution"] = "1hour"
        model["execution_replay_resolution"] = active
        model["execution_resolution_source"] = "data/pionex/execution_resolution_policy_v1.json"
        geo["geometry_model"] = model

        GEOMETRY_OUT_PATH.write_text(json.dumps(geo, indent=2), encoding="utf-8")

    if CAL_OUT_PATH.is_file():
        cal = json.loads(CAL_OUT_PATH.read_text(encoding="utf-8"))
        cal["execution_replay_resolution"] = active
        cal["execution_resolution_policy_status"] = policy.get("status")
        cal["execution_resolution_integrated"] = True
        CAL_OUT_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")


def main() -> None:
    policy = load_policy()
    status = str(policy.get("status") or "")
    active = str(policy.get("active_execution_resolution") or "1hour")

    if active not in SUPPORTED_EXECUTION_RESOLUTIONS:
        raise SystemExit(f"Unsupported promoted execution resolution: {active}")

    # A finer resolution may only be used after the prospective promotion gate.
    if active == "5min":
        if status != "PROMOTION_READY":
            raise SystemExit(
                f"5min requested by policy without PROMOTION_READY status: status={status}"
            )
        print("Phase 4D v2: preserving hourly regime selection and promoting execution replay to 5min.")
        base.mapped_paths = mapped_paths_5m
    else:
        print("Phase 4D v2: execution policy keeps 1hour replay; running legacy Phase 4D path engine unchanged.")

    base.main()
    annotate_outputs(policy, active)

    print("Phase 4D v2 execution integration complete.")
    print("Active execution resolution:", active)
    print("5m replay stats:", _STATS)


if __name__ == "__main__":
    main()
