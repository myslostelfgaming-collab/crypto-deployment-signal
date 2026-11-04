#!/usr/bin/env python3
"""Replay Phase 4.5 paired calibration with v2 quality diagnostics.

Research/read-only: reads repo diagnostics/state history and writes only a diagnostic JSON.
It never calls Pionex and cannot alter the live bot.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pionex_calibration_quality_v2 as q

ROOT = Path(__file__).resolve().parents[1]
DENSITY = ROOT / "data/diagnostics/pionex_grid_density_sweep_v1.json"
OUT = ROOT / "data/diagnostics/pionex_calibration_quality_v2.json"


def _load_legacy() -> dict[str, Any]:
    data = json.loads(DENSITY.read_text(encoding="utf-8"))
    cal = ((data.get("promotion_readiness") or {}).get("paired_calibration") or {})
    if not cal.get("windows"):
        raise SystemExit("paired calibration windows missing from density diagnostic")
    return cal


def _repo_context():
    """Best-effort feature/state adapters from the existing Phase 4.5 repo."""
    try:
        import build_pionex_grid_geometry_optimizer_v1 as base
    except Exception:
        return None, None

    try:
        states = base.phase4c.read_manual_states()
        audit = q.audit_state_pair_exclusions(states, min_h=base.CAL_MIN_H, max_h=base.CAL_MAX_H)
    except Exception:
        audit = None

    try:
        master = base.sim.build_master_candles()
        features = base.sim.load_eth_features()

        def enrich(window: dict[str, Any]) -> dict[str, Any]:
            raw = str(window.get("start_utc") or "")
            if not raw:
                return {}
            try:
                ts = int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
            except Exception:
                return {}
            feat = base.nearest_feature(features, ts)
            if feat is None:
                return {}
            out: dict[str, Any] = {}
            shape = base.path_shape_metrics(master, feat) or {}
            out.update(shape)
            # Feature schema has evolved; accept known transparent ATR variants.
            for key in ("atr14_pct", "atr_pct", "atr14_percent"):
                if feat.get(key) not in (None, ""):
                    out["atr14_pct"] = float(feat[key])
                    break
            return out

    except Exception:
        enrich = None
    return audit, enrich


def main() -> None:
    legacy = _load_legacy()
    audit, enricher = _repo_context()
    exclusions = (audit or {}).get("geometry_change_exclusions") or []
    upgraded = q.upgrade_calibration(
        legacy,
        exclusions=exclusions,
        feature_enricher=enricher,
        scale_bounds=tuple(legacy.get("scale_bounds") or (0.35, 1.50)),
    )
    v2 = upgraded["calibration_v2"]
    v2["state_pair_audit"] = audit or {
        "status": "UNAVAILABLE_IN_STANDALONE_REPLAY",
        "note": "Run inside the repository to audit canonical state transitions."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(v2, indent=2) + "\n", encoding="utf-8")

    hold = v2.get("holdout") or {}
    legacy_cmp = v2.get("legacy_global_holdout_comparison") or {}
    print("=== PIONEX CALIBRATION V2 REPLAY ===")
    print("Status:", (v2.get("quality") or {}).get("status"))
    print("Activity:", v2.get("activity"))
    print("Validation:", v2.get("validation"))
    print("Profit holdout:", hold.get("profit"))
    print("Rounds holdout:", hold.get("rounds"))
    print("Legacy comparator:", {"profit": legacy_cmp.get("profit"), "rounds": legacy_cmp.get("rounds")})
    print("Improvement:", v2.get("holdout_improvement_vs_legacy_pct"))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
