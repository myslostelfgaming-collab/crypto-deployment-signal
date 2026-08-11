#!/usr/bin/env python3

import json
from pathlib import Path

PATH = Path("data/diagnostics/model_tournament_v1.json")


def main() -> None:
    if not PATH.is_file():
        raise SystemExit(f"Missing {PATH}")

    payload = json.loads(PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "model_tournament_v1":
        raise SystemExit(f"Expected model_tournament_v1, got {payload.get('schema')}")
    if not payload.get("available"):
        raise SystemExit(f"Tournament unavailable: {payload.get('reason')}")
    if payload.get("status") != "EXPERIMENTAL_DIAGNOSTIC_ONLY":
        raise SystemExit(f"Unexpected tournament status: {payload.get('status')}")

    horizons = set((payload.get("tournaments", {}).get("by_horizon_independent") or {}).keys())
    expected = {"24", "48", "168", "336"}
    if horizons != expected:
        raise SystemExit(f"Independent horizons mismatch: found={sorted(horizons)}, expected={sorted(expected)}")

    for h in expected:
        rules = payload["tournaments"]["by_horizon_independent"][h].get("rules", {})
        for required_rule in ("similarity_v2", "mean_reversion_48h"):
            if required_rule not in rules:
                raise SystemExit(f"{h}h missing required rule {required_rule}")

    selector = payload.get("walk_forward_selector", {})
    if not selector.get("method"):
        raise SystemExit("Missing walk-forward selector method metadata")

    print("Model tournament v1 validated.")
    print("Short horizon selector:", selector.get("short_horizons_24_48"))
    print("Warnings:", payload.get("warnings", []))


if __name__ == "__main__":
    main()
