#!/usr/bin/env python3
"""
Full Phase 4D runner v2.

Reuses the existing validated full-pipeline integrity checks, substituting only:
- geometry v3 for out-of-grid recovery; and
- activity v2 for correct above/below-grid waiting-trigger state.
"""

from __future__ import annotations

import run_pionex_phase4d_full_v1 as legacy


_original_run_builder = legacy.run_builder


def _run_builder(label: str, relative_script: str) -> None:
    if relative_script == "scripts/build_pionex_grid_geometry_optimizer_v2.py":
        relative_script = "scripts/build_pionex_grid_geometry_optimizer_v3.py"
    elif relative_script == "scripts/build_pionex_grid_activity_v1.py":
        relative_script = "scripts/build_pionex_grid_activity_v2.py"
    _original_run_builder(label, relative_script)


def main() -> None:
    legacy.run_builder = _run_builder
    legacy.main()


if __name__ == "__main__":
    main()
