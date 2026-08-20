#!/usr/bin/env python3
"""Phase 4D.2 wrapper that corrects nearest-order state after a range escape."""

from __future__ import annotations

import build_pionex_grid_activity_v1 as base
from pionex_out_of_grid_support_v1 import corrected_initial_states


def main():
    base.sim.initial_states = corrected_initial_states
    base.main()
    print("Phase 4D.2 out-of-grid trigger-state correction applied.")


if __name__ == "__main__":
    main()
