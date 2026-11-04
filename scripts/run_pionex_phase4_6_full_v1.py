#!/usr/bin/env python3
"""Phase 4.6 wrapper: existing Phase 4.5 decision + calibration-v2 evidence.

No live-trading capability is added. Phase 4.5 safety and manual promotion gates remain
unchanged; the v2 layer is additive and can only qualify/downgrade forecast confidence.
"""
import run_pionex_phase4_5_full_v1 as phase45
import build_pionex_calibration_quality_v2 as calv2


def main() -> None:
    phase45.main()
    calv2.main()


if __name__ == "__main__":
    main()
