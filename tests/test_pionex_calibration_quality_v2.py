import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pionex_calibration_quality_v2 as q


def w(start, pp, ap, pr, ar, grids=35, lower=2500, upper=2850):
    return {
        "start_utc": start,
        "end_utc": start,
        "predicted_paired_profit_24h_raw_usdt": pp,
        "actual_grid_profit_24h_equiv_usdt": ap,
        "predicted_paired_rounds_24h_raw": pr,
        "actual_rounds_24h_equiv": ar,
        "grids": grids,
        "lower_usdt": lower,
        "upper_usdt": upper,
        "current_price_usdt": (lower + upper) / 2,
        "quantity_per_grid_eth": 0.00195,
    }


class CalibrationV2Tests(unittest.TestCase):
    def test_near_zero_mape_failure_is_guarded(self):
        row = w("2026-01-01T00:00:00+00:00", 0.20, 0.02, 10, 1)
        thresholds = {"zero_rounds_epsilon": 1e-9, "zero_profit_epsilon_usdt": 1e-9,
                      "low_rounds_max": 2.0, "low_profit_max_usdt": 0.03}
        self.assertEqual(q.classify_activity(row, thresholds), "LOW_ACTIVITY")
        raw = q.error_metrics([0.02], [0.20])
        self.assertAlmostEqual(raw["mae"], 0.18)
        self.assertAlmostEqual(raw["wape_pct"], 900.0)
        self.assertAlmostEqual(raw["mape_pct"], 900.0)
        guarded = q.error_metrics([0.02], [0.20], mape_mask=[False])
        self.assertIsNone(guarded["mape_pct"])
        self.assertAlmostEqual(guarded["mae"], 0.18)

    def test_zero_actual_no_nan_or_inf(self):
        row = w("2026-01-01T00:00:00+00:00", 0.1, 0.0, 12, 0)
        thresholds = {"zero_rounds_epsilon": 1e-9, "zero_profit_epsilon_usdt": 1e-9,
                      "low_rounds_max": 2.0, "low_profit_max_usdt": 0.03}
        self.assertEqual(q.classify_activity(row, thresholds), "ZERO_ACTIVITY")
        m = q.error_metrics([0.0], [12.0])
        self.assertIsNone(m["mape_pct"])
        self.assertIsNone(m["wape_pct"])
        self.assertTrue(math.isfinite(m["mae"]))

    def test_bias_positive_means_overprediction(self):
        m = q.error_metrics([1, 1, 1], [2, 2, 2])
        self.assertEqual(m["bias"], 1.0)

    def test_same_geometry_and_geometry_key(self):
        a = q.add_geometry_descriptors(w("2026-01-01T00:00:00+00:00", .2, .2, 10, 10))
        b = q.add_geometry_descriptors(w("2026-01-02T00:00:00+00:00", .2, .2, 10, 10))
        self.assertEqual(q.geometry_key(a), q.geometry_key(b))

    def test_geometry_change_is_not_same_key(self):
        a = q.add_geometry_descriptors(w("2026-01-01T00:00:00+00:00", .2, .2, 10, 10, grids=40))
        b = q.add_geometry_descriptors(w("2026-01-02T00:00:00+00:00", .2, .2, 10, 10, grids=35))
        self.assertNotEqual(q.geometry_key(a), q.geometry_key(b))

    def test_geometry_change_state_pair_is_explicitly_excluded(self):
        from datetime import datetime, timezone, timedelta
        start = {"_dt": datetime(2026, 1, 1, tzinfo=timezone.utc), "lower_limit_usdt": 2500, "upper_limit_usdt": 2850, "grids": 40, "quantity_per_grid_eth": 0.002}
        end = {"_dt": start["_dt"] + timedelta(hours=24), "lower_limit_usdt": 2500, "upper_limit_usdt": 2850, "grids": 35, "quantity_per_grid_eth": 0.002}
        r = q.classify_state_pair_for_calibration(start, end)
        self.assertFalse(r["eligible"])
        self.assertEqual(r["reason"], "CALIBRATION_WINDOW_EXCLUDED_GEOMETRY_CHANGE")
        self.assertIn("grids", r["changed_geometry_fields"])

    def test_clean_same_geometry_state_pair_is_eligible(self):
        from datetime import datetime, timezone, timedelta
        start = {"_dt": datetime(2026, 1, 1, tzinfo=timezone.utc), "lower_limit_usdt": 2500, "upper_limit_usdt": 2850, "grids": 35, "quantity_per_grid_eth": 0.002}
        end = {"_dt": start["_dt"] + timedelta(hours=23), "lower_limit_usdt": 2500, "upper_limit_usdt": 2850, "grids": 35, "quantity_per_grid_eth": 0.002}
        r = q.classify_state_pair_for_calibration(start, end)
        self.assertTrue(r["eligible"])
        self.assertEqual(r["reason"], "CALIBRATION_WINDOW_ELIGIBLE_SAME_GEOMETRY")

    def test_cohort_fallback_visible(self):
        fit = [q.add_geometry_descriptors(w(f"2026-01-{i+1:02d}T00:00:00+00:00", .2, .2, 10, 10, grids=22, lower=2190, upper=2630)) for i in range(10)]
        for row in fit:
            row["regime"] = {"volatility": "LOW", "path": "RANGING_MIXED", "density": "LOW", "center_skew": row["center_skew"]}
        target = q.add_geometry_descriptors(w("2026-02-01T00:00:00+00:00", .2, .2, 10, 10, grids=35))
        target["regime"] = {"volatility": "HIGH", "path": "TREND_UP", "density": "HIGH", "center_skew": target["center_skew"]}
        rows, meta = q.choose_cohort(fit, target, min_n=8)
        self.assertEqual(meta["fallback_level"], "GLOBAL")
        self.assertEqual(len(rows), 10)

    def test_chronological_holdout_never_uses_future(self):
        rows = [w(f"2026-01-{i+1:02d}T00:00:00+00:00", .2, .2, 10, 10) for i in range(25)]
        split = q.chronological_split(rows, max_holdout=6, prospective_n=3, min_fit=12)
        self.assertLess(split["fit"][-1]["start_utc"], split["holdout"][0]["start_utc"])
        self.assertLess(split["holdout"][-1]["start_utc"], split["prospective"][0]["start_utc"])

    def test_two_stage_activity_separates_productivity(self):
        rows = [w(f"2026-01-{i+1:02d}T00:00:00+00:00", .2, .1, 10, 5) for i in range(10)]
        f = q.fit_two_stage_factors(rows)
        self.assertAlmostEqual(f["rounds_scale"], 0.5, places=6)
        # Profit/cycle is unchanged: .1/5 == .2/10, so productivity scale ~1.
        self.assertAlmostEqual(f["productivity_scale"], 1.0, places=6)
        self.assertAlmostEqual(f["profit_scale"], 0.5, places=6)

    def test_upgrade_preserves_legacy_fields(self):
        rows = [w(f"2026-01-{i+1:02d}T00:00:00+00:00", .2, .2, 10, 10) for i in range(25)]
        legacy = {"schema": "pionex_paired_cycle_calibration_v1", "profit_scale_applied": 0.9, "rounds_scale_applied": 0.8, "holdout_validation": {"profit_mape_pct": 999.0, "rounds_mape_pct": 999.0}, "windows": rows}
        out = q.upgrade_calibration(legacy)
        self.assertEqual(out["profit_scale_applied"], 0.9)
        self.assertEqual(out["rounds_scale_applied"], 0.8)
        self.assertEqual(out["schema"], "pionex_paired_cycle_calibration_v1")
        self.assertIn("calibration_v2", out)
        self.assertEqual(out["calibration_quality_v2"]["mape_role"], "SECONDARY_DIAGNOSTIC_ONLY")


    def test_challenger_must_pareto_dominate_legacy_holdout(self):
        challenger = {
            "profit": {"wape_pct": 46.0, "mae": 0.14},
            "rounds": {"wape_pct": 44.0, "mae": 8.2},
        }
        champion = {
            "profit": {"wape_pct": 44.0, "mae": 0.13},
            "rounds": {"wape_pct": 43.0, "mae": 8.0},
        }
        gate = q.select_calibration_model(challenger, champion)
        self.assertFalse(gate["challenger_promoted"])
        self.assertEqual(gate["selected_model"], "LEGACY_GLOBAL_MEDIAN")
        self.assertFalse(gate["prospective_used_for_selection"])

        better = {
            "profit": {"wape_pct": 40.0, "mae": 0.12},
            "rounds": {"wape_pct": 41.0, "mae": 7.5},
        }
        gate2 = q.select_calibration_model(better, champion)
        self.assertTrue(gate2["challenger_promoted"])
        self.assertEqual(gate2["selected_model"], "V2_COHORT_TWO_STAGE")

    def test_target_factors_respect_legacy_champion_gate(self):
        rows = [
            q.add_geometry_descriptors(
                w(f"2026-01-{i+1:02d}T00:00:00+00:00", .2, .2, 10, 10)
            )
            for i in range(10)
        ]
        calibration = {
            "profit_scale_applied": 0.9,
            "rounds_scale_applied": 0.8,
            "calibration_v2": {
                "windows": rows,
                "regime_thresholds_fit_only": {},
                "model_selection": {
                    "selected_model": "LEGACY_GLOBAL_MEDIAN",
                    "challenger_promoted": False,
                },
            },
        }
        factors = q.target_factors(
            calibration,
            w("2026-02-01T00:00:00+00:00", .2, .2, 10, 10),
        )
        self.assertEqual(factors["profit_scale"], 0.9)
        self.assertEqual(factors["rounds_scale"], 0.8)
        self.assertEqual(factors["method"], "legacy_global_guarded_by_holdout")
        self.assertEqual(factors["cohort"]["fallback_level"], "LEGACY_GLOBAL_GUARDED")
        self.assertIn("challenger", factors)

    def test_mape_alone_does_not_force_weak_status(self):
        metrics = {
            "profit": {"wape_pct": 30.0, "mae": 0.04, "bias": 0.01, "mape_pct": 500.0},
            "rounds": {"wape_pct": 30.0, "mae": 3.0, "bias": 1.0, "mape_pct": 500.0},
        }
        status = q.quality_status(metrics, fit_n=30, holdout_n=6, low_share=0.1, cohort_fallbacks=["SIMILAR_GEOMETRY"] * 6)
        self.assertEqual(status["status"], "CALIBRATION_STRONG")
        self.assertEqual(status["mape_role"], "SECONDARY_DIAGNOSTIC_ONLY")

    def test_acceptable_wape_does_not_overstate_magnitude_precision(self):
        metrics = {
            "profit": {"wape_pct": 52.0, "mae": 0.10, "bias": 0.05},
            "rounds": {"wape_pct": 52.0, "mae": 9.0, "bias": 5.0},
        }
        status = q.quality_status(metrics, fit_n=34, holdout_n=6, low_share=0.18, cohort_fallbacks=["GLOBAL", "SIMILAR_GEOMETRY"])
        self.assertEqual(status["status"], "CALIBRATION_ACCEPTABLE")
        self.assertTrue(status["directional_signal_reliable"])
        self.assertFalse(status["magnitude_estimate_reliable"])


if __name__ == "__main__":
    unittest.main()
