#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
import unittest

from rescore_continuous import (
    SCORE_FLOOR,
    count_quality,
    residual_quality,
    rescore_result,
    weighted_geometric_mean,
)


CONFIG = {
    "score_weights": {
        "task": 0.15, "parabola": 0.20, "horizontal": 0.15,
        "vertical": 0.20, "pair_consistency": 0.30,
    },
    "thresholds": {
        "sync_frames": {"bad": 4.0}, "angle_error_deg": {"bad": 12.0},
        "landing_level_d": {"bad": 0.50},
        "camera_translation_diag": {"bad": 0.020},
        "parabola_rmse_d": {"bad": 0.30}, "parabola_p95_d": {"bad": 0.60},
        "x_fit_rmse_d": {"bad": 0.30}, "vx_robust_cv": {"bad": 0.20},
        "y_fit_rmse_d": {"bad": 0.30}, "vy_line_residual": {"bad": 0.20},
        "ay_robust_cv": {"bad": 0.35}, "range_ratio_error": {"bad": 0.15},
        "initial_speed_error": {"bad": 0.20}, "gravity_error": {"bad": 0.30},
    },
}


def synthetic_result(measurement_valid: bool = True, residual: float = 0.1) -> dict:
    per_ball = {
        slot: {
            "parabola_rmse_d": residual, "parabola_p95_d": residual,
            "x_fit_rmse_d": residual, "vx_robust_cv": residual,
            "y_fit_rmse_d": residual, "vy_line_residual": residual,
            "ay_robust_cv": residual,
        }
        for slot in ("upper", "lower")
    }
    return {
        "extract_success": True, "structural_ok": False,
        "measurement_valid": measurement_valid, "physics_pass": False,
        "scores": {"task": 1.0, "end_to_end": 0.0},
        "measurements": {"balls": {
            "upper": {"radius_robust_cv": residual},
            "lower": {"radius_robust_cv": residual},
        }},
        "metrics": {
            "M2": {"per_ball": per_ball}, "release_sync_frames": residual,
            "max_angle_error_deg": residual, "max_landing_level_error_d": residual,
            "M1_abs": residual, "initial_speed_error": residual,
            "gravity_symmetric_error": residual,
        },
        "qc": {
            "camera": {"max_translation_diag": residual},
            "balls": {
                "upper": {"backend_disagreement_median_radii": residual,
                          "fused_coverage": 1.0 - residual},
                "lower": {"backend_disagreement_median_radii": residual,
                          "fused_coverage": 1.0 - residual},
            },
        },
    }


class PrimitiveTests(unittest.TestCase):
    def test_residual_anchor_and_monotonicity(self) -> None:
        self.assertEqual(residual_quality(0.0, 2.0), 1.0)
        self.assertEqual(residual_quality(2.0, 2.0), 0.5)
        qualities = [residual_quality(value, 1.0) for value in (0, 0.1, 1, 10, 1e9)]
        self.assertTrue(all(a > b for a, b in zip(qualities, qualities[1:])))
        self.assertAlmostEqual(residual_quality(-2.0, 2.0), 0.5)
        self.assertEqual(residual_quality(float("inf"), 1.0), 0.0)
        self.assertEqual(residual_quality(1.0, 0.0), 0.0)

    def test_count_boundaries(self) -> None:
        self.assertEqual(count_quality(0, 4), SCORE_FLOOR)
        self.assertEqual(count_quality(2, 4), 0.5)
        self.assertEqual(count_quality(4, 4), 1.0)
        self.assertEqual(count_quality(8, 4), 1.0)
        self.assertEqual(count_quality(-2, 4), SCORE_FLOOR)
        self.assertEqual(count_quality(None, 4), SCORE_FLOOR)

    def test_weighted_gm_boundaries_and_monotonicity(self) -> None:
        self.assertEqual(weighted_geometric_mean([1, 1], [1, 1]), 1.0)
        self.assertAlmostEqual(weighted_geometric_mean([0, 0], [1, 1]), SCORE_FLOOR)
        low = weighted_geometric_mean([0.2, 0.4], [1, 1])
        high = weighted_geometric_mean([0.3, 0.4], [1, 1])
        self.assertGreater(high, low)
        with self.assertRaises(ValueError):
            weighted_geometric_mean([], [])


class ResultTests(unittest.TestCase):
    def test_valid_hard_failure_is_not_zeroed(self) -> None:
        source = synthetic_result(measurement_valid=True, residual=1000000.0)
        rescored = rescore_result(source, CONFIG)
        self.assertFalse(rescored["structural_ok"])
        self.assertGreaterEqual(rescored["scores"]["overall"], SCORE_FLOOR)
        self.assertGreater(rescored["scores"]["overall"], 0.0)
        self.assertEqual(rescored["scores_legacy"], source["scores"])

    def test_invalid_measurement_has_exact_zero_overall(self) -> None:
        rescored = rescore_result(synthetic_result(measurement_valid=False), CONFIG)
        self.assertEqual(rescored["scores"]["overall"], 0.0)
        for name, value in rescored["scores"].items():
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)

    def test_every_dimension_is_bounded(self) -> None:
        for residual in (0.0, 0.1, 1.0, 1e12, math.inf):
            rescored = rescore_result(synthetic_result(True, residual), CONFIG)
            for name, value in rescored["scores"].items():
                self.assertTrue(0.0 <= value <= 1.0, (name, value))

    def test_rescore_is_monotone_for_uniform_residual_worsening(self) -> None:
        good = rescore_result(synthetic_result(True, 0.01), CONFIG)["scores"]
        bad = rescore_result(synthetic_result(True, 0.20), CONFIG)["scores"]
        for name in good:
            self.assertGreater(good[name], bad[name], name)

    def test_status_and_physics_pass_are_preserved(self) -> None:
        source = synthetic_result(True, 0.1)
        source["extract_success"] = False
        source["physics_pass"] = True
        before = copy.deepcopy(source)
        rescored = rescore_result(source, CONFIG)
        self.assertEqual(rescored["extract_success"], before["extract_success"])
        self.assertEqual(rescored["measurement_valid"], before["measurement_valid"])
        self.assertEqual(rescored["physics_pass"], before["physics_pass"])
        self.assertEqual(rescored["score_version"], "continuous-0-1-v1")


if __name__ == "__main__":
    unittest.main()
