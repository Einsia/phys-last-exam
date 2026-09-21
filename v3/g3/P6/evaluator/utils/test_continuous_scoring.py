#!/usr/bin/env python3
"""Boundary, monotonicity, and no-threshold-jump tests for P6 rescoring."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import yaml

import rescore_continuous as scoring


HERE = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((HERE / "continuous_score_config_v1.yaml").read_text())


class ContinuousPrimitiveTests(unittest.TestCase):
    def test_residual_boundaries(self) -> None:
        for scale in (0.02, 0.18, 0.25, 0.35, 1.0, 12.0):
            self.assertEqual(scoring.residual_quality(0.0, scale), 1.0)
            self.assertAlmostEqual(scoring.residual_quality(scale, scale), 0.5, places=12)
            self.assertGreater(scoring.residual_quality(1e9 * scale, scale), 0.0)

    def test_evidence_boundaries(self) -> None:
        for scale in (1.0, 12.0, 24.0):
            self.assertEqual(scoring.evidence_quality(0.0, scale), 0.0)
            self.assertAlmostEqual(scoring.evidence_quality(scale, scale), 0.5, places=12)
            self.assertLess(scoring.evidence_quality(1e9 * scale, scale), 1.0)

    def test_coverage_legacy_boundary_is_half_quality(self) -> None:
        track_spec = CONFIG["dimensions"]["task_structure"]["components"]["track_coverage"]
        pattern_spec = CONFIG["dimensions"]["task_structure"]["components"]["pattern_coverage"]
        self.assertAlmostEqual(scoring.component_quality(0.85, track_spec), 0.5, places=12)
        self.assertAlmostEqual(scoring.component_quality(0.65, pattern_spec), 0.5, places=12)

    def test_monotonicity(self) -> None:
        residual = [scoring.residual_quality(value, 0.25) for value in (0, 0.1, 0.25, 1, 10)]
        evidence = [scoring.evidence_quality(value, 12) for value in (0, 1, 12, 24, 120)]
        self.assertTrue(all(a > b for a, b in zip(residual, residual[1:])))
        self.assertTrue(all(a < b for a, b in zip(evidence, evidence[1:])))

    def test_no_threshold_jump(self) -> None:
        for scale in (0.02, 0.18, 0.25, 0.35, 1.0, 12.0, 24.0):
            epsilon = max(1e-9, scale * 1e-7)
            left = scoring.residual_quality(scale - epsilon, scale)
            right = scoring.residual_quality(scale + epsilon, scale)
            self.assertLess(abs(left - right), 1e-6)
            left = scoring.evidence_quality(scale - epsilon, scale)
            right = scoring.evidence_quality(scale + epsilon, scale)
            self.assertLess(abs(left - right), 1e-6)

    def test_geometric_floor(self) -> None:
        value = scoring.weighted_geometric_mean({"a": 0.0, "b": 1.0}, {"a": 1, "b": 1}, 0.01)
        self.assertAlmostEqual(value, 0.1, places=12)


class FormalResultContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = HERE / "results_v1"
        cls.paths = sorted(cls.root.glob("P6_*_seed*/result.json"))

    def test_all_24_results(self) -> None:
        self.assertEqual(len(self.paths), 24)

    def test_score_contract_and_legacy_labels(self) -> None:
        for path in self.paths:
            result = json.loads(path.read_text())
            original_physics_label = result["physics_pass"]
            rescored = scoring.rescore_result(copy.deepcopy(result), CONFIG, "test-config-hash")
            self.assertEqual(rescored["physics_pass"], original_physics_label)
            self.assertIn("legacy_score", rescored)
            for name, value in rescored["scores"].items():
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)
            if rescored["measurement_valid"]:
                self.assertGreaterEqual(rescored["scores"]["overall"], 0.01)
            else:
                self.assertEqual(rescored["scores"]["overall"], 0.0)

    def test_dimension_monotonic_on_formal_metric_vector(self) -> None:
        result = json.loads(self.paths[0].read_text())
        metrics = result["metrics"]
        base, _ = scoring.score_metrics(metrics, CONFIG, True)
        worse = copy.deepcopy(metrics)
        worse["M1_rolling_ratio_abs"] = float(metrics["M1_rolling_ratio_abs"]) + 0.25
        changed, _ = scoring.score_metrics(worse, CONFIG, True)
        self.assertLess(changed["pure_rolling_ratio"], base["pure_rolling_ratio"])
        self.assertLess(changed["overall"], base["overall"])

    def test_rescore_does_not_import_tracking_stack(self) -> None:
        source = (HERE / "rescore_continuous.py").read_text(encoding="utf-8")
        self.assertNotIn("import cv2", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("read_video", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
