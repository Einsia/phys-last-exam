"""Pure-standard-library regression tests for G7 proxy-v1."""
from __future__ import annotations

import math
import unittest

from evaluator.contract import _error_proxy, _geometric_mean, _proxy_scores


class ProxyContractTests(unittest.TestCase):
    def test_error_anchor_and_monotonicity(self) -> None:
        scores = [_error_proxy(error, 0.10) for error in (0.0, 0.05, 0.10, 0.20)]
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[2], 0.5)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[3])

    def test_geometric_mean_has_hard_zero(self) -> None:
        self.assertEqual(_geometric_mean([0.8, 0.0]), 0.0)
        self.assertAlmostEqual(_geometric_mean([0.25, 1.0]), 0.5)

    def test_p12_ray_order_invariance(self) -> None:
        forward = {"per_ray_snell_residual": [0.02, 0.05, 0.10]}
        reverse = {"per_ray_snell_residual": list(reversed(forward["per_ray_snell_residual"]))}
        a = _proxy_scores("P12", 0.03, forward, {}, m1_available=True, m2_available=True)[1]
        b = _proxy_scores("P12", 0.03, reverse, {}, m1_available=True, m2_available=True)[1]
        self.assertTrue(math.isfinite(a))
        self.assertAlmostEqual(a, b, places=15)

    def test_p12_coverage_is_multiplicative(self) -> None:
        three = {"per_ray_snell_residual": [0.05, 0.05, 0.05]}
        two = {"per_ray_snell_residual": [0.05, 0.05]}
        full = _proxy_scores("P12", 0.03, three, {}, m1_available=True, m2_available=True)[1]
        partial = _proxy_scores("P12", 0.03, two, {}, m1_available=True, m2_available=True)[1]
        self.assertAlmostEqual(partial, (2.0 / 3.0) * full, places=15)


if __name__ == "__main__":
    unittest.main()
