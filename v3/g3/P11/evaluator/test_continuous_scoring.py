#!/usr/bin/env python3
"""Boundary, monotonicity, and continuity tests for continuous-0-1-v1."""

from __future__ import annotations

import copy
import json

from continuous_scoring import (
    DIMENSION_WEIGHTS,
    SCORING_VERSION,
    build_continuous_scores,
    migrate_record,
    q_residual,
    weighted_geometric_mean,
)


def fixture() -> dict:
    return {
        "video_id": "synthetic",
        "statuses": {"measurement_valid": True, "physics_pass": True, "structural_ok": True},
        "metrics": {
            "M1_snell_residual_abs": 0.12,
            "M2_intersection_disagreement_normalized": 0.025,
        },
        "geometry": {
            "surface": {"rmse_px": 3.0, "span_tank_fraction": 0.42},
            "incident_ray": {"rmse_px": 7.0, "span_tank_fraction": 0.09},
            "refracted_ray": {"rmse_px": 7.0, "span_tank_fraction": 0.10},
        },
        "temporal": {
            "stable_on_frame_indices": list(range(24)),
            "ray_stability": {
                "incident_angle_mad_deg": 4.0,
                "refracted_angle_mad_deg": 4.0,
                "sample_count": 7,
            },
        },
        "structure": {
            "camera": {
                "translation_diagonal_fraction": 0.02,
                "scale_error": 0.02,
                "rotation_deg": 1.0,
            }
        },
        "scores": {"overall_gated_score_0_100": 57.0, "M1_score_0_100": 45.0},
    }


def main() -> int:
    # Residual map boundaries and half-quality anchors.
    assert q_residual(0.0, 0.12) == 1.0
    assert abs(q_residual(0.12, 0.12) - 0.5) < 1.0e-12
    assert abs(q_residual(0.025, 0.025) - 0.5) < 1.0e-12
    assert 0.0 < q_residual(1.0e9, 0.12) < 1.0e-6
    sequence = [q_residual(value, 0.12) for value in (0.0, 0.03, 0.12, 0.30, 1.0)]
    assert all(left >= right for left, right in zip(sequence, sequence[1:])), sequence

    # No hard threshold step at either former pass threshold.
    epsilon = 1.0e-7
    assert abs(q_residual(0.12 - epsilon, 0.12) - q_residual(0.12 + epsilon, 0.12)) < 1.0e-5
    assert abs(q_residual(0.025 - epsilon, 0.025) - q_residual(0.025 + epsilon, 0.025)) < 1.0e-5

    # Geometric aggregation is bounded and the requested 0.01 floor prevents a
    # single zero input from forcing a valid result to zero.
    ones = {key: 1.0 for key in DIMENSION_WEIGHTS}
    assert abs(weighted_geometric_mean(ones, DIMENSION_WEIGHTS) - 1.0) < 1.0e-12
    one_zero = dict(ones)
    one_zero["M1_snell_quality"] = 0.0
    floored = weighted_geometric_mean(one_zero, DIMENSION_WEIGHTS)
    assert 0.0 < floored < 1.0

    record = fixture()
    scores = build_continuous_scores(record)
    assert scores["scoring_version"] == SCORING_VERSION
    assert all(0.0 <= value <= 1.0 for value in scores["dimensions"].values())
    assert 0.0 < scores["overall"] <= 1.0

    # Overall score is monotone in each residual when all other inputs are held
    # fixed, and remains continuous across the old pass boundaries.
    better = fixture()
    better["metrics"]["M1_snell_residual_abs"] = 0.03
    worse = fixture()
    worse["metrics"]["M1_snell_residual_abs"] = 0.30
    assert build_continuous_scores(better)["overall"] > build_continuous_scores(worse)["overall"]
    below = fixture()
    above = fixture()
    below["metrics"]["M1_snell_residual_abs"] = 0.12 - epsilon
    above["metrics"]["M1_snell_residual_abs"] = 0.12 + epsilon
    assert abs(build_continuous_scores(below)["overall"] - build_continuous_scores(above)["overall"]) < 1.0e-5
    below["metrics"]["M2_intersection_disagreement_normalized"] = 0.025 - epsilon
    above["metrics"]["M2_intersection_disagreement_normalized"] = 0.025 + epsilon
    assert abs(build_continuous_scores(below)["overall"] - build_continuous_scores(above)["overall"]) < 1.0e-5

    # Validity is the only zeroing gate. Hard physics labels are retained but
    # do not introduce a numeric score step.
    invalid = fixture()
    invalid["statuses"]["measurement_valid"] = False
    invalid["statuses"]["physics_pass"] = None
    assert build_continuous_scores(invalid)["overall"] == 0.0
    failing_label = fixture()
    failing_label["statuses"]["physics_pass"] = False
    assert build_continuous_scores(failing_label)["overall"] > 0.0

    # Migration preserves old scoring and all hard labels, and is idempotent.
    original = fixture()
    statuses_before = copy.deepcopy(original["statuses"])
    old_scores = copy.deepcopy(original["scores"])
    migrated = migrate_record(original)
    assert migrated["statuses"] == statuses_before
    assert migrated["legacy_scores_0_100"] == old_scores
    assert migrated["score_version"] == SCORING_VERSION
    assert migrated["scores"]["scoring_version"] == SCORING_VERSION
    first_legacy = copy.deepcopy(migrated["legacy_scores_0_100"])
    migrate_record(migrated)
    assert migrated["legacy_scores_0_100"] == first_legacy
    print(
        json.dumps(
            {
                "scoring_version": SCORING_VERSION,
                "tests": [
                    "q boundary anchors",
                    "monotonic residual map",
                    "no threshold discontinuity",
                    "0.01 geometric input floor",
                    "valid/invalid zero gate",
                    "legacy preservation and idempotence",
                ],
                "status": "PASS",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
