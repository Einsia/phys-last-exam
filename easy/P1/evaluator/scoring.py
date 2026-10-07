#!/usr/bin/env python3
"""Continuous 0-1 scoring for P1 measurements."""

from __future__ import annotations

import hashlib
import json

from continuous_scoring import SCORE_FLOOR, SCORE_VERSION, assert_score_contract, evidence_fraction, measurable_floor, residual_quality, weighted_geometric


DIMENSION_WEIGHTS = {
    "structure": 0.20,
    "monotonic_energy_loss": 0.20,
    "ballistic_arc_quality": 0.15,
    "restitution_consistency": 0.35,
    "spatial_stability": 0.10,
}

SCALES = {
    "height_monotonic_error": 0.015,
    "near_nondecreasing_fraction": 0.25,
    "time_monotonic_excess": 0.20,
    "arc_shape_rmse": 0.30,
    "apex_time_asymmetry": 0.30,
    "m1_height_time_consistency": 0.18,
    "restitution_cv": 0.35,
    "impact_x_spread_radii": 1.50,
    "arc_x_drift_radii": 2.00,
    "camera_drift_fraction": 0.020,
    "backend_disagreement_radii": 2.00,
}


def stable_payload(record: dict) -> str:
    payload = {
        "extract_success": record.get("extract_success"),
        "structural_ok": record.get("structural_ok"),
        "measurement_valid": record.get("measurement_valid"),
        "physics_pass": record.get("physics_pass"),
        "measurements": record.get("measurements"),
        "metrics": record.get("metrics"),
        "qc": record.get("qc"),
        "failure_reason": record.get("failure_reason"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score_record(record: dict) -> dict:
    before = stable_payload(record)
    if "legacy_scores_0_100" not in record:
        record["legacy_scores_0_100"] = record.get("scores", {})

    valid = bool(record.get("measurement_valid", record.get("extract_success")))
    metrics = record.get("metrics", {})
    measurements = record.get("measurements", {})
    tracking = record.get("qc", {}).get("tracking", {})
    arcs = float(measurements.get("complete_rebound_arc_count") or 0.0)
    required = float(measurements.get("required_rebound_arc_count") or 4.0)

    components = {
        "structure": {
            "complete_arc_fraction": evidence_fraction(arcs, required),
            "track_coverage": measurable_floor(float(tracking.get("fused_coverage") or 0.0)),
            "track_confidence": measurable_floor(float(tracking.get("confident_coverage") or 0.0)),
        },
        "monotonic_energy_loss": {
            "height_error": residual_quality(metrics.get("M2_monotonic_height_error"), SCALES["height_monotonic_error"]),
            "near_nondecreasing_fraction": residual_quality(metrics.get("M2_near_nondecreasing_height_fraction"), SCALES["near_nondecreasing_fraction"]),
            "time_error": residual_quality(metrics.get("M2_monotonic_time_excess"), SCALES["time_monotonic_excess"]),
        },
        "ballistic_arc_quality": {
            "arc_shape": residual_quality(metrics.get("arc_shape_normalized_rmse"), SCALES["arc_shape_rmse"]),
            "apex_symmetry": residual_quality(metrics.get("apex_time_asymmetry"), SCALES["apex_time_asymmetry"]),
        },
        "restitution_consistency": {
            "height_time_agreement": residual_quality(metrics.get("M1_height_time_restitution_consistency"), SCALES["m1_height_time_consistency"]),
            "coefficient_stability": residual_quality(metrics.get("M2_restitution_coefficient_cv"), SCALES["restitution_cv"]),
        },
        "spatial_stability": {
            "impact_position": residual_quality(metrics.get("impact_x_robust_spread_radii"), SCALES["impact_x_spread_radii"]),
            "arc_horizontal_drift": residual_quality(metrics.get("mean_arc_horizontal_drift_radii"), SCALES["arc_x_drift_radii"]),
            "camera": residual_quality(metrics.get("camera_drift_fraction_of_diagonal"), SCALES["camera_drift_fraction"]),
            "backend_agreement": residual_quality(metrics.get("best_backend_pair_disagreement_radii"), SCALES["backend_disagreement_radii"]),
        },
    }

    component_weights = {
        "structure": {"complete_arc_fraction": .60, "track_coverage": .20, "track_confidence": .20},
        "monotonic_energy_loss": {"height_error": .40, "near_nondecreasing_fraction": .40, "time_error": .20},
        "ballistic_arc_quality": {"arc_shape": .70, "apex_symmetry": .30},
        "restitution_consistency": {"height_time_agreement": .65, "coefficient_stability": .35},
        "spatial_stability": {"impact_position": .45, "arc_horizontal_drift": .25, "camera": .15, "backend_agreement": .15},
    }
    dimensions = {
        key: weighted_geometric(values, component_weights[key])
        for key, values in components.items()
    }
    overall = weighted_geometric(dimensions, DIMENSION_WEIGHTS) if valid else 0.0
    scores = {**dimensions, "overall": overall}
    assert_score_contract(scores.values(), valid, overall)

    record["score_version"] = SCORE_VERSION
    record["score_floor_for_measurable"] = SCORE_FLOOR
    record["score_scales"] = SCALES
    record["score_components"] = components if valid else {
        key: {name: 0.0 for name in values} for key, values in components.items()
    }
    record["scores"] = scores if valid else {key: 0.0 for key in scores}
    record["overall_score"] = overall
    if stable_payload(record) != before:
        raise AssertionError(f"rescoring changed measurements/status for {record.get('sample_id')}")
    return record
