#!/usr/bin/env python3
"""Continuous 0-1 scoring for P14 measurements."""

from __future__ import annotations

import hashlib
import json

from continuous_scoring import SCORE_FLOOR, SCORE_VERSION, assert_score_contract, evidence_fraction, high_quality, measurable_floor, residual_quality, weighted_geometric


DIMENSION_WEIGHTS = {"setup": .15, "structural": .20, "periodicity": .25, "pendulum_law": .40}
SCALES = {
    "initial_length_ratio_error": .12,
    "release_time_difference_s": .50,
    "pivot_drift_fraction": .045,
    "string_length_change_fraction": .15,
    "pivot_separation_change_fraction": .035,
    "sinusoid_r2_shortfall": .30,
    "period_cv": .20,
    "m1_pendulum_law": .20,
}


def stable_payload(record: dict) -> str:
    payload = {
        "status": record.get("status"),
        "metrics": record.get("metrics"),
        "pendulums": record.get("pendulums"),
        "camera_audit": record.get("camera_audit"),
        "errors": record.get("errors"),
        "warnings": record.get("warnings"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score_record(record: dict) -> dict:
    before = stable_payload(record)
    if "legacy_dimension_scores_0_100" not in record:
        record["legacy_dimension_scores_0_100"] = record.get("dimension_scores", {})
    if "legacy_overall_score_0_100" not in record:
        record["legacy_overall_score_0_100"] = record.get("overall_score", 0.0)

    status = record.get("status", {})
    valid = bool(status.get("measurement_valid"))
    metrics = record.get("metrics", {})
    pendulums = record.get("pendulums", {})
    short = pendulums.get("short", {})
    long = pendulums.get("long", {})
    short_period = short.get("period") or {}
    long_period = long.get("period") or {}

    length_ratio = metrics.get("initial_length_ratio_short_over_long")
    length_error = abs(float(length_ratio) - .5) if length_ratio is not None else None
    coverages = [short.get("coverage"), long.get("coverage")]
    coverage = min(float(x) for x in coverages if x is not None) if all(x is not None for x in coverages) else 0.0
    fit_values = [short_period.get("fit_r2"), long_period.get("fit_r2")]
    fit_r2 = min(float(x) for x in fit_values if x is not None) if all(x is not None for x in fit_values) else None
    cycles = min(float(short_period.get("observed_cycles") or 0.0), float(long_period.get("observed_cycles") or 0.0))

    components = {
        "setup": {
            "length_ratio": residual_quality(length_error, SCALES["initial_length_ratio_error"]),
            "release_sync": residual_quality(metrics.get("release_time_difference_s"), SCALES["release_time_difference_s"]),
            "tracking_coverage": measurable_floor(coverage),
        },
        "structural": {
            "pivot_stability": residual_quality(metrics.get("pivot_drift_short_length_fraction_p95"), SCALES["pivot_drift_fraction"]),
            "string_length_stability": residual_quality(metrics.get("string_length_relative_change_p95_max"), SCALES["string_length_change_fraction"]),
            "pivot_separation_stability": residual_quality(metrics.get("pivot_separation_relative_change_p95"), SCALES["pivot_separation_change_fraction"]),
        },
        "periodicity": {
            "sinusoid_fit": high_quality(fit_r2, 1.0, SCALES["sinusoid_r2_shortfall"]),
            "period_consistency": residual_quality(metrics.get("maximum_period_cv"), SCALES["period_cv"]),
            "observed_cycles": evidence_fraction(cycles, 1.0),
        },
        "pendulum_law": {
            "period_length_relation": residual_quality(metrics.get("m1_abs_residual"), SCALES["m1_pendulum_law"]),
        },
    }
    component_weights = {
        "setup": {"length_ratio": .55, "release_sync": .25, "tracking_coverage": .20},
        "structural": {"pivot_stability": .35, "string_length_stability": .40, "pivot_separation_stability": .25},
        "periodicity": {"sinusoid_fit": .45, "period_consistency": .35, "observed_cycles": .20},
        "pendulum_law": {"period_length_relation": 1.0},
    }
    dimensions = {key: weighted_geometric(values, component_weights[key]) for key, values in components.items()}
    overall = weighted_geometric(dimensions, DIMENSION_WEIGHTS) if valid else 0.0
    if not valid:
        components = {key: {name: 0.0 for name in values} for key, values in components.items()}
        dimensions = {key: 0.0 for key in dimensions}
    assert_score_contract([*dimensions.values(), overall], valid, overall)

    record["score_version"] = SCORE_VERSION
    record["score_floor_for_measurable"] = SCORE_FLOOR
    record["score_scales"] = SCALES
    record["score_components"] = components
    record["dimension_scores"] = dimensions
    record["overall_score"] = overall
    if stable_payload(record) != before:
        raise AssertionError(f"rescoring changed metrics/status for {record.get('sample_id')}")
    return record
