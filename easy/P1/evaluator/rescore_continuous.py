#!/usr/bin/env python3
"""Rescore frozen P1 measurements with continuous 0–1 quality functions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from continuous_scoring import (
    SCORE_FLOOR,
    SCORE_VERSION,
    assert_score_contract,
    evidence_fraction,
    measurable_floor,
    residual_quality,
    weighted_geometric,
)


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


def flatten(value: dict, prefix: str = "") -> dict:
    out: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(flatten(item, name))
        elif isinstance(item, list):
            out[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            out[name] = item
    return out


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


def self_test() -> None:
    assert residual_quality(0.0, .2) == 1.0
    assert residual_quality(.1, .2) > residual_quality(.2, .2) > residual_quality(.4, .2) > 0
    left = residual_quality(.2 - 1e-9, .2)
    right = residual_quality(.2 + 1e-9, .2)
    assert abs(left - right) < 1e-7
    assert_score_contract([0.01, 1.0], True, 0.01)
    assert_score_contract([0.0], False, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    self_test()
    paths = sorted((args.results / "json").glob("P1_*_seed*.json"))
    if len(paths) != 24:
        raise SystemExit(f"expected 24 P1 records, found {len(paths)}")
    records = []
    for path in paths:
        record = score_record(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)

    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with (args.results / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    values = [record["overall_score"] for record in records]
    valid_values = [record["overall_score"] for record in records if record.get("measurement_valid")]
    summary_path = args.results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    legacy_mean = summary.get("legacy_end_to_end_mean_score_0_100")
    if legacy_mean is None:
        legacy_mean = summary.get("end_to_end_mean_score")
    current_mean = statistics.fmean(values)
    summary.update({
        "score_version": SCORE_VERSION,
        "score_range": [0.0, 1.0],
        "zero_score_policy": "measurement_invalid_only",
        "end_to_end_mean_score": current_mean,
        "legacy_end_to_end_mean_score_0_100": legacy_mean,
        "continuous_score_mean_all": current_mean,
        "continuous_score_mean_valid": statistics.fmean(valid_values) if valid_values else 0.0,
        "continuous_score_median_valid": statistics.median(valid_values) if valid_values else 0.0,
        "continuous_score_min_valid": min(valid_values) if valid_values else 0.0,
        "continuous_score_max_valid": max(valid_values) if valid_values else 0.0,
        "continuous_score_zero_count": sum(value == 0 for value in values),
        "status_and_measurements_preserved": True,
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key.startswith("continuous_") or key in {"score_version", "continuous_score_zero_count"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
